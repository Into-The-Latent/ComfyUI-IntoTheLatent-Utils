# Whisper Loader node — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Downloads an official OpenAI Whisper checkpoint on first use into models/whisper/whisper-<name>/,
# loads it through transformers and keeps one model resident. The node emits the WHISPER *key*
# (model name, device), not the model — see resolve_handle(). Design:
# docs/superpowers/specs/2026-09-15-whisper-transcribe-design.md. transformers is imported lazily.
import gc
import os

import folder_paths
from comfy_api.latest import io

from .whisper_core import (
    DEFAULT_MODEL, DEVICE_CHOICES, DOWNLOAD_PATTERNS, MODEL_NAMES, MODEL_SIZES, MODELS, WhisperHandle,
    cache_key, dtype_for, missing_files, resolve_device, snapshot_dirname,
)

# Namespaced on purpose: ComfyUI matches links by type name globally, and a bare "WHISPER" from
# another pack would plug into our `model` input with a payload resolve_handle() cannot unpack.
WHISPER = io.Custom("ITL_WHISPER")

_MODELS_SUBDIR = "whisper"
folder_paths.add_model_folder_path(_MODELS_SUBDIR, os.path.join(folder_paths.models_dir, _MODELS_SUBDIR))

_CACHE: dict = {}          # cache_key -> WhisperHandle; at most one entry
INSTALL_HINT = ("Whisper needs the `transformers` package (>= 4.57, < 6) in ComfyUI's Python. Run ComfyUI "
                "Manager's 'Try fix' for ComfyUI-IntoTheLatent-Utils, or: pip install \"transformers>=4.57,<6\"")


def _snapshot_dir(name: str) -> str:
    # Built from models_dir directly: another pack may have registered its own "whisper" folder
    # first, and get_folder_paths("whisper")[0] would then point into that one.
    return os.path.join(folder_paths.models_dir, _MODELS_SUBDIR, snapshot_dirname(name))


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download
    snapshot_download(**kwargs)


def ensure_snapshot(name: str, ckpt_dir: str) -> str:
    """Download the checkpoint if any required file is missing; resumes partial downloads."""
    missing = missing_files(ckpt_dir)
    if not missing:
        return ckpt_dir
    repo = MODELS[name]
    print(f"[Whisper] downloading {repo} (~{MODEL_SIZES[name]}) to {ckpt_dir} ...")
    try:
        _snapshot_download(repo_id=repo, local_dir=ckpt_dir, allow_patterns=DOWNLOAD_PATTERNS)
    except Exception as e:
        raise RuntimeError(f"Whisper download to {ckpt_dir} failed: {e}") from e
    still = missing_files(ckpt_dir)
    if still:
        raise RuntimeError(f"Whisper download finished but files are missing in {ckpt_dir}: {', '.join(still)}")
    print("[Whisper] download done")
    return ckpt_dir


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def _load_pieces(ckpt_dir: str, device: str):
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    processor = WhisperProcessor.from_pretrained(ckpt_dir)
    # `dtype=` is the keyword on both transformers 4.57 and 5.x (verified).
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir, dtype=dtype_for(device)).to(device).eval()
    return processor, model


def _evict_all():
    _CACHE.clear()
    gc.collect()
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass


def unload():
    """Drop the resident Whisper model and give its memory back. ComfyUI's model manager cannot
    see `_CACHE`, so only this releases the weights (the transcribe node's `unload_after`)."""
    if _CACHE:
        print("[Whisper] unloading model")
    _evict_all()


def load_handle(name: str, device_choice: str) -> WhisperHandle:
    if name not in MODELS:
        raise ValueError(f"Unknown Whisper model {name!r}; expected one of {MODEL_NAMES}")
    device = resolve_device(device_choice, _cuda_available())
    key = cache_key(name, device)
    handle = _CACHE.get(key)
    if handle is not None:
        return handle
    ckpt_dir = ensure_snapshot(name, _snapshot_dir(name))
    _evict_all()
    processor, model = _load_pieces(ckpt_dir, device)
    handle = WhisperHandle(key, processor, model, device, dtype_for(device))
    _CACHE[key] = handle
    return handle


def resolve_handle(key) -> WhisperHandle:
    """WHISPER key (what the loader outputs) -> live handle: a cache hit is free, a miss reloads.
    The graph carries only this small tuple so ComfyUI's output cache holds no reference to the
    model tensors and unload() really frees them."""
    name, device = key
    return load_handle(name, device)


class ITLWhisperLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLWhisperLoader",
            display_name="ITL Whisper Loader",
            category="Into The Latent/audio",
            search_aliases=["whisper", "transcribe", "speech to text", "audio to text", "stt"],
            is_experimental=True,
            description="""
Loads an OpenAI Whisper speech-to-text model for ITL Whisper Transcribe.

First use downloads the checkpoint from Hugging Face into models/whisper/whisper-<model>/
(large-v3-turbo: 1.6 GB, large-v3: 3.1 GB, medium: 3.1 GB, small: 1 GB, base: 290 MB, tiny: 150 MB).
Runs in fp16 on CUDA (~2 GiB VRAM for large-v3-turbo) or fp32 on CPU.""",
            inputs=[
                io.Combo.Input("model", options=list(MODEL_NAMES), default=DEFAULT_MODEL,
                               tooltip="large-v3-turbo is the best speed/quality trade-off; large-v3 is the most "
                                       "accurate; smaller sizes are faster and less accurate."),
                io.Combo.Input("device", options=list(DEVICE_CHOICES), default="auto",
                               tooltip="auto = CUDA when available, else CPU."),
            ],
            outputs=[WHISPER.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model=DEFAULT_MODEL, device="auto") -> io.NodeOutput:
        return io.NodeOutput(load_handle(model, device).key)
