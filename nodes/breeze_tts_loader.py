# Breeze TTS Loader node — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Loads Breeze TTS 2 (BreezeBlue, 3B, English + Chinese) and hands the generate nodes a
# BREEZE_TTS key (the immutable cache_key tuple, not the loaded model — see resolve_handle()
# below). The first run downloads the Hugging Face snapshot (~7.2 GB) into
# models/breeze_tts/Breeze-TTS-2/; later runs find it there. Design:
# docs/superpowers/specs/2026-09-14-breeze-tts-design.md. All Breeze imports are lazy so the
# pack loads even when the `breeze-tts` fork is not installed.
import gc
import os

import folder_paths
from comfy_api.latest import io

from .breeze_tts_core import (
    DOWNLOAD_IGNORE, REPO_ID, SNAPSHOT_DIRNAME, BreezeHandle, cache_key, missing_snapshot_files,
)

BREEZE_TTS = io.Custom("BREEZE_TTS")

_MODELS_SUBDIR = "breeze_tts"
folder_paths.add_model_folder_path(_MODELS_SUBDIR, os.path.join(folder_paths.models_dir, _MODELS_SUBDIR))

_CACHE: dict = {}          # cache_key -> BreezeHandle; at most one entry (one 7 GB model resident)
_LICENSE_PRINTED = False
INSTALL_HINT = ("Breeze TTS is not installed. Run ComfyUI Manager's 'Try fix' for "
                "ComfyUI-IntoTheLatent-Utils, or: pip install -r custom_nodes/ComfyUI-IntoTheLatent-Utils/requirements.txt"
                " (pip installs the model code from GitHub, so `git` must be on PATH; needs transformers "
                ">= 4.57, < 6 and torch >= 2.3; the install never upgrades or replaces torch)")


def _snapshot_dir() -> str:
    return os.path.join(folder_paths.get_folder_paths(_MODELS_SUBDIR)[0], SNAPSHOT_DIRNAME)


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download
    snapshot_download(**kwargs)


def ensure_snapshot(ckpt_dir: str) -> str:
    """Download the weights if any required file is missing; resumes partial downloads."""
    missing = missing_snapshot_files(ckpt_dir)
    if not missing:
        return ckpt_dir
    print(f"[Breeze TTS] downloading {REPO_ID} (~7.2 GB) to {ckpt_dir} ...")
    try:
        _snapshot_download(repo_id=REPO_ID, local_dir=ckpt_dir, ignore_patterns=list(DOWNLOAD_IGNORE))
    except Exception as e:
        raise RuntimeError(f"Breeze TTS download to {ckpt_dir} failed: {e}") from e
    still = missing_snapshot_files(ckpt_dir)
    if still:
        raise RuntimeError(f"Breeze TTS download finished but files are missing in {ckpt_dir}: {', '.join(still)}")
    print("[Breeze TTS] download done")
    return ckpt_dir


def _require_fast_path_support(fast_path: bool):
    """fast_path captures CUDA graphs around transformers' StaticCache. transformers >= 5 advances
    that cache from a Python integer, which a replayed graph never re-executes, so capture would
    silently write every step into the same slot. Refuse instead of producing garbage."""
    if not fast_path:
        return
    import transformers
    major = int(str(transformers.__version__).split(".")[0])
    if major >= 5:
        raise RuntimeError("Breeze TTS fast_path needs transformers 4.57.x; this environment has "
                           f"transformers {transformers.__version__}. Turn fast_path off.")


def _require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Breeze TTS needs an NVIDIA GPU (CUDA); the upstream runtime has no CPU path.")


def _load_pieces(ckpt_dir: str, attention: str):
    try:
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    from pathlib import Path
    tokenizer, model, audio_tokenizer = load_runtime(Path(ckpt_dir), device=resolve_device(),
                                                     attn_implementation=attention)
    update_generation_config_for_breeze(model)
    return tokenizer, model, audio_tokenizer


def _runtime_factory(fast_path: bool):
    try:
        from breeze_models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e

    def build(model, audio_tokenizer, tokenizer, kwargs):
        config = FastStreamingConfig(fast_all=True if fast_path else None, **kwargs)
        return FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)
    return build


def _evict_all():
    _CACHE.clear()
    gc.collect()
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass


def unload():
    """Drop the resident Breeze model and give its VRAM back.

    ComfyUI's model manager does not know about `_CACHE`, so it can never evict the ~7.7 GiB
    itself: without this, a workflow that runs Breeze and then an image/video model OOMs on the
    second model. Called by the generate nodes' `unload_after` toggle and by the Unload node."""
    if _CACHE:
        print("[Breeze TTS] unloading model")
    _evict_all()


def load_handle(attention: str, fast_path: bool) -> BreezeHandle:
    global _LICENSE_PRINTED
    _require_cuda()
    _require_fast_path_support(fast_path)
    ckpt_dir = ensure_snapshot(_snapshot_dir())
    key = cache_key(ckpt_dir, attention, fast_path)
    handle = _CACHE.get(key)
    if handle is not None:
        return handle
    _evict_all()
    if not _LICENSE_PRINTED:
        print("[Breeze TTS] weights are licensed for research and non-commercial use only "
              "(BreezeBlue Research and Non-Commercial License).")
        _LICENSE_PRINTED = True
    tokenizer, model, audio_tokenizer = _load_pieces(ckpt_dir, attention)
    handle = BreezeHandle(key, tokenizer, model, audio_tokenizer, _runtime_factory(fast_path))
    _CACHE[key] = handle
    return handle


def resolve_handle(key) -> BreezeHandle:
    """Turn a BREEZE_TTS key (what the loader node now outputs) back into a live BreezeHandle.

    A `_CACHE` hit returns the resident handle for free; a miss reloads. The graph carries only
    the small immutable key so that ComfyUI's output cache holds no second reference to the model
    tensors; `_CACHE` itself is the one thing that keeps them resident, and only unload() (via
    the generate nodes' `unload_after` toggle or the Unload node) releases them — ComfyUI's own
    cache modes and model manager cannot see or evict this cache."""
    _, attention, fast_path = key
    return load_handle(attention, fast_path)


class ITLBreezeTTSLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLBreezeTTSLoader",
            display_name="ITL Breeze TTS Loader",
            category="Into The Latent/audio",
            search_aliases=["breeze", "tts", "text to speech", "voice clone"],
            is_experimental=True,
            description="""
Loads Breeze TTS 2 (BreezeBlue — English + Chinese text-to-speech, voice clone / design / direction).

First run downloads the weights (~7.2 GB) from Hugging Face into models/breeze_tts/Breeze-TTS-2/.
Needs an NVIDIA GPU: ~7.7 GiB VRAM (fast_path off) or ~14.4 GiB (fast_path on).
Weights are research / non-commercial (BreezeBlue license).""",
            inputs=[
                io.Combo.Input("attention", options=["sdpa", "eager"], default="sdpa",
                               tooltip="Attention kernel. 'sdpa' is faster; 'eager' is upstream's reference path."),
                io.Boolean.Input("fast_path", default=False,
                                 tooltip="Capture CUDA graphs for every stage (upstream --fast-all). Faster "
                                         "generation, ~2x VRAM, longer first run. Changing sampling settings "
                                         "re-captures."),
            ],
            outputs=[BREEZE_TTS.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, attention="sdpa", fast_path=False) -> io.NodeOutput:
        return io.NodeOutput(load_handle(attention, fast_path).key)


class ITLBreezeTTSUnload(io.ComfyNode):
    """Pass-through node that frees the Breeze model. Wire the generated AUDIO through it so the
    unload happens after synthesis and before the next model in the workflow loads."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLBreezeTTSUnload",
            display_name="ITL Breeze TTS Unload",
            category="Into The Latent/audio",
            search_aliases=["breeze", "tts", "unload", "free vram"],
            is_experimental=True,
            description="""
Frees the Breeze TTS model from VRAM (~7.7 GiB) and passes the audio through unchanged.

Put it between a Breeze generate node and whatever uses the audio next, so an image or video
model later in the workflow has the memory. The next Breeze node reloads the weights (~20 s).
The generate nodes' `unload_after` toggle does the same thing without an extra node.""",
            inputs=[io.Audio.Input("audio", tooltip="Passed through unchanged.")],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        unload()
        return io.NodeOutput(audio)
