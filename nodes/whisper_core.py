# Whisper engine — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# ComfyUI-free helpers behind the Whisper loader / transcribe nodes (design:
# docs/superpowers/specs/2026-09-15-whisper-transcribe-design.md). Testable with stubs: no
# comfy_api, no folder_paths, no weights. transformers is only ever touched through the
# `processor` / `model` objects on a WhisperHandle, which the loader builds.
from __future__ import annotations

import os

import numpy as np
import torch

from .breeze_tts_core import audio_to_mono_numpy

# name -> Hugging Face repo. Dict order is the dropdown order; the first entry is the default.
MODELS = {
    "large-v3-turbo": "openai/whisper-large-v3-turbo",   # 809 M params, 1.6 GB fp16 — best speed/quality
    "large-v3": "openai/whisper-large-v3",               # 1.55 B, 3.1 GB fp16
    "medium": "openai/whisper-medium",                   # 769 M, 3.1 GB fp32
    "small": "openai/whisper-small",                     # 244 M, 967 MB fp32
    "base": "openai/whisper-base",                       # 74 M, 290 MB fp32
    "tiny": "openai/whisper-tiny",                       # 39 M, 151 MB fp32
}
MODEL_NAMES = tuple(MODELS)
DEFAULT_MODEL = MODEL_NAMES[0]

# Approximate download size of model.safetensors per model (fp16 for the two large ones, fp32 otherwise).
MODEL_SIZES = {"large-v3-turbo": "1.6 GB", "large-v3": "3.1 GB", "medium": "3.1 GB",
               "small": "1 GB", "base": "290 MB", "tiny": "150 MB"}

# ISO 639-1 codes Whisper's tokenizer knows; "auto" (model-side detection) is added by the node.
LANGUAGES = ("en", "zh", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "uk", "tr", "ar", "hi",
             "ja", "ko", "vi", "id", "th", "sv", "da", "no", "fi", "cs", "el", "he", "hu", "ro")
DEVICE_CHOICES = ("auto", "cuda", "cpu")

# Only these are fetched. The large-v3 repo also holds Flax / TF / .bin / fp32 copies (~12 GB).
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
    "normalizer.json",
)
# The must-exist check and the download allow-list are the same set on purpose: every official
# openai/whisper-* repo ships all 11. If a checkpoint is ever added that lacks one, split the two
# lists rather than dropping the file from the allow-list.
DOWNLOAD_PATTERNS = list(REQUIRED_FILES)

TARGET_SR = 16000                     # Whisper's fixed input rate
LONG_FORM_SAMPLES = 30 * TARGET_SR    # above this, transformers' sequential long-form decoding


def snapshot_dirname(name: str) -> str:
    if name not in MODELS:
        raise ValueError(f"Unknown Whisper model {name!r}; expected one of {MODEL_NAMES}")
    return f"whisper-{name}"


def missing_files(ckpt_dir: str) -> list[str]:
    """Required files that are absent or empty (a killed download leaves 0-byte stubs)."""
    out = []
    for rel in REQUIRED_FILES:
        p = os.path.join(ckpt_dir, rel)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            out.append(rel)
    return out


def cache_key(name: str, device: str) -> tuple:
    return (str(name), str(device))


def resolve_device(choice: str, cuda_available: bool) -> str:
    if choice == "auto":
        return "cuda" if cuda_available else "cpu"
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not cuda_available:
            raise RuntimeError("Whisper device is set to 'cuda' but no CUDA device is available; use 'auto' or 'cpu'.")
        return "cuda"
    raise ValueError(f"Unknown device choice {choice!r}; expected one of {DEVICE_CHOICES}")


def dtype_for(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def _librosa_resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as e:
        raise ImportError("Whisper resampling needs `librosa` (in this pack's requirements.txt): "
                          "pip install -r custom_nodes/ComfyUI-IntoTheLatent-Utils/requirements.txt") from e
    return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)


def prepare_samples(audio: dict, resample=None) -> np.ndarray:
    """ComfyUI AUDIO dict -> 1-D float32 mono at 16 kHz (batch item 0, channels averaged)."""
    samples = audio_to_mono_numpy(audio["waveform"])
    if samples.size == 0:
        raise ValueError("audio is empty")
    sr = int(audio["sample_rate"])
    if sr != TARGET_SR:
        samples = (resample or _librosa_resample)(samples, sr, TARGET_SR)
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def normalise_text(text) -> str:
    return " ".join(text.split()) if isinstance(text, str) else ""


class WhisperHandle:
    """What the loader keeps resident: the processor + model and where they live."""

    def __init__(self, key, processor, model, device: str, dtype: torch.dtype):
        self.key = key
        self.processor = processor
        self.model = model
        self.device = device
        self.dtype = dtype


def transcribe(handle: WhisperHandle, audio: dict, language: str = "auto", resample=None) -> str:
    """One transcription. Verified call path on transformers 4.57 and 5.15 — see the plan's
    Global Constraints before changing any keyword here.

    <= 30 s: the feature extractor pads to Whisper's fixed 3000 mel frames (its default) and
    generate() runs short-form. > 30 s: `truncation=False, padding="longest"` keeps every frame
    and generate() runs transformers' sequential long-form decoding, which needs the attention
    mask and timestamps; condition_on_prev_tokens=False stops it looping on repeated phrases."""
    if language in (None, "", "auto"):
        lang = None
    elif language in LANGUAGES:
        lang = str(language)
    else:
        raise ValueError(f"Unknown language {language!r}; expected 'auto' or one of {LANGUAGES}")
    samples = prepare_samples(audio, resample=resample)
    if samples.shape[0] > LONG_FORM_SAMPLES:
        inputs = handle.processor(samples, sampling_rate=TARGET_SR, return_tensors="pt",
                                  truncation=False, padding="longest", return_attention_mask=True)
    else:
        inputs = handle.processor(samples, sampling_rate=TARGET_SR, return_tensors="pt")
    features = inputs["input_features"].to(handle.device, handle.dtype)
    extra = {}
    if "attention_mask" in inputs:
        extra["attention_mask"] = inputs["attention_mask"].to(handle.device)
    with torch.inference_mode():
        ids = handle.model.generate(features, task="transcribe", language=lang, return_timestamps=True,
                                    condition_on_prev_tokens=False, num_beams=1, **extra)
    text = handle.processor.batch_decode(ids, skip_special_tokens=True)[0]
    return normalise_text(text)
