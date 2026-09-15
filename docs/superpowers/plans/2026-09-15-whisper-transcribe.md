# Whisper Transcribe Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `ITL Whisper Loader` (auto-downloading an official OpenAI Whisper checkpoint into `models/whisper/`) and an `ITL Whisper Transcribe` node (ComfyUI `AUDIO` → transcript `STRING`) so Breeze TTS Clone / Direction get their `reference_text` without typing it.

**Architecture:** One pure engine module `nodes/whisper_core.py` (model table, file checks, device/dtype rules, audio preparation, `transcribe()`), unit-tested with stubs and no ComfyUI; two thin node modules `nodes/whisper_loader.py` (download + one-resident-model cache + `unload()`; emits the cache **key** as the `WHISPER` type, never the model) and `nodes/whisper_transcribe.py`. Whisper runs through `transformers` (`WhisperProcessor` + `WhisperForConditionalGeneration`), which is already a dependency of the pack via the Breeze fork and of ComfyUI itself. Imports of transformers are lazy so the pack always loads.

**Tech Stack:** Python 3.10+, ComfyUI v3 node API (`comfy_api.latest.io`), torch, numpy, librosa (resampling), huggingface_hub (download), transformers 4.57–5.x. Tests: pytest from the repo root (`addopts = --confcutdir=tests` already configured).

**Spec:** `docs/superpowers/specs/2026-09-15-whisper-transcribe-design.md`

## Global Constraints

- **Branch:** `feature/whisper-transcribe` (already created from `main`, spec committed). Commit after every task; commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Git identity is the repo-local `Into The Latent`; never commit as Chris / Little-God1983.
- **Never `pip install` into `E:\AI\ComfyUI\venv`** and never into the system python. Nothing in this plan needs a new package.
- **Two test commands** (both from the repo root `E:\Repos\ComfyUI-IntoTheLatent-Utils`):
  - Core (no ComfyUI): `python -m pytest tests/test_whisper_core.py -q` (system python 3.13, transformers 4.57.1, torch 2.9 CPU, librosa, soundfile, pytest — all present).
  - Nodes (needs ComfyUI): in Git Bash,
    `PYLIB='C:\Users\LITTLE~1\AppData\Local\Temp\claude\e--Repos-ComfyUI-IntoTheLatent-Utils\1bfb643d-b625-476a-a86b-3086bf6a2c9b\scratchpad\pylib'; PYTHONPATH="E:\AI\ComfyUI;$PYLIB" /e/AI/ComfyUI/venv/Scripts/python.exe -W ignore -m pytest tests/test_whisper_nodes.py -q`
    (that `pylib` folder holds pytest for the venv; verified today: the Breeze node tests pass with exactly this invocation). Run the full suite (`tests/`) the same way before the final commit.
- **No new requirement lines.** `requirements.txt` / `pyproject.toml` dependencies stay as they are; only `version` changes (→ `1.9.0`).
- **Node IDs / display names:** `ITLWhisperLoader` → "ITL Whisper Loader", `ITLWhisperTranscribe` → "ITL Whisper Transcribe". Category `Into The Latent/audio`, `is_experimental=True`.
- **Model table** (name → HF repo): `large-v3-turbo` → `openai/whisper-large-v3-turbo` (default), `large-v3`, `medium`, `small`, `base`, `tiny` → `openai/whisper-<name>`. Checkpoint dir: `<ComfyUI>/models/whisper/whisper-<name>/`.
- **Download allow-list is mandatory** (the `large-v3` repo also carries ~12 GB of Flax / TF / `.bin` / fp32 duplicates): `config.json`, `generation_config.json`, `model.safetensors`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `merges.txt`, `vocab.json`, `normalizer.json`.
- **Verified transformers call path (both 4.57.1 and 5.15.0, identical results — do not "improve" it):**
  - `WhisperForConditionalGeneration.from_pretrained(dir, dtype=<torch dtype>)` — the keyword is `dtype`, accepted on both versions.
  - ≤ 30 s of audio: `processor(samples, sampling_rate=16000, return_tensors="pt")` (default padding to 3000 mel frames; anything else makes the encoder raise "expects the mel input features to be of length 3000").
  - > 30 s: `processor(samples, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)` and pass `attention_mask` to `generate()` — transformers' sequential long-form decoding.
  - `model.generate(features, task="transcribe", language=<code or None>, return_timestamps=True, condition_on_prev_tokens=False, num_beams=1, [attention_mask=...])`, then `processor.batch_decode(ids, skip_special_tokens=True)[0]`.
- **Spec deviation, agreed:** the cache key is `(model_name, resolved_device)` rather than `(ckpt_dir, device)` — the directory is a pure function of the name, and `resolve_handle(key)` must be able to re-download after eviction, which needs the name. `device="auto"` resolves before keying, so `auto` and `cuda` share one cache entry.
- **Checkpoint directory** is built from `folder_paths.models_dir` directly (`<models_dir>/whisper/whisper-<name>/`), not from `get_folder_paths("whisper")[0]`: another pack may already have registered a `whisper` folder, and index 0 would then point into it. The folder is still registered with `add_model_folder_path` for discoverability.
- License header on new files: `# <Name> — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.`
- Whisper's `[transformers] A custom logits processor ... will take precedence` and `Ignoring clean_up_tokenization_spaces=True for BPE tokenizer` log lines on transformers 5 are expected noise; do not add code to silence them.

---

## File structure

- Create `nodes/whisper_core.py` — engine, no ComfyUI imports: `MODELS`, `MODEL_NAMES`, `DEFAULT_MODEL`, `LANGUAGES`, `DEVICE_CHOICES`, `REQUIRED_FILES`, `DOWNLOAD_PATTERNS`, `TARGET_SR`, `LONG_FORM_SAMPLES`, `snapshot_dirname`, `missing_files`, `cache_key`, `resolve_device`, `dtype_for`, `prepare_samples`, `normalise_text`, `WhisperHandle`, `transcribe`.
- Create `nodes/whisper_loader.py` — `WHISPER` io type, `INSTALL_HINT`, `_CACHE`, `ensure_snapshot`, `load_handle`, `resolve_handle`, `unload`, `ITLWhisperLoader`.
- Create `nodes/whisper_transcribe.py` — `ITLWhisperTranscribe`.
- Create `tests/test_whisper_core.py`, `tests/test_whisper_nodes.py`.
- Modify `__init__.py` (two imports, two mapping entries each), `nodes/breeze_tts_generate.py` (one tooltip string), `README.md` (new section + one sentence in the Breeze section + installation note), `pyproject.toml` (`version = "1.9.0"`).

---

### Task 1: Core constants, file check, device rules, audio preparation

**Files:**
- Create: `nodes/whisper_core.py`
- Test: `tests/test_whisper_core.py`

**Interfaces:**
- Consumes: `nodes.breeze_tts_core.audio_to_mono_numpy(waveform) -> np.ndarray` (existing; batch 0, channel mean, float32 1-D).
- Produces (used by Tasks 2–5):
  - `MODELS: dict[str, str]` name → repo id; `MODEL_NAMES: tuple[str, ...]` (dict order: `large-v3-turbo`, `large-v3`, `medium`, `small`, `base`, `tiny`); `DEFAULT_MODEL = "large-v3-turbo"`.
  - `LANGUAGES: tuple[str, ...]` (28 ISO codes, `en` first); `DEVICE_CHOICES = ("auto", "cuda", "cpu")`.
  - `REQUIRED_FILES: tuple[str, ...]`; `DOWNLOAD_PATTERNS: list[str]` (same 11 names, as a list for `allow_patterns`).
  - `TARGET_SR = 16000`; `LONG_FORM_SAMPLES = 30 * TARGET_SR`.
  - `snapshot_dirname(name) -> str` (`"whisper-" + name`, raises `ValueError` for unknown names).
  - `missing_files(ckpt_dir) -> list[str]` (absent or 0-byte).
  - `cache_key(name, device) -> tuple[str, str]`.
  - `resolve_device(choice, cuda_available) -> str` (`"cuda"` / `"cpu"`; `RuntimeError` for `cuda` without CUDA; `ValueError` for unknown choice).
  - `dtype_for(device) -> torch.dtype` (`float16` on cuda, `float32` otherwise).
  - `prepare_samples(audio, resample=None) -> np.ndarray` (float32 1-D at 16 kHz; `resample(samples, orig_sr, target_sr)` injectable, default librosa; `ValueError("audio is empty")`).
  - `normalise_text(text) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_whisper_core.py`:

```python
# Tests for the Whisper engine helpers — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import numpy as np
import pytest
import torch

from nodes.whisper_core import (
    DEFAULT_MODEL,
    DEVICE_CHOICES,
    DOWNLOAD_PATTERNS,
    LANGUAGES,
    LONG_FORM_SAMPLES,
    MODEL_NAMES,
    MODELS,
    REQUIRED_FILES,
    TARGET_SR,
    cache_key,
    dtype_for,
    missing_files,
    normalise_text,
    prepare_samples,
    resolve_device,
    snapshot_dirname,
)


def test_model_table():
    assert MODEL_NAMES == ("large-v3-turbo", "large-v3", "medium", "small", "base", "tiny")
    assert DEFAULT_MODEL == "large-v3-turbo"
    assert MODELS["large-v3-turbo"] == "openai/whisper-large-v3-turbo"
    assert all(repo == f"openai/whisper-{name}" for name, repo in MODELS.items())


def test_languages_and_devices():
    assert LANGUAGES[0] == "en" and "zh" in LANGUAGES and len(LANGUAGES) == 28
    assert len(set(LANGUAGES)) == len(LANGUAGES)
    assert "auto" not in LANGUAGES
    assert DEVICE_CHOICES == ("auto", "cuda", "cpu")


def test_required_files_and_patterns():
    assert "model.safetensors" in REQUIRED_FILES and "config.json" in REQUIRED_FILES
    assert not any(f.endswith((".bin", ".msgpack", ".h5")) for f in REQUIRED_FILES)
    assert list(REQUIRED_FILES) == DOWNLOAD_PATTERNS
    assert TARGET_SR == 16000 and LONG_FORM_SAMPLES == 480000


def test_snapshot_dirname():
    assert snapshot_dirname("tiny") == "whisper-tiny"
    with pytest.raises(ValueError, match="Unknown Whisper model"):
        snapshot_dirname("huge")


def test_missing_files(tmp_path):
    assert missing_files(str(tmp_path)) == list(REQUIRED_FILES)
    for rel in REQUIRED_FILES:
        (tmp_path / rel).write_bytes(b"x")
    assert missing_files(str(tmp_path)) == []
    (tmp_path / "model.safetensors").write_bytes(b"")     # 0-byte stub from a killed download
    assert missing_files(str(tmp_path)) == ["model.safetensors"]


def test_cache_key():
    assert cache_key("tiny", "cuda") == ("tiny", "cuda")
    assert cache_key("tiny", "cuda") == cache_key("tiny", "cuda")
    assert cache_key("tiny", "cpu") != cache_key("tiny", "cuda")


def test_resolve_device():
    assert resolve_device("auto", cuda_available=True) == "cuda"
    assert resolve_device("auto", cuda_available=False) == "cpu"
    assert resolve_device("cpu", cuda_available=True) == "cpu"
    assert resolve_device("cuda", cuda_available=True) == "cuda"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda", cuda_available=False)
    with pytest.raises(ValueError, match="device"):
        resolve_device("mps", cuda_available=True)


def test_dtype_for():
    assert dtype_for("cuda") is torch.float16
    assert dtype_for("cpu") is torch.float32


def test_prepare_samples_mono_at_16k_is_passthrough():
    wave = torch.arange(8, dtype=torch.float32)[None, None, :]
    calls = []
    out = prepare_samples({"waveform": wave, "sample_rate": 16000}, resample=lambda *a: calls.append(a))
    assert calls == [] and out.dtype == np.float32 and out.tolist() == list(range(8))


def test_prepare_samples_downmixes_and_resamples():
    wave = torch.stack([torch.ones(4), torch.zeros(4)])[None]     # [1, 2, 4] stereo
    seen = {}

    def fake_resample(samples, orig_sr, target_sr):
        seen["samples"], seen["orig"], seen["target"] = samples, orig_sr, target_sr
        return np.repeat(samples, 2)
    out = prepare_samples({"waveform": wave, "sample_rate": 8000}, resample=fake_resample)
    assert seen["samples"].tolist() == [0.5, 0.5, 0.5, 0.5]
    assert (seen["orig"], seen["target"]) == (8000, 16000)
    assert out.tolist() == [0.5] * 8 and out.dtype == np.float32


def test_prepare_samples_takes_batch_item_zero():
    wave = torch.stack([torch.full((1, 3), 1.0), torch.full((1, 3), 2.0)])   # [2, 1, 3]
    out = prepare_samples({"waveform": wave, "sample_rate": 16000})
    assert out.tolist() == [1.0, 1.0, 1.0]


def test_prepare_samples_rejects_empty():
    with pytest.raises(ValueError, match="audio is empty"):
        prepare_samples({"waveform": torch.zeros((1, 1, 0)), "sample_rate": 16000})


def test_normalise_text():
    assert normalise_text("  Hello,\n  world.  ") == "Hello, world."
    assert normalise_text("") == ""
    assert normalise_text(None) == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_whisper_core.py -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'nodes.whisper_core'`.

- [ ] **Step 3: Write the module**

Create `nodes/whisper_core.py`:

```python
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
    import librosa
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_whisper_core.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add nodes/whisper_core.py tests/test_whisper_core.py
git commit -m "feat(whisper): core constants, file check, device rules and audio preparation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `WhisperHandle` and `transcribe()`

**Files:**
- Modify: `nodes/whisper_core.py` (append)
- Test: `tests/test_whisper_core.py` (append)

**Interfaces:**
- Consumes: Task 1's `prepare_samples`, `normalise_text`, `LANGUAGES`, `LONG_FORM_SAMPLES`, `TARGET_SR`.
- Produces:
  - `class WhisperHandle(key, processor, model, device, dtype)` — plain attribute holder.
  - `transcribe(handle, audio, language="auto", resample=None) -> str`. `language` is `"auto"` or a code from `LANGUAGES` (anything else → `ValueError`). Calls `handle.processor(...)`, `handle.model.generate(...)`, `handle.processor.batch_decode(...)` exactly as in the Global Constraints.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_whisper_core.py`:

```python
from nodes.whisper_core import WhisperHandle, transcribe  # noqa: E402


class _StubProcessor:
    """Records the feature-extractor call, returns a features dict like transformers' BatchFeature."""

    def __init__(self, with_mask=False):
        self.calls = []
        self.with_mask = with_mask
        self.decoded = []

    def __call__(self, samples, **kw):
        self.calls.append((np.asarray(samples), kw))
        out = {"input_features": torch.zeros((1, 80, 3000))}
        if self.with_mask:
            out["attention_mask"] = torch.ones((1, 3000), dtype=torch.long)
        return out

    def batch_decode(self, ids, **kw):
        self.decoded.append((ids, kw))
        return ["  Hello,   world.\n"]


class _StubModel:
    def __init__(self):
        self.calls = []

    def generate(self, features, **kw):
        self.calls.append((features, kw))
        return torch.tensor([[1, 2, 3]])


def _handle(with_mask=False):
    return WhisperHandle(("tiny", "cpu"), _StubProcessor(with_mask), _StubModel(), "cpu", torch.float32)


def _audio(seconds, sr=16000):
    return {"waveform": torch.zeros((1, 1, int(seconds * sr))), "sample_rate": sr}


def test_handle_attributes():
    h = _handle()
    assert h.key == ("tiny", "cpu") and h.device == "cpu" and h.dtype is torch.float32
    assert isinstance(h.processor, _StubProcessor) and isinstance(h.model, _StubModel)


def test_transcribe_short_form_auto_language():
    h = _handle()
    text = transcribe(h, _audio(2.0))
    assert text == "Hello, world."
    samples, kw = h.processor.calls[0]
    assert samples.shape == (32000,) and kw == {"sampling_rate": 16000, "return_tensors": "pt"}
    features, gkw = h.model.calls[0]
    assert tuple(features.shape) == (1, 80, 3000) and features.dtype is torch.float32
    assert gkw == {"task": "transcribe", "language": None, "return_timestamps": True,
                   "condition_on_prev_tokens": False, "num_beams": 1}
    ids, dkw = h.processor.decoded[0]
    assert ids.tolist() == [[1, 2, 3]] and dkw == {"skip_special_tokens": True}


def test_transcribe_fixed_language():
    h = _handle()
    transcribe(h, _audio(1.0), language="zh")
    assert h.model.calls[0][1]["language"] == "zh"


def test_transcribe_rejects_unknown_language():
    with pytest.raises(ValueError, match="language"):
        transcribe(_handle(), _audio(1.0), language="xx")


def test_transcribe_long_form_uses_longest_padding_and_mask():
    h = _handle(with_mask=True)
    transcribe(h, _audio(31.0))
    _, kw = h.processor.calls[0]
    assert kw == {"sampling_rate": 16000, "return_tensors": "pt", "truncation": False,
                  "padding": "longest", "return_attention_mask": True}
    _, gkw = h.model.calls[0]
    assert gkw["attention_mask"].shape == (1, 3000) and gkw["return_timestamps"] is True


def test_transcribe_exactly_30s_is_short_form():
    h = _handle()
    transcribe(h, _audio(30.0))
    assert h.processor.calls[0][1] == {"sampling_rate": 16000, "return_tensors": "pt"}


def test_transcribe_resamples_before_the_processor():
    h = _handle()
    transcribe(h, _audio(1.0, sr=8000), resample=lambda s, o, t: np.repeat(s, 2))
    assert h.processor.calls[0][0].shape == (16000,)


def test_transcribe_moves_features_to_handle_device_and_dtype():
    h = WhisperHandle(("tiny", "cpu"), _StubProcessor(), _StubModel(), "cpu", torch.float16)
    transcribe(h, _audio(1.0))
    assert h.model.calls[0][0].dtype is torch.float16
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_whisper_core.py -q`
Expected: `ImportError: cannot import name 'WhisperHandle'`.

- [ ] **Step 3: Append the implementation**

Append to `nodes/whisper_core.py`:

```python


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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_whisper_core.py -q`
Expected: `21 passed`.

- [ ] **Step 5: Commit**

```bash
git add nodes/whisper_core.py tests/test_whisper_core.py
git commit -m "feat(whisper): WhisperHandle and transcribe() with short- and long-form paths

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Loader module and `ITLWhisperLoader`

**Files:**
- Create: `nodes/whisper_loader.py`
- Test: `tests/test_whisper_nodes.py`
- Reference pattern: `nodes/breeze_tts_loader.py` (same cache / evict / unload / key-only-output design).

**Interfaces:**
- Consumes: Task 1/2 names from `nodes.whisper_core`.
- Produces (used by Task 4):
  - `WHISPER = io.Custom("WHISPER")`.
  - `INSTALL_HINT: str`.
  - `ensure_snapshot(name, ckpt_dir) -> str`.
  - `load_handle(name, device_choice) -> WhisperHandle`; `resolve_handle(key) -> WhisperHandle` (key `(name, device)`); `unload() -> None`.
  - `class ITLWhisperLoader(io.ComfyNode)` with inputs `model`, `device`; output `WHISPER`; `execute` returns the handle's `key`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_whisper_nodes.py`:

```python
# Node-level tests for the Whisper nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
# Everything that needs weights or a GPU is stubbed; these only run inside a ComfyUI checkout.
import os
import re
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("comfy_api")  # only runs inside a ComfyUI environment
import torch  # noqa: E402

from nodes import whisper_core as core  # noqa: E402
from nodes import whisper_loader as loader  # noqa: E402


def test_loader_schema():
    s = loader.ITLWhisperLoader.define_schema()
    assert s.node_id == "ITLWhisperLoader" and s.display_name == "ITL Whisper Loader"
    assert s.category == "Into The Latent/audio"
    by_id = {i.id: i for i in s.inputs}
    assert list(by_id) == ["model", "device"]
    assert by_id["model"].options == list(core.MODEL_NAMES) and by_id["model"].default == core.DEFAULT_MODEL
    assert by_id["device"].options == list(core.DEVICE_CHOICES) and by_id["device"].default == "auto"
    assert [o.io_type for o in s.outputs] == ["WHISPER"]


def test_models_folder_registered():
    import folder_paths
    paths = folder_paths.get_folder_paths("whisper")
    assert any(p.replace("\\", "/").endswith("models/whisper") for p in paths)
    assert loader._snapshot_dir("tiny").replace("\\", "/").endswith("models/whisper/whisper-tiny")


def test_ensure_snapshot_skips_download_when_complete(tmp_path, monkeypatch):
    for rel in core.REQUIRED_FILES:
        (tmp_path / rel).write_bytes(b"x")
    monkeypatch.setattr(loader, "_snapshot_download", lambda **kw: pytest.fail("must not download"))
    assert loader.ensure_snapshot("tiny", str(tmp_path)) == str(tmp_path)


def test_ensure_snapshot_downloads_with_allow_list(tmp_path, monkeypatch):
    seen = {}

    def fake_download(**kw):
        seen.update(kw)
        for rel in core.REQUIRED_FILES:
            (tmp_path / rel).write_bytes(b"x")
    monkeypatch.setattr(loader, "_snapshot_download", fake_download)
    loader.ensure_snapshot("large-v3", str(tmp_path))
    assert seen["repo_id"] == "openai/whisper-large-v3" and seen["local_dir"] == str(tmp_path)
    assert seen["allow_patterns"] == core.DOWNLOAD_PATTERNS


def test_ensure_snapshot_reports_still_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_snapshot_download", lambda **kw: None)
    with pytest.raises(RuntimeError, match="model.safetensors"):
        loader.ensure_snapshot("tiny", str(tmp_path))


def test_ensure_snapshot_wraps_download_errors(tmp_path, monkeypatch):
    def boom(**kw):
        raise OSError("no network")
    monkeypatch.setattr(loader, "_snapshot_download", boom)
    with pytest.raises(RuntimeError, match=re.escape(str(tmp_path))):
        loader.ensure_snapshot("tiny", str(tmp_path))


def _stub_loading(monkeypatch, tmp_path, loads, cuda=True):
    loader._CACHE.clear()
    monkeypatch.setattr(loader, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(loader, "ensure_snapshot", lambda name, d: d)
    monkeypatch.setattr(loader, "_snapshot_dir", lambda name: str(tmp_path / name))
    monkeypatch.setattr(loader, "_load_pieces", lambda d, device: (loads.append((d, device)), ("proc", "model"))[1])
    monkeypatch.setattr(loader, "gc", SimpleNamespace(collect=lambda: None))


def test_load_handle_caches_resolves_auto_and_evicts(monkeypatch, tmp_path):
    loads = []
    _stub_loading(monkeypatch, tmp_path, loads, cuda=True)
    h1 = loader.load_handle("tiny", "auto")
    h2 = loader.load_handle("tiny", "cuda")
    assert h1 is h2 and isinstance(h1, core.WhisperHandle)
    assert h1.key == ("tiny", "cuda") and h1.device == "cuda" and h1.dtype is torch.float16
    assert h1.processor == "proc" and h1.model == "model"
    assert loads == [(str(tmp_path / "tiny"), "cuda")]
    h3 = loader.load_handle("base", "cpu")
    assert h3.key == ("base", "cpu") and h3.dtype is torch.float32
    assert list(loader._CACHE) == [h3.key]          # one resident model
    loader._CACHE.clear()


def test_load_handle_cpu_fallback_and_cuda_refusal(monkeypatch, tmp_path):
    loads = []
    _stub_loading(monkeypatch, tmp_path, loads, cuda=False)
    assert loader.load_handle("tiny", "auto").device == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        loader.load_handle("tiny", "cuda")
    loader._CACHE.clear()


def test_load_handle_rejects_unknown_model(monkeypatch, tmp_path):
    _stub_loading(monkeypatch, tmp_path, [], cuda=True)
    with pytest.raises(ValueError, match="Unknown Whisper model"):
        loader.load_handle("huge", "auto")


def test_resolve_handle_uses_load_handle(monkeypatch):
    seen = {}
    monkeypatch.setattr(loader, "load_handle", lambda name, device: seen.update(name=name, device=device) or "H")
    assert loader.resolve_handle(("small", "cpu")) == "H" and seen == {"name": "small", "device": "cpu"}


def test_unload_clears_cache(monkeypatch):
    loader._CACHE["k"] = object()
    monkeypatch.setattr(loader, "gc", SimpleNamespace(collect=lambda: None))
    loader.unload()
    assert loader._CACHE == {}


def test_loader_execute_returns_key(monkeypatch):
    sentinel = SimpleNamespace(key=("tiny", "cpu"))
    monkeypatch.setattr(loader, "load_handle", lambda name, device: sentinel)
    out = loader.ITLWhisperLoader.execute(model="tiny", device="cpu")
    assert out.args[0] == ("tiny", "cpu")


def test_load_pieces_import_error_names_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="transformers"):
        loader._load_pieces(str(tmp_path), "cpu")
```

- [ ] **Step 2: Run to verify failure**

Run (Git Bash, repo root):
`PYLIB='C:\Users\LITTLE~1\AppData\Local\Temp\claude\e--Repos-ComfyUI-IntoTheLatent-Utils\1bfb643d-b625-476a-a86b-3086bf6a2c9b\scratchpad\pylib'; PYTHONPATH="E:\AI\ComfyUI;$PYLIB" /e/AI/ComfyUI/venv/Scripts/python.exe -W ignore -m pytest tests/test_whisper_nodes.py -q`
Expected: collection error `No module named 'nodes.whisper_loader'`.

- [ ] **Step 3: Write the module**

Create `nodes/whisper_loader.py`:

```python
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
    DEFAULT_MODEL, DEVICE_CHOICES, DOWNLOAD_PATTERNS, MODEL_NAMES, MODELS, WhisperHandle, cache_key,
    dtype_for, missing_files, resolve_device, snapshot_dirname,
)

WHISPER = io.Custom("WHISPER")

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
    print(f"[Whisper] downloading {repo} to {ckpt_dir} ...")
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
```

- [ ] **Step 4: Run to verify pass**

Run the node test command from Step 2.
Expected: `13 passed`.

- [ ] **Step 5: Commit**

```bash
git add nodes/whisper_loader.py tests/test_whisper_nodes.py
git commit -m "feat(whisper): loader node with auto-download, one-resident cache and key-only WHISPER output

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `ITLWhisperTranscribe`

**Files:**
- Create: `nodes/whisper_transcribe.py`
- Test: `tests/test_whisper_nodes.py` (append)

**Interfaces:**
- Consumes: `whisper_loader.WHISPER`, `resolve_handle`, `unload`; `whisper_core.LANGUAGES`, `transcribe`.
- Produces: `class ITLWhisperTranscribe(io.ComfyNode)` — inputs `model` (WHISPER), `audio` (AUDIO), `language` (combo `auto` + LANGUAGES), `unload_after` (bool); output `STRING` named `text`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_whisper_nodes.py`:

```python
from nodes import whisper_transcribe as tr  # noqa: E402


def test_transcribe_schema():
    s = tr.ITLWhisperTranscribe.define_schema()
    assert s.node_id == "ITLWhisperTranscribe" and s.display_name == "ITL Whisper Transcribe"
    assert s.category == "Into The Latent/audio"
    by_id = {i.id: i for i in s.inputs}
    assert list(by_id) == ["model", "audio", "language", "unload_after"]
    assert by_id["model"].io_type == "WHISPER" and by_id["audio"].io_type == "AUDIO"
    assert by_id["language"].options == ["auto", *core.LANGUAGES] and by_id["language"].default == "auto"
    assert by_id["unload_after"].default is False
    assert [o.io_type for o in s.outputs] == ["STRING"]


def _capture_transcribe(monkeypatch):
    calls = {}

    def fake_transcribe(handle, audio, language="auto"):
        calls["handle"], calls["audio"], calls["language"] = handle, audio, language
        return "hello world"
    monkeypatch.setattr(tr, "transcribe", fake_transcribe)
    monkeypatch.setattr(tr, "resolve_handle", lambda key: "H:" + str(key))
    return calls


def test_transcribe_execute(monkeypatch):
    calls = _capture_transcribe(monkeypatch)
    audio = {"waveform": torch.zeros((1, 1, 4)), "sample_rate": 16000}
    out = tr.ITLWhisperTranscribe.execute(model=("tiny", "cpu"), audio=audio, language="zh")
    assert out.args[0] == "hello world"
    assert calls == {"handle": "H:('tiny', 'cpu')", "audio": audio, "language": "zh"}


def test_transcribe_unload_after(monkeypatch):
    _capture_transcribe(monkeypatch)
    order = []
    monkeypatch.setattr(tr, "transcribe", lambda *a, **kw: (order.append("transcribe"), "t")[1])
    monkeypatch.setattr(tr, "unload", lambda: order.append("unload"))
    audio = {"waveform": torch.zeros((1, 1, 4)), "sample_rate": 16000}
    tr.ITLWhisperTranscribe.execute(model=("tiny", "cpu"), audio=audio, language="auto")
    assert order == ["transcribe"]
    tr.ITLWhisperTranscribe.execute(model=("tiny", "cpu"), audio=audio, language="auto", unload_after=True)
    assert order == ["transcribe", "transcribe", "unload"]
```

- [ ] **Step 2: Run to verify failure**

Run the node test command.
Expected: `ImportError`/`ModuleNotFoundError` for `nodes.whisper_transcribe`.

- [ ] **Step 3: Write the module**

Create `nodes/whisper_transcribe.py`:

```python
# Whisper Transcribe node — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# AUDIO -> transcript STRING through a WHISPER model from ITL Whisper Loader. Built for feeding
# Breeze TTS Voice Clone / Direction their `reference_text`, but it is a general speech-to-text
# node. Engine: whisper_core.transcribe (design:
# docs/superpowers/specs/2026-09-15-whisper-transcribe-design.md §3).
from comfy_api.latest import io

from .whisper_core import LANGUAGES, transcribe
from .whisper_loader import WHISPER, resolve_handle, unload


class ITLWhisperTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLWhisperTranscribe",
            display_name="ITL Whisper Transcribe",
            category="Into The Latent/audio",
            search_aliases=["whisper", "transcribe", "speech to text", "audio to text", "stt"],
            is_experimental=True,
            description="""
Transcribes audio to text with Whisper (speech-to-text).

Wire the output into a Breeze TTS Voice Clone / Direction node's `reference_text`. Stereo is
downmixed, any sample rate is accepted, clips longer than 30 s are transcribed in full.
Punctuation and casing come from the model; check Chinese output for simplified vs. traditional
characters before using it as a Breeze transcript.""",
            inputs=[
                WHISPER.Input("model", tooltip="From ITL Whisper Loader."),
                io.Audio.Input("audio", tooltip="Batch item 0 is transcribed; stereo is downmixed."),
                io.Combo.Input("language", options=["auto", *LANGUAGES], default="auto",
                               tooltip="auto lets Whisper detect the language; pick a code when it guesses wrong."),
                io.Boolean.Input("unload_after", default=False,
                                 tooltip="Free the Whisper model after this node runs (the next Whisper node "
                                         "reloads it). Turn on when VRAM is tight."),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, model, audio, language="auto", unload_after=False) -> io.NodeOutput:
        text = transcribe(resolve_handle(model), audio, language=language)
        if unload_after:
            unload()
        return io.NodeOutput(text)
```

- [ ] **Step 4: Run to verify pass**

Run the node test command.
Expected: `16 passed`.

- [ ] **Step 5: Commit**

```bash
git add nodes/whisper_transcribe.py tests/test_whisper_nodes.py
git commit -m "feat(whisper): ITL Whisper Transcribe node (AUDIO -> STRING)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Registration, Breeze tooltip, README, version 1.9.0

**Files:**
- Modify: `__init__.py` (imports block, `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`)
- Modify: `nodes/breeze_tts_generate.py` (the `reference_text` tooltip in `_inputs`)
- Modify: `README.md` (Installation note; Breeze section; new "ITL Whisper Transcribe" section before "## Credits & License")
- Modify: `pyproject.toml` (`version`)
- Test: `tests/test_whisper_nodes.py` (append)

**Interfaces:**
- Consumes: `nodes.whisper_loader.ITLWhisperLoader`, `nodes.whisper_transcribe.ITLWhisperTranscribe`.
- Produces: the two nodes visible in ComfyUI.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_whisper_nodes.py`:

```python
def test_pack_registers_both_whisper_nodes():
    import importlib.util

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "itl_pack", os.path.join(repo_root, "__init__.py"), submodule_search_locations=[repo_root]
    )
    root = importlib.util.module_from_spec(spec)
    sys.modules["itl_pack"] = root
    try:
        spec.loader.exec_module(root)
        for node_id, display in [("ITLWhisperLoader", "ITL Whisper Loader"),
                                 ("ITLWhisperTranscribe", "ITL Whisper Transcribe")]:
            assert root.NODE_CLASS_MAPPINGS[node_id].define_schema().node_id == node_id
            assert root.NODE_DISPLAY_NAME_MAPPINGS[node_id] == display
    finally:
        for name in [n for n in sys.modules if n == "itl_pack" or n.startswith("itl_pack.")]:
            del sys.modules[name]


def test_breeze_reference_text_tooltip_points_at_whisper():
    from nodes import breeze_tts_generate as gen
    s = gen.ITLBreezeTTSVoiceClone.define_schema()
    tip = next(i for i in s.inputs if i.id == "reference_text").tooltip
    assert "Whisper" in tip
```

- [ ] **Step 2: Run to verify failure**

Run the node test command.
Expected: 2 failures — `KeyError: 'ITLWhisperLoader'` and the tooltip assertion.

- [ ] **Step 3: Register the nodes**

In `__init__.py`, after the `breeze_tts_generate` import block add:

```python
from .nodes.whisper_loader import ITLWhisperLoader
from .nodes.whisper_transcribe import ITLWhisperTranscribe
```

Append to `NODE_CLASS_MAPPINGS` (after `"ITLBreezeTTSVoiceDirectionAdvanced": ...,`):

```python
    "ITLWhisperLoader": ITLWhisperLoader,
    "ITLWhisperTranscribe": ITLWhisperTranscribe,
```

Append to `NODE_DISPLAY_NAME_MAPPINGS` (after the Direction Advanced line):

```python
    "ITLWhisperLoader": "ITL Whisper Loader",
    "ITLWhisperTranscribe": "ITL Whisper Transcribe",
```

- [ ] **Step 4: Point Breeze's `reference_text` at the new node**

In `nodes/breeze_tts_generate.py`, `_inputs()`, replace the `reference_text` tooltip:

```python
            io.String.Input("reference_text", multiline=True, default="",
                            tooltip="Exact transcript of reference_audio. Wrong text = wrong voice. "
                                    "ITL Whisper Transcribe can produce it from the clip."),
```

- [ ] **Step 5: README**

In `README.md`, Installation section, replace the sentence
`No extra dependencies (it uses Pillow, already shipped with ComfyUI).` with:

```markdown
No extra dependencies for the image / prompt / metadata nodes (Pillow ships with ComfyUI). The
Breeze TTS nodes need `pip install -r requirements.txt` (see their section); the Whisper nodes
run on the `transformers` package ComfyUI already has.
```

In the Breeze section, after the bullet list of the three modes, add:

```markdown
`reference_text` must be the clip's exact transcript — **ITL Whisper Transcribe** (below) produces it
from the reference audio.
```

Insert a new section immediately before `## Credits & License`:

```markdown
### ITL Whisper Transcribe (Loader + Audio to Text)

Speech-to-text with [OpenAI Whisper](https://github.com/openai/whisper) through `transformers`.
Two nodes:

- **ITL Whisper Loader** — picks the model (`large-v3-turbo` default, `large-v3`, `medium`,
  `small`, `base`, `tiny`) and the device (`auto` / `cuda` / `cpu`).
- **ITL Whisper Transcribe** — `AUDIO` in, transcript `STRING` out. `language` is `auto` or a
  fixed code (pick one when detection guesses wrong). `unload_after` frees the model.

Built to feed the Breeze TTS Clone / Direction nodes' `reference_text`, but it is a general
transcription node: stereo is downmixed, any sample rate is accepted, and clips longer than 30 s
are transcribed in full (sequential long-form decoding). Whisper's punctuation and casing are
used as-is. For Chinese, Whisper writes simplified or traditional characters depending on the
audio; check the text before using it as a Breeze transcript.

**First run** downloads the checkpoint from Hugging Face into `models/whisper/whisper-<model>/`
(only the safetensors + tokenizer files: `large-v3-turbo` 1.6 GB, `large-v3` 3.1 GB, `medium` 3.1 GB,
`small` 1 GB, `base` 290 MB, `tiny` 150 MB). Runs in fp16 on CUDA (`large-v3-turbo` ≈ 2 GiB VRAM,
`large-v3` ≈ 3.5 GiB) or fp32 on CPU. Whisper and Breeze can be resident together (≈ 10 GiB).

No extra install: it uses the `transformers` (>= 4.57) and `librosa` packages the Breeze nodes and
ComfyUI already require. Whisper weights are Apache 2.0; the node code is GPL-3.0 like the rest of
this pack.
```

- [ ] **Step 6: Version bump**

In `pyproject.toml`: `version = "1.9.0"`.

- [ ] **Step 7: Run everything**

Run the node test command with `tests/` instead of the single file (full suite under the venv), then `python -m pytest tests/test_whisper_core.py tests/test_breeze_tts_core.py -q` with the system python.
Expected: all pass (Breeze 59 + Whisper 18 + the rest); the two new tests green.

- [ ] **Step 8: Commit**

```bash
git add __init__.py nodes/breeze_tts_generate.py README.md pyproject.toml tests/test_whisper_nodes.py
git commit -m "feat(whisper): register the Whisper nodes, README section, point Breeze reference_text at it; bump to 1.9.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push -u origin feature/whisper-transcribe
```

---

### Task 6: Manual verification on the reference machine (not committed)

**Files:**
- Create (scratchpad only): `<scratchpad>/whisper_real_check.py`
- Uses: the user's ComfyUI at `E:\AI\ComfyUI` (venv: transformers 5.15, torch 2.11 cu130, RTX 5090); the system python (transformers 4.57.1, CPU); test clips in `E:\AI\ComfyUI\output\breeze_tts_test\` (`clone.wav` = "And this sentence was cloned from the first one.", `direction.wav` = "Now whisper this part very quietly.").

**Interfaces:**
- Consumes: `nodes.whisper_core` and `nodes.whisper_loader` from the repo (the loader needs `folder_paths` / `comfy_api` on the path: `PYTHONPATH="E:\AI\ComfyUI;E:\Repos\ComfyUI-IntoTheLatent-Utils"`).
- Produces: a pasted transcript log in the PR description.

- [ ] **Step 1: Write the check script**

```python
# whisper_real_check.py — real-model pass for the Whisper nodes. Run from the repo root.
import sys, time
import numpy as np, soundfile as sf, torch
from nodes import whisper_loader as loader
from nodes.whisper_core import transcribe

name = sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo"
device = sys.argv[2] if len(sys.argv) > 2 else "auto"
t = time.time()
h = loader.load_handle(name, device)
print(f"loaded {name} on {h.device} {h.dtype} in {time.time()-t:.1f}s; dir {loader._snapshot_dir(name)}")
assert loader.load_handle(name, device) is h, "second call must hit the cache"

def audio(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim == 2: y = y.T          # soundfile is [N, C]; ComfyUI is [B, C, N]
    else: y = y[None, :]
    return {"waveform": torch.from_numpy(np.ascontiguousarray(y))[None], "sample_rate": sr}

base = "E:/AI/ComfyUI/output/breeze_tts_test/"
for clip, lang in [("clone.wav", "auto"), ("clone.wav", "en"), ("direction.wav", "auto")]:
    t = time.time()
    print(f"{clip} lang={lang}: {transcribe(h, audio(base + clip), language=lang)!r}  ({time.time()-t:.1f}s)")

a = audio(base + "clone.wav")
w = a["waveform"]; gap = torch.zeros((1, 1, a["sample_rate"]))
reps = int(np.ceil(40 * a["sample_rate"] / (w.shape[-1] + gap.shape[-1])))
long = {"waveform": torch.cat([torch.cat([w, gap], dim=-1)] * reps, dim=-1), "sample_rate": a["sample_rate"]}
t = time.time()
text = transcribe(h, long)
print(f"long ({long['waveform'].shape[-1] / a['sample_rate']:.1f}s, {reps} repeats): "
      f"{text.count('cloned')} x 'cloned' in {time.time()-t:.1f}s")
if h.device == "cuda":
    before = torch.cuda.memory_allocated()
    loader.unload()
    print(f"unload: {before/2**30:.2f} GiB -> {torch.cuda.memory_allocated()/2**30:.2f} GiB allocated")
```

- [ ] **Step 2: Run on the ComfyUI venv (transformers 5.15, CUDA)**

Run (Git Bash, repo root):
`PYTHONPATH="E:\AI\ComfyUI;E:\Repos\ComfyUI-IntoTheLatent-Utils" /e/AI/ComfyUI/venv/Scripts/python.exe -W ignore <scratchpad>/whisper_real_check.py large-v3-turbo auto`
Expected: first run prints `[Whisper] downloading openai/whisper-large-v3-turbo to E:\AI\ComfyUI\models\whisper\whisper-large-v3-turbo ...` then `download done`; `loaded ... on cuda torch.float16`; the three short transcripts equal the sentences above (punctuation may differ slightly); long run reports `11 x 'cloned'`; unload drops allocated VRAM to ≈ 0. Check `E:\AI\ComfyUI\models\whisper\whisper-large-v3-turbo\` holds exactly the 11 allow-listed files (plus `.cache/`).

- [ ] **Step 3: Run on the system python (transformers 4.57.1, CPU)**

`PYTHONPATH="E:\AI\ComfyUI;E:\Repos\ComfyUI-IntoTheLatent-Utils" python -W ignore <scratchpad>/whisper_real_check.py tiny cpu`
(`tiny` keeps the CPU run short; it downloads 150 MB into `models/whisper/whisper-tiny/`.)
Expected: `loaded tiny on cpu torch.float32`; same transcripts; long run `11 x 'cloned'`.

- [ ] **Step 4: In ComfyUI**

In `E:\AI\ComfyUI\custom_nodes\ComfyUI-IntoTheLatent-Utils` run `git fetch && git checkout feature/whisper-transcribe && git pull`, restart ComfyUI, and build: Load Audio (a Breeze test clip) → ITL Whisper Transcribe (loader on `large-v3-turbo`) → ITL Breeze TTS Voice Clone `reference_text`, with the same clip as `reference_audio`. Queue twice: the second run must not print `[Whisper] downloading` or reload. Confirm the Clone output is speech.

- [ ] **Step 5: Open the PR**

`gh pr create --base main --head feature/whisper-transcribe` with the transcript logs from Steps 2–3 and the workflow result from Step 4 in the body; end the body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Use the Into-The-Latent GitHub identity (`gh auth switch --user Into-The-Latent` before, `gh auth switch --user Little-God1983` after).
