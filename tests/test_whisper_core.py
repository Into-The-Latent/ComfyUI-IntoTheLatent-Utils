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
