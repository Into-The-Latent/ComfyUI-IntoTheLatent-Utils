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
    MODEL_SIZES,
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
    assert set(MODEL_SIZES) == set(MODEL_NAMES)


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


def test_librosa_resample_import_error_names_requirements(monkeypatch):
    import sys
    from nodes.whisper_core import _librosa_resample
    monkeypatch.setitem(sys.modules, "librosa", None)
    with pytest.raises(ImportError, match="requirements.txt"):
        _librosa_resample(np.zeros(4, dtype=np.float32), 8000, 16000)


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
