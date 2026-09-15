# Tests for the Breeze TTS engine helpers — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from nodes.breeze_tts_core import (
    DEFAULT_SAMPLING,
    MODES,
    SamplingConfig,
    audio_to_mono_numpy,
    build_request,
    chunks_to_audio,
    write_reference_wav,
)


def test_modes():
    assert MODES == ("clone", "design", "direction")


def test_sampling_defaults_match_upstream():
    s = DEFAULT_SAMPLING
    assert (s.temperature, s.top_k, s.top_p, s.repetition_penalty, s.max_new_tokens) == (0.9, 50, 1.0, 1.1, 750)


def test_sampling_fast_config_kwargs_and_seq_len():
    kw = SamplingConfig(max_new_tokens=1500).fast_config_kwargs()
    assert kw["max_new_tokens"] == 1500
    assert kw["max_seq_len"] == 2012          # max_new_tokens + 512, at least 1024
    assert SamplingConfig().fast_config_kwargs()["max_seq_len"] == 1262
    assert SamplingConfig(max_new_tokens=100).fast_config_kwargs()["max_seq_len"] == 1024
    assert set(kw) == {"temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens", "max_seq_len"}


def test_sampling_is_hashable_and_comparable():
    assert SamplingConfig() == SamplingConfig()
    assert hash(SamplingConfig()) == hash(DEFAULT_SAMPLING)
    assert SamplingConfig(top_k=0) != DEFAULT_SAMPLING


def test_build_request_clone():
    r = build_request("clone", "hello", reference_text="ref words", has_reference_audio=True)
    assert r == {"id": "comfyui", "text": "hello", "speaker": "S0", "ref_text": "ref words"}


def test_build_request_design():
    r = build_request("design", "hello", instruction="a calm deep voice")
    assert r == {"id": "comfyui", "text": "hello", "speaker": "S0", "instruction": "a calm deep voice"}


def test_build_request_direction():
    r = build_request("direction", "hello", reference_text="ref", instruction="whisper", has_reference_audio=True)
    assert r == {"id": "comfyui", "text": "hello", "speaker": "S0", "ref_text": "ref", "instruction": "whisper"}


def test_build_request_strips_and_ignores_unused_inputs():
    r = build_request("design", "  hi  ", instruction=" x ", reference_text="ignored")
    assert r["text"] == "hi" and r["instruction"] == "x" and "ref_text" not in r


@pytest.mark.parametrize("mode,kwargs,missing", [
    ("clone", {"text": "", "reference_text": "r", "has_reference_audio": True}, "text"),
    ("clone", {"text": "t", "reference_text": " ", "has_reference_audio": True}, "reference_text"),
    ("clone", {"text": "t", "reference_text": "r", "has_reference_audio": False}, "reference_audio"),
    ("design", {"text": "t", "instruction": ""}, "instruction"),
    ("direction", {"text": "t", "reference_text": "r", "has_reference_audio": True, "instruction": None}, "instruction"),
    ("direction", {"text": "t", "instruction": "i", "reference_text": None, "has_reference_audio": True}, "reference_text"),
])
def test_build_request_missing_inputs(mode, kwargs, missing):
    with pytest.raises(ValueError, match=missing):
        build_request(mode, **kwargs)


def test_build_request_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        build_request("sing", "t")


def test_audio_to_mono_numpy_batched_stereo_means_channels():
    wav = torch.zeros((2, 2, 4))
    wav[0, 0] = 1.0   # batch 0 left = 1, right = 0 -> mean 0.5
    wav[1] = 9.0      # batch 1 must be ignored
    out = audio_to_mono_numpy(wav)
    assert out.shape == (4,) and out.dtype == np.float32
    assert np.allclose(out, 0.5)


def test_audio_to_mono_numpy_accepts_2d():
    out = audio_to_mono_numpy(torch.ones((1, 3)))
    assert out.shape == (3,)


def test_audio_to_mono_numpy_rejects_other_ranks():
    with pytest.raises(ValueError, match="waveform"):
        audio_to_mono_numpy(torch.ones(5))


def test_write_reference_wav_roundtrip(tmp_path):
    sr = 16000
    t = torch.linspace(0, 1, sr)
    audio = {"waveform": torch.stack([t, -t])[None], "sample_rate": sr}   # [1, 2, N]
    path = write_reference_wav(audio, str(tmp_path))
    assert os.path.dirname(path) == str(tmp_path) and os.path.basename(path).startswith("breeze_ref_")
    data, rate = sf.read(path, dtype="float32")
    assert rate == sr and data.ndim == 1 and len(data) == sr
    assert np.allclose(data, 0.0, atol=1e-4)   # channels cancel out


def test_chunks_to_audio_concatenates():
    out = chunks_to_audio([np.array([1, 2], np.float32), np.array([3], np.float32)], 24000)
    assert out["sample_rate"] == 24000
    assert out["waveform"].shape == (1, 1, 3) and out["waveform"].dtype == torch.float32
    assert out["waveform"][0, 0].tolist() == [1.0, 2.0, 3.0]


def test_chunks_to_audio_rejects_empty():
    with pytest.raises(ValueError, match="no audio"):
        chunks_to_audio([], 24000)
    with pytest.raises(ValueError, match="no audio"):
        chunks_to_audio([np.zeros(0, np.float32)], 24000)
