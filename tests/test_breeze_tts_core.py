# Tests for the Breeze TTS engine helpers — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import pytest

from nodes.breeze_tts_core import (
    DEFAULT_SAMPLING,
    MODES,
    SamplingConfig,
    build_request,
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
