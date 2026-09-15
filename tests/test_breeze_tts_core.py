# Tests for the Breeze TTS engine helpers — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import os
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from nodes.breeze_tts_core import (
    DEFAULT_SAMPLING,
    MODES,
    BreezeHandle,
    REQUIRED_SNAPSHOT_FILES,
    SamplingConfig,
    audio_to_mono_numpy,
    build_request,
    cache_key,
    chunks_to_audio,
    generate_audio,
    missing_snapshot_files,
    snapshot_is_complete,
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


def _touch_all(root):
    from pathlib import Path
    root = Path(root)
    for rel in REQUIRED_SNAPSHOT_FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_required_snapshot_files_cover_model_tokenizer_and_codec():
    assert "model-00001-of-00002.safetensors" in REQUIRED_SNAPSHOT_FILES
    assert "model-00002-of-00002.safetensors" in REQUIRED_SNAPSHOT_FILES
    assert "audio_tokenizer/model.safetensors" in REQUIRED_SNAPSHOT_FILES
    assert "tokenizer.json" in REQUIRED_SNAPSHOT_FILES


def test_snapshot_complete_and_missing(tmp_path):
    assert missing_snapshot_files(str(tmp_path)) == list(REQUIRED_SNAPSHOT_FILES)
    assert snapshot_is_complete(str(tmp_path)) is False
    _touch_all(tmp_path)
    assert snapshot_is_complete(str(tmp_path)) is True
    (tmp_path / "audio_tokenizer" / "model.safetensors").unlink()
    assert missing_snapshot_files(str(tmp_path)) == ["audio_tokenizer/model.safetensors"]


def test_snapshot_ignores_zero_byte_files(tmp_path):
    _touch_all(tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(b"")
    assert missing_snapshot_files(str(tmp_path)) == ["tokenizer.json"]


def test_cache_key_normalises_path(tmp_path):
    a = cache_key(str(tmp_path / "x" / ".."), "sdpa", True)
    b = cache_key(str(tmp_path), "sdpa", True)
    assert a == b and a[1:] == ("sdpa", True)
    assert cache_key(str(tmp_path), "eager", True) != b


def test_handle_runtime_for_caches_by_sampling():
    calls = []

    def factory(model, audio_tokenizer, tokenizer, kwargs):
        calls.append(kwargs)
        return object()

    h = BreezeHandle(("p", "sdpa", False), "tok", "model", "atok", factory)
    r1 = h.runtime_for(SamplingConfig())
    r2 = h.runtime_for(SamplingConfig())
    assert r1 is r2 and len(calls) == 1 and calls[0]["max_new_tokens"] == 750
    r3 = h.runtime_for(SamplingConfig(top_k=10))
    assert r3 is not r1 and len(calls) == 2 and calls[1]["top_k"] == 10
    r4 = h.runtime_for(SamplingConfig())          # previous config again -> rebuilt (only last is kept)
    assert r4 is not r1 and len(calls) == 3
    assert h.fast_path is False and h.tokenizer == "tok"


class _Runtime:
    sample_rate = 24000

    def __init__(self, log):
        self.log = log

    def iter_audio_chunks(self, inputs, *, request_id=None, seed=None):
        self.log.append(("iter", inputs, request_id, seed))
        yield SimpleNamespace(audio=np.array([0.1, 0.2], np.float32), is_final=False)
        yield SimpleNamespace(audio=np.array([0.3], np.float32), is_final=True)


def _stub(log):
    handle = BreezeHandle(("p", "sdpa", False), "tok", "model", "atok",
                          lambda m, a, t, kw: _Runtime(log))

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, *, guidance_scale,
                       guidance_scale_ref, guidance_scale_ins):
        log.append(("prepare", requests, template, guidance_scale, guidance_scale_ref, guidance_scale_ins))
        return {"prepared": True}

    api = SimpleNamespace(
        prepare_inputs=prepare_inputs,
        select_template_name=lambda r: "tpl:" + ",".join(sorted(k for k in r if k in ("ref_text", "instruction"))),
        get_template=lambda name: ("template", name),
        set_all_seeds=lambda s: log.append(("seed", s)),
    )
    return handle, api


def test_generate_audio_design_mode(tmp_path):
    log = []
    handle, api = _stub(log)
    out = generate_audio(handle, api, mode="design", text="hi", seed=7, cfg_scale=4.0,
                         instruction="deep voice", temp_dir=str(tmp_path))
    assert out["sample_rate"] == 24000 and out["waveform"].shape == (1, 1, 3)
    kinds = [e[0] for e in log]
    assert kinds == ["prepare", "seed", "iter"]
    _, requests, template, cfg, ref, ins = log[0]
    assert requests == [{"id": "comfyui", "text": "hi", "speaker": "S0", "instruction": "deep voice"}]
    assert template == ("template", "tpl:instruction") and (cfg, ref, ins) == (4.0, None, None)
    assert log[1] == ("seed", 7) and log[2][2:] == ("comfyui", 7)
    assert not list(tmp_path.iterdir())     # no temp file for design mode


def test_generate_audio_clone_writes_and_removes_temp_wav(tmp_path):
    log = []
    handle, api = _stub(log)
    seen = {}

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **kw):
        path = requests[0]["ref_audio_path"]
        seen["exists_during"] = os.path.isfile(path)
        seen["path"] = path
        return {}
    api.prepare_inputs = prepare_inputs
    ref = {"waveform": torch.zeros((1, 1, 800)), "sample_rate": 8000}
    generate_audio(handle, api, mode="clone", text="hi", seed=1, cfg_scale=1.0,
                   reference_audio=ref, reference_text="hi", temp_dir=str(tmp_path))
    assert seen["exists_during"] is True and seen["path"].startswith(str(tmp_path))
    assert not os.path.exists(seen["path"])


def test_generate_audio_removes_temp_wav_on_failure(tmp_path):
    log = []
    handle, api = _stub(log)

    def boom(*a, **k):
        raise RuntimeError("cuda oom")
    api.prepare_inputs = boom
    ref = {"waveform": torch.zeros((1, 1, 800)), "sample_rate": 8000}
    with pytest.raises(RuntimeError, match="cuda oom"):
        generate_audio(handle, api, mode="direction", text="hi", seed=1, cfg_scale=4.0,
                       reference_audio=ref, reference_text="hi", instruction="fast", temp_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_generate_audio_validates_before_touching_runtime(tmp_path):
    log = []
    handle, api = _stub(log)
    with pytest.raises(ValueError, match="reference_audio"):
        generate_audio(handle, api, mode="clone", text="hi", seed=1, cfg_scale=1.0,
                       reference_text="hi", temp_dir=str(tmp_path))
    assert log == []


def test_generate_audio_masks_seed_to_32_bits(tmp_path):
    log = []
    handle, api = _stub(log)
    generate_audio(handle, api, mode="design", text="hi", seed=2**40 + 7, cfg_scale=4.0,
                   instruction="deep voice", temp_dir=str(tmp_path))
    assert ("seed", 7) in log
    iter_entry = next(e for e in log if e[0] == "iter")
    assert iter_entry[3] == 7


def test_generate_audio_rejects_cfg_for_clone(tmp_path):
    log = []
    handle, api = _stub(log)
    ref = {"waveform": torch.zeros((1, 1, 800)), "sample_rate": 8000}
    with pytest.raises(ValueError, match="cfg_scale"):
        generate_audio(handle, api, mode="clone", text="hi", seed=1, cfg_scale=2.0,
                       reference_audio=ref, reference_text="hi", temp_dir=str(tmp_path))
    assert log == []


def test_generate_audio_uses_sampling_for_runtime(tmp_path):
    log = []
    kwargs_seen = []
    handle = BreezeHandle(("p", "sdpa", False), "tok", "model", "atok",
                          lambda m, a, t, kw: (kwargs_seen.append(kw), _Runtime(log))[1])
    _, api = _stub(log)
    generate_audio(handle, api, mode="design", text="hi", seed=1, cfg_scale=4.0, instruction="x",
                   sampling=SamplingConfig(max_new_tokens=300, top_p=0.8), temp_dir=str(tmp_path))
    assert kwargs_seen[0]["max_new_tokens"] == 300 and kwargs_seen[0]["top_p"] == 0.8
