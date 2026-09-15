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
