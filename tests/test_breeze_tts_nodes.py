# Node-level tests for the Breeze TTS nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
# Everything that needs weights or a GPU is stubbed; these only run inside a ComfyUI checkout.
import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("comfy_api")  # only runs inside a ComfyUI environment
import numpy as np  # noqa: E402
import torch  # noqa: E402

from nodes import breeze_tts_core as core  # noqa: E402
from nodes import breeze_tts_loader as loader  # noqa: E402


def test_loader_schema():
    s = loader.ITLBreezeTTSLoader.define_schema()
    assert s.node_id == "ITLBreezeTTSLoader" and s.display_name == "ITL Breeze TTS Loader"
    names = [i.id for i in s.inputs]
    assert names == ["attention", "fast_path"]
    assert [o.io_type for o in s.outputs] == ["BREEZE_TTS"]


def test_models_folder_registered():
    import folder_paths
    paths = folder_paths.get_folder_paths("breeze_tts")
    assert any(p.replace("\\", "/").endswith("models/breeze_tts") for p in paths)


def test_ensure_snapshot_skips_download_when_complete(tmp_path, monkeypatch):
    for rel in core.REQUIRED_SNAPSHOT_FILES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    monkeypatch.setattr(loader, "_snapshot_download", lambda **kw: pytest.fail("must not download"))
    assert loader.ensure_snapshot(str(tmp_path)) == str(tmp_path)


def test_ensure_snapshot_downloads_when_incomplete(tmp_path, monkeypatch):
    seen = {}

    def fake_download(**kw):
        seen.update(kw)
        for rel in core.REQUIRED_SNAPSHOT_FILES:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    monkeypatch.setattr(loader, "_snapshot_download", fake_download)
    loader.ensure_snapshot(str(tmp_path))
    assert seen["repo_id"] == core.REPO_ID and seen["local_dir"] == str(tmp_path)
    assert list(seen["ignore_patterns"]) == list(core.DOWNLOAD_IGNORE)


def test_ensure_snapshot_reports_still_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_snapshot_download", lambda **kw: None)
    with pytest.raises(RuntimeError, match="tokenizer.json"):
        loader.ensure_snapshot(str(tmp_path))


def test_load_handle_caches_and_evicts(monkeypatch, tmp_path):
    loader._CACHE.clear()
    loads = []
    monkeypatch.setattr(loader, "_require_cuda", lambda: None)
    monkeypatch.setattr(loader, "ensure_snapshot", lambda d: d)
    monkeypatch.setattr(loader, "_snapshot_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "_load_pieces", lambda d, attention: (loads.append(attention), ("tok", "model", "atok"))[1])
    monkeypatch.setattr(loader, "_runtime_factory", lambda fast_path: (lambda m, a, t, kw: "rt"))

    h1 = loader.load_handle("sdpa", False)
    h2 = loader.load_handle("sdpa", False)
    assert h1 is h2 and loads == ["sdpa"] and isinstance(h1, core.BreezeHandle)
    h3 = loader.load_handle("eager", False)
    assert h3 is not h1 and loads == ["sdpa", "eager"]
    assert list(loader._CACHE) == [h3.key]          # old entry evicted
    loader._CACHE.clear()


def test_loader_execute_returns_handle(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(loader, "load_handle", lambda attention, fast_path: sentinel)
    out = loader.ITLBreezeTTSLoader.execute(attention="sdpa", fast_path=False)
    assert out.args[0] is sentinel


def test_runtime_factory_import_error_names_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "breeze_models", None)
    monkeypatch.setitem(sys.modules, "breeze_models.fast_streaming", None)
    with pytest.raises(ImportError, match="not installed"):
        loader._runtime_factory(False)
