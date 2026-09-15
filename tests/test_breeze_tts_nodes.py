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


from nodes import breeze_tts_generate as gen  # noqa: E402

EXPECTED = {
    "ITLBreezeTTSVoiceClone": ("ITL Breeze TTS Voice Clone", ["model", "text", "reference_audio", "reference_text", "seed", "cfg_scale"]),
    "ITLBreezeTTSVoiceCloneAdvanced": ("ITL Breeze TTS Voice Clone Advanced", ["model", "text", "reference_audio", "reference_text", "seed", "cfg_scale", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]),
    "ITLBreezeTTSVoiceDesign": ("ITL Breeze TTS Voice Design", ["model", "text", "instruction", "seed", "cfg_scale"]),
    "ITLBreezeTTSVoiceDesignAdvanced": ("ITL Breeze TTS Voice Design Advanced", ["model", "text", "instruction", "seed", "cfg_scale", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]),
    "ITLBreezeTTSVoiceDirection": ("ITL Breeze TTS Voice Direction", ["model", "text", "reference_audio", "reference_text", "instruction", "seed", "cfg_scale"]),
    "ITLBreezeTTSVoiceDirectionAdvanced": ("ITL Breeze TTS Voice Direction Advanced", ["model", "text", "reference_audio", "reference_text", "instruction", "seed", "cfg_scale", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]),
}


@pytest.mark.parametrize("node_id", list(EXPECTED))
def test_generate_schemas(node_id):
    cls = gen.GENERATE_NODES[node_id]
    assert getattr(gen, node_id) is cls
    s = cls.define_schema()
    display, inputs = EXPECTED[node_id]
    assert s.node_id == node_id and s.display_name == display
    assert [i.id for i in s.inputs] == inputs
    assert [o.io_type for o in s.outputs] == ["AUDIO"]
    assert s.inputs[0].io_type == "BREEZE_TTS"
    cfg = next(i for i in s.inputs if i.id == "cfg_scale")
    assert cfg.default == (1.0 if "Clone" in node_id else 4.0)
    if "Advanced" in node_id:
        by_id = {i.id: i for i in s.inputs}
        assert (by_id["temperature"].default, by_id["top_k"].default, by_id["top_p"].default,
                by_id["repetition_penalty"].default, by_id["max_new_tokens"].default) == (0.9, 50, 1.0, 1.1, 750)


def _capture(monkeypatch, tmp_path):
    calls = {}

    def fake_generate(handle, api, **kw):
        calls["handle"], calls["api"], calls["kw"] = handle, api, kw
        return {"waveform": torch.zeros((1, 1, 10)), "sample_rate": 24000}
    monkeypatch.setattr(gen, "generate_audio", fake_generate)
    monkeypatch.setattr(gen, "_api", lambda: "API")
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    return calls


def test_design_normal_execute(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    out = gen.ITLBreezeTTSVoiceDesign.execute(model="H", text="hi", instruction="deep", seed=3, cfg_scale=4.0)
    assert out.args[0]["sample_rate"] == 24000
    kw = calls["kw"]
    assert calls["handle"] == "H" and calls["api"] == "API"
    assert kw["mode"] == "design" and kw["text"] == "hi" and kw["instruction"] == "deep"
    assert kw["seed"] == 3 and kw["cfg_scale"] == 4.0 and kw["temp_dir"] == str(tmp_path)
    assert kw["sampling"] == core.DEFAULT_SAMPLING
    assert kw["reference_audio"] is None and kw["reference_text"] is None


def test_direction_advanced_execute_passes_sampling(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    ref = {"waveform": torch.zeros((1, 1, 5)), "sample_rate": 8000}
    gen.ITLBreezeTTSVoiceDirectionAdvanced.execute(
        model="H", text="hi", reference_audio=ref, reference_text="hi", instruction="fast",
        seed=1, cfg_scale=2.5, temperature=0.5, top_k=10, top_p=0.9, repetition_penalty=1.3, max_new_tokens=300)
    kw = calls["kw"]
    assert kw["mode"] == "direction" and kw["reference_audio"] is ref and kw["reference_text"] == "hi"
    assert kw["sampling"] == core.SamplingConfig(0.5, 10, 0.9, 1.3, 300)


def test_clone_normal_execute(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    ref = {"waveform": torch.zeros((1, 1, 5)), "sample_rate": 8000}
    gen.ITLBreezeTTSVoiceClone.execute(model="H", text="hi", reference_audio=ref, reference_text="hi", seed=1, cfg_scale=1.0)
    assert calls["kw"]["mode"] == "clone" and calls["kw"]["instruction"] is None


def test_api_import_error_names_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "breeze_infer", None)
    monkeypatch.setitem(sys.modules, "breeze_infer.templates", None)
    with pytest.raises(ImportError, match="not installed"):
        gen._api()


def test_pack_registers_all_seven_breeze_nodes():
    import importlib.util

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "itl_pack", os.path.join(repo_root, "__init__.py"), submodule_search_locations=[repo_root]
    )
    root = importlib.util.module_from_spec(spec)
    sys.modules["itl_pack"] = root
    spec.loader.exec_module(root)

    ids = ["ITLBreezeTTSLoader", *EXPECTED]
    for node_id in ids:
        assert node_id in root.NODE_CLASS_MAPPINGS, node_id
        assert root.NODE_CLASS_MAPPINGS[node_id].define_schema().node_id == node_id
        assert node_id in root.NODE_DISPLAY_NAME_MAPPINGS
    assert root.NODE_DISPLAY_NAME_MAPPINGS["ITLBreezeTTSLoader"] == "ITL Breeze TTS Loader"
