# Integration test for the Save Metadata (Civitai) nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import pytest

pytest.importorskip("comfy_api")  # only runs inside a ComfyUI environment
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def test_basic_node_writes_parameters_chunk(tmp_path, monkeypatch):
    import folder_paths
    from nodes.save_civitai_metadata import ITLSaveCivitaiMetadata as Node

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(folder_paths, "get_full_path", lambda kind, name: None)  # skip hashing

    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a fox", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blur", "clip": ["1", 1]}},
        "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0],
              "negative": ["3", 0], "seed": 42, "steps": 25, "cfg": 7.0,
              "sampler_name": "euler", "scheduler": "normal"}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
        "7": {"class_type": "ITLSaveCivitaiMetadata", "inputs": {"images": ["6", 0]}},
    }

    class _Hidden:
        prompt = prompt
        extra_pnginfo = {"workflow": {"nodes": []}}
        unique_id = "7"

    monkeypatch.setattr(Node, "hidden", _Hidden, raising=False)

    import torch
    images = torch.zeros((1, 64, 48, 3))  # [batch, H, W, C]
    Node.execute(images, filename_prefix="ITLTest", save_workflow=False)

    pngs = list(tmp_path.glob("ITLTest_*.png"))
    assert pngs, "no PNG written"
    text = Image.open(str(pngs[0])).text
    params = text["parameters"]
    assert params.startswith("a fox")
    assert "Negative prompt: blur" in params
    assert "Steps: 25" in params and "Seed: 42" in params and "Size: 48x64" in params
    assert "workflow" not in text  # save_workflow was off


def test_ideogram_node_converts_moss_import():
    import json
    from nodes.ideogram4_nodes import ITLIdeogram4PromptBuilder as Ideo
    moss = ('{"summary":{"caption":"Poster."},"items":[{"label":"character","caption":"hero","bbox":[-416,-384,-288,384]}],'
            '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}')
    res = Ideo.execute(1024, 1024, "", {"style": "none"}, import_json=moss, import_mode="always", output_format="compact")
    out = json.loads(res.args[0])
    assert out["compositional_deconstruction"]["background"] == "Poster."
    assert out["compositional_deconstruction"]["elements"][0] == {"type": "obj", "bbox": [94, 125, 219, 875], "desc": "hero"}
    assert json.loads(res.ui["caption"][0]) == out                        # the editor mirrors the converted caption
    # a non-layout JSON object is ignored like prose: the editor output wins
    assert Ideo.execute(1024, 1024, "Editor wins.", {"style": "none"}, import_json='{"caption": "x"}', import_mode="always",
                        output_format="compact").args[0] == '{"compositional_deconstruction":{"background":"Editor wins.","elements":[]}}'
