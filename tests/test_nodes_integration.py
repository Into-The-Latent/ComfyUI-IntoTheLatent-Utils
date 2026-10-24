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
    assert json.loads(res.ui["caption"][0]) == json.loads(moss)           # the editor gets the raw import and converts it itself
    res = Ideo.execute(1024, 1024, "", {"style": "none"}, import_json=moss, import_mode="always", output_format="compact",
                       coord_mode="absolute", bbox_order="xy")             # the output honours the toggles; the mirror is unaffected
    assert json.loads(res.args[0])["compositional_deconstruction"]["elements"][0]["bbox"] == [128, 96, 896, 224]
    assert json.loads(res.ui["caption"][0]) == json.loads(moss)
    # a non-layout JSON object is ignored like prose: the editor output wins
    assert Ideo.execute(1024, 1024, "Editor wins.", {"style": "none"}, import_json='{"caption": "x"}', import_mode="always",
                        output_format="compact").args[0] == '{"compositional_deconstruction":{"background":"Editor wins.","elements":[]}}'


def test_moss_node_builds_layout_and_converts_ideogram_import():
    import json
    from nodes.moss_image_nodes import ITLMossImagePromptBuilder as Moss
    boxes = json.dumps([{"x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125, "type": "text", "text": "EXPLORE", "desc": "Large title.", "palette": []}])
    res = Moss.execute(1024, 1024, "Poster.", elements_data=boxes)
    out = json.loads(res.args[0])
    assert out["items"][0]["bbox"] == [-416, -384, -288, 384] and out["summary"] == {"caption": "Poster."}
    assert res.args[2] == [[{"x": 128, "y": 96, "width": 768, "height": 128}]] and res.args[3:5] == (1024, 1024)
    assert tuple(res.args[1].shape[1:]) == (1024, 1024, 3) and res.ui == {"dims": [1024, 1024]}
    assert Moss.execute(1024, 1024, "Just prose.").args[0] == "Just prose."                         # no items → plain text
    assert Moss.execute(1024, 1024, "Editor wins.", import_json="not json at all").args[0] == "Editor wins."
    ideo = ('{"compositional_deconstruction":{"background":"A beach","elements":[{"type":"obj","bbox":[200,100,600,400],"desc":"a crab"},'
            '{"type":"text","text":"later","desc":"unplaced"}]},"_itl":{"v":1,"boxes":[]}}')
    res = Moss.execute(1024, 1024, "ignored", import_json=ideo, import_mode="always")
    conv = json.loads(res.args[0])
    assert conv["summary"]["caption"] == "A beach" and [i["label"] for i in conv["items"]] == ["object"] and "_itl" not in res.args[0]
    assert json.loads(res.ui["caption"][0]) == json.loads(ideo)       # the editor gets the raw import and converts it itself (keeps the unplaced element)
    empty = '{"compositional_deconstruction":{"background":"Only a scene.","elements":[]}}'   # no items → plain prompt
    res = Moss.execute(1024, 1024, "ignored", import_json=empty, import_mode="always")
    assert res.args[0] == "Only a scene." and json.loads(res.ui["caption"][0]) == json.loads(empty)
    moss_in = json.dumps(conv)                                                                       # same-format import: rebuilt at the node's size
    assert json.loads(Moss.execute(1024, 1024, "ignored", import_json=moss_in, import_mode="always").args[0]) == conv
    rebuilt = json.loads(Moss.execute(1024, 768, "ignored", import_json=moss_in, import_mode="always").args[0])
    assert rebuilt["items"][0]["bbox"] == [-308, -410, 103, -102]   # pixels from the centre: inside 1024×768, so unchanged
    oversize = '{"summary":{"caption":"O."},"items":[{"label":"object","caption":"o","bbox":[-600,-600,600,600]}]}'
    assert json.loads(Moss.execute(1024, 768, "ignored", import_json=oversize, import_mode="always").args[0])["items"][0]["bbox"] == [-384, -512, 384, 512]
    draft = '{"summary":{"caption":"D."},"items":[{"label":"object","caption":"o","bbox":[128,96,896,224]}],"bbox_coord_space":"abs","bbox_format":"xyxy"}'
    assert json.loads(Moss.execute(1024, 1024, "ignored", import_json=draft, import_mode="always").args[0])["items"][0]["bbox"] == [-416, -384, -288, 384]
    no_items = '{"summary":{"caption":"Only a caption."},"items":[]}'
    assert Moss.execute(1024, 1024, "ignored", import_json=no_items, import_mode="always").args[0] == "Only a caption."
    assert json.loads(Moss.execute(1024, 1024, "Poster.", import_json=moss_in, elements_data=boxes).args[0]) == out   # "when empty" + regions: editor wins
