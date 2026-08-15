# Tests for the PNG metadata builder — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from PIL import Image
from nodes.civitai_metadata.png_info import build_pnginfo


def _roundtrip(info, tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (2, 2)).save(str(p), pnginfo=info)
    return Image.open(str(p)).text


def test_parameters_only(tmp_path):
    text = _roundtrip(build_pnginfo("hello params"), tmp_path)
    assert text["parameters"] == "hello params"
    assert "workflow" not in text
    assert "prompt" not in text


def test_with_workflow_and_prompt(tmp_path):
    info = build_pnginfo("p", extra_pnginfo={"workflow": {"nodes": []}}, prompt={"1": {"class_type": "X"}})
    text = _roundtrip(info, tmp_path)
    assert text["parameters"] == "p"
    assert '"nodes"' in text["workflow"]
    assert '"class_type"' in text["prompt"]
