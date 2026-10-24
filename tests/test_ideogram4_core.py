# Tests for the comfy-free Ideogram 4 format — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import json

from nodes.layout_formats import build_ideogram, parse_ideogram, compact, dumps, seed_boxes, bbox_outputs, loads_layout
from golden_ideogram4 import (
    GOLDEN_BOXES, PLAIN_COMPACT, PHOTO_PREFIX, ART_PREFIX, ABS_XY_BOX0, UNPLACED, DISABLED_AND_GROUP,
)

SCENE = dict(high_level_description="", background="A beach", style_kind="none", photo="", art_style="",
             aesthetics="", lighting="", medium="", palette=[])


def test_build_matches_golden():
    assert compact(build_ideogram(GOLDEN_BOXES, SCENE)) == PLAIN_COMPACT


def test_photo_style_and_hld():
    scene = dict(SCENE, high_level_description="Sunset", style_kind="photo", photo="35mm", aesthetics="moody",
                 medium="film", palette=["#112233"])
    assert compact(build_ideogram(GOLDEN_BOXES, scene)).startswith(PHOTO_PREFIX)


def test_art_style_key_order():
    scene = dict(SCENE, style_kind="art_style", art_style="ukiyo-e", lighting="soft")
    assert compact(build_ideogram(GOLDEN_BOXES, scene)).startswith(ART_PREFIX)


def test_absolute_xy_uses_width_height():
    out = build_ideogram(GOLDEN_BOXES, SCENE, coord_mode="absolute", bbox_order="xy", width=1024, height=768)
    assert out["compositional_deconstruction"]["elements"][0]["bbox"] == ABS_XY_BOX0


def test_unplaced_disabled_group():
    assert build_ideogram(UNPLACED, SCENE)["compositional_deconstruction"]["elements"] == [
        {"type": "obj", "desc": "a crab", "color_palette": ["#FF0000"]}]
    assert build_ideogram(DISABLED_AND_GROUP, SCENE)["compositional_deconstruction"]["elements"] == [
        {"type": "text", "bbox": [500, 500, 600, 750], "text": "HELLO", "desc": "title"}]


def test_pretty_keeps_scalar_arrays_inline():
    out = dumps(build_ideogram(GOLDEN_BOXES, SCENE))
    assert '"bbox": [200, 100, 600, 400]' in out and json.loads(out) == json.loads(PLAIN_COMPACT)


def test_parse_roundtrip():
    scene = dict(SCENE, high_level_description="Sunset", style_kind="photo", photo="35mm", aesthetics="moody",
                 medium="film", palette=["#112233"])
    boxes, back = parse_ideogram(build_ideogram(GOLDEN_BOXES, scene))
    assert [b["type"] for b in boxes] == ["obj", "text"] and boxes[1]["text"] == "HELLO"
    assert abs(boxes[0]["x"] - 0.1) < 1e-3 and abs(boxes[0]["h"] - 0.4) < 1e-3
    assert boxes[0]["palette"] == ["#FF0000"] and boxes[0]["desc"] == "a crab"
    assert back == scene
    assert parse_ideogram({"compositional_deconstruction": {"background": "b", "elements": [{"type": "obj", "desc": "x"}]}})[0][0]["nobbox"] is True


def test_parse_scene_kinds():
    assert parse_ideogram({"compositional_deconstruction": {"background": "b"}})[1]["style_kind"] == "none"
    assert parse_ideogram({"style_description": {"art_style": "x"}, "compositional_deconstruction": {}})[1]["style_kind"] == "art_style"


def test_seed_boxes_accepts_dict_flat_and_nested():
    one = {"x": 102, "y": 154, "width": 308, "height": 307}
    assert seed_boxes(one, 1024, 768) == seed_boxes([one], 1024, 768) == seed_boxes([[one]], 1024, 768)
    assert seed_boxes([one], 1024, 768)[0]["type"] == "obj"
    assert seed_boxes(None, 1024, 768) == [] and seed_boxes([], 1024, 768) == []


def test_bbox_outputs_normalizes_negative_sizes():
    assert bbox_outputs([{"x": 0.4, "y": 0.6, "w": -0.3, "h": -0.4, "type": "obj"}], 1024, 768) == \
        [[{"x": 102, "y": 154, "width": 307, "height": 307}]]
    assert bbox_outputs(UNPLACED, 1024, 768) == []


def test_loads_layout_repairs_fences_and_trailing_commas():
    assert loads_layout('```json\n{"a": [1, 2,], "b": "x,",}\n```') == {"a": [1, 2], "b": "x,"}
    assert loads_layout("not json") is None and loads_layout("") is None
