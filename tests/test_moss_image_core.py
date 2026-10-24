# Tests for the comfy-free MOSS-Image layout format — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.layout_formats import build_moss, parse_moss, moss_bbox, from_moss_bbox, compact, strs, moss_label

SCENE = dict(caption="A minimalist travel poster.", meta=[], canvas="none", palette=[])
BOX = {"x": 128 / 1024, "y": 96 / 1024, "w": 768 / 1024, "h": 128 / 1024}   # USERGUIDE "EXPLORE" title at 1024²
FRACS = {"x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125}


def test_userguide_example_bbox():
    assert moss_bbox(BOX, 1024, 1024) == [-416, -384, -288, 384]


def test_bbox_floor_ceil_and_centre_on_nonsquare():
    # 0.1*1024=102.4→102, 0.4*1024=409.6→410, 0.2*768=153.6→153, 0.6*768=460.8→461; minus 384 / 512
    assert moss_bbox({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, 1024, 768) == [153 - 384, 102 - 512, 461 - 384, 410 - 512]


def test_bbox_negative_size_and_overflow_are_normalized():
    assert moss_bbox({"x": 0.4, "y": 0.6, "w": -0.3, "h": -0.4}, 1024, 768) == moss_bbox({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, 1024, 768)
    assert moss_bbox({"x": -0.5, "y": -0.5, "w": 2.0, "h": 2.0}, 1024, 1024) == [-512, -512, 512, 512]


def test_zero_area_box_is_none():
    assert moss_bbox({"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.2}, 1024, 1024) is None
    assert moss_bbox({"x": 0.5, "y": 0.5, "w": 0.0004, "h": 0.2}, 1024, 1024) is not None   # floor/ceil keeps a sub-pixel box


def test_from_moss_bbox_spaces_and_orders():
    assert from_moss_bbox([-416, -384, -288, 384], 1024, 1024) == FRACS
    assert from_moss_bbox([96, 128, 224, 896], 1024, 1024, space="abs") == FRACS
    assert from_moss_bbox([128, 96, 896, 224], 1024, 1024, space="abs", fmt="xyxy") == FRACS
    assert from_moss_bbox([-900, -900, 900, 900], 1024, 1024) == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}   # oversize → clamped
    assert from_moss_bbox([100, 100, -100, -100], 1024, 1024)["w"] > 0                                   # reversed edges → positive size
    assert from_moss_bbox("nope", 1024, 1024) is None and from_moss_bbox([1, "x", 3, 4], 1024, 1024) is None


def test_build_key_order_and_compact():
    boxes = [dict(FRACS, type="text", text="EXPLORE", desc="Large title.", palette=["#1a2b3c"], tags="bold, uppercase"),
             {"x": 0.3, "y": 0.6, "w": 0.4, "h": 0.3, "type": "obj", "text": "", "desc": "A mountain.", "palette": []}]
    out = compact(build_moss(boxes, dict(caption="Poster.", meta=["poster", ""], canvas="single", palette=["#ffffff"]), 1024, 1024))
    assert out == ('{"summary":{"meta":["poster"],"canvas":"single","caption":"Poster.","palette":["#FFFFFF"]},'
                   '"items":[{"label":"text","caption":"Large title.","text":"EXPLORE","tags":["bold","uppercase"],"palette":["#1A2B3C"],"bbox":[-416,-384,-288,384]},'
                   '{"label":"object","caption":"A mountain.","bbox":[102,-205,410,205]}],'
                   '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}')


def test_build_skips_nobbox_group_disabled_zero_and_returns_none_when_empty():
    boxes = [{"x": 0, "y": 0, "w": 0.2, "h": 0.2, "type": "obj", "desc": "x", "nobbox": True},
             {"id": "g", "group": True, "x": 0, "y": 0, "w": 1, "h": 1, "type": "obj"},
             {"id": "d", "x": 0, "y": 0, "w": 0.5, "h": 0.5, "type": "obj", "desc": "x", "disabled": True},
             {"id": "c", "parent": "d", "x": 0, "y": 0, "w": 0.5, "h": 0.5, "type": "obj", "desc": "child of disabled"},
             {"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.5, "type": "obj", "desc": "zero"},
             "not a dict"]
    assert build_moss(boxes, SCENE, 1024, 1024) is None


def test_non_ascii_unescaped():
    boxes = [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.2, "type": "text", "text": "把日常，过成诗。", "desc": "标题", "palette": []}]
    assert "把日常，过成诗。" in compact(build_moss(boxes, SCENE, 1024, 1024))


def test_canvas_none_and_blank_meta_omitted_and_caps():
    box = {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.2, "type": "character", "desc": "x", "palette": ["#%06x" % i for i in range(7)]}
    out = build_moss([box], dict(SCENE, meta=[" ", ""], palette=["#%06x" % i for i in range(20)]), 1024, 1024)
    assert list(out["summary"]) == ["caption", "palette"] and len(out["summary"]["palette"]) == 16
    assert out["items"][0]["label"] == "character" and len(out["items"][0]["palette"]) == 5 and "text" not in out["items"][0]


def test_parse_moss():
    layout = {"summary": {"caption": "Poster.", "meta": ["poster"], "canvas": "single", "palette": ["#FFFFFF"]},
              "items": [{"label": "character", "caption": "A hero.", "tags": ["anime", "red"], "bbox": [-416, -384, -288, 384]},
                        {"label": "text", "caption": "t", "text": "HI", "bbox": "bad"},
                        "junk"],
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    boxes, scene = parse_moss(layout, 1024, 1024)
    assert boxes[0] == dict(FRACS, type="character", text="", desc="A hero.", tags="anime, red", palette=[])
    assert boxes[1]["nobbox"] is True and boxes[1]["type"] == "text" and boxes[1]["text"] == "HI" and len(boxes) == 2
    assert scene == {"caption": "Poster.", "meta": ["poster"], "canvas": "single", "palette": ["#FFFFFF"]}


def test_parse_moss_honours_space_and_format_and_defaults():
    layout = {"summary": {"caption": "x", "canvas": "weird"}, "items": [{"label": "object", "caption": "o", "bbox": [128, 96, 896, 224]}],
              "bbox_coord_space": "abs", "bbox_format": "xyxy"}
    boxes, scene = parse_moss(layout, 1024, 1024)
    assert {k: boxes[0][k] for k in FRACS} == FRACS and scene["canvas"] == "none" and scene["meta"] == []
    assert parse_moss({"items": [{"label": "obj", "bbox": [0, 0, 10, 10]}]}, 1024, 1024)[0][0]["type"] == "object"


def test_strs_and_label():
    assert strs("a, b\n c,,") == ["a", "b", "c"] and strs(["x", " ", 3]) == ["x", "3"] and strs(None) == []
    assert moss_label("obj") == "object" and moss_label("character") == "character" and moss_label(None) == "object"
    assert moss_label("text") == "text" and moss_label("object") == "object"
