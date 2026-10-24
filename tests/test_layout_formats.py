# Tests for format detection + Ideogram 4 <-> MOSS-Image conversion — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import pytest

from nodes.layout_formats import (
    detect_format, ideogram_scene_to_moss, moss_scene_to_ideogram, import_layout, build_ideogram, build_moss, compact,
)

IDEO = {"high_level_description": "Sunset",
        "style_description": {"aesthetics": "moody", "lighting": "", "photo": "35mm", "medium": "film", "color_palette": ["#112233"]},
        "compositional_deconstruction": {"background": "A beach", "elements": [
            {"type": "obj", "bbox": [200, 100, 600, 400], "desc": "a crab", "color_palette": ["#FF0000"]},
            {"type": "text", "bbox": [500, 500, 600, 750], "text": "HELLO", "desc": "title"}]}}


def ideo_scene(**kw):
    s = dict(high_level_description="", background="", style_kind="none", photo="", art_style="", aesthetics="",
             lighting="", medium="", palette=[])
    s.update(kw)
    return s


def test_detect():
    assert detect_format(IDEO) == "ideogram"
    assert detect_format({"summary": {"caption": "x"}, "items": []}) == "moss"
    assert detect_format({"caption": "x"}) is None and detect_format("str") is None and detect_format(None) is None


@pytest.mark.parametrize("scene,expected", [
    (ideo_scene(high_level_description="Sunset", background="A beach", style_kind="photo", photo="35mm", aesthetics="moody",
                medium="film", palette=["#112233"]), "Sunset A beach Style: 35mm, moody, film."),
    (ideo_scene(background="A beach", style_kind="none", aesthetics="moody"), "A beach"),
    (ideo_scene(background="A beach", style_kind="art_style"), "A beach"),
    (ideo_scene(style_kind="art_style", art_style="ukiyo-e", lighting="soft"), "Style: ukiyo-e, soft."),
    (ideo_scene(), ""),
])
def test_caption_join(scene, expected):
    out = ideogram_scene_to_moss(scene)
    assert out["caption"] == expected and out["meta"] == [] and out["canvas"] == "none" and out["palette"] == scene["palette"]


def test_moss_scene_to_ideogram():
    s = moss_scene_to_ideogram(dict(caption="Poster.", meta=["poster"], canvas="single", palette=["#FFFFFF"]))
    assert s == ideo_scene(background="Poster.", palette=["#FFFFFF"])


def test_ideogram_to_moss_to_ideogram_within_one_grid_unit():
    boxes, scene, conv = import_layout(IDEO, "moss", 1024, 768)
    assert conv and [b["type"] for b in boxes] == ["object", "text"]
    moss = build_moss(boxes, scene, 1024, 768)
    assert moss["summary"]["caption"] == "Sunset A beach Style: 35mm, moody, film." and moss["summary"]["palette"] == ["#112233"]
    back_boxes, back_scene, conv2 = import_layout(moss, "ideogram", 1024, 768)
    assert conv2 and back_scene["background"] == "Sunset A beach Style: 35mm, moody, film." and back_scene["palette"] == ["#112233"]
    out = build_ideogram(back_boxes, back_scene)["compositional_deconstruction"]["elements"]
    assert len(out) == 2
    for got, want in zip(out, IDEO["compositional_deconstruction"]["elements"]):
        assert all(abs(g - w) <= 1 for g, w in zip(got["bbox"], want["bbox"]))
        assert got.get("text") == want.get("text") and got["desc"] == want["desc"] and got.get("color_palette") == want.get("color_palette")


def test_same_format_import_is_not_converted():
    boxes, scene, conv = import_layout(IDEO, "ideogram", 1024, 768)
    assert not conv and scene["style_kind"] == "photo" and boxes[0]["type"] == "obj"
    moss = {"summary": {"caption": "P."}, "items": [{"label": "character", "caption": "hero", "bbox": [-100, -100, 100, 100]}]}
    boxes, scene, conv = import_layout(moss, "moss", 1024, 1024)
    assert not conv and boxes[0]["type"] == "character" and scene["caption"] == "P."


def test_tags_survive_editor_roundtrip_and_character_becomes_obj():
    moss = {"summary": {"caption": "P."}, "items": [{"label": "character", "caption": "hero", "tags": ["anime"], "bbox": [-100, -100, 100, 100]}]}
    boxes, scene, _ = import_layout(moss, "ideogram", 1024, 1024)
    assert boxes[0]["type"] == "obj" and boxes[0]["tags"] == "anime"
    assert "tags" not in compact(build_ideogram(boxes, scene))
    boxes2, _, _ = import_layout(build_ideogram(boxes, scene), "moss", 1024, 1024)   # obj→object; tags are not in Ideogram JSON
    assert boxes2[0]["type"] == "object" and "tags" not in boxes2[0]


def test_sidecar_never_reaches_moss():
    ideo = dict(IDEO, _itl={"v": 1, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1, "type": "obj", "group": True, "id": "g"}]})
    boxes, scene, _ = import_layout(ideo, "moss", 1024, 1024)
    assert "_itl" not in compact(build_moss(boxes, scene, 1024, 1024)) and len(boxes) == 2


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        import_layout({"caption": "x"}, "moss", 1024, 1024)
    with pytest.raises(ValueError):
        import_layout(IDEO, "qwen", 1024, 1024)
