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


def test_sidecar_boxes_drive_the_import_and_never_reach_the_prompt():
    # A clipboard/template sidecar carries the full editor layout: tags, foreign labels, groups, disabled flags.
    # The import uses those boxes (labels normalised) so the prompt agrees with what the editor shows.
    side = [{"id": "g", "x": 0, "y": 0, "w": 1, "h": 1, "type": "obj", "group": True, "name": "G"},
            {"id": "a", "parent": "g", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "type": "character", "desc": "hero", "tags": "anime", "palette": []},
            {"id": "b", "x": 0.5, "y": 0.5, "w": 0.25, "h": 0.1, "type": "text", "text": "HELLO", "desc": "title", "disabled": True}]
    ideo = dict(IDEO, _itl={"v": 1, "boxes": side})
    boxes, scene, _ = import_layout(ideo, "moss", 1024, 1024)
    assert [b["type"] for b in boxes] == ["object", "character", "text"] and boxes[0]["group"] is True   # labels normalised to MOSS
    out = build_moss(boxes, scene, 1024, 1024)
    assert "_itl" not in compact(out) and out["items"] == [{"label": "character", "caption": "hero", "tags": ["anime"], "bbox": [-308, -410, 103, -102]}]
    boxes, _, _ = import_layout(ideo, "ideogram", 1024, 1024)
    assert [b["type"] for b in boxes] == ["obj", "obj", "text"] and boxes[1]["tags"] == "anime"
    assert len(import_layout(dict(IDEO, _itl={"v": 1, "boxes": []}), "moss", 1024, 1024)[0]) == 2   # empty sidecar → the plain elements


def test_non_string_scene_values_are_ignored_not_fatal():
    from nodes.layout_formats import parse_ideogram, parse_moss, ideogram_scene_to_moss
    cap = {"high_level_description": ["Sunset"], "style_description": {"photo": 5, "aesthetics": {"x": 1}, "lighting": "soft"},
           "compositional_deconstruction": {"background": None, "elements": [{"type": "text", "text": 7, "desc": ["d"], "bbox": [0, 0, 100, 100]}]}}
    boxes, scene = parse_ideogram(cap)
    assert scene["high_level_description"] == "" and scene["photo"] == "5" and scene["aesthetics"] == "" and scene["lighting"] == "soft"
    assert boxes[0]["text"] == "7" and boxes[0]["desc"] == ""
    assert compact(build_ideogram(boxes, scene)).startswith('{"style_description":{"aesthetics":"","lighting":"soft","photo":"5"')
    assert ideogram_scene_to_moss(scene)["caption"] == "Style: 5, soft."
    _, ms = parse_moss({"summary": {"caption": ["x"], "meta": "a,b", "canvas": 3}, "items": []}, 1024, 1024)
    assert ms == {"caption": "", "meta": ["a", "b"], "canvas": "none", "palette": []}


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        import_layout({"caption": "x"}, "moss", 1024, 1024)
    with pytest.raises(ValueError):
        import_layout(IDEO, "qwen", 1024, 1024)
