# Golden test for the Ideogram 4 Prompt Builder node — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
# ComfyUI env only (see docs/superpowers/plans/2026-10-24-moss-image-prompt-builder.md, Task 1): freezes the
# node's `prompt` output so the layout_formats refactor can prove it changed nothing.
import json
import pytest

pytest.importorskip("comfy_api")
from nodes.ideogram4_nodes import ITLIdeogram4PromptBuilder as Node  # noqa: E402
from golden_ideogram4 import (  # noqa: E402  (tests/ is on sys.path: pytest prepends the test dir)
    GOLDEN_BOXES, ELEMS, PLAIN_COMPACT, PHOTO_PREFIX, ART_PREFIX, ABS_XY_BOX0, UNPLACED, DISABLED_AND_GROUP,
)


def run(**kw):
    args = dict(width=1024, height=768, background="A beach", style={"style": "none"}, elements_data=ELEMS,
                output_format="compact")
    args.update(kw)
    return Node.execute(**args).args[0]


def test_plain_compact():
    assert run() == PLAIN_COMPACT


def test_photo_style_and_hld():
    out = run(high_level_description="Sunset", style={"style": "photo", "photo": "35mm"}, aesthetics="moody",
              medium="film", style_palette_data='["#112233"]')
    assert out.startswith(PHOTO_PREFIX)


def test_art_style_key_order():
    out = run(style={"style": "art_style", "art_style": "ukiyo-e"}, lighting="soft")
    assert out.startswith(ART_PREFIX)


def test_absolute_xy():
    out = json.loads(run(coord_mode="absolute", bbox_order="xy"))
    assert out["compositional_deconstruction"]["elements"][0]["bbox"] == ABS_XY_BOX0


def test_unplaced_element_has_no_bbox():
    out = json.loads(run(elements_data=json.dumps(UNPLACED)))
    assert out["compositional_deconstruction"]["elements"] == [{"type": "obj", "desc": "a crab", "color_palette": ["#FF0000"]}]


def test_disabled_and_group_boxes_are_skipped():
    out = json.loads(run(elements_data=json.dumps(DISABLED_AND_GROUP)))
    assert out["compositional_deconstruction"]["elements"] == [{"type": "text", "bbox": [500, 500, 600, 750], "text": "HELLO", "desc": "title"}]


def test_pretty_keeps_scalar_arrays_inline():
    out = run(output_format="pretty")
    assert '"bbox": [200, 100, 600, 400]' in out and out.startswith("{\n    \"compositional_deconstruction\": {\n")
    assert json.loads(out) == json.loads(PLAIN_COMPACT)


def test_wired_import_always_strips_sidecar():
    imported = json.loads(PLAIN_COMPACT)
    imported["_itl"] = {"v": 1, "boxes": GOLDEN_BOXES}
    out = run(elements_data="", import_json=json.dumps(imported), import_mode="always")
    assert out == PLAIN_COMPACT


def test_bboxes_output_and_dims():
    res = Node.execute(width=1024, height=768, background="A beach", style={"style": "none"}, elements_data=ELEMS,
                       output_format="compact")
    assert res.args[2] == [[{"x": 102, "y": 154, "width": 307, "height": 307}, {"x": 512, "y": 384, "width": 256, "height": 77}]]
    assert res.args[3:5] == (1024, 768) and tuple(res.args[1].shape[1:]) == (768, 1024, 3)
