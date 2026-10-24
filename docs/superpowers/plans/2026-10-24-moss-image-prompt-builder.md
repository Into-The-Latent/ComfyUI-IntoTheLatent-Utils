# MOSS-Image Prompt Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second layout node, `ITLMossImagePromptBuilder`, that emits MOSS-Image's layout JSON from the same bbox editor as the Ideogram 4 node, with cross-format import both ways and the Ideogram output provably unchanged.

**Architecture:** All format logic (Ideogram build/parse, MOSS build/parse, detection, conversion) moves into a new comfy-free module `nodes/layout_formats.py`; both node files become thin schema + `execute` wrappers. The JS editor `web/js/ideogram4_prompt_builder.js` stays the single editor implementation and reads a per-node spec object looked up by node name; `web/js/moss_image_prompt_builder.js` only holds the MOSS spec.

**Tech Stack:** Python 3.12, ComfyUI `comfy_api.latest.io` (v3 node API), pytest; ES-module frontend on LiteGraph (`window.comfyAPI.app`).

**Spec:** `docs/superpowers/specs/2026-10-23-moss-image-prompt-builder-design.md`

## Global Constraints

- One issue, one branch `feature/moss-image-prompt-builder`, one PR; commit and push after every task (`gh auth switch --user Into-The-Latent` before any push).
- Ideogram `prompt` output byte-identical before/after (Task 1 golden test guards it).
- No new dependencies, no torch floors, no env var / network / subprocess use (registry rules).
- `nodes/layout_formats.py` imports nothing from `comfy_api`, `torch`, PIL or numpy.
- MOSS bbox: `[ymin, xmin, ymax, xmax]`, pixel edges `floor(round(v,9))` for min / `ceil(round(v,9))` for max, minus `H//2` / `W//2`; `bbox_coord_space:"abs_centralized"`, `bbox_format:"yxyx"`.
- MOSS key orders: summary `meta?, canvas?, caption, palette?`; item `label, caption, text?, tags?, palette?, bbox`; root `summary, items, bbox_coord_space, bbox_format`.
- Compact output = `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`; pretty = existing `_dumps`.
- Ideogram→MOSS caption join: `<high_level_description> <background> Style: <photo|art_style>, <aesthetics>, <lighting>, <medium>.` — empties dropped, single-space joined, `Style:` sentence omitted when kind is `none` or all style fields blank.
- MOSS→Ideogram: `summary.caption` → `background`; `high_level_description` empty; style `none`; `summary.palette` → style palette; `character` → `obj`; `meta`/`canvas` dropped; `tags` kept on the box, never exported.
- Labels: Ideogram boxes `obj|text`; MOSS boxes `object|character|text`. Palette caps 5 per box, 16 scene.
- Templates stay in `itl/ideogram4/templates`; sidecar `_itl` emitted when any box has `group`, `parent`, non-empty `tags`, or a label outside the node's own set.
- Version `1.13.0`.
- Comfy-dependent tests run in the ComfyUI venv per the runner in Task 1; comfy-free tests run with the system Python: `python -m pytest tests/<file> -v` from the repo root.

## Review Focus

1. A box whose edges collapse to the same pixel after rounding (w or h = 0): a MOSS consumer rejects a zero-area bbox, so the item is skipped like an unplaced one — test in Task 3.
2. Non-ASCII captions (Chinese text) must reach the model unescaped — `ensure_ascii=False` test in Task 3.
3. `import_json` carrying an Ideogram caption with the `_itl` sidecar wired into the MOSS node — the sidecar must not leak into the MOSS output — test in Task 4.
4. `import_json` that is plain prose (an LLM reply, not JSON) on the MOSS node — ignored, editor output used, no exception — test in Task 5.
5. A MOSS layout whose boxes exceed the node's canvas (layout made for a larger size) — clamped into 0..1, never negative sizes — test in Task 3.

---

### Task 1: Issue, branch, golden test for the current Ideogram output

**Files:**
- Create: `tests/test_ideogram4_node_golden.py`
- Create: `tests/golden_ideogram4.py` — plain module (no pytest imports, no `importorskip`) holding `GOLDEN_BOXES`, `ELEMS`, and the frozen expected strings (`PLAIN_COMPACT`, `PHOTO_PREFIX`, …) so the comfy-free test in Task 2 can import them under the system Python
- Create (scratch, not committed): `<scratchpad>/run_comfy_tests.py`
- Commit: `docs/superpowers/specs/2026-10-23-moss-image-prompt-builder-design.md`

**Interfaces:**
- Produces: the golden fixture `GOLDEN_BOXES` and the expected strings reused by Task 2's comfy-free test.

- [ ] **Step 1: Issue and branch**

```bash
gh auth switch --user Into-The-Latent
gh issue create --title "MOSS-Image Prompt Builder: second layout node on the shared bbox editor" --body "Design: docs/superpowers/specs/2026-10-23-moss-image-prompt-builder-design.md. Separate node ITLMossImagePromptBuilder sharing the Ideogram 4 editor; cross-format import both ways; Ideogram output unchanged."
git checkout -b feature/moss-image-prompt-builder
git add docs/superpowers/specs/2026-10-23-moss-image-prompt-builder-design.md
git commit -m "docs: MOSS-Image prompt builder design"
```

- [ ] **Step 2: Write the comfy-env runner (scratch)**

`<scratchpad>/run_comfy_tests.py` — per the repo's test notes: `sys.path.insert(0, "E:/Repos/ComfyUI-IntoTheLatent-Utils")` **before** `sys.path.insert(0, "E:/AI/ComfyUI")`, then insert `<scratchpad>/pytest_site`, `os.chdir(repo)`, `pytest.main(sys.argv[1:])`. Install pytest once: `E:/AI/ComfyUI/venv/Scripts/python.exe -m pip install --target <scratchpad>/pytest_site pytest`.

- [ ] **Step 3: Write the golden test**

```python
# tests/test_ideogram4_node_golden.py — ComfyUI env only; freezes the Ideogram 4 prompt output.
import json, pytest
pytest.importorskip("comfy_api")
from nodes.ideogram4_nodes import ITLIdeogram4PromptBuilder as Node

from tests.golden_ideogram4 import GOLDEN_BOXES, ELEMS   # GOLDEN_BOXES = the two boxes below; ELEMS = json.dumps(GOLDEN_BOXES)
# GOLDEN_BOXES = [
#     {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4, "type": "obj", "text": "", "desc": "a crab", "palette": ["#ff0000"]},
#     {"x": 0.5, "y": 0.5, "w": 0.25, "h": 0.1, "type": "text", "text": "HELLO", "desc": "title", "palette": []},
# ]

def run(**kw):
    args = dict(width=1024, height=768, background="A beach", style={"style": "none"}, elements_data=ELEMS, output_format="compact")
    args.update(kw)
    return Node.execute(**args).args[0]

def test_plain_compact():
    assert run() == ('{"compositional_deconstruction":{"background":"A beach","elements":['
                     '{"type":"obj","bbox":[200,100,600,400],"desc":"a crab","color_palette":["#FF0000"]},'
                     '{"type":"text","bbox":[500,500,600,750],"text":"HELLO","desc":"title"}]}}')

def test_photo_style_and_hld():
    out = run(high_level_description="Sunset", style={"style": "photo", "photo": "35mm"}, aesthetics="moody", medium="film",
              style_palette_data='["#112233"]')
    assert out.startswith('{"high_level_description":"Sunset","style_description":{"aesthetics":"moody","lighting":"","photo":"35mm","medium":"film","color_palette":["#112233"]},"compositional_deconstruction"')

def test_absolute_xy():
    out = json.loads(run(coord_mode="absolute", bbox_order="xy"))
    assert out["compositional_deconstruction"]["elements"][0]["bbox"] == [102, 154, 410, 461]
```

Add, in the same style, one test each for: `art_style` kind (key order `aesthetics, lighting, medium, art_style`), an unplaced box (`"nobbox": True` → no `bbox` key), a disabled box and a group box (both absent from `elements`), `output_format="pretty"` (assert equality with `json.dumps(json.loads(out), ensure_ascii=False, indent=4)` **after** collapsing scalar arrays — simplest: assert `out` contains `"bbox": [200, 100, 600, 400]` on one line), and a wired `import_json` in mode `always` whose `_itl` key is stripped from the output. For those, run the current code once, read the output, and freeze it as the literal; the hand-computed three above sanity-check the frozen ones.

- [ ] **Step 4: Run it in the ComfyUI env**

Run: `E:/AI/ComfyUI/venv/Scripts/python.exe <scratchpad>/run_comfy_tests.py tests/test_ideogram4_node_golden.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add tests/test_ideogram4_node_golden.py tests/golden_ideogram4.py
git commit -m "test(ideogram4): golden prompt output before the layout_formats refactor"
git push -u origin feature/moss-image-prompt-builder
```

---

### Task 2: `nodes/layout_formats.py` — move the Ideogram format out of the node

**Files:**
- Create: `nodes/layout_formats.py`
- Create: `tests/test_ideogram4_core.py`
- Modify: `nodes/ideogram4_nodes.py` (lines 156–250 move out; `execute` 353–470 shrinks)

**Interfaces:**
- Produces (all in `layout_formats`, comfy-free):
  - `norm_bbox(box, sx=1000, sy=1000, order="yx") -> list[int]` (today's `_norm_bbox`)
  - `palette(colors) -> list[str]`, `dumps(v, lvl=0) -> str`, `parse_json_list(s) -> list`, `repair_json(s) -> str`, `loads_layout(s) -> dict | None` (today's `_loads_caption`, log tag `[ITLLayout]`), `by_id(boxes)`, `eff_disabled(box, by_id)`
  - `compact(v) -> str` = `json.dumps(v, ensure_ascii=False, separators=(",", ":"))`
  - `IdeogramScene = dict` with keys `high_level_description, background, style_kind ("none"|"photo"|"art_style"), photo, art_style, aesthetics, lighting, medium, palette (list[str])`
  - `build_ideogram(boxes, scene, *, coord_mode="normalized", bbox_order="yx", width=1024, height=1024) -> dict` — today's `execute` lines 394–434, with `bsx, bsy` / `border` derived inside
  - `parse_ideogram(cap) -> tuple[list[dict], IdeogramScene]` — today's `_caption_to_boxes` plus the scene fields (`style_kind` from which of `photo`/`art_style` is a string, as the JS `applyCaption` does)
  - `seed_boxes(bboxes, width, height) -> list[dict]` — today's lines 366–379 (returns `[]` when `bboxes` is falsy)
  - `bbox_outputs(boxes, width, height) -> list[list[dict]]` — today's lines 444–461
- `ideogram4_nodes.py` keeps `_render_preview`, `_bg_image(image) -> PIL.Image | None` (today's lines 436–440, new helper), and imports the rest from `layout_formats`.

- [ ] **Step 1: Write the comfy-free test**

```python
# tests/test_ideogram4_core.py
from nodes.layout_formats import build_ideogram, parse_ideogram, compact, norm_bbox, seed_boxes, bbox_outputs
from tests.golden_ideogram4 import GOLDEN_BOXES, PLAIN_COMPACT   # shared fixture module (Task 1)

SCENE = dict(high_level_description="", background="A beach", style_kind="none", photo="", art_style="",
             aesthetics="", lighting="", medium="", palette=[])

def test_build_matches_golden():
    assert compact(build_ideogram(GOLDEN_BOXES, SCENE)) == <the literal from test_plain_compact>

def test_absolute_xy_uses_width_height():
    out = build_ideogram(GOLDEN_BOXES, SCENE, coord_mode="absolute", bbox_order="xy", width=1024, height=768)
    assert out["compositional_deconstruction"]["elements"][0]["bbox"] == [102, 154, 410, 461]

def test_parse_roundtrip():
    boxes, scene = parse_ideogram(build_ideogram(GOLDEN_BOXES, SCENE))
    assert [b["type"] for b in boxes] == ["obj", "text"] and boxes[1]["text"] == "HELLO"
    assert abs(boxes[0]["x"] - 0.1) < 1e-3 and scene["background"] == "A beach" and scene["style_kind"] == "none"

def test_seed_boxes_accepts_dict_flat_and_nested():
    one = {"x": 102, "y": 154, "width": 308, "height": 307}
    assert seed_boxes(one, 1024, 768) == seed_boxes([one], 1024, 768) == seed_boxes([[one]], 1024, 768)
    assert seed_boxes(None, 1024, 768) == []

def test_bbox_outputs_normalizes_negative_sizes():
    assert bbox_outputs([{"x": 0.4, "y": 0.6, "w": -0.3, "h": -0.4, "type": "obj"}], 1024, 768) == [[{"x": 102, "y": 154, "width": 307, "height": 307}]]
```

Plus the photo / art_style / unplaced / disabled / group cases from Task 1 against `build_ideogram`.

- [ ] **Step 2: Run, expect ImportError**

Run: `python -m pytest tests/test_ideogram4_core.py -v` — Expected: FAIL, `No module named 'nodes.layout_formats'`.

- [ ] **Step 3: Create `nodes/layout_formats.py` and cut `ideogram4_nodes.py` over**

Move the functions verbatim (rename by dropping the underscore; keep `_norm_bbox = norm_bbox` style aliases in `ideogram4_nodes.py` is unnecessary — nothing else imports them). `execute` becomes: parse `elements_data` → `seed_boxes` fallback → `loads_layout(import_json)` → scene dict from the widgets → `caption = {k: v for k, v in imported.items() if k != "_itl"}` / `parse_ideogram` when `used_import`, else `build_ideogram(...)` → `_render_preview` → `bbox_outputs` → `ui`. Module header keeps the KJNodes GPL attribution in both files.

- [ ] **Step 4: Run both test files**

Run: `python -m pytest tests/test_ideogram4_core.py -v` — Expected: PASS.
Run: `E:/AI/ComfyUI/venv/Scripts/python.exe <scratchpad>/run_comfy_tests.py tests/test_ideogram4_node_golden.py -v` — Expected: PASS, unchanged.

- [ ] **Step 5: Commit and push**

```bash
git add nodes/layout_formats.py nodes/ideogram4_nodes.py tests/test_ideogram4_core.py
git commit -m "refactor(ideogram4): move the caption format into comfy-free layout_formats"
git push
```

---

### Task 3: MOSS format — `build_moss` / `parse_moss`

**Files:**
- Modify: `nodes/layout_formats.py`
- Create: `tests/test_moss_image_core.py`

**Interfaces:**
- Produces:
  - `CANVASES = ("single", "multi_panel", "collage")`, `MOSS_LABELS = ("object", "character", "text")`
  - `strs(v) -> list[str]` — list or comma/newline-separated string → stripped non-empty strings
  - `moss_label(t) -> str` — `"text"`→`"text"`, `"character"`→`"character"`, anything else (`"obj"`, `"object"`, `None`) → `"object"`
  - `moss_bbox(box, width, height) -> list[int] | None` — None when the rounded box has zero width or height
  - `from_moss_bbox(bb, width, height, space="abs_centralized", fmt="yxyx") -> dict | None` — `{x, y, w, h}` fractions clamped to 0..1, None for malformed input
  - `MossScene = dict` with keys `caption (str), meta (list[str]), canvas (str, "none" or one of CANVASES), palette (list[str])`
  - `build_moss(boxes, scene, width, height) -> dict | None` — None when no item survives (caller uses `scene["caption"]` as plain text)
  - `parse_moss(layout, width, height) -> tuple[list[dict], MossScene]` — boxes carry `type` from `moss_label`, `desc` from `caption`, `text`, `tags` (string, comma-joined), `palette`; items without a usable bbox get the unplaced placeholder (`nobbox: True`, same stagger as `parse_ideogram`)

- [ ] **Step 1: Write the tests**

```python
# tests/test_moss_image_core.py
from nodes.layout_formats import build_moss, parse_moss, moss_bbox, from_moss_bbox, compact, strs, moss_label

SCENE = dict(caption="A minimalist travel poster.", meta=[], canvas="none", palette=[])

def test_userguide_example_bbox():
    # USERGUIDE: EXPLORE title at 1024² → [-416,-384,-288,384]; i.e. top-left px (128,96)-(896,224)
    assert moss_bbox({"x": 128/1024, "y": 96/1024, "w": 768/1024, "h": 128/1024}, 1024, 1024) == [-416, -384, -288, 384]

def test_bbox_floor_ceil_and_centre_on_nonsquare():
    # 0.1*1024=102.4→102, 0.4*1024=409.6→410, 0.2*768=153.6→153, 0.6*768=460.8→461; minus 384 / 512
    assert moss_bbox({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, 1024, 768) == [153-384, 102-512, 461-384, 410-512]

def test_zero_area_box_is_none():
    assert moss_bbox({"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.2}, 1024, 1024) is None

def test_from_moss_bbox_spaces_and_orders():
    assert from_moss_bbox([-416, -384, -288, 384], 1024, 1024) == {"x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125}
    assert from_moss_bbox([96, 128, 224, 896], 1024, 1024, space="abs") == {"x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125}
    assert from_moss_bbox([128, 96, 896, 224], 1024, 1024, space="abs", fmt="xyxy") == {"x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125}
    assert from_moss_bbox([-900, -900, 900, 900], 1024, 1024) == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}   # oversize → clamped
    assert from_moss_bbox("nope", 1024, 1024) is None

def test_build_key_order_and_compact():
    boxes = [{"x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125, "type": "text", "text": "EXPLORE", "desc": "Large title.", "palette": ["#1a2b3c"], "tags": "bold, uppercase"},
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
             {"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.5, "type": "obj", "desc": "zero"}]
    assert build_moss(boxes, SCENE, 1024, 1024) is None

def test_non_ascii_unescaped():
    boxes = [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.2, "type": "text", "text": "把日常，过成诗。", "desc": "标题", "palette": []}]
    assert "把日常" in compact(build_moss(boxes, SCENE, 1024, 1024))

def test_canvas_none_and_blank_meta_omitted():
    out = build_moss([{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.2, "type": "obj", "desc": "x", "palette": []}], SCENE, 1024, 1024)
    assert list(out["summary"]) == ["caption"]

def test_parse_moss():
    layout = {"summary": {"caption": "Poster.", "meta": ["poster"], "canvas": "single", "palette": ["#FFFFFF"]},
              "items": [{"label": "character", "caption": "A hero.", "tags": ["anime", "red"], "bbox": [-416, -384, -288, 384]},
                        {"label": "text", "caption": "t", "text": "HI", "bbox": "bad"}],
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    boxes, scene = parse_moss(layout, 1024, 1024)
    assert boxes[0] == {"type": "character", "text": "", "desc": "A hero.", "tags": "anime, red", "palette": [], "x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125}
    assert boxes[1]["nobbox"] is True and boxes[1]["type"] == "text"
    assert scene == {"caption": "Poster.", "meta": ["poster"], "canvas": "single", "palette": ["#FFFFFF"]}

def test_strs_and_label():
    assert strs("a, b\n c,,") == ["a", "b", "c"] and strs(["x", " ", 3]) == ["x", "3"] and strs(None) == []
    assert moss_label("obj") == "object" and moss_label("character") == "character" and moss_label(None) == "object"
```

- [ ] **Step 2: Run, expect ImportError** — `python -m pytest tests/test_moss_image_core.py -v`.

- [ ] **Step 3: Implement in `layout_formats.py`**

`moss_bbox`: clamp edges to 0..1 (sorted, like `bbox_outputs`), then `floor(round(y1*H, 9)) - H//2` etc.; return None if `ymin >= ymax or xmin >= xmax`. `build_moss`: skip `group`, `eff_disabled`, `nobbox`, and `moss_bbox(...) is None`; summary via the key order above with `palette(...)` upper-casing and `[:16]`; item palette `[:5]`.

- [ ] **Step 4: Run** — Expected: PASS.

- [ ] **Step 5: Commit and push**

```bash
git add nodes/layout_formats.py tests/test_moss_image_core.py
git commit -m "feat(layout): MOSS-Image layout build/parse with centred-pixel bboxes"
git push
```

---

### Task 4: Detection and cross-format conversion; the Ideogram node accepts MOSS imports

**Files:**
- Modify: `nodes/layout_formats.py`, `nodes/ideogram4_nodes.py` (`execute` import path)
- Create: `tests/test_layout_formats.py`

**Interfaces:**
- Produces:
  - `detect_format(obj) -> str | None` — `"moss"` if `obj.get("items")` is a list, `"ideogram"` if `"compositional_deconstruction"` in obj, else None
  - `ideogram_scene_to_moss(scene: IdeogramScene) -> MossScene`, `moss_scene_to_ideogram(scene: MossScene) -> IdeogramScene`
  - `boxes_to_labels(boxes, target: "ideogram"|"moss") -> list[dict]` — copies with `type` normalised (`obj`↔`object`, `character`→`obj` for ideogram); other keys (incl. `tags`) untouched
  - `import_layout(obj, target, width, height) -> tuple[list[dict], IdeogramScene | MossScene, bool]` — parses `obj` in its detected format, converts scene + labels when the format ≠ `target`; third value is `True` when a conversion happened. Raises `ValueError` when `detect_format` is None.
- `ideogram4_nodes.execute`: when `used_import` and the import was converted, `caption = build_ideogram(boxes, scene, coord_mode=…, bbox_order=…, width, height)` and `ui["caption"]` mirrors that built dict; same-format import keeps today's passthrough.

- [ ] **Step 1: Write the tests**

```python
# tests/test_layout_formats.py
import pytest
from nodes.layout_formats import detect_format, ideogram_scene_to_moss, moss_scene_to_ideogram, import_layout, build_ideogram, build_moss, parse_ideogram, compact

IDEO = {"high_level_description": "Sunset", "style_description": {"aesthetics": "moody", "lighting": "", "photo": "35mm", "medium": "film", "color_palette": ["#112233"]},
        "compositional_deconstruction": {"background": "A beach", "elements": [
            {"type": "obj", "bbox": [200, 100, 600, 400], "desc": "a crab", "color_palette": ["#FF0000"]},
            {"type": "text", "bbox": [500, 500, 600, 750], "text": "HELLO", "desc": "title"}]}}

def test_detect():
    assert detect_format(IDEO) == "ideogram"
    assert detect_format({"summary": {"caption": "x"}, "items": []}) == "moss"
    assert detect_format({"caption": "x"}) is None and detect_format("str") is None

@pytest.mark.parametrize("scene,expected", [
    (dict(high_level_description="Sunset", background="A beach", style_kind="photo", photo="35mm", art_style="", aesthetics="moody", lighting="", medium="film", palette=["#112233"]),
     "Sunset A beach Style: 35mm, moody, film."),
    (dict(high_level_description="", background="A beach", style_kind="none", photo="", art_style="", aesthetics="moody", lighting="", medium="", palette=[]), "A beach"),
    (dict(high_level_description="", background="A beach", style_kind="art_style", photo="", art_style="", aesthetics="", lighting="", medium="", palette=[]), "A beach"),
    (dict(high_level_description="", background="", style_kind="art_style", photo="", art_style="ukiyo-e", aesthetics="", lighting="soft", medium="", palette=[]), "Style: ukiyo-e, soft."),
])
def test_caption_join(scene, expected):
    out = ideogram_scene_to_moss(scene)
    assert out["caption"] == expected and out["meta"] == [] and out["canvas"] == "none" and out["palette"] == scene["palette"]

def test_moss_scene_to_ideogram():
    s = moss_scene_to_ideogram(dict(caption="Poster.", meta=["poster"], canvas="single", palette=["#FFFFFF"]))
    assert s == dict(high_level_description="", background="Poster.", style_kind="none", photo="", art_style="", aesthetics="", lighting="", medium="", palette=["#FFFFFF"])

def test_ideogram_to_moss_to_ideogram_within_one_grid_unit():
    boxes, scene, conv = import_layout(IDEO, "moss", 1024, 768)
    assert conv and [b["type"] for b in boxes] == ["object", "text"]
    moss = build_moss(boxes, scene, 1024, 768)
    back_boxes, back_scene, conv2 = import_layout(moss, "ideogram", 1024, 768)
    assert conv2 and back_scene["background"] == "Sunset A beach Style: 35mm, moody, film."
    out = build_ideogram(back_boxes, back_scene)["compositional_deconstruction"]["elements"]
    for got, want in zip(out, IDEO["compositional_deconstruction"]["elements"]):
        assert all(abs(g - w) <= 1 for g, w in zip(got["bbox"], want["bbox"]))
        assert got.get("text") == want.get("text") and got["desc"] == want["desc"] and got.get("color_palette") == want.get("color_palette")

def test_same_format_import_is_not_converted():
    boxes, scene, conv = import_layout(IDEO, "ideogram", 1024, 768)
    assert not conv and scene["style_kind"] == "photo" and boxes[0]["type"] == "obj"

def test_tags_and_character_survive_editor_roundtrip():
    moss = {"summary": {"caption": "P."}, "items": [{"label": "character", "caption": "hero", "tags": ["anime"], "bbox": [-100, -100, 100, 100]}]}
    boxes, scene, _ = import_layout(moss, "ideogram", 1024, 1024)
    assert boxes[0]["type"] == "obj" and boxes[0]["tags"] == "anime"
    assert "tags" not in compact(build_ideogram(boxes, scene))
    boxes2, _, _ = import_layout(build_ideogram(boxes, scene), "moss", 1024, 1024)   # label is obj→object; tags are not in Ideogram JSON
    assert boxes2[0]["type"] == "object" and "tags" not in boxes2[0]

def test_sidecar_never_reaches_moss():
    ideo = dict(IDEO, _itl={"v": 1, "boxes": [{"x": 0, "y": 0, "w": 1, "h": 1, "type": "obj", "group": True, "id": "g"}]})
    boxes, scene, _ = import_layout(ideo, "moss", 1024, 1024)
    assert "_itl" not in compact(build_moss(boxes, scene, 1024, 1024)) and len(boxes) == 2

def test_unknown_format_raises():
    with pytest.raises(ValueError):
        import_layout({"caption": "x"}, "moss", 1024, 1024)
```

- [ ] **Step 2: Run, expect ImportError** — `python -m pytest tests/test_layout_formats.py -v`.

- [ ] **Step 3: Implement in `layout_formats.py`; wire into `ideogram4_nodes.execute`**

`import_layout` ignores the `_itl` key (parses the plain caption only — the sidecar is an editor concern). In `execute`: `imported = loads_layout(import_json)`; if not None and `detect_format(imported)` is None → treat as no import (today's behaviour for non-captions). The ui mirror for a converted import is `dumps(built_caption)`.

- [ ] **Step 4: Run all three Python test files + the golden** — Expected: PASS; golden unchanged.

- [ ] **Step 5: Commit and push**

```bash
git add nodes/layout_formats.py nodes/ideogram4_nodes.py tests/test_layout_formats.py
git commit -m "feat(layout): detect + convert between Ideogram 4 and MOSS-Image layouts"
git push
```

---

### Task 5: The MOSS node

**Files:**
- Create: `nodes/moss_image_nodes.py`
- Modify: `__init__.py:14-15, 32-33, 57-58`
- Modify: `tests/test_nodes_integration.py` (append)

**Interfaces:**
- Consumes: Task 3/4 functions; `_render_preview`, `_bg_image` from `ideogram4_nodes`.
- Produces: `ITLMossImagePromptBuilder` with `node_id="ITLMossImagePromptBuilder"`, `display_name="ITL MOSS-Image Prompt Builder"`, `category="Into The Latent/text"`, `search_aliases=["moss", "moss-image", "layout", "bbox", "prompt builder", "json prompt"]`, `is_experimental=True`.
  - Inputs, in order: `width`, `height` (Int, default 1024, min 64, max 16384, step 16), `caption` (String multiline), `meta` (String multiline), `canvas` (Combo `["none","single","multi_panel","collage"]`, default `"none"`), `image` (optional), `import_json` (optional, force_input), `style_palette_data`, `elements_data` (socketless, advanced), `bg_brightness` (Int 25, socketless, advanced), `import_mode` (Combo), `output_format` (String `"compact"`, socketless, advanced), `bboxes` (BoundingBox optional, force_input).
  - Outputs: `prompt`, `preview`, `bboxes`, `width`, `height`.
  - `execute(cls, width, height, caption, meta="", canvas="none", style_palette_data="", elements_data="", import_json="", import_mode="when empty", output_format="compact", bboxes=None, image=None, bg_brightness=25)`.
  - `ui = {"dims": [width, height]}` plus `boxes` when seeded and `caption` (= `dumps(layout)` or the plain caption string when no items) when an import was used — same contract as the Ideogram node.

- [ ] **Step 1: Write the integration test (ComfyUI env)**

```python
# append to tests/test_nodes_integration.py
def test_moss_node_builds_layout_and_converts_ideogram_import():
    import json
    from nodes.moss_image_nodes import ITLMossImagePromptBuilder as Moss
    boxes = json.dumps([{"x": 0.125, "y": 0.09375, "w": 0.75, "h": 0.125, "type": "text", "text": "EXPLORE", "desc": "Large title.", "palette": []}])
    out = Moss.execute(1024, 1024, "Poster.", elements_data=boxes).args[0]
    assert json.loads(out)["items"][0]["bbox"] == [-416, -384, -288, 384]
    assert Moss.execute(1024, 1024, "Just prose.").args[0] == "Just prose."           # no items → plain text
    assert Moss.execute(1024, 1024, "Editor wins.", import_json="not json at all").args[0] == "Editor wins."
    ideo = '{"compositional_deconstruction":{"background":"A beach","elements":[{"type":"obj","bbox":[200,100,600,400],"desc":"a crab"}]}}'
    conv = json.loads(Moss.execute(1024, 1024, "ignored", import_json=ideo, import_mode="always").args[0])
    assert conv["summary"]["caption"] == "A beach" and conv["items"][0]["label"] == "object"

def test_ideogram_node_converts_moss_import():
    import json
    from nodes.ideogram4_nodes import ITLIdeogram4PromptBuilder as Ideo
    moss = '{"summary":{"caption":"Poster."},"items":[{"label":"character","caption":"hero","bbox":[-416,-384,-288,384]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'
    out = json.loads(Ideo.execute(1024, 1024, "", {"style": "none"}, import_json=moss, import_mode="always", output_format="compact").args[0])
    assert out["compositional_deconstruction"]["background"] == "Poster."
    assert out["compositional_deconstruction"]["elements"][0] == {"type": "obj", "bbox": [94, 125, 219, 875], "desc": "hero"}
```

- [ ] **Step 2: Run in the ComfyUI env, expect ImportError** for `nodes.moss_image_nodes`.

- [ ] **Step 3: Create `nodes/moss_image_nodes.py`; register in `__init__.py`**

Module docstring states the MOSS format summary and that the editor is shared with (and derived from) the Ideogram builder, GPL-3.0 KJNodes attribution like `ideogram4_nodes.py`. Description text (for the node's tooltip) must say: boxes are in pixels of `width`/`height`, so generate at exactly that size; an imported MOSS layout is read at the node's current width/height.

- [ ] **Step 4: Run** `tests/test_nodes_integration.py` and the golden in the ComfyUI env — Expected: PASS.

- [ ] **Step 5: Smoke-load the pack** in the live ComfyUI (`E:\AI\ComfyUI\custom_nodes\ComfyUI-IntoTheLatent-Utils` is a separate clone: `git -C <that clone> fetch && git -C <that clone> checkout feature/moss-image-prompt-builder`), start ComfyUI, confirm the node appears under *Into The Latent/text* with no import errors in the console.

- [ ] **Step 6: Commit and push**

```bash
git add nodes/moss_image_nodes.py __init__.py tests/test_nodes_integration.py
git commit -m "feat(moss-image): MOSS-Image Prompt Builder node"
git push
```

---

### Task 6: Spec-ify the editor (Ideogram spec = today's behaviour)

**Files:**
- Modify: `web/js/ideogram4_prompt_builder.js` — registration (L517–521), toolbar toggles (L530–532, 611–624, 3400-ish restore), scene-field lists (L2518, 3244), type picker (L2977), labels (L1391, 2344, 2386, 3071), group/new-box defaults (L1860, 2011), bbox readout (`normBboxJS`, `updateBboxLabel`, `commitGridEdit` L2871–2900), serializer/importer (L2529–2727), sidecar (L2600–2602), `updateTokens` (L2605), `MAX_*` constants, style-palette row label.

**Interfaces:**
- Produces (module-level in `ideogram4_prompt_builder.js`):
  - `export const SPECS = new Map()` keyed by node name; `export function registerLayoutSpec(spec)`.
  - Spec shape (exact keys):
    ```js
    { nodeName, sceneFields: string[], labels: string[], defaultLabel, nativeLabels: Set,
      toolbar: { coordToggle: bool, orderToggle: bool },
      maxElemColors, maxStyleColors, paletteLabel, tokenCap,
      outHint: string,                       // title of the "out:" readout field
      outBbox: (box, W, H, widgets) => number[],        // what the prompt will contain for this box
      outToBox: (arr, W, H, widgets) => {x,y,w,h}|null,  // inverse, for the editable readout
      build: (ctx) => object,                // ctx = { boxes, getW, stylePalette, isDisabled, widgets }
      buildPlain: (ctx) => string|null,      // MOSS: caption text when build() yields no items; Ideogram: null
      apply: (obj, format, ctx) => void,     // ctx += { setBoxes, setWidgetVal, setStylePalette, W, H }
      hasTags: bool, requiresBbox: bool,     // requiresBbox: unplaced boxes get a "not exported" hint (MOSS)
      importPrompt: string, importError: string }
    ```
  - `detectFormat(obj)` (mirror of Python), `tryParseCaption(text)` returns `{obj, format}` or null, accepted for both formats on both nodes.
  - `needsSidecar(boxes, spec)` = any box has `group`, `parent != null`, non-empty `tags`, or `!spec.nativeLabels.has(box.type)`.
  - The extension registers `beforeRegisterNodeDef` once and bails unless `SPECS.has(nodeData.name)`; `onNodeCreated` captures `const spec = SPECS.get(node.type)`.
- The Ideogram spec (`IDEOGRAM_SPEC`, defined and registered at the bottom of the file): `sceneFields` = the six names at L2518, `labels ["obj","text"]`, `defaultLabel "obj"`, toggles both true, caps 5/16, `tokenCap 2048`, `build` = today's `buildCaptionObject`, `apply` = today's `applyCaption` for format `"ideogram"` and the Python-mirroring conversion (`mossSceneToIdeogram`, labels `object|character → obj`, `fromMossBbox`) for `"moss"`.

- [ ] **Step 1: Refactor** so every Ideogram-specific site reads `spec`. Keep the file's behaviour identical; move nothing else. `tryParseCaption` now returns `{obj, format}`; `loadCaption` / `insertCaptionBoxes` / `applyImported` pass `format` to `spec.apply`.

- [ ] **Step 2: On-screen check (BenQ monitor)** on the live ComfyUI clone: load an existing workflow with the Ideogram node — regions, groups, dock geometry, style palette restore; draw a box, Copy, Paste into a fresh Ideogram node; toggle `absolute` + `xy` and read the `out:` field; run the graph and compare `prompt` with the frozen golden string for the same layout. Expected: identical to before; no console errors.

- [ ] **Step 3: Paste a MOSS layout** (the USERGUIDE example) into the Ideogram node → one `text` box at (128,96)-(896,224) px with text `EXPLORE`, background widget = the summary caption. Expected: boxes land where the readout says; `prompt` output is Ideogram JSON.

- [ ] **Step 4: Commit and push**

```bash
git add web/js/ideogram4_prompt_builder.js
git commit -m "refactor(editor): drive the bbox editor from a per-node layout spec; accept MOSS imports"
git push
```

---

### Task 7: MOSS spec, tags field, sidecar rule

**Files:**
- Create: `web/js/moss_image_prompt_builder.js`
- Modify: `web/js/ideogram4_prompt_builder.js` (inspector: `tags` field under `desc` when `spec.hasTags`; `needsSidecar`; token colour thresholds scaled from `tokenCap`; "unplaced → not exported" hint in the region list when `spec.requiresBbox`).

**Interfaces:**
- Consumes: `registerLayoutSpec`, spec shape from Task 6.
- Produces: `MOSS_SPEC` — `nodeName "ITLMossImagePromptBuilder"`, `sceneFields ["caption","meta","canvas"]`, `labels ["object","character","text"]`, `defaultLabel "object"`, toggles both false, caps 5/16, `paletteLabel "palette"`, `tokenCap 4096`, `outHint "Exported bbox (centred pixels): ymin, xmin, ymax, xmax — editable"`, `outBbox = mossBbox` (floor/ceil/centre, mirror of Python), `outToBox = fromMossBbox`, `build` mirrors `build_moss`, `buildPlain` returns the caption widget text, `apply` handles `"moss"` directly and `"ideogram"` via the caption-join mirror (`ideogramSceneToMoss`), `hasTags true`, `requiresBbox true`, `importPrompt "Paste MOSS-Image layout or Ideogram 4 caption JSON:"`, `importError "Not a layout JSON (needs 'items' or 'compositional_deconstruction')."`.
- Clear-all on the MOSS node resets `canvas` to `"none"` (combo), the other two to `""`.
- `onSerialize` blob (`o.ideo`) is unchanged in shape; `coordMode`/`bboxOrder` are simply `undefined` on the MOSS node.

- [ ] **Step 1: Write `moss_image_prompt_builder.js`** — imports `{ registerLayoutSpec }` from `./ideogram4_prompt_builder.js`, defines the bbox/scene mirrors and `MOSS_SPEC`, registers it. GPL header like the other JS files.

- [ ] **Step 2: Add the `tags` field, `requiresBbox` hint and the sidecar rule** in the shared editor.

- [ ] **Step 3: On-screen check (BenQ)**: fresh MOSS node — draw three boxes (object / character / text with text), fill caption, meta `poster, minimal`, canvas `single`, tags on one box; run → `prompt` equals `compact(build_moss(...))` for the same boxes (compare against a Python one-liner with the box fractions read from `elements_data`); the `out:` field shows centred pixels and editing it moves the box; toolbar shows only the compact toggle; token colour thresholds at 4096. Copy from the MOSS node → Paste into the Ideogram node → `character` became `obj`, tags kept (visible after pasting back into a MOSS node), background = caption. Save a template on the MOSS node, load it on the Ideogram node and vice versa. Wire Ideogram `prompt` → MOSS `import_json` with `always` → run → MOSS output follows edits on the Ideogram node.

- [ ] **Step 4: Commit and push**

```bash
git add web/js/moss_image_prompt_builder.js web/js/ideogram4_prompt_builder.js
git commit -m "feat(moss-image): editor spec, per-box tags, cross-format paste and templates"
git push
```

---

### Task 8: README, version, PR

**Files:**
- Modify: `README.md` (new section after the Ideogram 4 Prompt Builder section ~L96; Credits paragraph L424–427), `pyproject.toml:4` → `1.13.0`.

- [ ] **Step 1: README section "ITL MOSS-Image Prompt Builder"** covering: what it emits (the JSON shape with one text + one object item), the fields (`caption`, `meta`, `canvas`, per-box label/caption/text/tags/palette), "boxes are pixels of width/height — generate at exactly that size; an imported MOSS layout is read at the node's current width/height", no items → plain caption, cross-format Paste/templates/`import_json` in both nodes with the caption-join rule and the "leave the Ideogram toggles at their defaults when wiring Ideogram → MOSS" caveat, and that the Resolution Selector drives it. Credits: both builders derived from KJNodes' editor. `tests/test_requirements.py` must still pass (`python -m pytest tests/test_requirements.py -v`).

- [ ] **Step 2: Full test run** — `python -m pytest tests -v` (system Python) and the ComfyUI-env runner on `tests/test_ideogram4_node_golden.py tests/test_nodes_integration.py`. Expected: all PASS (comfy tests skip under system Python).

- [ ] **Step 3: Commit, push, PR**

```bash
git add README.md pyproject.toml
git commit -m "docs(moss-image): README section; 1.13.0"
git push
gh auth switch --user Into-The-Latent
gh pr create --title "MOSS-Image Prompt Builder on the shared bbox editor (1.13.0)" --body "Closes #<issue>. <summary: separate node, shared editor spec, layout_formats module, cross-format import both ways, golden test proves Ideogram output unchanged, on-screen checks done on the BenQ>

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

Then the user's UI check and merge.
