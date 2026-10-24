# MOSS-Image Prompt Builder — second layout node on the shared bbox editor

**Date:** 2026-10-23
**Nodes:** new `ITLMossImagePromptBuilder` (`nodes/moss_image_nodes.py`), existing
`ITLIdeogram4PromptBuilder` (`nodes/ideogram4_nodes.py`), shared editor
`web/js/ideogram4_prompt_builder.js`, new comfy-free `nodes/layout_formats.py`.

## Problem

MOSS-Image (OpenMOSS, released 2026-10-16) reads a layout prompt that is structurally the same idea
as Ideogram 4's caption JSON: one scene description plus a list of boxed elements, each with a
label, a description, optional text to render and a `[ymin, xmin, ymax, xmax]` bbox. Our Ideogram 4
Prompt Builder already has the editor people want for this — the canvas, and above all the region
list on the left (select / reorder / group / disable) that no other MOSS builder has — but it only
speaks Ideogram's JSON.

The two formats differ in field names, in the scene-level fields, and in the bbox coordinate space
(Ideogram: 0–1000 grid; MOSS: absolute pixels with the canvas centre as origin). The editor's own
box state (`{x, y, w, h}` fractions, type, text, desc, palette) is format-free, so the same editor
can drive both — only the serializers, the importers and a handful of labels differ.

The decision (made in conversation, after costing both options) is a **separate node that shares
the editor**, not a format dropdown on the Ideogram node. Reasons: the scene-level widgets differ
(hiding them per mode is a hack this editor already fights), a MOSS user would not find a node called
"Ideogram 4", and a mode switch doubles the test matrix on a node that already ships.

## Goals

1. A new node **ITL MOSS-Image Prompt Builder** with the full editor (canvas, region list, groups,
   docks, swatches, live preview, templates, copy/paste) and MOSS-specific scene fields, producing
   MOSS-Image's documented layout JSON.
2. **One editor implementation.** `ideogram4_prompt_builder.js` becomes format-agnostic, driven by a
   per-node spec. No copy of the file.
3. **Cross-format import both ways.** Each node's Paste, `import_json` socket, template load and
   "insert boxes" accept either format and convert it to the node's own. Wiring the Ideogram node's
   `prompt` into the MOSS node's `import_json` (mode `always`) is a live Ideogram → MOSS converter.
4. **The Ideogram node's output does not change.** Byte-for-byte, proven by a golden test written
   before the refactor.
5. The format logic (both builders, both parsers, the converter) lives in a comfy-free module with
   tests, and the JS mirrors it.

## Non-goals

- No model loading, sampling or conditioning nodes for MOSS-Image. ComfyUI core and the official
  pack cover that; this node only produces the prompt string.
- No "MOSS-Image" profile in the Resolution Selector (follow-up issue; the selector pushes
  width/height by widget name, so it already drives the new node).
- No support for the older gallery schema (`scene_type` / `gen_brief` / `elements` with
  `identity`) that some official sample prompts use. `summary` + `items` is the documented format.
- The Style Wizard stays Ideogram-only.
- No change to the templates' storage location or file format.

## Anti-goals

- Changing any Ideogram output, key order, rounding or behaviour as a side effect of the refactor.
- A second copy of the editor, or `if (isMoss)` branches sprinkled through it. The editor reads a
  spec; the spec is the only place a format is named.
- Per-mode widget hiding on either node.

---

## 1. The MOSS-Image layout format

Documented in the official repo (`USERGUIDE.md`) and enforced by the authors' compiler
(`skills/moss-image-prompt-enhancer/scripts/compile_prompt.py`); ComfyUI core's
`comfy_extras/nodes_moss_image.py` emits the same thing.

```json
{"summary":{"meta":["poster"],"canvas":"single","caption":"…whole scene and style…","palette":["#1A2B3C"]},
 "items":[
   {"label":"text","caption":"Large dark-blue uppercase title.","text":"EXPLORE","bbox":[-416,-384,-288,384]},
   {"label":"object","caption":"A flat blue mountain.","tags":["illustration"],"palette":["#3366CC"],"bbox":[-192,-288,384,288]}
 ],
 "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```

- `summary.caption` — required. `meta` (string list), `canvas` (`single` | `multi_panel` |
  `collage`) and `palette` are optional.
- `items[]` — each needs `label` (`object` | `character` | `text`), `caption`, `bbox`. `text` items
  need `text`. `tags` and `palette` are optional string lists.
- `bbox` — `[ymin, xmin, ymax, xmax]` in **absolute pixels from the canvas centre**; a 1024×1024
  canvas spans −512..+512 on both axes. The prompt carries no width/height, so the boxes are only
  right at the resolution the image is generated at.
- Serialized compact, one line, `ensure_ascii=False`.
- With no items the prompt is plain text (ComfyUI core does the same: returns the caption as-is).

### bbox conversion (editor fractions → MOSS)

Same rounding as ComfyUI core and the official compiler: enclose the box in whole pixels, then shift
the origin.

```
ymin = floor(y1 · H) − H//2      xmin = floor(x1 · W) − W//2
ymax = ceil (y2 · H) − H//2      xmax = ceil (x2 · W) − W//2
```
(`round(v, 9)` before floor/ceil, as the reference does, so 0.3·1024 = 307.2 → 307 and float noise
such as 307.0000000001 does not ceil to 308.) Edges are clamped into the canvas first.

Reverse (MOSS → fractions): `x = (xmin + W//2) / W`, etc., clamped to 0..1. If the layout says
`bbox_coord_space: "abs"` (top-left origin, the draft form the authors' skill uses) the shift is
skipped; if `bbox_format: "xyxy"` the pairs are swapped before use.

## 2. Field mapping between the two formats

| Ideogram 4 | MOSS-Image | Conversion |
|---|---|---|
| `high_level_description` | `summary.caption` | Ideogram→MOSS: joined into the caption, see §5 |
| `compositional_deconstruction.background` | `summary.caption` | Ideogram→MOSS: joined; MOSS→Ideogram: caption becomes `background` |
| `style_description.{photo\|art_style, aesthetics, lighting, medium}` | — | Ideogram→MOSS: joined into the caption; MOSS→Ideogram: style `none` |
| `style_description.color_palette` | `summary.palette` | 1:1 |
| — | `summary.meta`, `summary.canvas` | MOSS only; dropped going to Ideogram |
| `elements[]` | `items[]` | |
| `type: "obj"` | `label: "object"` or `"character"` | `obj`→`object`; `object`/`character`→`obj` |
| `type: "text"`, `text` | `label: "text"`, `text` | 1:1 |
| `desc` | `caption` | 1:1 |
| `color_palette` (≤5) | `palette` | 1:1 (Ideogram side keeps its cap of 5) |
| — | `tags` | MOSS only; kept on the editor box, never exported by the Ideogram node |
| element without `bbox` ("unplaced") | not allowed | skipped on MOSS export |
| bbox 0–1000 `yxyx` | bbox centred px `yxyx` | §1 |

## 3. The new node

```
ITLMossImagePromptBuilder   display "ITL MOSS-Image Prompt Builder"   category "Into The Latent/text"
search_aliases: moss, moss-image, layout, bbox, prompt builder, json prompt
```

**Inputs** (same shape as the Ideogram node except the scene fields):

| name | type | notes |
|---|---|---|
| `width`, `height` | INT, default 1024, step 16 | the pixel canvas the bboxes are measured in; outputs mirror them |
| `caption` | STRING multiline | → `summary.caption`; the whole prompt when there are no placed boxes |
| `meta` | STRING | comma/newline-separated → `summary.meta`; blank = omitted |
| `canvas` | COMBO `none, single, multi_panel, collage` default `none` | `none` = omitted |
| `image` | IMAGE optional | editor/preview background, unchanged |
| `import_json` | STRING optional, force_input | either format, see §5 |
| `import_mode` | COMBO `when empty, always` | unchanged semantics |
| `bboxes` | BOUNDING_BOX optional | seeds the editor, unchanged |
| `elements_data`, `style_palette_data`, `bg_brightness`, `output_format` | hidden data widgets | as today; `style_palette_data` holds `summary.palette` |

Not present: `coord_mode`, `bbox_order` (MOSS fixes both), `high_level_description`,
`background`, `style`, `aesthetics`, `lighting`, `medium`.

**Outputs:** `prompt` (STRING), `preview` (IMAGE), `bboxes` (BOUNDING_BOX), `width`, `height` —
identical to the Ideogram node so the downstream wiring (SAM3 / crop / EmptyLatent) is the same.

**Output rules**

- Placed, enabled, non-group boxes become items in editor order. Key order per item:
  `label, caption, text?, tags?, palette?, bbox`. Summary key order: `meta?, canvas?, caption, palette?`.
- No items → `prompt` is the `caption` text, untouched (not JSON).
- `output_format` `compact` (default) is one line; `pretty` uses the existing `_dumps` layout. Same
  toolbar toggle as today, the other two toggles do not exist on this node.
- Unplaced boxes (`nobbox`) are skipped on export; the region list shows them with the existing
  "unplaced" styling plus a hint that MOSS needs a box.
- Groups and disabled boxes behave exactly as on the Ideogram node (skipped).
- Palette caps stay the editor's UI caps (5 per item, 16 on the summary). MOSS documents no cap.

## 4. The shared editor

`web/js/ideogram4_prompt_builder.js` keeps its name (it is the editor's home and the derived-work
attribution points at it) and registers for both node names. Everything format-bound is read from a
**spec** looked up by `nodeData.name`:

```js
{
  nodeName,                 // "ITLIdeogram4PromptBuilder" | "ITLMossImagePromptBuilder"
  sceneFields,              // widget names cleared by "Clear all" and tracked for dirty state
  labels,                   // ["obj","text"] | ["object","character","text"]   (type picker order)
  defaultLabel,             // "obj" | "object"
  isText: (box) => …,
  toolbar: { coordToggle, orderToggle },   // true/false — MOSS has neither
  maxElemColors, maxStyleColors, paletteLabel,   // "style colors" | "palette"
  tokenCap,                 // 2048 (Ideogram) | 4096 (MOSS text encoder truncates there)
  readout: { pxHint, outHint, outFormat: (box, W, H) => [..] },   // inspector bbox fields
  build:  (editorState, widgets) => layoutObject,       // mirrors layout_formats.build_*
  apply:  (layoutObject, editorState, widgets) => …,     // mirrors layout_formats.parse_* + convert
  hasTags,                  // show the per-box "tags" field (MOSS)
  importPrompt, importError // the strings in doImport()
}
```

The Ideogram spec is a verbatim extraction of today's behaviour. The MOSS spec lives in a new
`web/js/moss_image_prompt_builder.js` that imports the shared module and registers its spec (ES
modules; `utility.js` is already imported this way). The ~25 sites that currently name Ideogram
things (serializer/importer block ~L2516–2709, the three scene-field name lists, the five `obj`/`text`
sites, the toolbar toggles, `normBboxJS`, the constants, the registration check) become spec reads.

Format detection on import (`tryParseCaption` and the Python twin): an object with an `items` array
is MOSS; one with `compositional_deconstruction` is Ideogram; anything else is rejected with the
existing error path. Both nodes accept both; the spec's `apply` receives the parsed object plus the
detected format and converts when they differ.

**Per-box `tags`** (MOSS only): one text field in the inspector under `desc`, comma-separated,
stored as `box.tags` (string). The Ideogram node never shows or exports it but keeps it on the box,
so a layout can go Ideogram → MOSS → Ideogram inside the editor without losing it.

**Sidecar rule (my call):** today the `_itl` clipboard/template sidecar is only emitted when a box
has `group` or `parent`. It is now emitted whenever any box holds something the node's native
format cannot express: `group`, `parent`, a non-empty `tags`, or a label outside the node's own set
(`character` on the Ideogram node). Plain scenes stay clean, and the `prompt` output is untouched —
the anti-goal covers the prompt, not the clipboard form. That is what makes `tags` and the label
survive a copy/paste or template round trip through the other node.

**Labels on the box:** `box.type` stays the storage field. Ideogram boxes hold `obj`/`text`, MOSS
boxes hold `object`/`character`/`text`. `apply` normalises to the receiving node's label set
(`obj`↔`object`, `character`→`obj` when going to Ideogram). Templates and the sidecar can therefore
hold either and still load into either node.

**Templates** stay in `itl/ideogram4/templates`, shared by both nodes. A template is a layout JSON in
whichever format saved it (plus the sidecar); loading it in the other node goes through the same
cross-format `apply`. Renaming the directory would orphan existing templates, so it does not move.

**Token estimate:** same ~chars/4 heuristic, colour thresholds scaled to `tokenCap`.

**Unchanged and shared without modification:** region list, groups, multi-select, docks/pinning,
swatch rows, live preview / Grab BG, Resolution Selector push (it targets widgets by name), clipboard
copy/paste, the dirty-size dance in `loadCaption`.

## 5. Cross-format conversion (`nodes/layout_formats.py`, mirrored in JS)

Comfy-free module, pure functions on plain dicts and the editor box list:

- `detect_format(obj) -> "ideogram" | "moss" | None`
- `build_ideogram(boxes, scene, *, coord_mode, bbox_order) -> dict` — moved out of
  `ideogram4_nodes.execute`, behaviour unchanged (golden test).
- `parse_ideogram(cap) -> (boxes, scene)` — today's `_caption_to_boxes` plus the scene fields.
- `build_moss(boxes, scene, width, height) -> dict | None` (None = no items → plain text).
- `parse_moss(layout, width, height) -> (boxes, scene)`.
- `convert(obj, src, dst, width, height) -> dict` — composed from the above.

`scene` is a small dict (`caption`/`background`/`high_level_description`/style fields/palette/
meta/canvas, only the keys the format has).

**Ideogram → MOSS scene:** `summary.caption` is built from the Ideogram scene fields, empties
dropped, in this order and with this joining (my call — the builder may not invent another):

```
<high_level_description>  <background>  Style: <photo|art_style>, <aesthetics>, <lighting>, <medium>.
```
Sentences are joined with a single space; the `Style:` sentence is omitted when the style kind is
`none` or every style field is blank. `style_description.color_palette` → `summary.palette`.
`meta` and `canvas` are left empty.

**MOSS → Ideogram scene:** `summary.caption` → `background` (Ideogram requires it; the "whole scene"
sentence is the closest fit), `high_level_description` empty, style `none`,
`summary.palette` → style palette. `meta`, `canvas` dropped. `character` → `obj`.

**Width/height on import:** a MOSS layout carries no canvas size, so parsing uses the receiving
node's current `width`/`height`. A mismatch shifts the boxes; the import does not try to guess. The
node description says so.

**Ideogram toggles on a cross-format import (my call):** Ideogram JSON carries no marker for
`coord_mode` / `bbox_order`, so `parse_ideogram` assumes the Ideogram defaults (0–1000 grid,
`yxyx`) — exactly what today's Ideogram → Ideogram import already assumes. No heuristic (a value
> 1000 cannot detect `xy` order and fails on canvases ≤ 1000 px). An Ideogram node with
`absolute` or `xy` switched on, wired into the MOSS node, converts wrong; the README says to leave
the toggles at their defaults when wiring the two nodes. In the other direction the converted
`prompt` is built by `build_ideogram` from the editor boxes, so it honours the Ideogram node's own
toggles like any editor-built output (a same-format import still passes through untouched).

**`import_mode = always` with a cross-format import:** the `prompt` output is the *converted*
layout in the node's own format (today, a same-format import passes through minus `_itl`; that stays
for same-format). The editor mirrors the converted result, as it does today.

## 6. Python layout after the change

- `nodes/layout_formats.py` — new, comfy-free: everything in §5, plus `_norm_bbox`, `moss_bbox`,
  `from_moss_bbox`, `_dumps`, `_repair_json`/`loads_layout`, `_palette`, `strs` (string-list
  splitter). `ideogram4_nodes.py` re-exports what the wizard or tests already import from it.
- `nodes/ideogram4_nodes.py` — thin: schema + `execute` calling `layout_formats`. Three pieces of
  `execute` become shared helpers: `seed_boxes(bboxes, width, height)`, `bbox_outputs(boxes, width,
  height)`, `bg_image(image)` — plus `_render_preview`, `_by_id`, `_eff_disabled` already there.
- `nodes/moss_image_nodes.py` — new node, ~150 lines, imports those helpers.
- `__init__.py` — register the node and display name. `pyproject.toml` → `1.13.0`.
- `README.md` — new node section next to the Ideogram one (fields, output rules, the width/height
  caveat, cross-format import); the Credits paragraph names both builders as derived from
  KJNodes' editor (GPL-3.0 attribution stays in the JS header).

## 7. Tests (`tests/`, comfy-free unless marked)

1. **Golden test — written first, against the current code**, before any refactor. Two layers
   (my call; `ideogram4_nodes.py` imports `comfy_api`, torch and PIL at module level and `execute`
   always renders the preview, and the repo has no `comfy_api` stub):
   - `tests/test_ideogram4_node_golden.py` — ComfyUI-env only (`pytest.importorskip("comfy_api")`,
     run in the `E:\AI\ComfyUI\venv` per the repo's test notes), calls `execute` with `image=None`
     and asserts the exact `prompt` string for a representative caption (style photo, style
     art_style, none; text + obj elements; palettes; an unplaced element; a disabled box; a group;
     `coord_mode absolute`; `bbox_order xy`; compact and pretty). It keeps calling `execute` and
     must pass unchanged after the refactor — run it before and after.
   - `tests/test_ideogram4_core.py` — comfy-free, added with the refactor, the same cases against
     `layout_formats.build_ideogram`, so the bare-repo run covers the serializer from then on.
2. `test_layout_formats.py`: `detect_format`; `moss_bbox` against the USERGUIDE example
   (`EXPLORE` → `[-416,-384,-288,384]` at 1024²) and a non-square canvas; `from_moss_bbox` for
   `abs_centralized`/`abs` and `yxyx`/`xyxy`; round trip editor → MOSS → editor within 1 px;
   Ideogram → MOSS → Ideogram keeps every box within ±1 grid unit and every text/desc/palette;
   the caption join rules (each empty-field case); `character` → `obj`; `tags` survive a round
   trip through the editor box list.
3. `test_moss_image_core.py`: `build_moss` key order and compact serialization; no items → `None`;
   `nobbox`, group and disabled boxes skipped; `meta` splitting; `canvas none` omitted;
   summary palette.
4. `test_nodes_integration.py` gets one case per node (ComfyUI env only, `importorskip`): execute
   with a wired cross-format `import_json` in mode `always` and check the `prompt` output.

JS has no test harness in this repo; the JS mirrors are verified by the on-screen check (BenQ
monitor, per the global agreement): draw three boxes on each node, copy from one, paste into the
other, compare the `prompt` output to the Python result for the same layout.

## 8. What's left to the builder

- Exact split of the JS spec object and where the shared `registerExtension` lives (one extension
  registering both names, or one per file) — as long as there is one editor implementation.
- Inspector layout for the `tags` field and the `canvas` combo's placement among the widgets.
- Whether `nodes/layout_formats.py` also absorbs `_render_preview` (it needs PIL/torch, so
  probably not).
- Commit granularity on the single PR. Suggested order: golden test → refactor Ideogram onto
  `layout_formats` (tests green, output identical) → `build_moss`/`parse_moss`/`convert` with tests
  → MOSS node + registration → editor spec-ification → MOSS spec + `tags` field → README/version.

## 9. Delivery

One issue, one branch `feature/moss-image-prompt-builder`, one PR, version `1.13.0`. The on-screen
check and the merge are the user's.
