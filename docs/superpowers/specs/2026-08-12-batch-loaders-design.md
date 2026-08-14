# Batch Loaders — design spec

> **Status: APPROVED** (brainstormed and signed off 2026-08-12). Companion visual brief:
> https://claude.ai/code/artifact/188ce1f7-c952-44c0-81d1-63b18bedf5b1

## Goal

Drop multiple image or audio files straight onto a node on the ComfyUI canvas and get **one output
socket group per file** — `image_1`, `image_2`, … — so several distinct files can feed different
places in one workflow at once. No folder paths, no one-file-per-run index walk (that pattern is
already covered by AI2Go Prompt Batch).

## Node set

Four nodes, two Simple/Advanced pairs (the pack's existing Save Metadata (Civitai) pattern):

| node_id | Display name | Per file | Always |
|---|---|---|---|
| `AI2GoBatchImageLoader` | AI2Go Batch Image Loader | `image_N` (IMAGE) | `count` (INT) |
| `AI2GoBatchImageLoaderAdvanced` | AI2Go Batch Image Loader Advanced | `image_N`, `mask_N` (MASK), `filename_N` (STRING) | `count` (INT) |
| `AI2GoBatchAudioLoader` | AI2Go Batch Audio Loader | `audio_N` (AUDIO) | `count` (INT) |
| `AI2GoBatchAudioLoaderAdvanced` | AI2Go Batch Audio Loader Advanced | `audio_N`, `filename_N` (STRING) | `count` (INT) |

Categories: `AI2Go/image` and `AI2Go/audio`. Both image nodes carry the downscale controls; audio
nodes do not.

**Why pairs instead of checkboxes:** ComfyUI validates every wire by socket *position* —
`execution.py:934` (v0.32.0) indexes the class's static output list, so slot K must carry the type
declared at slot K, always. A checkbox hiding the mask sockets would slide filenames into
mask-typed slots and the workflow would be rejected (`return_type_mismatch`). Choosing which node
to drop *is* the checkbox.

## Hard constraints (verified against ComfyUI v0.32.0 / frontend 1.48.7 source)

1. **Outputs are static per class** — `_io.py:1060`: `DynamicOutput` is an empty placeholder,
   never implemented; inputs can `Autogrow`, outputs cannot. A "growing" node therefore declares
   its ceiling up front and the front-end trims the visible tail.
2. **Trimming only works from the end** — because validation is positional, the used sockets must
   form one contiguous run from slot 0. This forces the output order below.
3. **Stock Load Audio never resamples** — `nodes_audio.py:380` returns
   `{"waveform": Tensor[1,C,S], "sample_rate": native}` as read. We match it.
4. **Stock mask quirks** — `nodes.py:1786`: mask = `1.0 - alpha`; a file with no alpha channel
   yields a 64×64 zero tensor (NOT image-sized). We copy both quirks so our masks behave like
   everyone else's.

## Output slot order

`count` occupies slot 0 (it must survive trimming). Then one group per file, **grouped per file**,
not per type — this is what keeps used slots contiguous:

- Image simple: `count, image_1 … image_8` → 9 declared.
- Image Advanced: `count, image_1, mask_1, filename_1, image_2, … filename_8` → 25 declared.
- Audio simple: `count, audio_1 … audio_8` → 9 declared.
- Audio Advanced: `count, audio_1, filename_1, … filename_8` → 17 declared.

**Ceiling: 8 files per node.** Fixed at build time; raising it later appends outputs (append-only,
old workflows keep working). Unused slots return `None` and are hidden on the canvas.

## File storage & data flow

- Dropped files are uploaded **once, at drop time**, to ComfyUI's `input/` folder via the same
  upload endpoint the stock upload widget uses (`/upload/image`; the exact route the frontend's
  `UploadType.audio` widget calls is confirmed during implementation). The name **returned by the
  endpoint** is authoritative (ComfyUI may rename on collision).
- The workflow stores **filenames only**, in a hidden `files_json` widget — a JSON array of
  `{"name": str, "subfolder": str, "type": "input"}` — the single source of truth for save,
  restore, and execution (the Prompt Batch pattern).
- Row thumbnails/audio metadata come free from ComfyUI's `/view` endpoint.
- Uploaded originals are never modified; downscaling happens at run time in `execute()`.
- Caching: `fingerprint_inputs` hashes the file list plus each file's size/mtime so a changed file
  on disk re-runs the node (stock LoadAudio hashes contents; size+mtime is enough here and cheap
  for 8 files).

## Front-end behavior (`web/js/batch_loader.js`, shared by all four nodes)

- **Drop zone** at the top of the node accepts multiple files; an **＋ Add** button opens a file
  picker (multi-select). Only files whose type matches the node (image / audio) are accepted;
  rejects are reported in the status line, not silently dropped.
- **Rows**: one per file — thumbnail + pixel size for images, waveform glyph + duration for audio
  (the file's native sample rate is not shown in the row: browsers resample on decode, and parsing
  container headers per format isn't worth it in v1 — the `AUDIO` socket still carries the native
  rate); a ⠿ drag grip to reorder; a ✕ to remove. **Sort by name** button for one-click folder
  order. No inline audio player in v1.
- **List order = socket order**: row 1 feeds `image_1`/`audio_1`, and so on. Files land in drop
  order.
- **Grow/trim**: after any list change the JS trims `node.outputs` to `count` + the loaded groups
  (and re-adds up to the declared ceiling when files are added back). *This is the one unproven
  piece — the implementation plan starts with a spike proving trim/re-add against frontend 1.48.7,
  including save/load round-trips of a trimmed node.*
- **Delete = shift up + warn**: removing file K promotes every later file one group up (the list
  never has holes). If any promoted-into or vacated socket had a connection, the status line warns,
  naming the moved file(s), e.g. `⚠ Removing city.jpg moved studio_cat.webp to image_2 — check
  your wires.` No blocking dialogs (pack rule): warnings are status-line text.
- **Over-ceiling**: dropping more files than free slots loads the first free-slot-count of them and
  warns `only 8 files fit — N skipped`.
- **Pack pitfalls honored**: scalars mirrored by name into `node.properties` (widgets_values
  save/restore mismatch); INT widgets guarded via `serializeValue` (`''` → default); no
  `confirm()`/`alert()` in widget callbacks. No Clear All button in v1 — per-row ✕ suffices for a
  list of at most 8.

## Downscaling (image nodes only, simple AND Advanced)

Two widgets, appended at the end of the schema:

- `downscale_mode` (COMBO): **`off` (default)** / `fit` / `crop` / `stretch`. Off = images pass
  through exactly as dropped.
- `max_size` (INT, default 1200, min 8, max 8192): the size threshold/target. Ignored while mode
  is `off`.

Semantics (trigger: longest edge > `max_size`; smaller images are never upscaled by `fit`/`crop`):

| mode | result for a 2400×1200 source, max_size 1200 |
|---|---|
| `off` | 2400×1200 — untouched |
| `fit` | 1200×600 — aspect kept, longest edge capped at `max_size` |
| `crop` | 1200×1200 — largest centered square (side = shorter edge), then downscaled to ≤ `max_size` |
| `stretch` | 1200×1200 — both sides forced to `max_size`; note the shorter side may be enlarged (inherent to forcing a square) |

- Resampling: Pillow `LANCZOS` (Pillow is already a pack dependency).
- The Advanced node's masks are resized/cropped/stretched with the exact same geometry as their
  image (the 64×64 no-alpha placeholder is left as is — stock behavior).
- Audio is never resampled or altered — each `AUDIO` output keeps its file's native sample rate
  (stock behavior; combiner nodes downstream reconcile rates themselves).

## Run-time behavior & errors

- `execute()` parses `files_json`, loads each file from `input/`, applies downscaling (image
  nodes, mode ≠ off), and returns `count` + one group per file; unused slots return `None`.
- **No files loaded** → raise `ValueError("No files loaded — drop images/audio onto the node.")`
  (matches Prompt Batch's fail-fast on an empty list).
- **File missing from `input/`** (deleted since upload, workflow moved machines) → raise
  `ValueError` naming the missing file.
- **Unreadable/corrupt file** → raise `ValueError` naming the file and position.
- Image decode: Pillow, EXIF-transposed, RGB, float32 0–1, shape `[1,H,W,C]`. Mask: inverted
  alpha as above. Audio decode: PyAV (bundled with ComfyUI), float32 PCM, `[1,C,S]` +
  `sample_rate` — mirroring the stock loaders.

## Architecture

| file | contents |
|---|---|
| `nodes/batch_loader_core.py` | comfy-free: `files_json` parse/validate, shift-on-delete list ops, fit/crop/stretch geometry math (pure functions returning target sizes/boxes) |
| `nodes/batch_image_loader.py` | `AI2GoBatchImageLoader` + `AI2GoBatchImageLoaderAdvanced` (one file holds the pair, like `save_civitai_metadata.py`) |
| `nodes/batch_audio_loader.py` | `AI2GoBatchAudioLoader` + `AI2GoBatchAudioLoaderAdvanced` |
| `web/js/batch_loader.js` | shared front-end, parameterized by node type (drop zone, rows, upload, trim, warnings) |
| `tests/test_batch_loader_core.py` | pytest for the core module, run from repo root (no ComfyUI/torch in dev env) |

Registration: four entries appended to `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` in
`__init__.py`.

## Testing

- **pytest (runs in dev env)**: `files_json` parsing (valid/malformed/empty/wrong types),
  shift-on-delete ordering, over-ceiling clamp, and the downscale geometry table above (fit, crop,
  stretch, off; upscale-never cases; already-small images; exact 2400×1200→1200×600 style cases).
- **JS**: ESM syntax check only (`node --check` false-passes ESM — validate as module per pack
  memory); live behavior (drop, trim, save/load round-trip, delete warning) is verified by the
  user in a running ComfyUI, listed as a manual checklist in the implementation plan.
- **Live spike first**: prove output trim/re-add + save/load on frontend 1.48.7 before building
  the rest.

## Out of scope (v1)

- Inline audio player in rows.
- Folder-path loading (drag-and-drop and file picker only).
- More than 8 files per node (drop two loader nodes instead).
- Audio resampling / a `pad` resize mode / exposing the resample filter as a widget.
- Batched-tensor output (single IMAGE carrying all files).

## Risks

1. **Front-end trim/re-add of outputs on frontend 1.48.7** — standard LiteGraph surgery used by
   other packs, but unproven here; hence the spike. Fallback if it fails outright: ship Advanced
   nodes as fixed 8-group nodes (all sockets always visible, unused return `None`) — behavior
   identical, just a taller node.
2. **Upload route for audio** — assumed `/upload/image` (generic); confirmed in the spike.
3. **Reordering rows rewires meaning** just like deletion does; the same status-line warning
   covers drag-reorder of rows whose sockets are already connected.

## Amendments (2026-08-14)

- **Batch → Multi rename**: `AI2GoBatch*Loader*` node_ids/files became `AI2GoMulti*Loader*` /
  `multi_*` — "Batch" already means the Prompt Batch index-walk pattern in this pack, and these
  nodes do something different (parallel sockets), so the old name was misleading.
- **Downscale mode labels** became plain words (`keep aspect ratio` / `crop to square` /
  `stretch to square`) instead of `fit`/`crop`/`stretch`, per pack rule to lead with plain
  language over jargon.
- **`output_slots` added** (Combo: `auto`/1-8, front-end only): the Python side always returns a
  fully padded output list regardless of file count, so socket count was purely a display
  decision — pinning it lets sockets, and the wires on them, survive file-list edits instead of
  shrinking (and destroying a wire) whenever a file is removed.
- **Drop zone doubles as the file picker**: the separate ＋ Add button was removed; clicking the
  drop zone now opens the same file picker, so there is one field for both gestures.
