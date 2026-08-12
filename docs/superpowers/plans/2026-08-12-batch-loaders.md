# Batch Loaders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Four ComfyUI nodes (Batch Image Loader + Advanced, Batch Audio Loader + Advanced) that accept multiple dropped files and emit one output socket group per file, with optional image downscaling.

**Architecture:** Comfy-free logic (files_json parsing, shift-on-delete, downscale geometry) lives in `nodes/batch_loader_core.py` and is fully pytest-covered in the dev environment (no ComfyUI/torch installed there). Thin `io.ComfyNode` wrappers declare a fixed ceiling of 8 file-groups of outputs; a single shared front-end (`web/js/batch_loader.js`) uploads dropped files to ComfyUI's `input/`, keeps a hidden `files_json` widget as the single source of truth, and trims/re-grows the visible output sockets to match the file count.

**Tech Stack:** Python 3.10+, Pillow (LANCZOS), PyAV + torch (only inside ComfyUI), ComfyUI v3 node API (`comfy_api.latest.io`), vanilla ES-module JS against ComfyUI frontend 1.48.7.

**Spec:** `docs/superpowers/specs/2026-08-12-batch-loaders-design.md` — read it first; it records the verified ComfyUI constraints (positional output validation, no dynamic outputs, stock mask/audio quirks) that this plan builds on.

## Global Constraints

- Ceiling: **8 files per node** (`MAX_FILES = 8`). Never raise without appending outputs at the end.
- Output slot order: `count` at slot 0, then **grouped per file** (`image_1, mask_1, filename_1, image_2, …`). Used slots must stay contiguous from slot 0 — the front-end trims only from the end.
- Downscale modes: `off` (default) / `fit` / `crop` / `stretch`; trigger only when `max(w, h) > max_size`; `fit`/`crop` never upscale; masks get identical geometry; audio is never resampled.
- Every new file starts with the pack's GPL header comment (`# … — part of ComfyUI-AI2Go-Utils. GPL-3.0.` / JS block comment equivalent).
- Widgets: `files_json` is widget #0 (positional-safe); any new widget is **appended** at schema end; scalar widgets are mirrored by name into `node.properties` on serialize (frontend saves widgets_values at absolute indices but restores compacted); INT widgets get a `serializeValue` guard (`''` → default).
- No `confirm()`/`alert()` in widget callbacks — status-line text only. No Clear All button in v1.
- JS validation: `node --check` on a `.js` false-passes ES-module errors — always copy to `.mjs` first (exact command in the JS tasks).
- Run pytest from the repo root: `python -m pytest tests/ -v`. Integration tests use `pytest.importorskip("comfy_api")` and SKIP in the dev env — that is expected and green.
- Python changes need a ComfyUI **server restart** to test live; JS needs a **hard refresh** (Ctrl+F5).

---

### Task 1: Core — `parse_files`

**Files:**
- Create: `nodes/batch_loader_core.py`
- Test: `tests/test_batch_loader_core.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces: `MAX_FILES: int = 8`, `DOWNSCALE_MODES: tuple = ("off", "fit", "crop", "stretch")`, and `parse_files(raw: str) -> list[dict]` where each dict is `{"name": str, "subfolder": str, "type": "input"}`. Raises `ValueError` with a human-readable message on anything malformed. Tasks 4, 5 call `parse_files`; Task 7 mirrors it in JS.

- [ ] **Step 1: Write the failing tests**

```python
# Batch Loader core tests — part of ComfyUI-AI2Go-Utils. GPL-3.0.
import json

import pytest

from nodes.batch_loader_core import MAX_FILES, parse_files


def test_parse_valid_list():
    raw = json.dumps([
        {"name": "fox.png", "subfolder": "", "type": "input"},
        {"name": "city.jpg"},  # subfolder/type optional
    ])
    files = parse_files(raw)
    assert files == [
        {"name": "fox.png", "subfolder": "", "type": "input"},
        {"name": "city.jpg", "subfolder": "", "type": "input"},
    ]


def test_parse_keeps_subfolder():
    files = parse_files(json.dumps([{"name": "a.png", "subfolder": "refs"}]))
    assert files[0]["subfolder"] == "refs"


@pytest.mark.parametrize("raw", ["", "   ", "[]"])
def test_parse_empty_raises(raw):
    with pytest.raises(ValueError, match="No files loaded"):
        parse_files(raw)


def test_parse_malformed_json_raises():
    with pytest.raises(ValueError, match="Malformed files_json"):
        parse_files("[{oops")


def test_parse_non_list_raises():
    with pytest.raises(ValueError, match="JSON array"):
        parse_files('{"name": "a.png"}')


@pytest.mark.parametrize("entry", [42, "a.png", {}, {"name": ""}, {"name": 3}])
def test_parse_bad_entry_raises(entry):
    with pytest.raises(ValueError, match="File #1"):
        parse_files(json.dumps([entry]))


def test_parse_over_ceiling_raises():
    raw = json.dumps([{"name": f"f{i}.png"} for i in range(MAX_FILES + 1)])
    with pytest.raises(ValueError, match="at most 8"):
        parse_files(raw)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_batch_loader_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nodes.batch_loader_core'`

- [ ] **Step 3: Write the implementation**

```python
# Batch Loader core — part of ComfyUI-AI2Go-Utils. GPL-3.0.
#
# Pure (comfy-free) logic shared by the four Batch Loader nodes (nodes/batch_image_loader.py,
# nodes/batch_audio_loader.py). parse_files is mirrored in web/js/batch_loader.js — keep in sync.
import json

MAX_FILES = 8
DOWNSCALE_MODES = ("off", "fit", "crop", "stretch")


def parse_files(raw):
    """Parse the files_json widget into ``[{"name", "subfolder", "type"}, ...]``.

    Raises ``ValueError`` with a human-readable message on anything malformed, on an
    empty list, and on more than ``MAX_FILES`` entries.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("No files loaded — drop files onto the node.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed files_json: {e.msg} (line {e.lineno}, column {e.colno}).") from e
    if not isinstance(data, list):
        raise ValueError('Expected a JSON array of files, e.g. [{"name": "fox.png"}, ...].')
    if not data:
        raise ValueError("No files loaded — drop files onto the node.")
    if len(data) > MAX_FILES:
        raise ValueError(f"{len(data)} files listed — the node supports at most {MAX_FILES}.")

    files = []
    for i, entry in enumerate(data):
        where = f"File #{i + 1}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: each entry must be an object with a 'name' field.")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{where}: 'name' must be a non-empty string.")
        subfolder = entry.get("subfolder") or ""
        if not isinstance(subfolder, str):
            raise ValueError(f"{where}: 'subfolder' must be a string.")
        files.append({"name": name, "subfolder": subfolder, "type": "input"})
    return files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_batch_loader_core.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add nodes/batch_loader_core.py tests/test_batch_loader_core.py
git commit -m "feat: batch loader core - files_json parsing"
```

---

### Task 2: Core — `downscale_size`

**Files:**
- Modify: `nodes/batch_loader_core.py` (append)
- Test: `tests/test_batch_loader_core.py` (append)

**Interfaces:**
- Consumes: `DOWNSCALE_MODES` from Task 1.
- Produces: `downscale_size(width: int, height: int, mode: str, max_size: int) -> tuple[tuple[int, int], tuple[int, int, int, int] | None]` — returns `((target_w, target_h), crop_box)`; `crop_box` is `(left, top, right, bottom)` to apply **before** resizing, or `None`. Task 4 applies this to Pillow images and their alpha channels.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_batch_loader_core.py`)

```python
from nodes.batch_loader_core import downscale_size


def test_downscale_off_untouched():
    assert downscale_size(2400, 1200, "off", 1200) == ((2400, 1200), None)


def test_downscale_below_threshold_untouched():
    for mode in ("fit", "crop", "stretch"):
        assert downscale_size(800, 600, mode, 1200) == ((800, 600), None)


def test_downscale_exactly_max_untouched():
    assert downscale_size(1200, 1200, "fit", 1200) == ((1200, 1200), None)


def test_fit_caps_longest_edge():
    assert downscale_size(2400, 1200, "fit", 1200) == ((1200, 600), None)
    assert downscale_size(1200, 2400, "fit", 1200) == ((600, 1200), None)


def test_fit_never_returns_zero():
    (w, h), box = downscale_size(5000, 2, "fit", 1200)
    assert (w, h) == (1200, 1) and box is None


def test_crop_centers_square_then_caps():
    # shorter edge 1200 -> centered 1200x1200 box, already <= max
    assert downscale_size(2400, 1200, "crop", 1200) == ((1200, 1200), (600, 0, 1800, 1200))
    # shorter edge 800 < max -> square of 800, no upscale
    assert downscale_size(2400, 800, "crop", 1200) == ((800, 800), (800, 0, 1600, 800))
    # shorter edge 2000 > max -> crop 2000-box, downscale to 1200
    assert downscale_size(3000, 2000, "crop", 1200) == ((1200, 1200), (500, 0, 2500, 2000))


def test_stretch_forces_square():
    assert downscale_size(2400, 1200, "stretch", 1200) == ((1200, 1200), None)


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="downscale_mode"):
        downscale_size(100, 100, "zoom", 1200)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_batch_loader_core.py -v -k downscale or fit or crop or stretch`
Expected: FAIL — `ImportError: cannot import name 'downscale_size'`

- [ ] **Step 3: Write the implementation** (append to `nodes/batch_loader_core.py`)

```python
def downscale_size(width, height, mode, max_size):
    """Compute the downscale geometry for one image.

    Returns ``((target_w, target_h), crop_box)``; ``crop_box`` is ``(left, top, right,
    bottom)`` to apply before resizing, or ``None``. Triggers only when the longest edge
    exceeds ``max_size``; ``fit`` and ``crop`` never upscale. ``stretch`` forces a
    ``max_size`` square (its shorter side may grow — inherent to forcing a square).
    """
    if mode not in DOWNSCALE_MODES:
        raise ValueError(f"Unknown downscale_mode {mode!r} — expected one of {DOWNSCALE_MODES}.")
    if mode == "off" or max(width, height) <= max_size:
        return ((width, height), None)

    if mode == "fit":
        scale = max_size / max(width, height)
        return ((max(1, round(width * scale)), max(1, round(height * scale))), None)

    if mode == "crop":
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        target = min(side, max_size)
        return ((target, target), (left, top, left + side, top + side))

    # stretch
    return ((max_size, max_size), None)
```

- [ ] **Step 4: Run the full core test file**

Run: `python -m pytest tests/test_batch_loader_core.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add nodes/batch_loader_core.py tests/test_batch_loader_core.py
git commit -m "feat: batch loader core - fit/crop/stretch downscale geometry"
```

---

### Task 3: Core — `remove_file`

**Files:**
- Modify: `nodes/batch_loader_core.py` (append)
- Test: `tests/test_batch_loader_core.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `remove_file(files: list, index: int) -> tuple[list, list[str]]` — new list without entry `index`, plus the names that shifted up a slot (callers use this to word the wire warning). Task 10 mirrors this logic in JS.

- [ ] **Step 1: Write the failing tests** (append)

```python
from nodes.batch_loader_core import remove_file


def _files(*names):
    return [{"name": n, "subfolder": "", "type": "input"} for n in names]


def test_remove_middle_shifts_up():
    new, moved = remove_file(_files("a.png", "b.png", "c.png"), 1)
    assert [f["name"] for f in new] == ["a.png", "c.png"]
    assert moved == ["c.png"]


def test_remove_last_moves_nothing():
    new, moved = remove_file(_files("a.png", "b.png"), 1)
    assert [f["name"] for f in new] == ["a.png"]
    assert moved == []


def test_remove_does_not_mutate_input():
    original = _files("a.png", "b.png")
    remove_file(original, 0)
    assert len(original) == 2


def test_remove_bad_index_raises():
    import pytest
    with pytest.raises(IndexError):
        remove_file(_files("a.png"), 5)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_batch_loader_core.py -v -k remove`
Expected: FAIL — `ImportError: cannot import name 'remove_file'`

- [ ] **Step 3: Implement** (append)

```python
def remove_file(files, index):
    """Remove ``files[index]`` without leaving a hole.

    Returns ``(new_files, moved_names)`` where ``moved_names`` are the files that were
    promoted one slot — the caller uses them to warn about disturbed wires.
    """
    if not 0 <= index < len(files):
        raise IndexError(f"file index {index} out of range (have {len(files)}).")
    new_files = files[:index] + files[index + 1:]
    moved_names = [f["name"] for f in files[index + 1:]]
    return new_files, moved_names
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all PASS (pre-existing tests untouched)

- [ ] **Step 5: Commit**

```bash
git add nodes/batch_loader_core.py tests/test_batch_loader_core.py
git commit -m "feat: batch loader core - shift-on-delete list op"
```

---

### Task 4: Batch Image Loader nodes (simple + Advanced)

**Files:**
- Create: `nodes/batch_image_loader.py`
- Test: `tests/test_batch_image_loader.py` (integration — SKIPS in dev env, runs inside ComfyUI)

**Interfaces:**
- Consumes: `parse_files`, `downscale_size`, `MAX_FILES`, `DOWNSCALE_MODES` from `nodes.batch_loader_core`; `folder_paths.get_input_directory()`; `comfy_api.latest.io`.
- Produces: classes `AI2GoBatchImageLoader`, `AI2GoBatchImageLoaderAdvanced` (Task 6 registers them); module function `_run_image(files_json: str, downscale_mode: str, max_size: int, advanced: bool) -> list` returning the padded output list (`[count, image_1, (mask_1, filename_1,) …]`, length 9 or 25, unused slots `None`) — tests target this, `execute` just wraps it in `io.NodeOutput`.

- [ ] **Step 1: Write the integration test**

```python
# Integration test for the Batch Image Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
import json

import pytest

pytest.importorskip("comfy_api")  # only runs inside a ComfyUI environment
import torch  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    return tmp_path


def test_advanced_images_masks_filenames(input_dir):
    from nodes.batch_image_loader import _run_image
    Image.new("RGBA", (32, 16), (255, 0, 0, 128)).save(input_dir / "a.png")
    Image.new("RGB", (8, 8), (0, 255, 0)).save(input_dir / "b.png")
    out = _run_image(json.dumps([{"name": "a.png"}, {"name": "b.png"}]), "off", 1200, advanced=True)

    assert len(out) == 25 and out[0] == 2
    img1, mask1, name1 = out[1], out[2], out[3]
    assert img1.shape == (1, 16, 32, 3)           # [1,H,W,C]
    assert mask1.shape == (1, 16, 32)             # inverted alpha, image-sized
    assert torch.allclose(mask1, torch.full_like(mask1, 1.0 - 128 / 255), atol=1e-3)
    assert name1 == "a.png"
    img2, mask2 = out[4], out[5]
    assert img2.shape == (1, 8, 8, 3)
    assert mask2.shape == (1, 64, 64) and mask2.max() == 0   # no alpha -> stock 64x64 zeros
    assert out[7] is None                          # padding starts after file 2's group


def test_simple_layout(input_dir):
    from nodes.batch_image_loader import _run_image
    Image.new("RGB", (8, 8), (0, 0, 255)).save(input_dir / "c.png")
    out = _run_image(json.dumps([{"name": "c.png"}]), "off", 1200, advanced=False)
    assert len(out) == 9 and out[0] == 1
    assert out[1].shape == (1, 8, 8, 3) and out[2] is None


def test_downscale_fit_applies(input_dir):
    from nodes.batch_image_loader import _run_image
    Image.new("RGB", (2400, 1200), (9, 9, 9)).save(input_dir / "big.png")
    out = _run_image(json.dumps([{"name": "big.png"}]), "fit", 1200, advanced=False)
    assert out[1].shape == (1, 600, 1200, 3)


def test_missing_file_raises(input_dir):
    from nodes.batch_image_loader import _run_image
    with pytest.raises(ValueError, match="gone.png"):
        _run_image(json.dumps([{"name": "gone.png"}]), "off", 1200, advanced=False)
```

- [ ] **Step 2: Run — verify it SKIPS in the dev env (not ERRORS)**

Run: `python -m pytest tests/test_batch_image_loader.py -v`
Expected: `SKIPPED (could not import 'comfy_api')` for every test. An ERROR or collection failure means the import guard is wrong — fix before continuing.

- [ ] **Step 3: Write the node module**

```python
# Batch Image Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
#
# Two loaders that emit one output socket group per dropped file (design:
# docs/superpowers/specs/2026-08-12-batch-loaders-design.md). Simple = image_N + count;
# Advanced adds mask_N + filename_N. Outputs are declared at the MAX_FILES ceiling and the
# front-end (web/js/batch_loader.js) trims the unused tail — slot order is count first, then
# grouped per file, so used slots stay contiguous (ComfyUI validates output types by position).
# Mask quirks copied from stock LoadImage: mask = 1.0 - alpha; no alpha -> 64x64 zeros.
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy_api.latest import io

from .batch_loader_core import DOWNSCALE_MODES, MAX_FILES, downscale_size, parse_files


def _input_path(f):
    """Resolve a files_json entry to a path inside the input directory (traversal-safe)."""
    base = os.path.abspath(folder_paths.get_input_directory())
    path = os.path.abspath(os.path.join(base, f["subfolder"], f["name"]))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"File path escapes the input folder: {f['name']!r}")
    return path


def _load_image(path, mode, max_size):
    """Load one image -> (IMAGE [1,H,W,3], MASK). Applies downscale geometry to both."""
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ValueError(f"Could not read image {os.path.basename(path)!r}: {e}") from e

    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")

    (tw, th), box = downscale_size(rgb.width, rgb.height, mode, max_size)
    if box is not None:
        rgb = rgb.crop(box)
        alpha = alpha.crop(box) if alpha is not None else None
    if (tw, th) != rgb.size:
        rgb = rgb.resize((tw, th), Image.LANCZOS)
        alpha = alpha.resize((tw, th), Image.LANCZOS) if alpha is not None else None

    image = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0)[None,]
    if alpha is not None:
        mask = 1.0 - torch.from_numpy(np.array(alpha).astype(np.float32) / 255.0)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32)  # stock LoadImage placeholder
    return image, mask.unsqueeze(0)


def _run_image(files_json, downscale_mode, max_size, advanced):
    """Shared engine for both nodes. Returns the padded output list (count first)."""
    files = parse_files(files_json)
    group = 3 if advanced else 1
    outputs = [None] * (1 + MAX_FILES * group)
    outputs[0] = len(files)
    for i, f in enumerate(files):
        path = _input_path(f)
        if not os.path.isfile(path):
            raise ValueError(f"File not found in the input folder: {f['name']!r} — re-add it to the node.")
        image, mask = _load_image(path, downscale_mode, max_size)
        base = 1 + i * group
        outputs[base] = image
        if advanced:
            outputs[base + 1] = mask
            outputs[base + 2] = f["name"]
    return outputs


def _fingerprint(files_json, downscale_mode, max_size):
    """Cache key: file list + size/mtime per file, so an on-disk change re-runs the node."""
    try:
        files = parse_files(files_json)
    except ValueError:
        return files_json
    sig = [str(downscale_mode), str(max_size)]
    for f in files:
        try:
            st = os.stat(_input_path(f))
            sig.append(f"{f['subfolder']}/{f['name']}:{st.st_size}:{st.st_mtime_ns}")
        except (OSError, ValueError):
            sig.append(f"{f['subfolder']}/{f['name']}:missing")
    return "|".join(sig)


def _image_inputs():
    return [
        io.String.Input(
            "files_json", default="[]",
            tooltip="Authoritative file list as JSON. Hidden in the UI and kept in sync by the "
                    "front-end — drop files onto the node instead of editing this.",
        ),
        io.Combo.Input(
            "downscale_mode", options=list(DOWNSCALE_MODES), default="off",
            tooltip="off = images pass through untouched. fit = shrink keeping shape so the longest "
                    "edge is max_size. crop = centered square, at most max_size. stretch = force a "
                    "max_size square (ignores shape). Never upscales (fit/crop); masks follow their image.",
        ),
        io.Int.Input(
            "max_size", default=1200, min=8, max=8192, step=8,
            tooltip="Size threshold/target in pixels for downscaling. Ignored while downscale_mode is off.",
        ),
    ]


def _image_outputs(advanced):
    outs = [io.Int.Output(display_name="count")]
    for i in range(1, MAX_FILES + 1):
        outs.append(io.Image.Output(display_name=f"image_{i}"))
        if advanced:
            outs.append(io.Mask.Output(display_name=f"mask_{i}"))
            outs.append(io.String.Output(display_name=f"filename_{i}"))
    return outs


class AI2GoBatchImageLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoBatchImageLoader",
            display_name="AI2Go Batch Image Loader",
            category="AI2Go/image",
            search_aliases=["batch", "load", "images", "multi", "drop", "upload"],
            description="Drop up to 8 images onto the node; each gets its own image_N output "
                        "socket (sockets appear/disappear with the list). Optional downscaling: "
                        "set downscale_mode to fit/crop/stretch and images larger than max_size "
                        "shrink on load. count = number of files loaded.",
            inputs=_image_inputs(),
            outputs=_image_outputs(advanced=False),
        )

    @classmethod
    def execute(cls, files_json="[]", downscale_mode="off", max_size=1200) -> io.NodeOutput:
        return io.NodeOutput(*_run_image(files_json, downscale_mode, max_size, advanced=False))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", downscale_mode="off", max_size=1200):
        return _fingerprint(files_json, downscale_mode, max_size)


class AI2GoBatchImageLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoBatchImageLoaderAdvanced",
            display_name="AI2Go Batch Image Loader Advanced",
            category="AI2Go/image",
            search_aliases=["batch", "load", "images", "multi", "drop", "upload", "mask", "filename"],
            description="Batch Image Loader plus a mask_N (inverted alpha, stock LoadImage "
                        "behavior) and filename_N output per file.",
            inputs=_image_inputs(),
            outputs=_image_outputs(advanced=True),
        )

    @classmethod
    def execute(cls, files_json="[]", downscale_mode="off", max_size=1200) -> io.NodeOutput:
        return io.NodeOutput(*_run_image(files_json, downscale_mode, max_size, advanced=True))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", downscale_mode="off", max_size=1200):
        return _fingerprint(files_json, downscale_mode, max_size)
```

- [ ] **Step 4: Syntax-check and run the suite**

Run: `python -m py_compile nodes/batch_image_loader.py && python -m pytest tests/ -v`
Expected: compile clean; new tests SKIPPED; everything else PASS.

- [ ] **Step 5: Commit**

```bash
git add nodes/batch_image_loader.py tests/test_batch_image_loader.py
git commit -m "feat: batch image loader nodes (simple + advanced)"
```

---

### Task 5: Batch Audio Loader nodes (simple + Advanced)

**Files:**
- Create: `nodes/batch_audio_loader.py`
- Test: `tests/test_batch_audio_loader.py` (integration — SKIPS in dev env)

**Interfaces:**
- Consumes: `parse_files`, `MAX_FILES` from core; `_input_path` pattern (re-implemented locally — see note in code); PyAV (`av`) and torch, both bundled with ComfyUI.
- Produces: classes `AI2GoBatchAudioLoader`, `AI2GoBatchAudioLoaderAdvanced`; `_run_audio(files_json: str, advanced: bool) -> list` (length 9 or 17; `AUDIO` values are `{"waveform": Tensor[1,C,S], "sample_rate": int}` at the file's native rate).

- [ ] **Step 1: Write the integration test**

```python
# Integration test for the Batch Audio Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
import json
import wave

import pytest

pytest.importorskip("comfy_api")
pytest.importorskip("av")


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    return tmp_path


def _write_wav(path, sample_rate=22050, n=2205):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x10" * n)


def test_advanced_audio_and_filenames(input_dir):
    from nodes.batch_audio_loader import _run_audio
    _write_wav(input_dir / "vo.wav", sample_rate=22050)
    _write_wav(input_dir / "bed.wav", sample_rate=44100)
    out = _run_audio(json.dumps([{"name": "vo.wav"}, {"name": "bed.wav"}]), advanced=True)

    assert len(out) == 17 and out[0] == 2
    a1, name1, a2 = out[1], out[2], out[3]
    assert a1["sample_rate"] == 22050          # native rate, no resampling
    assert a2["sample_rate"] == 44100
    assert a1["waveform"].ndim == 3 and a1["waveform"].shape[0] == 1   # [1,C,S]
    assert name1 == "vo.wav"
    assert out[5] is None                       # padding after file 2's group


def test_simple_layout(input_dir):
    from nodes.batch_audio_loader import _run_audio
    _write_wav(input_dir / "one.wav")
    out = _run_audio(json.dumps([{"name": "one.wav"}]), advanced=False)
    assert len(out) == 9 and out[0] == 1
    assert out[1]["sample_rate"] == 22050 and out[2] is None


def test_missing_file_raises(input_dir):
    from nodes.batch_audio_loader import _run_audio
    with pytest.raises(ValueError, match="gone.wav"):
        _run_audio(json.dumps([{"name": "gone.wav"}]), advanced=False)
```

- [ ] **Step 2: Run — verify SKIPPED in dev env**

Run: `python -m pytest tests/test_batch_audio_loader.py -v`
Expected: all `SKIPPED`, zero errors.

- [ ] **Step 3: Write the node module**

```python
# Batch Audio Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
#
# Two loaders that emit one output socket group per dropped file (design:
# docs/superpowers/specs/2026-08-12-batch-loaders-design.md). Simple = audio_N + count;
# Advanced adds filename_N. Audio is never resampled — each AUDIO output keeps its file's
# native sample rate, exactly like stock LoadAudio. The decoder below mirrors
# comfy_extras/nodes_audio.py load() (ComfyUI, GPL-3.0 — same license as this pack).
import os

import av
import torch

import folder_paths
from comfy_api.latest import io

from .batch_loader_core import MAX_FILES, parse_files


def _input_path(f):
    """Resolve a files_json entry to a path inside the input directory (traversal-safe)."""
    base = os.path.abspath(folder_paths.get_input_directory())
    path = os.path.abspath(os.path.join(base, f["subfolder"], f["name"]))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"File path escapes the input folder: {f['name']!r}")
    return path


def _f32_pcm(wav):
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def _load_audio(path):
    """Decode one file -> {"waveform": Tensor[1,C,S], "sample_rate": native_rate}."""
    try:
        with av.open(path) as af:
            if not af.streams.audio:
                raise ValueError("no audio stream in the file")
            stream = af.streams.audio[0]
            sr = stream.codec_context.sample_rate
            frames = []
            for frame in af.decode(streams=stream.index):
                buf = torch.from_numpy(frame.to_ndarray())
                if buf.shape[0] != frame.layout.nb_channels:
                    buf = buf.view(-1, frame.layout.nb_channels).t()
                frames.append(buf)
            if not frames:
                raise ValueError("file decoded to zero samples")
            wav = _f32_pcm(torch.cat(frames, dim=1))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read audio {os.path.basename(path)!r}: {e}") from e
    return {"waveform": wav.unsqueeze(0), "sample_rate": sr}


def _run_audio(files_json, advanced):
    """Shared engine for both nodes. Returns the padded output list (count first)."""
    files = parse_files(files_json)
    group = 2 if advanced else 1
    outputs = [None] * (1 + MAX_FILES * group)
    outputs[0] = len(files)
    for i, f in enumerate(files):
        path = _input_path(f)
        if not os.path.isfile(path):
            raise ValueError(f"File not found in the input folder: {f['name']!r} — re-add it to the node.")
        base = 1 + i * group
        outputs[base] = _load_audio(path)
        if advanced:
            outputs[base + 1] = f["name"]
    return outputs


def _fingerprint(files_json):
    try:
        files = parse_files(files_json)
    except ValueError:
        return files_json
    sig = []
    for f in files:
        try:
            st = os.stat(_input_path(f))
            sig.append(f"{f['subfolder']}/{f['name']}:{st.st_size}:{st.st_mtime_ns}")
        except (OSError, ValueError):
            sig.append(f"{f['subfolder']}/{f['name']}:missing")
    return "|".join(sig)


def _audio_inputs():
    return [
        io.String.Input(
            "files_json", default="[]",
            tooltip="Authoritative file list as JSON. Hidden in the UI and kept in sync by the "
                    "front-end — drop files onto the node instead of editing this.",
        ),
    ]


def _audio_outputs(advanced):
    outs = [io.Int.Output(display_name="count")]
    for i in range(1, MAX_FILES + 1):
        outs.append(io.Audio.Output(display_name=f"audio_{i}"))
        if advanced:
            outs.append(io.String.Output(display_name=f"filename_{i}"))
    return outs


class AI2GoBatchAudioLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoBatchAudioLoader",
            display_name="AI2Go Batch Audio Loader",
            category="AI2Go/audio",
            search_aliases=["batch", "load", "audio", "multi", "drop", "upload", "wav", "mp3"],
            description="Drop up to 8 audio files onto the node; each gets its own audio_N "
                        "output socket (sockets appear/disappear with the list). Audio is never "
                        "resampled — each output keeps its file's native sample rate. count = "
                        "number of files loaded.",
            inputs=_audio_inputs(),
            outputs=_audio_outputs(advanced=False),
        )

    @classmethod
    def execute(cls, files_json="[]") -> io.NodeOutput:
        return io.NodeOutput(*_run_audio(files_json, advanced=False))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]"):
        return _fingerprint(files_json)


class AI2GoBatchAudioLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoBatchAudioLoaderAdvanced",
            display_name="AI2Go Batch Audio Loader Advanced",
            category="AI2Go/audio",
            search_aliases=["batch", "load", "audio", "multi", "drop", "upload", "filename"],
            description="Batch Audio Loader plus a filename_N output per file.",
            inputs=_audio_inputs(),
            outputs=_audio_outputs(advanced=True),
        )

    @classmethod
    def execute(cls, files_json="[]") -> io.NodeOutput:
        return io.NodeOutput(*_run_audio(files_json, advanced=True))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]"):
        return _fingerprint(files_json)
```

Note: `_input_path` is duplicated (7 lines) rather than shared, because `batch_loader_core.py` must stay comfy-free (`folder_paths` doesn't exist in the dev env) and a third "comfy-shared" module for one helper isn't worth it. If a third user appears later, extract then.

- [ ] **Step 4: Syntax-check and run the suite**

Run: `python -m py_compile nodes/batch_audio_loader.py && python -m pytest tests/ -v`
Expected: compile clean; audio tests SKIPPED; everything else PASS.

- [ ] **Step 5: Commit**

```bash
git add nodes/batch_audio_loader.py tests/test_batch_audio_loader.py
git commit -m "feat: batch audio loader nodes (simple + advanced)"
```

---

### Task 6: Register the four nodes

**Files:**
- Modify: `__init__.py`

**Interfaces:**
- Consumes: the four classes from Tasks 4–5.
- Produces: mapping keys `AI2GoBatchImageLoader`, `AI2GoBatchImageLoaderAdvanced`, `AI2GoBatchAudioLoader`, `AI2GoBatchAudioLoaderAdvanced` — Task 7's JS keys off these exact node ids.

- [ ] **Step 1: Add imports and mapping entries**

Add after the existing imports in `__init__.py`:

```python
from .nodes.batch_image_loader import AI2GoBatchImageLoader, AI2GoBatchImageLoaderAdvanced
from .nodes.batch_audio_loader import AI2GoBatchAudioLoader, AI2GoBatchAudioLoaderAdvanced
```

Append inside `NODE_CLASS_MAPPINGS` (keys MUST match each schema's `node_id`):

```python
    "AI2GoBatchImageLoader": AI2GoBatchImageLoader,
    "AI2GoBatchImageLoaderAdvanced": AI2GoBatchImageLoaderAdvanced,
    "AI2GoBatchAudioLoader": AI2GoBatchAudioLoader,
    "AI2GoBatchAudioLoaderAdvanced": AI2GoBatchAudioLoaderAdvanced,
```

Append inside `NODE_DISPLAY_NAME_MAPPINGS`:

```python
    "AI2GoBatchImageLoader": "AI2Go Batch Image Loader",
    "AI2GoBatchImageLoaderAdvanced": "AI2Go Batch Image Loader Advanced",
    "AI2GoBatchAudioLoader": "AI2Go Batch Audio Loader",
    "AI2GoBatchAudioLoaderAdvanced": "AI2Go Batch Audio Loader Advanced",
```

- [ ] **Step 2: Verify**

Run: `python -m py_compile __init__.py && python -m pytest tests/ -v`
Expected: compile clean, suite green (the dev env never imports `__init__.py` — pytest's `--confcutdir=tests` keeps it out of collection).

- [ ] **Step 3: Commit**

```bash
git add __init__.py
git commit -m "feat: register batch loader nodes"
```

---

### Task 7: Front-end skeleton — trim/grow sockets + minimal upload

**Files:**
- Create: `web/js/batch_loader.js`

**Interfaces:**
- Consumes: node ids from Task 6; `chainCallback` from `web/js/utility.js`; hidden `files_json` widget from Tasks 4–5 schemas.
- Produces: extension `AI2Go.BatchLoader`; per-node state `node._blRows` (array of `{name, subfolder, type}`); helpers `syncOutputs(node, cfg, fileCount)`, `syncJson(node)`, `setStatus(text, color)`, `uploadFile(file) -> Promise<{name, subfolder, type}>`, `parseFiles(raw)` (JS mirror of core `parse_files`). Task 9–10 extend this file.

- [ ] **Step 1: Write the skeleton**

```js
/*
 * Part of ComfyUI-AI2Go-Utils.
 *
 * Shared front-end for the four Batch Loader nodes. GPL-3.0, like the rest of the pack.
 *
 * Files dropped/picked are uploaded once to ComfyUI's input/ folder; the hidden `files_json`
 * widget (a JSON array of {name, subfolder, type}) is the single source of truth for save,
 * restore and execution — the Prompt Batch pattern. The Python schema declares MAX_FILES
 * output groups; syncOutputs() trims node.outputs to `count` + the loaded groups and re-adds
 * up to the ceiling when files come back. Trimming only ever cuts from the end — ComfyUI
 * validates output types by slot position, so used slots must stay contiguous from slot 0.
 * parseFiles mirrors parse_files in nodes/batch_loader_core.py — keep the two in sync.
 */
import { chainCallback } from "./utility.js";
const { app } = window.comfyAPI.app;

const MAX_FILES = 8;

// group: [prefix, TYPE] per output within one file's group, in schema order.
const NODES = {
  AI2GoBatchImageLoader:         { kind: "image", group: [["image_", "IMAGE"]] },
  AI2GoBatchImageLoaderAdvanced: { kind: "image", group: [["image_", "IMAGE"], ["mask_", "MASK"], ["filename_", "STRING"]] },
  AI2GoBatchAudioLoader:         { kind: "audio", group: [["audio_", "AUDIO"]] },
  AI2GoBatchAudioLoaderAdvanced: { kind: "audio", group: [["audio_", "AUDIO"], ["filename_", "STRING"]] },
};

// ── Mirror of parse_files in nodes/batch_loader_core.py ──
function parseFiles(raw) {
  const text = (raw || "").trim();
  if (!text) return { ok: true, files: [] }; // empty is fine in the UI; Python rejects at run time
  let data;
  try { data = JSON.parse(text); } catch (e) { return { ok: false, error: "Malformed files_json: " + e.message }; }
  if (!Array.isArray(data)) return { ok: false, error: "Expected a JSON array of files." };
  const files = [];
  for (let i = 0; i < data.length && i < MAX_FILES; i++) {
    const e = data[i];
    if (!e || typeof e !== "object" || typeof e.name !== "string" || !e.name.trim()) {
      return { ok: false, error: `File #${i + 1}: 'name' must be a non-empty string.` };
    }
    files.push({ name: e.name, subfolder: typeof e.subfolder === "string" ? e.subfolder : "", type: "input" });
  }
  return { ok: true, files };
}

const findWidget = (node, name) => node.widgets?.find((w) => w.name === name);

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

// Trim node.outputs to count + fileCount groups; re-add (in schema order) up to the ceiling.
// removeOutput disconnects any links on the removed slot — that is the intended behavior.
function syncOutputs(node, cfg, fileCount) {
  const want = 1 + Math.min(fileCount, MAX_FILES) * cfg.group.length;
  while (node.outputs.length > want) node.removeOutput(node.outputs.length - 1);
  while (node.outputs.length < want) {
    const slot = node.outputs.length;                     // next slot index to create
    const gi = (slot - 1) % cfg.group.length;             // position within the file's group
    const fi = Math.floor((slot - 1) / cfg.group.length) + 1;  // 1-based file number
    const [prefix, type] = cfg.group[gi];
    node.addOutput(prefix + fi, type);
  }
  node.setDirtyCanvas?.(true, true);
}

// Upload one File to ComfyUI's input/ via the same endpoint the stock upload widget uses.
// The endpoint is generic despite its name; the form field is "image" even for audio.
async function uploadFile(file) {
  const form = new FormData();
  form.append("image", file);
  form.append("type", "input");
  const api = window.comfyAPI?.api?.api;
  const res = api?.fetchApi
    ? await api.fetchApi("/upload/image", { method: "POST", body: form })
    : await fetch("/upload/image", { method: "POST", body: form });
  if (res.status !== 200) throw new Error(`upload of ${file.name} failed (HTTP ${res.status})`);
  const data = await res.json();
  return { name: data.name, subfolder: data.subfolder || "", type: "input" };
}

app.registerExtension({
  name: "AI2Go.BatchLoader",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = NODES[nodeData?.name];
    if (!cfg) return;

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const jsonW = findWidget(node, "files_json");
      hideWidget(jsonW);
      node._blRows = [];

      // INT guard (image nodes): a number widget can serialize '' which fails INT validation.
      const maxSizeW = findWidget(node, "max_size");
      if (maxSizeW) {
        maxSizeW.serializeValue = () => {
          const v = parseInt(maxSizeW.value, 10);
          return Number.isFinite(v) && v >= 8 ? v : 1200;
        };
      }

      // Status line (read-only DOM widget).
      const statusEl = document.createElement("div");
      statusEl.style.cssText = "width:100%;box-sizing:border-box;padding:3px 6px;text-align:center;line-height:1.4;font:12px sans-serif;";
      const setStatus = (text, color) => { statusEl.textContent = text; statusEl.style.color = color; node.setDirtyCanvas?.(true, true); };
      node._blSetStatus = setStatus;

      function syncJson() {
        if (jsonW) jsonW.value = JSON.stringify(node._blRows);
      }
      node._blSyncJson = syncJson;
      node._blSyncOutputs = () => syncOutputs(node, cfg, node._blRows.length);

      async function addFiles(fileList) {
        const files = [...fileList].filter((f) => f.type.startsWith(cfg.kind + "/"));
        const rejected = fileList.length - files.length;
        const free = MAX_FILES - node._blRows.length;
        const taking = files.slice(0, free);
        const skipped = files.length - taking.length;
        for (const f of taking) {
          try {
            node._blRows.push(await uploadFile(f));
          } catch (e) {
            setStatus("❌ " + e.message, "#e0555a");
            break;
          }
        }
        syncJson();
        node._blSyncOutputs();
        node._blRender?.();
        const parts = [`${node._blRows.length} file${node._blRows.length === 1 ? "" : "s"} loaded`];
        if (skipped) parts.push(`only ${MAX_FILES} fit — ${skipped} skipped`);
        if (rejected) parts.push(`${rejected} not ${cfg.kind} — ignored`);
        setStatus((skipped || rejected ? "⚠ " : "✅ ") + parts.join("; "), skipped || rejected ? "#e0a03c" : "#46b4e6");
      }
      node._blAddFiles = addFiles;

      const addBtn = node.addWidget("button", cfg.kind === "image" ? "＋ Add images" : "＋ Add audio", null, () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.multiple = true;
        picker.accept = cfg.kind + "/*";
        picker.onchange = () => picker.files?.length && addFiles(picker.files);
        picker.click();
      });
      addBtn.serialize = false;

      node.addDOMWidget("batch_loader_status", "info", statusEl, { serialize: false });
      setStatus(`Drop ${cfg.kind} files here or press ＋ Add.`, "#8a8a8a");

      // Fresh node: no files yet -> trim the declared ceiling down to just `count`.
      node._blSyncOutputs();
    });

    // After a workflow loads: rebuild rows from the restored files_json, then re-trim.
    // (Serialized nodes save their trimmed outputs, so this is normally a no-op — it heals
    // hand-edited or older workflows.)
    chainCallback(nodeType.prototype, "onConfigure", function () {
      const node = this;
      requestAnimationFrame(() => {
        const res = parseFiles(findWidget(node, "files_json")?.value);
        node._blRows = res.ok ? res.files : [];
        node._blSyncJson?.();
        node._blSyncOutputs?.();
        node._blRender?.();
      });
    });
  },
});
```

- [ ] **Step 2: Validate as an ES module** (`node --check` on `.js` false-passes — pack memory)

```bash
cp web/js/batch_loader.js "$TMPDIR/batch_loader.mjs" 2>/dev/null || cp web/js/batch_loader.js /tmp/batch_loader.mjs
node --check /tmp/batch_loader.mjs
```
Expected: no output (clean parse). On Windows PowerShell: `Copy-Item web/js/batch_loader.js $env:TEMP/batch_loader.mjs; node --check $env:TEMP/batch_loader.mjs`.

- [ ] **Step 3: Commit**

```bash
git add web/js/batch_loader.js
git commit -m "feat: batch loader front-end skeleton - socket trim/grow + minimal upload"
```

---

### Task 8: LIVE SPIKE — user verifies trim/grow on frontend 1.48.7

**Files:** none (verification gate; the user runs this in their ComfyUI).

This is the spec's mandated spike: prove socket trim/re-add + save/load **before** building the rest of the UI. Everything after this task is UI polish on a proven mechanism.

- [ ] **Step 1: Hand the user this checklist** (they run it; restart the ComfyUI server first — new Python; hard-refresh the browser — new JS):

1. Add **AI2Go Batch Image Loader Advanced** to a graph → it should show only the `count` output (25 declared, trimmed to 1).
2. Press **＋ Add images**, pick 2 images → sockets `image_1, mask_1, filename_1, image_2, mask_2, filename_2` appear.
3. Wire `image_2` into a PreviewImage. Save the workflow. Reload the browser page and reopen the workflow → sockets and the wire must survive.
4. Queue the graph → PreviewImage shows the second image; no validation errors.
5. Add a third image after the wire exists → two more socket groups appear, existing wire untouched.
6. Repeat steps 1–2 with **AI2Go Batch Audio Loader** and one `.wav`/`.mp3` → `audio_1` appears; wire into a SaveAudio/PreviewAudio node and queue.

- [ ] **Step 2: Decision gate**

- All six pass → continue to Task 9.
- Trim/re-add misbehaves (sockets wrong after reload, links break, addOutput names drift) → STOP and apply the spec's fallback: delete `syncOutputs` + its call sites so all declared sockets stay visible (unused return `None`); the rest of the plan proceeds unchanged. Record which path was taken in the commit message.

---

### Task 9: Front-end — drop zone

**Files:**
- Modify: `web/js/batch_loader.js`

**Interfaces:**
- Consumes: `node._blAddFiles` from Task 7.
- Produces: a drop-zone DOM element per node (`node._blDropEl`), inserted before the status widget; Task 10's row list renders directly under it.

- [ ] **Step 1: Add shared styles + drop zone** — inside `onNodeCreated`, before the `addBtn` creation, insert:

```js
      // ── Drop zone (DOM widget). stopPropagation beats ComfyUI's global drop handler,
      // which would otherwise try to load the files as a workflow. ──
      const dropEl = document.createElement("div");
      dropEl.className = "ai2go-bl-drop";
      dropEl.textContent = cfg.kind === "image" ? "Drop images here" : "Drop audio here";
      for (const ev of ["dragenter", "dragover"]) {
        dropEl.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropEl.classList.add("over"); });
      }
      dropEl.addEventListener("dragleave", () => dropEl.classList.remove("over"));
      dropEl.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation();
        dropEl.classList.remove("over");
        if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
      });
      node._blDropEl = dropEl;
      node.addDOMWidget("batch_loader_drop", "drop", dropEl, { serialize: false });
```

And add a `ensureStyles()` function at module level (call it once inside `beforeRegisterNodeDef`, like `web/js/prompt_batch.js` does):

```js
function ensureStyles() {
  if (document.getElementById("ai2go-bl-style")) return;
  const s = document.createElement("style");
  s.id = "ai2go-bl-style";
  s.textContent = `
  .ai2go-bl-drop{box-sizing:border-box;width:100%;padding:10px;margin:2px 0;text-align:center;
    font:11.5px -apple-system,"Segoe UI",Roboto,sans-serif;color:#7ab8e6;background:#1d2733;
    border:1px dashed #46b4e6;border-radius:8px;cursor:copy}
  .ai2go-bl-drop.over{background:#24384c;border-style:solid}
  .ai2go-bl{display:flex;flex-direction:column;gap:5px;width:100%;box-sizing:border-box;
    font:12px/1.4 -apple-system,"Segoe UI",Roboto,sans-serif;color:#d3d3d0}
  .ai2go-bl .bl-row{display:flex;align-items:center;gap:7px;background:#262625;
    border:1px solid #3a3a38;border-radius:8px;padding:5px 7px}
  .ai2go-bl .bl-row.bl-drag{opacity:.45}
  .ai2go-bl .bl-row.bl-over{border-color:#46b4e6;box-shadow:0 0 0 1px #46b4e6 inset}
  .ai2go-bl .bl-grip{color:#6d6d68;font-size:14px;cursor:grab;user-select:none;flex:none}
  .ai2go-bl .bl-num{flex:none;width:18px;height:18px;border-radius:50%;background:#333331;
    color:#8b8b86;font:600 10px/18px ui-monospace,Consolas,monospace;text-align:center}
  .ai2go-bl .bl-thumb{flex:none;width:34px;height:34px;border-radius:4px;object-fit:cover;background:#1a1a19}
  .ai2go-bl .bl-wave{flex:none;width:34px;height:34px;border-radius:4px;background:#13332b;
    color:#46cca8;font-size:15px;line-height:34px;text-align:center}
  .ai2go-bl .bl-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11.5px}
  .ai2go-bl .bl-meta{flex:none;color:#8b8b86;font:10px ui-monospace,Consolas,monospace}
  .ai2go-bl .bl-x{flex:none;color:#6d6d68;font-size:13px;cursor:pointer;padding:2px}
  .ai2go-bl .bl-x:hover{color:#c86b6b}
  .ai2go-bl .bl-empty{padding:6px;text-align:center;color:#6d6d68;font-size:11px}
  `;
  document.head.appendChild(s);
}
```

- [ ] **Step 2: Validate as ESM**

Run: `cp web/js/batch_loader.js /tmp/batch_loader.mjs && node --check /tmp/batch_loader.mjs`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add web/js/batch_loader.js
git commit -m "feat: batch loader drop zone"
```

---

### Task 10: Front-end — rows UI (previews, delete-with-warning, reorder, sort)

**Files:**
- Modify: `web/js/batch_loader.js`

**Interfaces:**
- Consumes: `node._blRows`, `node._blSyncJson`, `node._blSyncOutputs`, `node._blSetStatus`, styles from Task 9; `remove_file` semantics from Task 3 (mirrored here).
- Produces: `node._blRender()` (already called by Task 7's `addFiles`/`onConfigure`); a **Sort by name** button; property mirror for `downscale_mode`/`max_size`.

- [ ] **Step 1: Add the rows widget** — inside `onNodeCreated`, after the drop zone block:

```js
      // ── Rows list (DOM widget): one row per file, in socket order. ──
      const listEl = document.createElement("div");
      listEl.className = "ai2go-bl";
      const rowsWidget = node.addDOMWidget("batch_loader_rows", "rows", listEl, { serialize: false });
      let dragIndex = -1;

      // Auto-fit node height to the rows (measured; the prompt_batch pattern).
      function fitToContent() {
        const h = Math.max(listEl.scrollHeight, 8);
        rowsWidget.computeSize = () => [node.size?.[0] || 300, h + 8];
        const want = node.computeSize?.();
        if (want) node.setSize([node.size[0], want[1]]);
        node.setDirtyCanvas?.(true, true);
      }
      let lastFitH = 0;
      const ro = new ResizeObserver(() => {
        const h = listEl.scrollHeight;
        if (h && h !== lastFitH) { lastFitH = h; fitToContent(); }
      });
      ro.observe(listEl);
      chainCallback(node, "onRemoved", () => ro.disconnect());

      const viewUrl = (f) =>
        `/view?filename=${encodeURIComponent(f.name)}&type=input&subfolder=${encodeURIComponent(f.subfolder)}`;

      // Any wire on any group socket means a reorder/delete changes what flows where.
      const anyGroupWired = () =>
        (node.outputs || []).slice(1).some((o) => o.links && o.links.length);

      function removeAt(k) {
        const removed = node._blRows[k].name;
        const moved = node._blRows.slice(k + 1).map((f) => f.name);   // mirror of core remove_file
        const start = 1 + k * cfg.group.length;
        const disturbed = (node.outputs || []).slice(start).some((o) => o.links && o.links.length);
        node._blRows.splice(k, 1);
        node._blSyncJson(); node._blSyncOutputs(); render();
        if (disturbed && moved.length) {
          node._blSetStatus(`⚠ Removed ${removed} — ${moved.join(", ")} moved up a slot. Check your wires.`, "#e0a03c");
        } else {
          node._blSetStatus(`Removed ${removed}.`, "#8a8a8a");
        }
      }

      function render() {
        listEl.replaceChildren();
        if (!node._blRows.length) {
          const empty = document.createElement("div");
          empty.className = "bl-empty";
          empty.textContent = "No files loaded.";
          listEl.appendChild(empty);
          return;
        }
        node._blRows.forEach((f, k) => {
          const row = document.createElement("div");
          row.className = "bl-row";

          const grip = document.createElement("span");
          grip.className = "bl-grip";
          grip.textContent = "⠿";
          grip.title = "Drag to reorder";
          grip.addEventListener("mousedown", () => { row.draggable = true; });
          row.addEventListener("mouseup", () => { row.draggable = false; });
          row.addEventListener("dragstart", (e) => { dragIndex = k; e.dataTransfer.effectAllowed = "move"; row.classList.add("bl-drag"); });
          row.addEventListener("dragend", () => { row.draggable = false; dragIndex = -1; row.classList.remove("bl-drag"); listEl.querySelectorAll(".bl-over").forEach((n) => n.classList.remove("bl-over")); });
          row.addEventListener("dragover", (e) => { e.preventDefault(); if (dragIndex > -1 && dragIndex !== k) row.classList.add("bl-over"); });
          row.addEventListener("dragleave", () => row.classList.remove("bl-over"));
          row.addEventListener("drop", (e) => {
            e.preventDefault(); e.stopPropagation();
            row.classList.remove("bl-over");
            if (dragIndex > -1 && dragIndex !== k) {
              const wired = anyGroupWired();
              const [movedRow] = node._blRows.splice(dragIndex, 1);
              node._blRows.splice(k, 0, movedRow);
              node._blSyncJson(); render();
              if (wired) node._blSetStatus("⚠ Reordered — sockets now carry different files. Check your wires.", "#e0a03c");
            }
          });

          const num = document.createElement("span");
          num.className = "bl-num";
          num.textContent = String(k + 1);

          let preview;
          const meta = document.createElement("span");
          meta.className = "bl-meta";
          if (cfg.kind === "image") {
            preview = document.createElement("img");
            preview.className = "bl-thumb";
            preview.src = viewUrl(f);
            preview.addEventListener("load", () => { meta.textContent = `${preview.naturalWidth}×${preview.naturalHeight}`; });
          } else {
            preview = document.createElement("span");
            preview.className = "bl-wave";
            preview.textContent = "♪";
            const probe = new Audio();
            probe.preload = "metadata";
            probe.src = viewUrl(f);
            probe.addEventListener("loadedmetadata", () => {
              const s = Math.round(probe.duration);
              meta.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
            });
          }

          const name = document.createElement("span");
          name.className = "bl-name";
          name.textContent = f.name;
          name.title = (f.subfolder ? f.subfolder + "/" : "") + f.name;

          const x = document.createElement("span");
          x.className = "bl-x";
          x.textContent = "✕";
          x.title = "Remove this file";
          x.addEventListener("click", () => removeAt(k));

          row.append(grip, num, preview, name, meta, x);
          listEl.appendChild(row);
        });
      }
      node._blRender = render;

      const sortBtn = node.addWidget("button", "Sort by name", null, () => {
        if (node._blRows.length < 2) return;
        const wired = anyGroupWired();
        node._blRows.sort((a, b) => a.name.localeCompare(b.name));
        node._blSyncJson(); render();
        node._blSetStatus(wired ? "⚠ Sorted — sockets now carry different files. Check your wires." : "Sorted by name.", wired ? "#e0a03c" : "#8a8a8a");
      });
      sortBtn.serialize = false;

      render();
```

- [ ] **Step 2: Add the property mirror** (module level, next to the `onConfigure` chain; the widgets_values save/restore mismatch — scalars after buttons shift a slot on load):

```js
    const MIRRORED = ["downscale_mode", "max_size"];   // present on image nodes only; findWidget just misses on audio
    chainCallback(nodeType.prototype, "onSerialize", function (o) {
      const mirror = {};
      for (const name of MIRRORED) {
        const w = findWidget(this, name);
        if (w) mirror[name] = w.value;
      }
      o.properties = o.properties || {};
      o.properties.ai2go_bl = mirror;
    });
```

And extend the existing `onConfigure` callback: before the `parseFiles` line, add

```js
        const mirror = arguments[0]?.properties?.ai2go_bl;
        if (mirror && typeof mirror === "object") {
          for (const name of MIRRORED) {
            const w = findWidget(node, name);
            if (w && mirror[name] !== undefined) w.value = mirror[name];
          }
        }
```

(Note: `onConfigure` receives the serialized node as its first argument — capture it as a named parameter if the existing chain doesn't already.)

- [ ] **Step 3: Validate as ESM**

Run: `cp web/js/batch_loader.js /tmp/batch_loader.mjs && node --check /tmp/batch_loader.mjs`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add web/js/batch_loader.js
git commit -m "feat: batch loader rows UI - previews, delete warning, reorder, sort"
```

---

### Task 11: README + version bump

**Files:**
- Modify: `README.md` (insert new section between "AI2Go Prompt Batch" and "AI2Go Resolution Selector" sections)
- Modify: `pyproject.toml` (version `1.6.1` → `1.7.0`)

- [ ] **Step 1: Add the README section**

```markdown
### AI2Go Batch Loaders (Image & Audio)

Drop **multiple files onto one node** and every file gets its **own output socket** — wire each
image or clip somewhere different in the same run. Four nodes, two flavors each:

| Node | Per file | Always |
|---|---|---|
| **Batch Image Loader** | `image_N` | `count` |
| **Batch Image Loader Advanced** | `image_N` + `mask_N` + `filename_N` | `count` |
| **Batch Audio Loader** | `audio_N` | `count` |
| **Batch Audio Loader Advanced** | `audio_N` + `filename_N` | `count` |

- **Up to 8 files per node.** Sockets appear as you add files and disappear as you remove them.
  Files are uploaded into ComfyUI's `input/` folder (like the stock Load Image), so saved
  workflows survive restarts.
- **Rows** show a thumbnail + pixel size (images) or a duration (audio). Drag the ⠿ grip to
  reorder, ✕ to remove, **Sort by name** for folder order. Row order = socket order.
- **Removing or reordering files shifts what each socket carries** — the node warns you in its
  status line whenever a change touches a socket that has a wire, so check your connections.
- **Downscaling (image nodes):** set `downscale_mode` to `fit` (shrink keeping shape), `crop`
  (centered square) or `stretch` (forced square) and any image whose longest edge exceeds
  `max_size` is shrunk on load — `off` (the default) passes images through untouched. Never
  upscales; masks are resized with their image; originals in `input/` are never modified.
- **Masks** (Advanced): inverted alpha, exactly like stock Load Image — files without an alpha
  channel yield the stock 64×64 empty mask.
- **Audio is never resampled** — each `audio_N` keeps its file's native sample rate.
```

- [ ] **Step 2: Bump the version**

In `pyproject.toml`: `version = "1.6.1"` → `version = "1.7.0"`.

- [ ] **Step 3: Verify and commit**

Run: `python -m pytest tests/ -v` (green) — then:

```bash
git add README.md pyproject.toml
git commit -m "docs: README section for batch loaders; bump to 1.7.0"
```

---

### Task 12: Final live verification (user checklist)

**Files:** none. The dev env has no ComfyUI — this is the user-run acceptance pass. Restart the ComfyUI server (Python changed) and hard-refresh the browser (JS changed) first.

- [ ] **Step 1: Hand the user this checklist**

**Image nodes**
1. Simple node: drop 3 mixed-size images onto the drop zone → 3 rows with thumbnails + pixel sizes; sockets `count, image_1..3` only.
2. Advanced node: same 3 files → triplet sockets per file; wire `image_1` and `mask_1` (use a PNG with transparency) into PreviewImage — mask previews as the inverted alpha.
3. Set `downscale_mode = fit`, `max_size = 1200`, queue with a >1200px image → output is capped at 1200 on the longest edge; with `off` it comes through full-size. `crop` gives a centered square.
4. Delete the middle row while `image_3` is wired → warning names the moved file; `image_3` socket disappears; the wire that pointed at it is gone.
5. Drag-reorder rows and press Sort by name with a wire attached → warning appears both times.
6. Drop 10 files at once → 8 load, status says 2 skipped. Drop a `.txt` → ignored with a notice.
7. Save the workflow; restart the ComfyUI server; reload the page; reopen → rows, sockets, wires, `downscale_mode`/`max_size` values all intact; queue runs clean.
8. Delete one of the files from the `input/` folder on disk, queue → clear error naming the file.

**Audio nodes**
9. Simple node: drop 2 audio files of different sample rates → rows show durations; wire `audio_1` into a PreviewAudio/SaveAudio and queue — plays at the correct (native) rate.
10. Advanced node: `filename_1` wired into a text/display node shows the original filename.

- [ ] **Step 2: Fix-and-retest loop**

Any failure: fix, re-run the affected checklist item, and only then continue.

- [ ] **Step 3: Close out**

When the checklist passes, this plan is done. (Publishing to the Comfy Registry / pushing the release is the user's existing "new version push" flow, not part of this plan.)
