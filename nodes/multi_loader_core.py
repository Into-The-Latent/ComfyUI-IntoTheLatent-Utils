# Multi Loader core — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Pure (comfy-free) logic shared by the four Multi Loader nodes (nodes/multi_image_loader.py,
# nodes/multi_audio_loader.py). parse_files is mirrored in web/js/multi_loader.js — keep in sync.
#
# "Multi" here means several files emitted in a single run — distinct from ITLPromptBatch's
# "Batch", which walks one entry per queued run.
import json

MAX_FILES = 8
DOWNSCALE_MODES = ("off", "keep aspect ratio", "crop to square", "stretch to square")


def parse_files(raw):
    """Parse the files_json widget into ``[{"name", "subfolder", "type", "enabled"}, ...]``.

    ``enabled`` is optional per entry and defaults to ``True`` (every pre-existing
    files_json — without the key — keeps working). A row with ``enabled: false`` still
    occupies its socket position; the loader nodes skip loading it and emit ``None`` in its
    place instead of shifting later rows up.

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
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{where}: 'enabled' must be a boolean.")
        files.append({"name": name, "subfolder": subfolder, "type": "input", "enabled": bool(enabled)})
    return files


def downscale_size(width, height, mode, max_size):
    """Compute the downscale geometry for one image.

    Returns ``((target_w, target_h), crop_box)``; ``crop_box`` is ``(left, top, right,
    bottom)`` to apply before resizing, or ``None``. Triggers only when the longest edge
    exceeds ``max_size``; ``keep aspect ratio`` and ``crop to square`` never upscale.
    ``stretch to square`` forces a ``max_size`` square (its shorter side may grow —
    inherent to forcing a square).
    """
    if mode not in DOWNSCALE_MODES:
        raise ValueError(f"Unknown downscale_mode {mode!r} — expected one of {DOWNSCALE_MODES}.")
    if mode == "off" or max(width, height) <= max_size:
        return ((width, height), None)

    if mode == "keep aspect ratio":
        scale = max_size / max(width, height)
        return ((max(1, round(width * scale)), max(1, round(height * scale))), None)

    if mode == "crop to square":
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        target = min(side, max_size)
        return ((target, target), (left, top, left + side, top + side))

    # stretch to square
    return ((max_size, max_size), None)


def resample_frame_indices(src_count, src_fps, target_fps):
    """Map a target-rate frame sequence back onto source frame indices, preserving duration.

    Mirrors VideoHelperSuite's force_rate semantics: retiming to a lower rate drops frames,
    retiming to a higher rate duplicates them, but the real running time (source frame count
    / source fps) is always kept — this is not a frame-count change, it's a playback-rate
    change.

    ``target_fps <= 0`` means "off" (0 is the widget's off value) — returns ``None`` so the
    caller passes the source through untouched instead of decoding it. ``src_fps <= 0`` is an
    unusable source rate and raises. ``src_count == 0`` returns ``[]`` (nothing to retime).
    """
    if target_fps <= 0:
        return None
    if src_fps <= 0:
        raise ValueError(f"Unusable source frame rate: {src_fps}")
    if src_count == 0:
        return []
    n = max(1, round(src_count / src_fps * target_fps))
    return [min(src_count - 1, int(i * src_fps / target_fps)) for i in range(n)]


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
