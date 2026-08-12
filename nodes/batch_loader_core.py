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
