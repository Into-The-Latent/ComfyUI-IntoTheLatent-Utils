# Multi Image Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
#
# Two loaders that emit one output socket group per dropped file (design:
# docs/superpowers/specs/2026-08-12-batch-loaders-design.md). Simple = image_N + count;
# Advanced adds mask_N + filename_N. Outputs are declared at the MAX_FILES ceiling and the
# front-end (web/js/multi_loader.js) trims the unused tail — slot order is count first, then
# grouped per file, so used slots stay contiguous (ComfyUI validates output types by position).
# Mask quirks copied from stock LoadImage: mask = 1.0 - alpha; no alpha -> 64x64 zeros.
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from comfy_api.latest import io

from .multi_loader_core import DOWNSCALE_MODES, MAX_FILES, downscale_size, parse_files


def _input_path(f):
    """Resolve a files_json entry to a path inside the input directory (traversal-safe)."""
    base = os.path.abspath(folder_paths.get_input_directory())
    path = os.path.abspath(os.path.join(base, f["subfolder"], f["name"]))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"File path escapes the input folder: {f['name']!r}")
    return path


def _load_image(path, mode, max_size, pos):
    """Load one image -> (IMAGE [1,H,W,3], MASK). Applies downscale geometry to both.

    ``pos`` is the file's 1-based position in files_json, included in error messages so a
    failure can be traced back to a specific row in the UI.
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ValueError(f"File #{pos}: Could not read image {os.path.basename(path)!r}: {e}") from e

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
            raise ValueError(f"File #{i + 1} ({f['name']!r}): not found in the input folder — re-add it to the node.")
        image, mask = _load_image(path, downscale_mode, max_size, i + 1)
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
            tooltip="off = images pass through untouched. keep aspect ratio = shrink keeping shape so "
                    "the longest edge is max_size. crop to square = centered square, at most max_size. "
                    "stretch to square = force a max_size square (ignores shape). Never upscales (keep "
                    "aspect ratio/crop to square); masks follow their image.",
        ),
        io.Int.Input(
            "max_size", default=1200, min=8, max=8192, step=8,
            tooltip="Size threshold/target in pixels for downscaling. Ignored while downscale_mode is off.",
        ),
        io.Combo.Input(
            "output_slots", options=["auto", "1", "2", "3", "4", "5", "6", "7", "8"], default="auto",
            tooltip="How many output sockets to show. 'auto' follows the number of loaded files, so "
                    "sockets appear and disappear as you edit the list. Pick a fixed number to keep the "
                    "sockets (and your wires) in place while you swap files around — extra sockets with "
                    "no file behind them output nothing, so don't wire more than you load.",
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


class AI2GoMultiImageLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoMultiImageLoader",
            display_name="AI2Go Multi Image Loader",
            category="AI2Go/image",
            search_aliases=["batch", "load", "images", "multi", "drop", "upload"],
            description="Drop up to 8 images onto the node; each gets its own image_N output "
                        "socket (sockets appear/disappear with the list, or pin output_slots to a "
                        "fixed count so wires survive file edits). Optional downscaling: set "
                        "downscale_mode to keep aspect ratio/crop to square/stretch to square and "
                        "images larger than max_size shrink on load. count = number of files loaded.",
            inputs=_image_inputs(),
            outputs=_image_outputs(advanced=False),
        )

    @classmethod
    def execute(cls, files_json="[]", downscale_mode="off", max_size=1200, output_slots="auto") -> io.NodeOutput:
        # output_slots is front-end-only (see web/js/multi_loader.js): it only picks how many
        # sockets are shown, not what the sockets carry. Accepted here only so it serializes /
        # the socket exists.
        return io.NodeOutput(*_run_image(files_json, downscale_mode, max_size, advanced=False))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", downscale_mode="off", max_size=1200, output_slots="auto"):
        return _fingerprint(files_json, downscale_mode, max_size)


class AI2GoMultiImageLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoMultiImageLoaderAdvanced",
            display_name="AI2Go Multi Image Loader Advanced",
            category="AI2Go/image",
            search_aliases=["batch", "load", "images", "multi", "drop", "upload", "mask", "filename"],
            description="Multi Image Loader plus a mask_N (inverted alpha, stock LoadImage "
                        "behavior) and filename_N output per file.",
            inputs=_image_inputs(),
            outputs=_image_outputs(advanced=True),
        )

    @classmethod
    def execute(cls, files_json="[]", downscale_mode="off", max_size=1200, output_slots="auto") -> io.NodeOutput:
        # output_slots is front-end-only (see web/js/multi_loader.js): it only picks how many
        # sockets are shown, not what the sockets carry. Accepted here only so it serializes /
        # the socket exists.
        return io.NodeOutput(*_run_image(files_json, downscale_mode, max_size, advanced=True))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", downscale_mode="off", max_size=1200, output_slots="auto"):
        return _fingerprint(files_json, downscale_mode, max_size)
