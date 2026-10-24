# Ideogram 4 Prompt Builder node — part of ComfyUI-IntoTheLatent-Utils.
#
# Derived from ComfyUI-KJNodes (nodes/ideogram4_nodes.py) by Kijai,
# licensed under GPL-3.0; this file and pack remain GPL-3.0.
# Upstream: https://github.com/kijai/ComfyUI-KJNodes
#
"""Ideogram 4 prompt builder.

A single self-contained node with a visual bbox editor: draw regions on a blank
canvas, set each region's type/desc/text/color palette, and assemble the Ideogram 4 JSON caption prompt.
"""

import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

from comfy_api.latest import io

from . import layout_formats as lf


_FONT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fonts", "FreeMono.ttf")


def _hex_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)) if len(h) == 6 else (255, 255, 255)


def _readable(rgb):
    # Lighten toward white if too dark, so box-colored text stays legible on the dark canvas.
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum < 130:
        t = (130 - lum) / (255 - lum)
        r, g, b = round(r + (255 - r) * t), round(g + (255 - g) * t), round(b + (255 - b) * t)
    return (r, g, b)


def _font(size):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except Exception:
        try:
            return ImageFont.load_default(size)
        except Exception:
            return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split():
            test = word if not line else line + " " + word
            if line and draw.textlength(test, font=font) > max_w:
                lines.append(line)
                line = word
            else:
                line = test
        lines.append(line)
    return lines


def _bg_image(image):
    # IMAGE tensor -> PIL (first frame) for the preview background, None when absent/unreadable.
    if image is None:
        return None
    try:
        return Image.fromarray((image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    except Exception:
        return None


def _render_preview(boxes, width, height, bg=None, brightness=50):
    # Render the regions + prompts over the reference image (or a black canvas).
    if bg is not None:
        iw, ih = bg.size
        long_edge = max(iw, ih)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
        base = bg.convert("RGB").resize((rw, rh), Image.LANCZOS)
        if brightness < 100:                                # dim to match the editor's brightness slider
            base = ImageEnhance.Brightness(base).enhance(max(0.0, brightness / 100.0))
        img = base.convert("RGBA")
    else:
        long_edge = max(width, height)
        scale = min(1.0, 1024 / long_edge) if long_edge > 0 else 1.0
        rw = max(1, round(width * scale))
        rh = max(1, round(height * scale))
        img = Image.new("RGBA", (rw, rh), (0, 0, 0, 255))    # black so the overlay composites cleanly
    overlay = Image.new("RGBA", (rw, rh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fs = max(10, round(rh / 64))
    font = _font(fs)
    tag_font = _font(max(9, fs - 2))
    lh = fs + 2

    ids = lf.by_id(boxes)
    for i, box in enumerate(boxes):                          # enumerate the full list: tags show the editor's region numbers
        if not isinstance(box, dict) or box.get("nobbox") or box.get("group") or lf.eff_disabled(box, ids):
            continue                                        # skip unplaced elements, editor-only groups, and disabled
        palette = [c for c in (box.get("palette") or []) if c]
        r, g, b = _hex_rgb(palette[0]) if palette else (140, 140, 140)   # box = first palette color, else grey
        x1 = max(0, min(rw, round(box.get("x", 0) * rw)))
        y1 = max(0, min(rh, round(box.get("y", 0) * rh)))
        x2 = max(0, min(rw, round((box.get("x", 0) + box.get("w", 0)) * rw)))
        y2 = max(0, min(rh, round((box.get("y", 0) + box.get("h", 0)) * rh)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=2)

        pal5 = palette[:5]                                   # palette shown as a strip along the top edge
        if pal5 and (x2 - x1) > 2:
            sh = max(5, fs // 2)
            seg = (x2 - x1) / len(pal5)
            for p, hexc in enumerate(pal5):
                sx = x1 + round(p * seg)
                draw.rectangle([sx, y1, x1 + round((p + 1) * seg), y1 + sh], fill=_hex_rgb(hexc))

        etype = "text" if box.get("type") == "text" else "obj"
        tag = str(i + 1).zfill(2)
        tw = draw.textlength(tag, font=tag_font)
        draw.rectangle([x1, y1, x1 + tw + 6, y1 + fs + 2], fill=(r, g, b, 255))  # tag chip = box color
        tagfill = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x1 + 3, y1 + 1), tag, fill=tagfill, font=tag_font)

        body = box.get("desc", "") or ""
        if etype == "text" and box.get("text"):
            body = '"%s"%s' % (box["text"], " — " + body if body else "")
        if body and (x2 - x1) > 8:
            ty = y1 + fs + 5
            for line in _wrap(draw, body, font, x2 - x1 - 8):
                if ty > y2:
                    break
                draw.text((x1 + 4, ty), line, fill=_readable((r, g, b)) + (255,), font=font)
                ty += lh

    img = Image.alpha_composite(img, overlay).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class ITLIdeogram4PromptBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLIdeogram4PromptBuilder",
            display_name="ITL Ideogram 4 Prompt Builder",
            category="Into The Latent/text",
            search_aliases=["ideogram", "caption", "bbox", "prompt builder", "json prompt"],
            is_experimental=True,
            description="""
Visual prompt builder for Ideogram 4's structured JSON caption format.

Drag on the canvas to draw regions; select a region to set its type (obj/text),  
description, text, and color palette. Set the background and optional style fields  
as widgets. Outputs the assembled caption JSON string.  

bbox is normalized to a 0-1000 grid as [ymin, xmin, ymax, xmax]; width/height set
the canvas aspect ratio.

Canvas controls:
- Drag: draw a new region
- Ctrl/Cmd-drag: force-draw a new region even on top of an existing one
- Click: select a region · Alt-click: cycle overlapping regions
- Double-click: edit the description inline
- Right-click: region list (select / delete / duplicate / reorder, top = front)
- Del / Backspace: remove the selected region
- Ctrl/Cmd + C / V / D: copy / paste / duplicate the selected region
- bbox fields (px / out) next to obj/text are editable

Color swatches:
- Click: edit · Drag: reorder · Right-click: remove
- Hover + Ctrl/Cmd + C / V: copy / paste the hex
- "+": add a color (uses the clipboard color if it is one)

Toolbar:
- Live: use the live sampling preview as the background (and grab the final result)
- Grab BG / Clear BG: use the last generated image as the background
- brightness slider, token estimate, and Copy / Paste / Clear all""",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas aspect width (also the pixel grid the bbox is measured in). Ideogram 4 needs multiples of 16."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Canvas aspect height (also the pixel grid the bbox is measured in). Ideogram 4 needs multiples of 16."),
                io.String.Input("high_level_description", multiline=True, default="",
                                tooltip="Optional one-line overview of the whole image (blank = omitted)."),
                io.String.Input("background", multiline=True, default="",
                                tooltip="Required scene background description."),
                io.DynamicCombo.Input("style", options=[
                    io.DynamicCombo.Option("none", []),
                    io.DynamicCombo.Option("photo", [
                        io.String.Input("photo", default=""),
                    ]),
                    io.DynamicCombo.Option("art_style", [
                        io.String.Input("art_style", default=""),
                    ]),
                ]),
                io.String.Input("aesthetics", default="", tooltip="Style descriptor (blank = omitted)."),
                io.String.Input("lighting", default="", tooltip="Style descriptor (blank = omitted)."),
                io.String.Input("medium", default="", tooltip="Style descriptor (blank = omitted)."),
                io.Image.Input("image", optional=True,
                               tooltip="Optional reference image shown as the editor background (and behind the preview)."),
                io.String.Input("import_json", default="", optional=True, force_input=True,
                                tooltip="Optional: a full caption JSON. When connected, it loads into the editor "
                                        "and drives the output per 'import_mode'."),
                io.String.Input("style_palette_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized style color palette from the editor (managed by the node UI)."),
                io.String.Input("elements_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized regions from the editor (managed by the node UI)."),
                io.Int.Input("bg_brightness", default=25, min=0, max=100, socketless=True, advanced=True,
                             tooltip="Background image brightness % (managed by the node UI slider)."),
                io.Combo.Input("import_mode", options=["when empty", "always"], default="when empty",
                               tooltip="How a wired import_json is used: 'when empty' only seeds the editor while "
                                       "it has no regions (then the editor wins, so you can edit); 'always' makes "
                                       "the wired JSON authoritative so its changes always propagate to the output."),
                io.String.Input("output_format", default="compact", socketless=True, advanced=True,
                                tooltip="Output JSON formatting (set via the editor toolbar): 'compact' (default, what "
                                        "Ideogram 4 expects) or 'pretty' (indented, for readability)."),
                io.String.Input("coord_mode", default="normalized", socketless=True, advanced=True,
                                tooltip="bbox coordinate space (set via the editor toolbar): 'normalized' (default, the "
                                        "0-1000 grid Ideogram 4 expects) or 'absolute' (pixels, scaled by width/height "
                                        "— NON-STANDARD for Ideogram, for other tools that want pixel coords)."),
                io.String.Input("bbox_order", default="yx", socketless=True, advanced=True,
                                tooltip="bbox axis order (set via the editor toolbar): 'yx' (default, Ideogram's "
                                        "[ymin,xmin,ymax,xmax]) or 'xy' ([xmin,ymin,xmax,ymax], the standard x1,y1,x2,y2 "
                                        "that Qwen-VL and most detectors use)."),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                                     tooltip="Optional pixel-space boxes ({x, y, width, height}) used to seed the "
                                             "editor's regions when it has none. Ignored once regions exist."),
            ],
            outputs=[
                io.String.Output(display_name="prompt"),
                io.Image.Output(display_name="preview"),
                io.BoundingBox.Output(display_name="bboxes"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )


    @classmethod
    def execute(cls, width, height, background, style,
                high_level_description="", aesthetics="", lighting="", medium="",
                style_palette_data="", elements_data="", import_json="", import_mode="when empty",
                output_format="pretty", coord_mode="normalized", bbox_order="yx", bboxes=None, image=None, bg_brightness=25) -> io.NodeOutput:
        if import_mode not in ("when empty", "always"):      # old workflows saved before this widget existed
            import_mode = "when empty"
        dump = lf.dumps if output_format == "pretty" else lf.compact
        boxes = lf.parse_json_list(elements_data)
        boxes_seeded = False
        if not boxes:
            boxes = lf.seed_boxes(bboxes, width, height)
            boxes_seeded = bool(boxes)

        imported = lf.loads_layout(import_json)              # strict parse, then a lenient repair fallback
        src = lf.detect_format(imported)                     # None -> not a layout: ignored like prose

        # Use the wired import_json directly per import_mode: "always" -> authoritative (its changes
        # always propagate); "when empty" -> only seed the editor while it has no regions, then the
        # editor wins so manual edits stick. The editor mirrors it via ui when used.
        used_import = src is not None and (import_mode == "always" or not boxes)

        if used_import and src == "ideogram":
            caption = {k: v for k, v in imported.items() if k != "_itl"}   # strip the editor-only sidecar from the prompt output
            boxes, _ = lf.parse_ideogram(imported)
            mirror = imported
        else:
            if used_import:                                 # a MOSS-Image layout: convert, then build like editor output
                boxes, scene, _ = lf.import_layout(imported, "ideogram", width, height)
            else:
                kind = style["style"]                       # "none" | "photo" | "art_style"
                scene = lf.ideogram_scene(high_level_description=high_level_description, background=background,
                                          style_kind=kind, photo=style.get("photo", ""), art_style=style.get("art_style", ""),
                                          aesthetics=aesthetics, lighting=lighting, medium=medium,
                                          palette=lf.parse_json_list(style_palette_data))
            caption = lf.build_ideogram(boxes, scene, coord_mode=coord_mode, bbox_order=bbox_order,
                                        width=width, height=height)
            mirror = caption
        preview = _render_preview(boxes, width, height, _bg_image(image), bg_brightness)
        bboxes_out = lf.bbox_outputs(boxes, width, height)

        # ui: send the resolved width/height so the editor canvas can follow connected
        # inputs; import_json (if wired) loads into the editor (output reflects editor only).
        ui = {"dims": [width, height]}
        if boxes_seeded:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import:                                       # mirror the import in the editor (only when used)
            ui["caption"] = [lf.dumps(mirror)]
        return io.NodeOutput(dump(caption), preview, bboxes_out, width, height, ui=ui)
