# MOSS-Image Prompt Builder node — part of ComfyUI-IntoTheLatent-Utils.
#
# Shares its visual editor with the Ideogram 4 Prompt Builder (nodes/ideogram4_nodes.py,
# web/js/ideogram4_prompt_builder.js), which is derived from ComfyUI-KJNodes by Kijai,
# licensed under GPL-3.0; this file and pack remain GPL-3.0.
# Upstream: https://github.com/kijai/ComfyUI-KJNodes
#
"""MOSS-Image layout prompt builder.

MOSS-Image (OpenMOSS) reads a layout as a JSON prompt: a ``summary`` (caption, optional meta tags, canvas kind,
palette) plus ``items``, each with a label (object / character / text), a caption, the text to render, optional
tags / palette, and a bbox in absolute pixels measured from the canvas centre (``[ymin, xmin, ymax, xmax]``,
``bbox_coord_space`` "abs_centralized", ``bbox_format`` "yxyx"). The prompt carries no canvas size, so the boxes
are only right at the width/height the image is generated at. Same output as ComfyUI core's
comfy_extras/nodes_moss_image.py, built from the shared bbox editor instead of Create Bounding Boxes.
"""

import json

from comfy_api.latest import io

from . import layout_formats as lf
from .ideogram4_nodes import _render_preview, _bg_image


class ITLMossImagePromptBuilder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLMossImagePromptBuilder",
            display_name="ITL MOSS-Image Prompt Builder",
            category="Into The Latent/text",
            search_aliases=["moss", "moss-image", "layout", "bbox", "prompt builder", "json prompt"],
            is_experimental=True,
            description="""
Visual prompt builder for MOSS-Image's layout JSON (summary + items with pixel bboxes).

Drag on the canvas to draw regions; select a region to set its label (object / character / text),
caption, text, tags and color palette. The caption widget describes the whole scene; meta and canvas
are optional summary fields. Outputs the layout JSON — or the plain caption when no region is placed.

bbox is in absolute pixels of width/height measured from the canvas centre, [ymin, xmin, ymax, xmax]:
generate at exactly this width/height. An imported MOSS layout is read at the node's current
width/height too. Paste / import_json / templates accept MOSS-Image and Ideogram 4 JSON alike;
an Ideogram caption is converted (its scene fields are joined into the summary caption).

Same editor as the Ideogram 4 Prompt Builder: see that node for the canvas, Overview and group controls.""",
            inputs=[
                io.Int.Input("width", default=1024, min=64, max=16384, step=16,
                             tooltip="Generation width in pixels — the bboxes are measured in it. MOSS-Image needs multiples of 16."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=16,
                             tooltip="Generation height in pixels — the bboxes are measured in it. MOSS-Image needs multiples of 16."),
                io.String.Input("caption", multiline=True, default="",
                                tooltip="summary.caption: the whole scene and style. Used as the plain prompt when no region is placed."),
                io.String.Input("meta", multiline=True, default="",
                                tooltip="summary.meta: comma/newline-separated medium or genre tags (blank = omitted)."),
                io.Combo.Input("canvas", options=["none"] + list(lf.CANVASES), default="none",
                               tooltip="summary.canvas: single, multi_panel or collage ('none' = omitted)."),
                io.Image.Input("image", optional=True,
                               tooltip="Optional reference image shown as the editor background (and behind the preview)."),
                io.String.Input("import_json", default="", optional=True, force_input=True,
                                tooltip="Optional: a MOSS-Image layout or Ideogram 4 caption JSON. When connected, it loads "
                                        "into the editor and drives the output per 'import_mode'."),
                io.String.Input("style_palette_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized summary palette from the editor (managed by the node UI)."),
                io.String.Input("elements_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized regions from the editor (managed by the node UI)."),
                io.Int.Input("bg_brightness", default=25, min=0, max=100, socketless=True, advanced=True,
                             tooltip="Background image brightness % (managed by the node UI slider)."),
                io.Combo.Input("import_mode", options=["when empty", "always"], default="when empty",
                               tooltip="How a wired import_json is used: 'when empty' only seeds the editor while "
                                       "it has no regions (then the editor wins, so you can edit); 'always' makes "
                                       "the wired JSON authoritative so its changes always propagate to the output."),
                io.String.Input("output_format", default="compact", socketless=True, advanced=True,
                                tooltip="Output JSON formatting (set via the editor toolbar): 'compact' (default, one line, "
                                        "what MOSS-Image reads) or 'pretty' (indented, for readability)."),
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
    def execute(cls, width, height, caption, meta="", canvas="none", style_palette_data="", elements_data="",
                import_json="", import_mode="when empty", output_format="compact", bboxes=None, image=None,
                bg_brightness=25) -> io.NodeOutput:
        if import_mode not in ("when empty", "always"):
            import_mode = "when empty"
        dump = lf.dumps if output_format == "pretty" else lf.compact
        boxes = lf.parse_json_list(elements_data)
        boxes_seeded = False
        if not boxes:
            boxes = lf.seed_boxes(bboxes, width, height)
            boxes_seeded = bool(boxes)

        imported = lf.loads_layout(import_json)
        src = lf.detect_format(imported)                     # None -> not a layout: ignored like prose
        used_import = src is not None and (import_mode == "always" or not boxes)

        if used_import:
            # Either format is parsed (a MOSS layout at THIS node's width/height, an Ideogram caption converted)
            # and rebuilt like editor output, so the prompt's pixel bboxes always match the width/height outputs.
            boxes, scene, _ = lf.import_layout(imported, "moss", width, height)
        else:
            scene = lf.moss_scene(caption=caption, meta=meta, canvas=canvas,
                                  palette=lf.parse_json_list(style_palette_data))
        layout = lf.build_moss(boxes, scene, width, height)
        prompt = dump(layout) if layout is not None else scene["caption"].strip()   # no placed items -> plain text (stripped like summary.caption)
        preview = _render_preview(boxes, width, height, _bg_image(image), bg_brightness)
        bboxes_out = lf.bbox_outputs(boxes, width, height)

        ui = {"dims": [width, height]}
        if boxes_seeded:
            ui["boxes"] = [json.dumps(boxes)]
        if used_import:                                       # mirror the raw import in the editor (only when used); the editor
            ui["caption"] = [lf.dumps(imported)]              # converts an Ideogram caption itself (layout_formats.js), keeping
        return io.NodeOutput(prompt, preview, bboxes_out, width, height, ui=ui)
