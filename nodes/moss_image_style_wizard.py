# MOSS-Image Style Wizard node — part of ComfyUI-IntoTheLatent-Utils.
#
# Companion to the MOSS-Image Prompt Builder; shares its chip-picker with the Ideogram 4 Style Wizard.
# GPL-3.0, like the rest of the pack. Upstream lineage: https://github.com/kijai/ComfyUI-KJNodes
#
"""MOSS-Image style wizard.

The same "click-together" chip-picker as the Ideogram 4 Style Wizard (web/js/ideogram4_style_wizard.js drives
both), aimed at the ITL MOSS-Image Prompt Builder. MOSS-Image has no style fields, so on close the wizard writes
Medium + Art Style as summary ``meta`` tags and Aesthetics + Lighting + Photo as one "Style: ..." sentence at the
end of the builder's caption (web/js/moss_style.js), replacing only what it wrote itself.

The ``style`` output is wired into the builder's ``import_json`` only as the anchor that tells the wizard which
node to write into; it emits an empty string at run time, which the builder ignores. ``wizard_data`` persists the
picker state (and what the wizard last wrote) inside the workflow.
"""

from comfy_api.latest import io


class ITLMossImageStyleWizard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLMossImageStyleWizard",
            display_name="ITL MOSS-Image Style Wizard",
            category="Into The Latent/text",
            search_aliases=["moss", "moss-image", "style", "wizard", "aesthetics", "lighting", "art style", "medium"],
            is_experimental=True,
            description="""
Visual chip-picker for MOSS-Image styles — the same chips as the Ideogram 4 Style Wizard.

Click "Open Style Wizard" and toggle chips per category (aesthetics, lighting, medium, photo, art style);
each category also has an editable text field. On close the picks are written into a connected
ITL MOSS-Image Prompt Builder:
- Medium + Art Style -> the builder's meta tags
- Aesthetics + Lighting + Photo -> one "Style: ..." sentence at the end of the caption
The wizard only replaces what it wrote itself; your own caption text and meta tags stay.

Wire this node's "style" output into the Prompt Builder's "import_json" input — that link is only used to
find the target. This node emits an empty string at run time, so it never overwrites the builder.""",
            inputs=[
                io.String.Input("wizard_data", default="", socketless=True, advanced=True,
                                tooltip="Serialized chip-picker state (managed by the node UI)."),
            ],
            outputs=[
                io.String.Output(display_name="style"),
            ],
        )

    @classmethod
    def execute(cls, wizard_data="") -> io.NodeOutput:
        # The wizard does its work in the frontend (writeback on close); the output only anchors the link.
        return io.NodeOutput("")
