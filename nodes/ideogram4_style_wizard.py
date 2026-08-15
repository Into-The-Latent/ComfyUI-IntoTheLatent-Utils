# Ideogram 4 Style Wizard node — part of ComfyUI-IntoTheLatent-Utils.
#
# Companion to the Ideogram 4 Prompt Builder. GPL-3.0, like the rest of the pack.
# Upstream lineage: https://github.com/kijai/ComfyUI-KJNodes
#
"""Ideogram 4 style wizard.

A tiny "click-together" helper for the Ideogram 4 ``style_description`` fields. The whole UI lives in
the frontend (web/js/ideogram4_style_wizard.js): a button opens a chip-picker where the user toggles
aesthetics / lighting / medium / photo / art_style terms. On close the wizard writes those values
straight into a connected ITL Ideogram 4 Prompt Builder's widgets.

The node's own ``style`` output is wired into the builder's ``import_json`` only as the anchor that
tells the wizard which node to write into; it deliberately emits an empty string at runtime so the
builder keeps building from its own widgets (and the user's bounding boxes / color palette are never
wiped by the import). ``wizard_data`` just persists the picker state inside the workflow.
"""

from comfy_api.latest import io


class ITLIdeogram4StyleWizard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLIdeogram4StyleWizard",
            display_name="ITL Ideogram 4 Style Wizard",
            category="Into The Latent/text",
            search_aliases=["ideogram", "style", "wizard", "aesthetics", "lighting", "art_style"],
            is_experimental=True,
            description="""
Visual chip-picker for Ideogram 4's style_description fields.

Click "Open Style Wizard" and toggle one or many chips per category (aesthetics, lighting,
medium, photo, art_style); each category also has an editable text field and a live JSON preview.
On close, the selections are written straight into a connected ITL Ideogram 4 Prompt Builder.

Wire this node's "style" output into the Prompt Builder's "import_json" input — that link is only
used to find the target. This node emits an empty string at run time, so it never overwrites the
builder's bounding boxes, color palette, or high_level_description; it only fills the style fields.""",
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
        # The wizard does its work in the frontend (writeback on close). The data output stays empty so
        # a wire into the builder's import_json is inert and never clobbers the editor's regions.
        return io.NodeOutput("")
