# Resolution Selector node — part of ComfyUI-IntoTheLatent-Utils.
#
# A standalone companion to the Ideogram 4 Prompt Builder. GPL-3.0, like the rest of the pack.
#
"""Resolution selector.

Computes a valid width/height from one of four modes and a target aspect ratio:

- raw        : type width/height directly (snapped to the profile's / snap_multiple's multiple).
- auto       : pick an aspect ratio; edit either side and the other follows.
- megapixel  : pick a target megapixel count + aspect ratio; both sides are computed.
- input      : the aspect ratio is detected from the width/height inputs (normally connected) and
               kept exactly; the megapixel count sets the size. Ignores aspect_ratio/orientation.

A *profile* selects the rules. "default" does no model clamping (its snap multiple is the
`snap_multiple` widget, default 8). "Ideogram 4" snaps to 16 and clamps 256-2048 px per side.
A landscape-only aspect list + an `orientation` toggle (driven by the JS flip button) cover both
orientations without duplicate entries. All math lives in resolution_core so it is unit-testable
without ComfyUI, and the editor JS mirrors it.
"""

from comfy_api.latest import io

from .resolution_core import (
    ASPECT_PRESETS, DEFAULT_PROFILE, PROFILES, aspect_label, aspect_options, exceeds_cap,
    resolve_dims, ui_payload,
)

_DEFAULT_ASPECT = aspect_label(*ASPECT_PRESETS[0])   # "1:1 (Square)"


class ITLResolutionSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLResolutionSelector",
            display_name="ITL Resolution Selector",
            category="Into The Latent/image",
            search_aliases=["resolution", "aspect ratio", "megapixel", "ideogram", "width", "height", "size"],
            is_experimental=True,
            description="""
Pick a valid resolution by mode + aspect ratio, output width/height as INT.

- profile: 'default' (no clamp; snaps to the 'snap_multiple' field) or a model ruleset like
  'Ideogram 4' (multiples of 16, 256-2048 px per side).
- resolution_mode: 'raw' (type width/height), 'auto' (pick a ratio; edit one side, the other
  follows), 'megapixel' (target megapixels + ratio; both computed), or 'input' (keep the exact
  aspect ratio of the width/height inputs and size it to the megapixel target).
- aspect_ratio lists square + landscape ratios; the '⟷' flip button (orientation) makes the
  portrait versions. megapixels feeds the megapixel and input modes.

Connect width/height (e.g. from an image's dimensions) and the node switches to 'input' mode
automatically, keeping that ratio exactly at whatever megapixel count you ask for.

Wire width/height into the ITL Ideogram 4 Prompt Builder's width/height inputs; edits push into
the builder's canvas live (they also apply on execution).""",
            inputs=[
                io.Combo.Input("profile", options=list(PROFILES.keys()), default=DEFAULT_PROFILE,
                               tooltip="Model ruleset. 'default' = no clamp, snaps to 'snap_multiple'. "
                                       "'Ideogram 4' = multiples of 16, 256-2048 px."),
                io.Combo.Input("resolution_mode", options=["raw", "auto", "megapixel", "input"], default="megapixel",
                               tooltip="'raw' = type width/height; 'auto' = pick a ratio and edit either side; "
                                       "'megapixel' = pick a target megapixels + ratio; 'input' = keep the exact "
                                       "ratio of the width/height inputs at the megapixel target (selected "
                                       "automatically when both are connected). All snap to the profile."),
                io.Combo.Input("aspect_ratio", options=aspect_options(), default=_DEFAULT_ASPECT,
                               tooltip="Target aspect ratio for 'auto' and 'megapixel'. Square + landscape only; "
                                       "use the flip button (orientation) for portrait. Ignored by 'input', which "
                                       "detects the ratio from the width/height inputs."),
                io.Float.Input("megapixels", default=1.0, min=0.1, max=16.0, step=0.1,
                               tooltip="Target size in megapixels for 'megapixel' and 'input' modes. (Ideogram 4 "
                                       "still clamps to ~4.19 MP at 2048x2048.)"),
                io.Int.Input("width", default=1024, min=64, max=16384, step=8,
                             tooltip="Width. Editable in 'raw' and 'auto'; computed in 'megapixel'. In 'input' it is "
                                     "the SOURCE width the aspect ratio is detected from — connect it. Snapped to the multiple."),
                io.Int.Input("height", default=1024, min=64, max=16384, step=8,
                             tooltip="Height. Editable in 'raw' and 'auto'; computed in 'megapixel'. In 'input' it is "
                                     "the SOURCE height the aspect ratio is detected from — connect it. Snapped to the multiple."),
                io.Int.Input("snap_multiple", default=8, min=1, max=1024, step=1,
                             tooltip="Round each side to a multiple of this. Most diffusion models require "
                                     "multiples of 8, so keep it at 8 unless your model needs otherwise. "
                                     "Ignored by model profiles that define their own multiple (Ideogram 4 = 16)."),
                io.Combo.Input("orientation", options=["landscape", "portrait"], default="landscape",
                               tooltip="Landscape or portrait. Toggled by the '⟷' flip button in the editor; "
                                       "portrait transposes the selected ratio (16:9 -> 9:16)."),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, profile=DEFAULT_PROFILE, snap_multiple=8, resolution_mode="megapixel",
                aspect_ratio=_DEFAULT_ASPECT, orientation="landscape", megapixels=1.0,
                width=1024, height=1024) -> io.NodeOutput:
        args = (profile, resolution_mode, aspect_ratio, orientation, snap_multiple, megapixels,
                width, height)
        w, h = resolve_dims(*args)
        # Run-time feedback for the editor readout: `dims` is what we resolved, `src` the width/
        # height the ratio was detected from, `clamped` whether the profile's cap cut it down — in
        # 'input' mode all three arrive over links, so the front-end cannot know any of them until
        # now (without the flag a clamped result would show no warning at all). Same ui/onExecuted
        # path as ideogram4_nodes. A cached node does not re-execute, so the readout keeps its
        # previous values. Payload shape lives in resolution_core, where tests can reach it.
        return io.NodeOutput(w, h, ui=ui_payload(w, h, width, height, exceeds_cap(*args)))
