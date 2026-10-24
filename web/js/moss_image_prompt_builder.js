// MOSS-Image Prompt Builder — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
//
// The MOSS-Image spec for the shared bbox editor in ideogram4_prompt_builder.js (which is derived from
// ComfyUI-KJNodes by Kijai, GPL-3.0). Everything format-specific for nodes/moss_image_nodes.py lives here:
// scene widgets (caption / meta / canvas), the object / character / text labels, per-box tags, the centred-pixel
// bbox readout and the serializer (layout_formats.js mirrors nodes/layout_formats.py).

import { registerLayoutSpec } from './ideogram4_prompt_builder.js';
import * as LF from './layout_formats.js';

const MOSS_SPEC = {
  nodeName: "ITLMossImagePromptBuilder", format: "moss", title: "MOSS-Image editor",
  sceneFields: ["caption", "meta", "canvas"],
  sceneDefaults: { canvas: "none" },
  labels: LF.MOSS_LABELS, defaultLabel: "object", nativeLabels: new Set(LF.MOSS_LABELS),
  toolbar: { coordToggle: false, orderToggle: false },      // MOSS fixes the coordinate space and axis order
  maxElemColors: 5, maxStyleColors: 16, paletteLabel: "Palette:", tokenCap: 4096,   // the Qwen3.5 text encoder truncates at 4096
  outHint: "Exported bbox in pixels from the canvas centre: ymin, xmin, ymax, xmax — editable",
  outBbox: (b, W, H) => LF.mossBbox(b, W, H) || [],
  outToBox: (nums, W, H) => LF.fromMossBbox(nums, W, H),
  sceneFromWidgets: ({ getW, stylePalette }) =>
    LF.mossScene({ caption: getW("caption"), meta: getW("meta"), canvas: getW("canvas"), palette: stylePalette }),
  // allowEmpty: the clipboard / templates / token estimate always see a layout object; the node itself emits the
  // plain caption when no item is placed (plainPrompt mirrors that for the token estimate).
  build: (boxes, scene, { width, height }) => LF.buildMoss(boxes, scene, width, height, { allowEmpty: true }),
  plainPrompt: (obj) => (obj.items.length ? null : obj.summary.caption),
  applyScene: (scene, { setWidgetVal }) => {
    setWidgetVal("caption", scene.caption || "");
    setWidgetVal("meta", (scene.meta || []).join(", "));
    setWidgetVal("canvas", LF.CANVASES.includes(scene.canvas) ? scene.canvas : "none");
  },
  hasTags: true, requiresBbox: true,
  alwaysSidecar: true,                                       // pixel bboxes: the sidecar's fraction boxes make a copy size-independent
  importPrompt: "Paste MOSS-Image layout (or Ideogram 4 caption) JSON:",
  importError: "Not a layout JSON (needs 'items' or 'compositional_deconstruction').",
};
registerLayoutSpec(MOSS_SPEC);
