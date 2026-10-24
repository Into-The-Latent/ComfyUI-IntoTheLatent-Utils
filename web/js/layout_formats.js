// Layout prompt formats — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
//
// Frontend mirror of nodes/layout_formats.py: the same box list ({x, y, w, h} fractions + type/text/desc/palette/
// tags and the editor-only id/parent/group/disabled/nobbox) and scene dicts, the same key orders and rounding,
// so the editor's token estimate, clipboard, templates and bbox readouts match what the Python node emits.
// Keep the two files in step: every function here has a namesake there.

// ── generic ──────────────────────────────────────────────────────────────────────────────────

export function detectFormat(obj) {
  // "moss" (an items list), "ideogram" (compositional_deconstruction), else null.
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return null;
  if (Array.isArray(obj.items)) return "moss";
  if ("compositional_deconstruction" in obj) return "ideogram";
  return null;
}

export function upperPalette(colors) {
  // ["#rrggbb", ...] -> ["#RRGGBB", ...] in order, dropping empties.
  return (colors || []).filter((c) => c).map((c) => String(c).toUpperCase());
}

export function strs(v) {
  // A list of non-empty strings; a comma/newline separated string is split (the editor's meta/tags fields).
  if (typeof v === "string") v = v.split(/[,\n]/);
  return (Array.isArray(v) ? v : []).map((t) => String(t).trim()).filter(Boolean);
}

function effDisabled(box, byId) {
  // True if the box or any ancestor has `disabled` — disabling a group disables its members.
  let cur = box, guard = 0;
  while (cur && typeof cur === "object" && guard++ < 1000) {
    if (cur.disabled) return true;
    if (cur.parent == null) return false;
    cur = byId.get(cur.parent);
  }
  return false;
}

export function exportedBoxes(boxes) {
  // The boxes that reach a prompt: objects that are neither groups nor (effectively) disabled.
  const byId = new Map();
  for (const b of boxes) if (b && typeof b === "object" && b.id) byId.set(b.id, b);
  return boxes.filter((b) => b && typeof b === "object" && !b.group && !effDisabled(b, byId));
}

export function unplacedBox(idx) {
  // Placeholder for an element without a bbox: small, staggered, flagged so export omits bbox.
  const k = (idx || 0) % 6;
  return { x: 0.03 + k * 0.035, y: 0.03 + k * 0.035, w: 0.22, h: 0.14, nobbox: true };
}

// ── Ideogram 4 ───────────────────────────────────────────────────────────────────────────────

export function normBbox(box, sx = 1000, sy = 1000, order = "yx") {
  // {x, y, w, h} fractions -> bbox; x scaled by sx, y by sy. Default: the Ideogram 0-1000 grid.
  // order "yx" -> [ymin, xmin, ymax, xmax] (Ideogram); "xy" -> [xmin, ymin, xmax, ymax] (Qwen/standard).
  const cx = (v) => Math.max(0, Math.min(sx, Math.round(v * sx)));
  const cy = (v) => Math.max(0, Math.min(sy, Math.round(v * sy)));
  let ymin = cy(box.y || 0), xmin = cx(box.x || 0), ymax = cy((box.y || 0) + (box.h || 0)), xmax = cx((box.x || 0) + (box.w || 0));
  if (ymin > ymax) [ymin, ymax] = [ymax, ymin];
  if (xmin > xmax) [xmin, xmax] = [xmax, xmin];
  return order === "xy" ? [xmin, ymin, xmax, ymax] : [ymin, xmin, ymax, xmax];
}

export function ideogramScene({ high_level_description = "", background = "", style_kind = "none", photo = "", art_style = "",
                                aesthetics = "", lighting = "", medium = "", palette = [] } = {}) {
  return { high_level_description, background, style_kind: (style_kind === "photo" || style_kind === "art_style") ? style_kind : "none",
           photo, art_style, aesthetics, lighting, medium, palette: (palette || []).slice() };
}

export function buildIdeogram(boxes, scene, { coordMode = "normalized", bboxOrder = "yx", width = 1024, height = 1024 } = {}) {
  const [bsx, bsy] = coordMode === "absolute" ? [width, height] : [1000, 1000];
  const border = bboxOrder === "xy" ? "xy" : "yx";
  const cap = {};
  if ((scene.high_level_description || "").trim()) cap.high_level_description = scene.high_level_description;
  const kind = scene.style_kind || "none";
  if (kind !== "none") {
    // Every style key present (in order) once a style is chosen; only color_palette is conditional.
    const sd = { aesthetics: scene.aesthetics || "", lighting: scene.lighting || "" };
    if (kind === "photo") { sd.photo = scene.photo || ""; sd.medium = scene.medium || ""; }
    else { sd.medium = scene.medium || ""; sd.art_style = scene.art_style || ""; }
    const pal = upperPalette(scene.palette);
    if (pal.length) sd.color_palette = pal;
    cap.style_description = sd;
  }
  const elements = exportedBoxes(boxes).map((b) => {
    const etype = b.type === "text" ? "text" : "obj";
    const el = { type: etype };                             // key order matters
    if (!b.nobbox) el.bbox = normBbox(b, bsx, bsy, border);  // unplaced elements omit bbox
    if (etype === "text") el.text = b.text || "";
    el.desc = b.desc || "";
    const pal = upperPalette(b.palette);
    if (pal.length) el.color_palette = pal.slice(0, 5);
    return el;
  });
  cap.compositional_deconstruction = { background: scene.background || "", elements };
  return cap;
}

export function parseIdeogram(cap) {
  // Caption -> {boxes, scene}; bbox assumed on the 0-1000 grid in [ymin, xmin, ymax, xmax] order.
  const cd = (cap && cap.compositional_deconstruction) || {};
  const boxes = [];
  (Array.isArray(cd.elements) ? cd.elements : []).forEach((el, i) => {
    if (!el || typeof el !== "object") return;
    const box = { type: el.type === "text" ? "text" : "obj", text: el.text || "", desc: el.desc || "",
                  palette: Array.isArray(el.color_palette) ? el.color_palette.slice() : [] };
    const bb = el.bbox;
    if (Array.isArray(bb) && bb.length === 4) {
      const [ymin, xmin, ymax, xmax] = bb;
      Object.assign(box, { x: xmin / 1000, y: ymin / 1000, w: (xmax - xmin) / 1000, h: (ymax - ymin) / 1000 });
    } else Object.assign(box, unplacedBox(i));
    boxes.push(box);
  });
  const sd = (cap.style_description && typeof cap.style_description === "object") ? cap.style_description : {};
  const kind = typeof sd.photo === "string" ? "photo" : typeof sd.art_style === "string" ? "art_style" : "none";
  const scene = ideogramScene({ high_level_description: cap.high_level_description || "", background: cd.background || "",
                                style_kind: kind, photo: sd.photo || "", art_style: sd.art_style || "", aesthetics: sd.aesthetics || "",
                                lighting: sd.lighting || "", medium: sd.medium || "", palette: sd.color_palette || [] });
  return { boxes, scene };
}

// ── MOSS-Image ───────────────────────────────────────────────────────────────────────────────
// {"summary": {meta?, canvas?, caption, palette?}, "items": [{label, caption, text?, tags?, palette?, bbox}],
//  "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"} — bbox = [ymin, xmin, ymax, xmax] in absolute
// pixels measured from the canvas centre.

export const CANVASES = ["single", "multi_panel", "collage"];
export const MOSS_LABELS = ["object", "character", "text"];
export const MOSS_SPACE = "abs_centralized", MOSS_FORMAT = "yxyx";

export function mossLabel(t) { return (t === "text" || t === "character") ? t : "object"; }

export function mossScene({ caption = "", meta = [], canvas = "none", palette = [] } = {}) {
  return { caption, meta: strs(meta), canvas: CANVASES.includes(canvas) ? canvas : "none", palette: (palette || []).slice() };
}

function edges(box) {
  // {x, y, w, h} fractions -> clamped [x1, y1, x2, y2] fractions, negative sizes normalized.
  const x = +box.x || 0, y = +box.y || 0, w = +box.w || 0, h = +box.h || 0;
  const [x1, x2] = [x, x + w].sort((a, b) => a - b), [y1, y2] = [y, y + h].sort((a, b) => a - b);
  return [x1, y1, x2, y2].map((v) => Math.max(0, Math.min(1, v)));
}

const r9 = (v) => Math.round(v * 1e9) / 1e9;   // round(…, 9): float noise never ceils 307.0000000001 to 308

export function mossBbox(box, width, height) {
  // Editor box -> [ymin, xmin, ymax, xmax]: whole-pixel edges enclosing the box, origin at the canvas centre.
  // null when the box has no area after rounding — MOSS consumers reject zero-size boxes.
  const [x1, y1, x2, y2] = edges(box);
  const hx = Math.floor(width / 2), hy = Math.floor(height / 2);
  const bb = [Math.floor(r9(y1 * height)) - hy, Math.floor(r9(x1 * width)) - hx,
              Math.ceil(r9(y2 * height)) - hy, Math.ceil(r9(x2 * width)) - hx];
  return (bb[0] < bb[2] && bb[1] < bb[3]) ? bb : null;
}

export function fromMossBbox(bb, width, height, space = MOSS_SPACE, fmt = MOSS_FORMAT) {
  // Layout bbox -> editor {x, y, w, h} fractions clamped into the canvas, or null for malformed input.
  if (!Array.isArray(bb) || bb.length !== 4) return null;
  const n = bb.map(Number);
  if (n.some((v) => !Number.isFinite(v))) return null;
  let [y1, x1, y2, x2] = String(fmt).toLowerCase() === "xyxy" ? [n[1], n[0], n[3], n[2]] : n;
  if (String(space || MOSS_SPACE).includes("central")) {
    y1 += Math.floor(height / 2); y2 += Math.floor(height / 2); x1 += Math.floor(width / 2); x2 += Math.floor(width / 2);
  }
  const c = (v, d) => Math.max(0, Math.min(1, v / d));
  [x1, x2] = [c(x1, width), c(x2, width)].sort((a, b) => a - b);
  [y1, y2] = [c(y1, height), c(y2, height)].sort((a, b) => a - b);
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

export function buildMoss(boxes, scene, width, height, { allowEmpty = false } = {}) {
  // Editor boxes + scene -> layout, or null when no item survives (the node then emits the caption as plain
  // text). allowEmpty keeps an items-less layout object for the editor's clipboard/templates.
  const items = [];
  for (const b of exportedBoxes(boxes)) {
    if (b.nobbox) continue;
    const bb = mossBbox(b, width, height);
    if (!bb) continue;
    const label = mossLabel(b.type);
    const item = { label, caption: b.desc || "" };
    if (label === "text") item.text = b.text || "";
    const tags = strs(b.tags), pal = upperPalette(b.palette);
    if (tags.length) item.tags = tags;
    if (pal.length) item.palette = pal.slice(0, 5);
    item.bbox = bb;
    items.push(item);
  }
  if (!items.length && !allowEmpty) return null;
  const summary = {};
  const meta = strs(scene.meta);
  if (meta.length) summary.meta = meta;
  if (CANVASES.includes(scene.canvas)) summary.canvas = scene.canvas;
  summary.caption = (scene.caption || "").trim();
  const pal = upperPalette(scene.palette);
  if (pal.length) summary.palette = pal.slice(0, 16);
  return { summary, items, bbox_coord_space: MOSS_SPACE, bbox_format: MOSS_FORMAT };
}

export function parseMoss(layout, width, height) {
  // Layout -> {boxes, scene}; bboxes are read against the given width/height (the layout carries no size).
  const space = layout.bbox_coord_space, fmt = layout.bbox_format;
  const boxes = [];
  (Array.isArray(layout.items) ? layout.items : []).forEach((it, i) => {
    if (!it || typeof it !== "object") return;
    const box = { type: mossLabel(it.label), text: String(it.text || ""), desc: String(it.caption || ""),
                  tags: strs(it.tags).join(", "), palette: strs(it.palette) };
    Object.assign(box, fromMossBbox(it.bbox, width, height, space || MOSS_SPACE, fmt || MOSS_FORMAT) || unplacedBox(i));
    boxes.push(box);
  });
  const s = (layout.summary && typeof layout.summary === "object") ? layout.summary : {};
  return { boxes, scene: mossScene({ caption: String(s.caption || ""), meta: s.meta, canvas: s.canvas, palette: strs(s.palette) }) };
}

// ── conversion ───────────────────────────────────────────────────────────────────────────────

export function ideogramSceneToMoss(scene) {
  // summary.caption = "<hld> <background> Style: <photo|art_style>, <aesthetics>, <lighting>, <medium>." — empties
  // dropped, single-space joined; the Style sentence only when a kind is chosen and a field is filled.
  const parts = [(scene.high_level_description || "").trim(), (scene.background || "").trim()];
  const kind = scene.style_kind || "none";
  if (kind === "photo" || kind === "art_style") {
    const fields = [scene[kind], scene.aesthetics, scene.lighting, scene.medium].map((f) => (f || "").trim()).filter(Boolean);
    if (fields.length) parts.push("Style: " + fields.join(", ") + ".");
  }
  return mossScene({ caption: parts.filter(Boolean).join(" "), palette: scene.palette || [] });
}

export function mossSceneToIdeogram(scene) {
  // The whole-scene caption is the closest fit for Ideogram's required background; meta/canvas have no home.
  return ideogramScene({ background: scene.caption || "", palette: scene.palette || [] });
}

export function boxesToLabels(boxes, target) {
  // Copies with `type` normalised to the target's label set; every other key (incl. tags) is kept.
  return boxes.map((b) => ({ ...b, type: target === "ideogram" ? (b.type === "text" ? "text" : "obj") : mossLabel(b.type) }));
}

export function importLayout(obj, target, width, height) {
  // Parse a layout of either format into {boxes, scene, converted} for the `target` builder ("ideogram" | "moss");
  // scene + labels are converted when the formats differ. Throws when the object is not a layout.
  const src = detectFormat(obj);
  if (!src || (target !== "ideogram" && target !== "moss")) throw new Error("not a layout JSON (needs 'items' or 'compositional_deconstruction')");
  const { boxes, scene } = src === "ideogram" ? parseIdeogram(obj) : parseMoss(obj, width, height);
  if (src === target) return { boxes, scene, converted: false };
  return { boxes: boxesToLabels(boxes, target), converted: true,
           scene: target === "moss" ? ideogramSceneToMoss(scene) : mossSceneToIdeogram(scene) };
}
