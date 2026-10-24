# Layout prompt formats — part of ComfyUI-IntoTheLatent-Utils.
#
# The Ideogram 4 caption logic here was extracted from nodes/ideogram4_nodes.py, which is derived from
# ComfyUI-KJNodes (nodes/ideogram4_nodes.py) by Kijai, licensed under GPL-3.0; this file and pack remain GPL-3.0.
# Upstream: https://github.com/kijai/ComfyUI-KJNodes
#
"""Comfy-free layout formats shared by the prompt-builder nodes.

The editor stores boxes as ``{x, y, w, h}`` fractions of the canvas (0-1) plus ``type``, ``text``, ``desc``,
``palette`` (and editor-only keys: ``id``, ``parent``, ``group``, ``disabled``, ``nobbox``, ``tags``). This module
turns that box list plus a small ``scene`` dict into a model's layout JSON and back. No comfy_api / torch / PIL
imports, so it is testable with the system Python.

Ideogram 4: ``{high_level_description?, style_description?, compositional_deconstruction: {background, elements}}``
with bbox on a 0-1000 grid as ``[ymin, xmin, ymax, xmax]``.
"""

import json
import logging
import re


# ── generic helpers ────────────────────────────────────────────────────────────────────────────

def compact(v):
    # The one-line form every model consumer reads (ensure_ascii=False keeps CJK text intact).
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def dumps(v, lvl=0):
    # Like json.dumps(ensure_ascii=False, indent=4), but scalar arrays stay on one line.
    pad, end = "    " * (lvl + 1), "    " * lvl
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return "[]"
        if all(not isinstance(x, (dict, list)) for x in v):
            return "[" + ", ".join(dumps(x, lvl) for x in v) + "]"
        return "[\n" + ",\n".join(pad + dumps(x, lvl + 1) for x in v) + "\n" + end + "]"
    if isinstance(v, dict):
        if not v:
            return "{}"
        items = [pad + json.dumps(k, ensure_ascii=False) + ": " + dumps(val, lvl + 1) for k, val in v.items()]
        return "{\n" + ",\n".join(items) + "\n" + end + "}"
    return json.dumps(v, ensure_ascii=False)


def parse_json_list(s):
    if s:
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    return []


def repair_json(s):
    # Slice out the outermost {...} (drops ``` fences / prose), then strip trailing commas before
    # } or ] — the leading "(...)" alt matches whole strings first, so quoted commas are untouched.
    i, j = s.find("{"), s.rfind("}")
    t = s[i:j + 1] if (i != -1 and j > i) else s
    return re.sub(r'("(?:[^"\\]|\\.)*")|,(\s*[}\]])', lambda m: m.group(1) or m.group(2), t)


def loads_layout(s):
    # Parse a layout dict; on failure retry once with the lenient repair. Returns dict or None.
    for cand in ((s, repair_json(s)) if s and s.strip() else ()):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                if cand is not s:
                    logging.warning("[ITLLayout] import_json had errors; recovered with lenient parse")
                return v
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def palette(colors):
    # ["#rrggbb", ...] (or autogrow dict) -> ["#RRGGBB", ...] in order, dropping empties.
    if isinstance(colors, dict):
        colors = colors.values()
    return [c.upper() for c in colors if c]


def by_id(boxes):
    return {b.get("id"): b for b in boxes if isinstance(b, dict) and b.get("id")}


def eff_disabled(box, ids):
    # True if the box or any ancestor has "disabled" — disabling a group disables its members.
    cur, seen = box, 0
    while isinstance(cur, dict) and seen < 1000:
        if cur.get("disabled"):
            return True
        parent = cur.get("parent")
        if parent is None:
            return False
        cur = ids.get(parent)
        seen += 1
    return False


def exported_boxes(boxes):
    # The boxes that reach a prompt: dicts that are neither groups nor (effectively) disabled.
    ids = by_id(boxes)
    return [b for b in boxes if isinstance(b, dict) and not b.get("group") and not eff_disabled(b, ids)]


def seed_boxes(bboxes, width, height):
    # A wired BoundingBox input ({x, y, width, height} pixels: one dict, a flat list, or per-frame nesting)
    # -> editor boxes. Used to seed the editor when it has no regions.
    if not bboxes:
        return []
    if isinstance(bboxes, dict):                             # a single BoundingBox is a bare {x,y,width,height} dict
        frame = [bboxes]
    elif isinstance(bboxes[0], (list, tuple)):
        frame = bboxes[0]                                    # per-frame nesting: [[box, ...], ...]
    else:
        frame = bboxes                                       # flat list of boxes
    boxes = []
    for bb in frame:
        if not isinstance(bb, dict):
            continue
        boxes.append({"x": bb.get("x", 0) / width, "y": bb.get("y", 0) / height,
                      "w": bb.get("width", 0) / width, "h": bb.get("height", 0) / height,
                      "type": "obj", "text": "", "desc": "", "palette": []})
    return boxes


def bbox_outputs(boxes, width, height):
    # Pixel-space bboxes ({x, y, width, height}) for SAM3 / BoundingBox consumers, nested per frame
    # (list[list[dict]] — the canonical BoundingBox shape: bboxes[frame] -> list of boxes). [] when empty.
    out = []
    for box in exported_boxes(boxes):
        if box.get("nobbox"):
            continue
        x, y = box.get("x", 0.0), box.get("y", 0.0)
        bw, bh = box.get("w", 0.0), box.get("h", 0.0)
        if bw < 0:
            x += bw
            bw = -bw
        if bh < 0:
            y += bh
            bh = -bh
        out.append({"x": round(x * width), "y": round(y * height),
                    "width": round(bw * width), "height": round(bh * height)})
    return [out] if out else []


# ── Ideogram 4 ─────────────────────────────────────────────────────────────────────────────────

def norm_bbox(box, sx=1000, sy=1000, order="yx"):
    # {x, y, w, h} fractions (0-1) -> bbox; x scaled by sx, y by sy.
    # Default sx=sy=1000 is the Ideogram-standard 0-1000 grid; pass (width, height) for absolute pixels.
    # order="yx" -> [ymin, xmin, ymax, xmax] (Ideogram); "xy" -> [xmin, ymin, xmax, ymax] (Qwen/standard).
    def cx(v):
        return max(0, min(sx, round(v * sx)))

    def cy(v):
        return max(0, min(sy, round(v * sy)))
    x, y, w, h = box.get("x", 0.0), box.get("y", 0.0), box.get("w", 0.0), box.get("h", 0.0)
    ymin, xmin, ymax, xmax = cy(y), cx(x), cy(y + h), cx(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [xmin, ymin, xmax, ymax] if order == "xy" else [ymin, xmin, ymax, xmax]


def ideogram_scene(high_level_description="", background="", style_kind="none", photo="", art_style="",
                   aesthetics="", lighting="", medium="", palette=None):
    # The scene dict build_ideogram consumes (all the non-box fields of a caption).
    return {"high_level_description": high_level_description, "background": background,
            "style_kind": style_kind if style_kind in ("photo", "art_style") else "none",
            "photo": photo, "art_style": art_style, "aesthetics": aesthetics, "lighting": lighting,
            "medium": medium, "palette": list(palette or [])}


def build_ideogram(boxes, scene, *, coord_mode="normalized", bbox_order="yx", width=1024, height=1024):
    # bbox scale: 0-1000 grid (Ideogram-standard) or absolute pixels (width/height) when coord_mode=="absolute".
    # bbox axis order: "yx" (Ideogram) or "xy" (Qwen/standard x1,y1,x2,y2).
    bsx, bsy = (width, height) if coord_mode == "absolute" else (1000, 1000)
    border = "xy" if bbox_order == "xy" else "yx"
    caption = {}
    if scene.get("high_level_description", "").strip():
        caption["high_level_description"] = scene["high_level_description"]

    kind = scene.get("style_kind", "none")
    if kind != "none":
        # The verifier requires every style key present (in order) once a style is
        # chosen; only color_palette is conditional. Emit blanks rather than omit.
        sd = {"aesthetics": scene.get("aesthetics", ""), "lighting": scene.get("lighting", "")}
        # photo: ...photo, medium...  |  art_style: ...medium, art_style...  (key order)
        if kind == "photo":
            sd["photo"] = scene.get("photo", "")
            sd["medium"] = scene.get("medium", "")
        else:
            sd["medium"] = scene.get("medium", "")
            sd["art_style"] = scene.get("art_style", "")
        pal = palette(scene.get("palette") or [])
        if pal:
            sd["color_palette"] = pal
        caption["style_description"] = sd

    elements = []
    for box in exported_boxes(boxes):
        etype = "text" if box.get("type") == "text" else "obj"
        elem = {"type": etype}                               # key order matters
        if not box.get("nobbox"):                            # unplaced elements omit bbox
            elem["bbox"] = norm_bbox(box, bsx, bsy, border)
        if etype == "text":
            elem["text"] = box.get("text", "")
        elem["desc"] = box.get("desc", "")
        pal = palette(box.get("palette", []))
        if pal:
            elem["color_palette"] = pal[:5]
        elements.append(elem)

    caption["compositional_deconstruction"] = {
        "background": scene.get("background", ""),
        "elements": elements,
    }
    return caption


def unplaced_box(idx):
    # Placeholder for an element without a bbox: small, staggered, flagged so export omits bbox.
    k = (idx or 0) % 6
    return {"x": 0.03 + k * 0.035, "y": 0.03 + k * 0.035, "w": 0.22, "h": 0.14, "nobbox": True}


def parse_ideogram(cap):
    # Caption dict -> (editor boxes, scene). Boxes: {x,y,w,h, type, text, desc, palette}; bbox assumed on the
    # 0-1000 grid in [ymin, xmin, ymax, xmax] order (the Ideogram defaults — the JSON carries no marker).
    cd = cap.get("compositional_deconstruction") or {}
    boxes = []
    for i, el in enumerate(cd.get("elements") or []):
        if not isinstance(el, dict):
            continue
        box = {"type": "text" if el.get("type") == "text" else "obj",
               "text": el.get("text", "") or "", "desc": el.get("desc", "") or "",
               "palette": list(el.get("color_palette") or [])}
        bb = el.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            ymin, xmin, ymax, xmax = bb
            box.update(x=xmin / 1000.0, y=ymin / 1000.0,
                       w=(xmax - xmin) / 1000.0, h=(ymax - ymin) / 1000.0)
        else:
            box.update(unplaced_box(i))
        boxes.append(box)
    sd = cap.get("style_description") or {}
    if not isinstance(sd, dict):
        sd = {}
    kind = "photo" if isinstance(sd.get("photo"), str) else "art_style" if isinstance(sd.get("art_style"), str) else "none"
    scene = ideogram_scene(high_level_description=cap.get("high_level_description") or "",
                           background=cd.get("background") or "", style_kind=kind,
                           photo=sd.get("photo") or "", art_style=sd.get("art_style") or "",
                           aesthetics=sd.get("aesthetics") or "", lighting=sd.get("lighting") or "",
                           medium=sd.get("medium") or "", palette=sd.get("color_palette") or [])
    return boxes, scene
