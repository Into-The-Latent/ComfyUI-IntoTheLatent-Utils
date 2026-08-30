# Comfy-free resolution math for the ITL Resolution Selector — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
"""Pure resolution math, import-clean (no comfy_api) so it is unit-testable from the repo root.

The node class in resolution_selector.py imports from here, and the editor JS
(web/js/resolution_selector.js) mirrors this math so the readout and the INT outputs always agree.
"""
import re
from math import log

# A profile whose `max` == BIG has no real per-side cap -> never shows the "clamped" warning.
BIG = 16384

# Per-profile rules. `mult` of None means "use the node's snap_multiple widget" (the `default`
# profile). Keep in sync with PROFILES in web/js/resolution_selector.js.
PROFILES = {
    "default":    {"mult": None, "min": 1,   "max": BIG},
    "Ideogram 4": {"mult": 16,   "min": 256, "max": 2048},
}
DEFAULT_PROFILE = "default"

# (ratio, name) — square + landscape only (W >= H). Portrait counterparts come from the flip
# button / `orientation`, so 1:1 is listed exactly once. Keep in sync with the JS ASPECT_PRESETS.
ASPECT_PRESETS = [
    ("1:1", "Square"),
    ("5:4", "Large Format"),
    ("4:3", "Standard"),
    ("3:2", "Photo"),
    ("16:10", "Monitor"),
    ("16:9", "Widescreen"),
    ("2:1", "Panorama"),
    ("21:9", "Cinemascope"),
    ("3:1", "Wide Panorama"),
]

_AR_RE = re.compile(r"\s*(\d+)\s*:\s*(\d+)")


def aspect_label(ratio, name):
    return f"{ratio} ({name})"


def aspect_options():
    return [aspect_label(r, n) for r, n in ASPECT_PRESETS]


def parse_ar(s):
    # Leading "W:H" from a bare ratio ("16:9") or a label ("16:9 (Widescreen)") -> width/height
    # float. 1.0 on anything malformed. Tolerates old saved values (bare ratios).
    m = _AR_RE.match(str(s))
    if not m:
        return 1.0
    a, b = float(m.group(1)), float(m.group(2))
    return a / b if b else 1.0


def effective_ar(aspect, orientation):
    # Presets are always landscape/square (W >= H); portrait inverts to H:W.
    ar = parse_ar(aspect)
    return (1.0 / ar) if (orientation == "portrait" and ar) else ar


def _prof(name):
    return PROFILES.get(name, PROFILES[DEFAULT_PROFILE])


def profile_clamps(name):
    # True when the profile has a real per-side cap (shows the "clamped to keep aspect" warning).
    return _prof(name)["max"] < BIG


# Widget max for snap_multiple; keep in sync with the io.Int.Input in resolution_selector.py and the
# clampSnap helper in web/js/resolution_selector.js.
SNAP_MAX = 1024


def clamp_snap_multiple(v):
    """Coerce the snap_multiple widget value to a usable multiple in [1, SNAP_MAX].

    Anything non-numeric or < 1 heals to the default 8. This matters because a ComfyUI number widget
    can serialize an empty string (an emptied/uninitialised spinner), and `int('')` then fails
    ComfyUI's INT validation and rejects the WHOLE node — even for a profile like Ideogram 4 that
    ignores snap_multiple. Healing to a valid multiple here (and, on the front-end, at serialization)
    keeps a stray value from ever breaking a profile's output or blocking a queue. Model profiles
    ignore snap_multiple entirely (see _rules). Mirrors clampSnap in web/js/resolution_selector.js."""
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        return 8
    if n < 1:
        return 8
    return min(SNAP_MAX, n)


def _rules(name, snap_multiple):
    # Effective {mult, min, max}. `default` reads its multiple from the widget; models use their own.
    p = _prof(name)
    mult = p["mult"] if p["mult"] else clamp_snap_multiple(snap_multiple)
    return {"mult": mult, "min": p["min"], "max": p["max"]}


def _snap(v, p):
    m = p["mult"]
    return int(min(p["max"], max(p["min"], round((float(v) if v else 0.0) / m) * m)))


def _fit_w(tw, ar, p):
    # Largest width with aspect `ar` whose width AND height both fit [min, max], aspect preserved —
    # so hitting the per-side cap keeps the ratio (16:9 -> 2048x1152, not 2048x2048).
    lo, hi = p["min"], p["max"]
    wlo, whi = max(lo, lo * ar), min(hi, hi * ar)
    w = min(hi, max(lo, tw)) if wlo > whi else min(whi, max(wlo, tw))
    w = _snap(w, p)
    return w, _snap(w / ar if ar else w, p)


def detect_ar(width, height):
    """Aspect ratio of the raw width/height inputs, exactly as given — never snapped to a preset.

    Returns 1.0 (square) for anything unusable: a zero/negative side, a non-numeric widget value,
    None. The fallback is deliberately square rather than the `aspect_ratio` widget — that widget is
    hidden in `input` mode, and a stale hidden value silently steering the output is precisely the
    failure this mode exists to prevent. Mirrors detectAR in web/js/resolution_selector.js."""
    try:
        w, h = float(width), float(height)
    except (TypeError, ValueError):
        return 1.0
    return (w / h) if (w > 0 and h > 0) else 1.0


def nearest_preset(ar):
    """Closest ASPECT_PRESETS entry to `ar` -> (ratio, name, orientation). Display only.

    Distances are compared in log space, so a ratio is judged proportionally: 2.66 reads as nearer
    3:1 than 21:9 (2.333) even though absolute distance says otherwise, which stops the wide end of
    the list from swallowing its neighbours just because the numbers there are bigger. Presets are
    landscape/square only, so a ratio below 1.0 is inverted and reported as portrait. This NEVER
    feeds resolve_dims — `input` mode keeps the exact detected ratio. Mirrors nearestPreset in
    web/js/resolution_selector.js."""
    try:
        v = float(ar)
    except (TypeError, ValueError):
        v = 1.0
    if v <= 0:
        v = 1.0
    orientation = "landscape"
    if v < 1.0:
        v, orientation = 1.0 / v, "portrait"
    ratio, name = min(ASPECT_PRESETS, key=lambda p: abs(log(v / parse_ar(p[0]))))
    return ratio, name, orientation


def resolve_dims(profile, mode, aspect, orientation, snap_multiple, mp, width, height):
    # Mirror of the editor JS math so the INT outputs are correct even headless / via the API.
    p = _rules(profile, snap_multiple)
    if mode == "raw":
        return _snap(width, p), _snap(height, p)
    # `input`: the ratio comes from the width/height inputs themselves (normally connected), so the
    # aspect_ratio and orientation widgets are ignored — a portrait source simply gives ar < 1, and
    # _fit_w already handles that. Defined without reference to whether the values arrived over a
    # link, so the mode resolves identically headless and via the API.
    ar = detect_ar(width, height) if mode == "input" else effective_ar(aspect, orientation)
    if mode in ("megapixel", "input"):        # sized by the megapixel target
        tw = (max(0.0, float(mp)) * 1_000_000.0 * ar) ** 0.5
    else:                                    # auto: width drives (JS keeps both sides consistent)
        tw = float(width)
    return _fit_w(tw, ar, p)
