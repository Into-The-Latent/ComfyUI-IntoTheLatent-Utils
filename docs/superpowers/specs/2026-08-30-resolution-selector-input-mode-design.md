# Resolution Selector — `input` mode (width/height driven, aspect auto-detected)

**Date:** 2026-08-30
**Node:** `ITLResolutionSelector` (`nodes/resolution_core.py`, `nodes/resolution_selector.py`, `web/js/resolution_selector.js`)

## Problem

When an image already exists upstream, its aspect ratio is the thing worth preserving — but the
node has no way to learn it. Today the ratio can only come from the `aspect_ratio` dropdown, a list
of nine named presets. A 1512×1080 source (1.400) has to be forced into 4:3 (1.333) or 3:2 (1.500);
either choice crops or letterboxes downstream.

What's missing is: *take the ratio from the incoming width/height, keep it exactly, and let me
choose the size in megapixels* — with `snap_multiple` and the profile rules still honoured.

## Goals

1. A new **`input`** resolution mode: aspect ratio = `width / height` **exactly**, size from the
   `megapixels` target, snapped/clamped by the profile and `snap_multiple`.
2. Drive it by **connecting the existing `width`/`height` widget sockets** — no new node inputs.
3. **Auto-switch** to `input` when both sockets are linked; revert when either is unlinked.
4. An **auto-detect readout** showing the ratio that was detected and the nearest named preset.
5. Existing modes (`raw`, `auto`, `megapixel`) keep their current behaviour bit-for-bit.

## Non-goals

- No new `IMAGE`/`LATENT` input that derives dimensions from a tensor. Width/height INTs only.
- No "match the source megapixels" option — the size always comes from the `megapixels` widget.
- No change to the profile table, the aspect preset list, or the flip/orientation mechanism.

---

## 1. Why a new mode instead of repurposing `auto`

`auto` today means *pick a ratio, type one side, the other follows*; `megapixels` is hidden and
ignored. Full-auto means *ratio comes from w/h, megapixels drives the size*. Same saved widget
values, different output: a workflow saved as `auto` / 16:9 / width 1920 yields 1920×1080 today and
would yield 1333×750 (mp default 1.0) after such a redefinition. So `input` is added as a fourth
mode value and `auto` is left alone.

Critically, `input` is a **real mode, not a front-end state**. Its definition makes no reference to
whether width/height arrived over a link, so it resolves identically headless and over the API. The
front-end merely *selects* it for you; the auto-switch is a convenience, not load-bearing behaviour.

## 2. Why the existing sockets, and what that forces

Reusing the `width`/`height` widget sockets is how every other Comfy node exposes this, and it
degrades gracefully: with only *one* side linked the node stays in its current mode, where a linked
width plus a chosen ratio in `auto` already means "height follows".

It forces one change: `applyVisibility()` currently hides the width/height widgets in `megapixel`
mode, which is the node's **default** mode. A hidden widget row takes its input dot with it, so out
of the box there would be nothing to plug into. **Width/height therefore stay visible in every
mode**; in `megapixel` mode they display the computed result, the way `EmptyLatentImage` and friends
always show them.

## 3. Core math (`nodes/resolution_core.py`)

`resolve_dims` splits "where does the ratio come from" away from "how is the target sized", so
`megapixel` and `input` share one sizing path:

```python
if mode == "raw":
    return _snap(width, p), _snap(height, p)
ar = detect_ar(width, height) if mode == "input" else effective_ar(aspect, orientation)
if mode in ("megapixel", "input"):
    tw = (max(0.0, float(mp)) * 1_000_000.0 * ar) ** 0.5
else:                                       # auto: width drives
    tw = float(width)
return _fit_w(tw, ar, p)
```

`_fit_w` is unchanged, so the profile cap still preserves the ratio (a 16:9-ish source under
Ideogram 4 gives 2048×1152, not 2048×2048) and the existing clamp warning still applies.

**New helpers:**

- `detect_ar(width, height) -> float` — `w / h`, returning **1.0** when either side is `<= 0`,
  non-numeric, or `None`. The fallback is deliberately *square*, not the `aspect_ratio` widget:
  `aspect_ratio` is hidden in `input` mode, and a stale hidden widget silently steering the output
  is exactly the failure this mode exists to avoid.
- `nearest_preset(ar) -> (ratio, name, orientation)` — nearest entry in `ASPECT_PRESETS`, compared
  in **log space** (`abs(log(ar / preset_ar))`) so 3:1 and 1:1 are judged on equal footing rather
  than by absolute distance. Ratios below 1.0 are inverted first and reported as `"portrait"`.
  **Display only** — it never influences the computed dimensions.

`input` ignores `aspect` and `orientation` entirely; a portrait source simply produces `ar < 1`,
which `_fit_w` already handles.

## 4. Node schema (`nodes/resolution_selector.py`)

- `resolution_mode` options become `["raw", "auto", "megapixel", "input"]`. Default stays
  `megapixel`.
- **No new inputs.** `width`/`height` keep their declarations; their tooltips gain: in `input` mode
  they are the *source* dimensions, normally connected.
- `execute` returns run-time feedback alongside the INTs, mirroring `ideogram4_nodes.py`:

  ```python
  return io.NodeOutput(w, h, ui={"dims": [w, h], "src": [src_w, src_h]})
  ```

  `dims` is the resolved output; `src` is what the auto-detect readout reflects. The front-end
  derives the ratio and the nearest preset from `src` with its mirrored helper, keeping the payload
  minimal and the display logic on one side.
- The node `description` and the `resolution_mode` tooltip document the new mode.

Known limitation: an unchanged node is served from cache and does not re-execute, so no `executed`
event fires and the readout keeps its previous values.

## 5. Front-end (`web/js/resolution_selector.js`)

Mirrors the core math as always (`detectAR`, `nearestPreset` alongside `parseAR` / `effAR`).

**Auto-switch.** A helper reports whether the inputs named `width` and `height` both have
`link != null`. The already-chained `onConnectionsChange` then:

- both linked, mode is not `input` → stash the current mode on `node._resPrevMode`, set
  `resolution_mode` to `input`;
- not both linked, mode is `input` → restore `node._resPrevMode`, falling back to `megapixel`.

`node._resApply` calls the same helper, so a saved workflow whose links are restored on load lands
in `input` mode without the user touching anything.

**Visibility** in `input` mode: show `megapixels`, `snap_multiple` (default profile only) and
width/height; hide `aspect_ratio` and the flip button — the ratio comes from the source and carries
its own orientation. Plus the change from section 2: width/height are no longer hidden in
`megapixel` mode.

**In `input` mode the JS never writes computed dimensions into the width/height widgets.** There
they are *source* fields, not destinations. Overwriting a typed 1512×1080 with 1352×966 on every
keystroke would fight the user, and writing to a linked widget is meaningless anyway since the link
value wins at queue time. Keeping both cases identical also keeps `recalcDims` simple. The result
appears in the readout and on the output slots.

**Readout** in `input` mode:

| State | Line |
|---|---|
| Manual (no links) | `1352 × 966    1.31 MP    ratio 1.400 (≈ 4:3 Standard)` — fully live, the source dims are known |
| Linked, not yet run | `→ resolves at run time    target 1.00 MP    ratio auto-detected from inputs` |
| Linked, after a run | `1888 × 1064    2.01 MP    ratio 1.774 (≈ 16:9 Widescreen)` from the `ui` payload |

**Pushing to connected nodes.** While linked and un-run the JS has no real dimensions, so
`pushToTargets` is skipped rather than pushing stale widget values into a connected Prompt Builder
canvas. A new `onExecuted` handler stores `dims` / `src`, refreshes the readout and pushes the real
dimensions.

## 6. Testing

`tests/test_resolution_core.py` gains, all against the comfy-free core:

- exact ratio preserved at a megapixel target (1512×1080 source → output ratio ≈ 1.400, area ≈ the
  target, both sides multiples of 8);
- `input` ignores `aspect_ratio` and `orientation` (passing `16:9` / `portrait` changes nothing);
- `snap_multiple` honoured (mult 64 → both sides multiples of 64);
- Ideogram 4 clamps to 2048 while holding the detected ratio;
- degenerate sources (0, negative, `None`, `""`) fall back to square;
- `detect_ar` and `nearest_preset` unit cases, including portrait inversion and the log-space
  choice;
- a property test that `input` from a 1920×1080 source equals `megapixel` at `16:9 (Widescreen)`.

## 7. Docs

README's **ITL Resolution Selector** section: add `input` to the Modes bullet, and note in the
readout bullet that connecting width/height switches the node into `input` mode and reports the
detected ratio after a run.
