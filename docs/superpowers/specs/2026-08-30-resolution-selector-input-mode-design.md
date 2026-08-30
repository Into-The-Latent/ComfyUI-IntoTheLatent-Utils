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

**Two hardening changes the review pulled in, both of which `input` mode is what makes reachable:**

- `clamp_megapixels` — `float(mp)` was unguarded, so an emptied `megapixels` widget raised
  `ValueError` and rejected the whole node: the same failure class `clamp_snap_multiple` exists to
  prevent, now on the mode the node auto-selects. Heals to the 1.0 default, clamps to the widget's
  `[0.1, 16.0]`, and gets the matching `serializeValue` guard on the front-end.
- A **floor** in `_rules`: `max(min, mult)` rather than the profile's bare `min`. With `min == 1` on
  the `default` profile, an extreme detected ratio drove `w / ar` to round to 0 and `_snap` returned
  **1** — not a multiple of anything, breaking the `snap_multiple` guarantee (512:1 source at
  `snap_multiple` 64 gave height 1). The presets stop at 3:1 so this was unreachable before; an
  arbitrary upstream ratio reaches `_fit_w` now. A model floor larger than its multiple still wins.

`exceeds_cap(...)` reports whether the ideal size overflowed the profile cap. It shares `_target()`
with `resolve_dims`, so "did we clamp?" cannot drift from the dimensions themselves. It exists
because the editor computes that warning from a source it cannot see in linked `input` mode — without
it a clamped result displayed the cut-down dimensions and said nothing.

## 4. Node schema (`nodes/resolution_selector.py`)

- `resolution_mode` options become `["raw", "auto", "megapixel", "input"]`. Default stays
  `megapixel`.
- **No new inputs.** `width`/`height` keep their declarations; their tooltips gain: in `input` mode
  they are the *source* dimensions, normally connected.
- `execute` returns run-time feedback alongside the INTs, mirroring `ideogram4_nodes.py`:

  ```python
  return io.NodeOutput(w, h, ui=ui_payload(w, h, width, height, exceeds_cap(*args)))
  #  -> {"dims": [w, h], "src": [src_w, src_h], "clamped": [bool]}
  ```

  `dims` is the resolved output; `src` is what the auto-detect readout reflects; `clamped` drives the
  warning line. The front-end derives the ratio and the nearest preset from `src` with its mirrored
  helper, keeping the payload minimal and the display logic on one side.

  `ui_payload` lives in `resolution_core`, not in the node module, for one reason: the node module
  cannot be imported without ComfyUI, so a test of it would be permanently skipped in this repo. The
  front-end is the payload's only other consumer, so a silent rename would break the readout with
  nothing failing anywhere. In the core it is pinned by tests that actually run. Every field is
  coerced — this is cosmetic, and must never fail `execute()` after `resolve_dims` has succeeded
  (`int(float('inf'))` raises `OverflowError`, which is not a `ValueError`).
- The node `description` and the `resolution_mode` tooltip document the new mode.

Known limitation: an unchanged node is served from cache and does not re-execute, so no `executed`
event fires and the readout keeps its previous values.

## 5. Front-end (`web/js/resolution_selector.js`)

Mirrors the core math as always (`detectAR`, `nearestPreset` alongside `parseAR` / `effAR`).

**Auto-switch.** A helper reports whether the inputs named `width` and `height` both have
`link != null`. The already-chained `onConnectionsChange` then, **on input-side changes only**
(`type === 1`; wiring an *output* into a Prompt Builder must never re-impose a mode):

- rising edge — both became linked, mode is not `input` → stash the current mode on
  `node._resPrevMode`, set `resolution_mode` to `input`;
- falling edge — both were linked and no longer are, mode is `input` **and `_resPrevMode` is set** →
  hand the stashed mode back.

Edge-triggering and that `_resPrevMode` gate are both load-bearing, because manual `input` (typed
width/height, no links) is a first-class state:

- level-triggering would re-impose `input` on every later connection event, undoing a mode the user
  deliberately picked while the sockets stayed linked;
- restoring without the gate would rewrite a hand-picked — or freshly loaded — `input` to
  `megapixel` the moment anything touched the connections, silently swapping a 1512×1080 source's
  detected 1.400 for the hidden `aspect_ratio` widget's 1:1.

Choosing a mode by hand clears `_resPrevMode`, so the user's choice outlives a later disconnect. At
load `node._resApply` only *records* the link state and never switches: a saved workflow's mode
belongs to the user, and rewriting it would change what an existing graph outputs.

**Visibility** in `input` mode: show `megapixels`, `snap_multiple` (default profile only) and
width/height; hide `aspect_ratio` and the flip button — the ratio comes from the source and carries
its own orientation. Plus the change from section 2: width/height are no longer hidden in
`megapixel` mode.

That visibility change has two consequences of its own, both handled rather than left implicit:

- In `megapixel` mode the two widgets are *outputs* of the computation — `recalcDims` overwrites
  them — so an edit there could never stick. They are marked `disabled` in that mode and their
  callbacks skip the recalc, instead of silently bouncing back what the user typed.
- The sockets can now be wired in a mode that never reads them. `megapixel` computes both sides from
  `megapixels × ratio`, so a link into either side is dropped; the readout says so explicitly
  ("width/height inputs are ignored in megapixel mode — connect both for 'input' mode") rather than
  ignoring the connection in silence.

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
canvas. A new `onExecuted` handler stores `dims` / `src` / `clamped`, refreshes the readout and
pushes the real dimensions.

**Invalidating the last run.** `_resLast` is cleared whenever `recalcDims` runs in linked `input`
mode. Every path into `recalcDims` is a real change — `megapixels`, `profile`, `snap_multiple`, a
rewire — and each one makes the last run's dimensions answer a question nobody asked any more.
Without this, dragging `megapixels` from 2.0 to 4.0 kept printing the 2 MP result *and* pushed those
stale numbers into the connected canvas: exactly the outcome `outDims()`'s null case exists to
prevent.

Note the two link tests differ on purpose: **both** sides linked selects `input` mode, but **either**
side linked suppresses the live preview. One linked side is enough for the editor's numbers to
diverge from what the backend will see (the link value wins at queue time), so it shows "resolves at
run time" rather than a preview it cannot stand behind.

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

From the review, also in the core tests: `clamp_megapixels` healing and range; `resolve_dims`
surviving an emptied `megapixels` widget; no side ever snapping below one multiple (with the model
floor still winning); `exceeds_cap` agreeing with what `resolve_dims` actually did, as a property
across a grid of targets and sources; and the `ui_payload` shape plus its coercion of stray values.

## 7. Docs

README's **ITL Resolution Selector** section: add `input` to the Modes bullet, and note in the
readout bullet that connecting width/height switches the node into `input` mode and reports the
detected ratio after a run.
