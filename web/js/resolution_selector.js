/*
 * Part of ComfyUI-IntoTheLatent-Utils.
 *
 * Front-end for the ITL Resolution Selector node. GPL-3.0, like the rest of the pack.
 *
 * Four modes (raw / auto / megapixel / input) compute a profile-valid width/height. The math mirrors
 * nodes/resolution_core.py so the readout/UI and the INT outputs always agree. A landscape-only
 * aspect list + a "⟷" flip button (orientation) cover both orientations without duplicate entries.
 * Edits auto-push the dims into any connected node's width/height widgets and fire their callbacks —
 * which, for the ITL Ideogram 4 Prompt Builder, refreshes its editor canvas.
 *
 * `input` mode keeps the exact aspect ratio of the width/height INPUTS at a chosen megapixel count.
 * Connecting both sockets selects it automatically. Their values then live upstream, so the editor
 * cannot know them before the graph runs: the node reports them back via ui/onExecuted instead.
 */
import { chainCallback } from "./utility.js";
const { app } = window.comfyAPI.app;

// A profile whose max == BIG has no real cap -> never shows the "clamped" warning.
const BIG = 16384;
// Keep in sync with PROFILES in nodes/resolution_core.py. mult null = use the snap_multiple widget.
const PROFILES = {
  "default":    { mult: null, min: 1,   max: BIG },
  "Ideogram 4": { mult: 16,   min: 256, max: 2048 },
};
const DEFAULT_PROFILE = "default";

// (ratio, name) — square + landscape only. Keep in sync with ASPECT_PRESETS in resolution_core.py.
const ASPECT_PRESETS = [
  ["1:1", "Square"], ["5:4", "Large Format"], ["4:3", "Standard"], ["3:2", "Photo"],
  ["16:10", "Monitor"], ["16:9", "Widescreen"], ["2:1", "Panorama"], ["21:9", "Cinemascope"],
  ["3:1", "Wide Panorama"],
];
const aspectLabel = (r, n) => `${r} (${n})`;

const parseAR = (s) => { const m = /^\s*(\d+)\s*:\s*(\d+)/.exec(String(s || "")); return m && +m[2] ? (+m[1]) / (+m[2]) : 1; };
const effAR = (aspect, orient) => { const ar = parseAR(aspect); return (orient === "portrait" && ar) ? 1 / ar : ar; };
const profClamps = (name) => (PROFILES[name] || PROFILES[DEFAULT_PROFILE]).max < BIG;
// Coerce the snap_multiple widget value to a usable multiple in [1,1024]; anything non-numeric or
// < 1 (an emptied number widget serializes "", 0, null, a negative) heals to the default 8. ComfyUI
// coerces the widget to INT at queue time and int("") rejects the whole node — even for a profile
// that ignores snap_multiple. Mirrors clamp_snap_multiple in nodes/resolution_core.py exactly.
const clampSnap = (v) => { const n = Math.trunc(Number(v)); return (Number.isFinite(n) && n >= 1) ? Math.min(1024, n) : 8; };
function effRules(name, snapMult) {
  const p = PROFILES[name] || PROFILES[DEFAULT_PROFILE];
  const mult = p.mult ? p.mult : clampSnap(snapMult);
  return { mult, min: p.min, max: p.max };
}
const snap = (v, p) => Math.min(p.max, Math.max(p.min, Math.round((Number(v) || 0) / p.mult) * p.mult));
function fitW(tw, ar, p) {
  const wlo = Math.max(p.min, p.min * ar), whi = Math.min(p.max, p.max * ar);
  const w = snap(wlo > whi ? Math.min(p.max, Math.max(p.min, tw)) : Math.min(whi, Math.max(wlo, tw)), p);
  return [w, snap(ar ? w / ar : w, p)];
}
// Ratio of the width/height INPUTS exactly as given — never snapped to a preset. 1 (square) on
// anything unusable. Deliberately not the aspect_ratio widget, which is hidden in input mode and
// would steer the output from a value the user cannot see. Mirrors detect_ar in resolution_core.py.
const detectAR = (w, h) => {
  const a = Number(w), b = Number(h);
  return (Number.isFinite(a) && Number.isFinite(b) && a > 0 && b > 0) ? a / b : 1;
};
// Closest preset to `ar` -> [ratio, name, orientation], compared in LOG space so a ratio is judged
// proportionally (2.66 reads as nearer 3:1 than 21:9, though absolute distance says otherwise).
// Sub-1.0 ratios invert and report portrait. Display only — the dims keep the exact detected ratio.
// Mirrors nearest_preset in nodes/resolution_core.py.
function nearestPreset(ar) {
  let v = Number(ar);
  if (!Number.isFinite(v) || v <= 0) v = 1;
  let orientation = "landscape";
  if (v < 1) { v = 1 / v; orientation = "portrait"; }
  let best = ASPECT_PRESETS[0], bestD = Infinity;
  for (const p of ASPECT_PRESETS) {
    const d = Math.abs(Math.log(v / parseAR(p[0])));
    if (d < bestD) { bestD = d; best = p; }
  }
  return [best[0], best[1], orientation];
}
const flipRatio = (r) => { const m = /(\d+)\s*:\s*(\d+)/.exec(String(r)); return m ? `${m[2]}:${m[1]}` : r; };

app.registerExtension({
  name: "ITL.ResolutionSelector",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "ITLResolutionSelector") return;

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const findW = (n) => node.widgets?.find((w) => w.name === n);
      const profileWidget = findW("profile");
      const snapWidget = findW("snap_multiple");
      const modeWidget = findW("resolution_mode");
      const arWidget = findW("aspect_ratio");
      const orientWidget = findW("orientation");
      const mpWidget = findW("megapixels");
      const wWidget = findW("width");
      const hWidget = findW("height");

      const resMode = () => (modeWidget ? modeWidget.value : "megapixel");
      const profName = () => profileWidget?.value || DEFAULT_PROFILE;
      const orient = () => (orientWidget?.value === "portrait" ? "portrait" : "landscape");
      const currentDims = () => ({ w: parseInt(wWidget?.value, 10) || 0, h: parseInt(hWidget?.value, 10) || 0 });

      // ── input mode: is this node being driven by connected width/height sockets? ──
      const inputLinked = (name) => {
        const inp = node.inputs?.find((i) => i.name === name);
        return !!inp && inp.link != null;
      };
      const bothLinked = () => inputLinked("width") && inputLinked("height");   // -> auto-select input mode
      // Even ONE linked side means the editor cannot know what the backend will see (the link value
      // wins at queue time), so there is nothing honest to preview or push until the graph runs.
      const anyLinked = () => inputLinked("width") || inputLinked("height");

      // The node's current OUTPUT dims. In every mode but `input` those are simply the width/height
      // widgets. In `input` the widgets hold the *source* instead, so the result is computed aside
      // into _resIn — and while the sockets are linked only a run can know it (_resLast, filled by
      // onExecuted). null means "linked but never run": nothing truthful to show or push yet.
      function outDims() {
        if (resMode() !== "input") return currentDims();
        if (node._resIn) return { w: node._resIn.w, h: node._resIn.h };
        if (node._resLast) return { w: node._resLast.dims[0], h: node._resLast.dims[1] };
        return null;
      }

      // Toggle a native widget's visibility while keeping it serializable (the proven prompt-builder
      // trick): saved computeSize is restored on show, [0,-4] collapses it on hide.
      function setWidgetVisible(w, vis) {
        if (!w) return;
        if (!vis) {
          if (!w._resHidden) { w._resPrevCompute = w.computeSize; w._resHidden = true; }
          w.hidden = true;
          w.computeSize = () => [0, -4];
        } else if (w._resHidden) {
          w.hidden = false;
          w.computeSize = w._resPrevCompute;
          w._resHidden = false;
        }
      }

      // Recompute width/height for the active mode, snapping/clamping to the profile. `driver` is the
      // side the user just edited in auto mode ("w"|"h"). Re-entrancy guarded.
      function recalcDims(driver) {
        if (node._resCalc || !wWidget || !hWidget) return;
        const mode = resMode(), p = effRules(profName(), snapWidget?.value), clamps = profClamps(profName());
        node._resCalc = true;
        let warn = false;
        try {
          if (mode === "input") {
            // The width/height widgets are the SOURCE of the ratio here, never destinations:
            // overwriting them would fight the user's typing, and writing into a linked widget is
            // meaningless anyway since the link value wins at queue time. So the result goes to
            // _resIn. When a side is linked its value lives upstream — nothing to compute until the
            // node runs and onExecuted reports back.
            node._resIn = null;
            if (!anyLinked()) {
              const src = currentDims();
              const ar = detectAR(src.w, src.h);
              const tw = Math.sqrt(Math.max(0, parseFloat(mpWidget?.value) || 0) * 1e6 * ar);
              warn = clamps && (tw > p.max || (ar ? tw / ar : 0) > p.max);
              const [w, h] = fitW(tw, ar, p);
              node._resIn = { w, h, src: [src.w, src.h] };
            }
          } else if (mode === "raw") {              // raw: literal sides, snapped + per-axis clamped
            const w = Number(wWidget.value) || 0, h = Number(hWidget.value) || 0;
            warn = clamps && (w > p.max || h > p.max);
            wWidget.value = snap(w, p);
            hWidget.value = snap(h, p);
          } else {                                  // auto / megapixel: aspect-locked, ratio preserved at cap
            const ar = effAR(arWidget?.value, orient());
            let tw;
            if (mode === "megapixel") tw = Math.sqrt(Math.max(0, parseFloat(mpWidget?.value) || 0) * 1e6 * ar);
            else if (driver === "h") tw = (Number(hWidget.value) || 0) * ar;   // auto, height edited
            else tw = Number(wWidget.value) || 0;                              // auto, width edited / ratio change
            warn = clamps && (tw > p.max || (ar ? tw / ar : 0) > p.max);       // ideal side exceeds the cap
            const [w, h] = fitW(tw, ar, p);
            wWidget.value = w;
            hWidget.value = h;
          }
        } finally { node._resCalc = false; node._resWarn = warn; }
      }

      // The auto-detect field: the ratio actually detected from the source, plus the nearest named
      // preset for orientation ("auto 1.400 ≈ 4:3 Standard"). The preset is a label only — the dims
      // keep the exact ratio, which is the whole point of the mode.
      function detectedSuffix() {
        const src = node._resIn?.src || node._resLast?.src;
        if (!src || !(src[0] > 0) || !(src[1] > 0)) return "    ·    ratio auto-detected from inputs";
        const ar = detectAR(src[0], src[1]);
        const [ratio, name, orientation] = nearestPreset(ar);
        return `    ·    auto ${ar.toFixed(3)} ≈ ${orientation === "portrait" ? flipRatio(ratio) : ratio} ${name}`;
      }

      function updateReadout() {
        if (!resLine) return;
        const p = effRules(profName(), snapWidget?.value), d = outDims();
        if (!d) {                                   // input mode, linked, not run yet
          const target = (parseFloat(mpWidget?.value) || 0).toFixed(2);
          resLine.textContent = `→ resolves at run time    target ${target} MP${detectedSuffix()}`;
          warnLine.style.display = "none";
          return;
        }
        const mp = (d.w * d.h / 1e6).toFixed(2);
        let suffix = "";
        if (resMode() === "input") suffix = detectedSuffix();
        else if (resMode() !== "raw") {             // show the effective ratio + orientation
          const m = /(\d+)\s*:\s*(\d+)/.exec(arWidget?.value || "");
          if (m) {
            const a = +m[1], b = +m[2];
            if (a === b) suffix = `    ·    ${a}:${b} Square`;
            else if (orient() === "portrait") suffix = `    ·    ${b}:${a} Portrait`;
            else suffix = `    ·    ${a}:${b} Landscape`;
          }
        }
        resLine.textContent = `${d.w} × ${d.h}    ${mp} MP${suffix}`;
        if (node._resWarn) {
          warnLine.textContent = `⚠ ${profName()} max ${p.max} × ${p.max} px — clamped to keep aspect`;
          warnLine.style.display = "";
        } else {
          warnLine.style.display = "none";
        }
      }

      // Show/hide the mode-relevant widgets (snap_multiple only for the default profile; orientation
      // is always hidden — driven by the flip button; the flip button itself hides in raw mode, where
      // there is no aspect to flip, and in input mode, where the source carries its own orientation),
      // then relayout the node.
      function applyVisibility() {
        const mode = resMode(), isInput = mode === "input";
        setWidgetVisible(arWidget, mode !== "raw" && !isInput);   // input detects the ratio instead
        setWidgetVisible(mpWidget, mode === "megapixel" || isInput);
        // width/height stay visible in EVERY mode: hiding a widget hides its input socket with it,
        // and those sockets are how input mode is fed. In megapixel mode they show the result.
        setWidgetVisible(wWidget, true);
        setWidgetVisible(hWidget, true);
        setWidgetVisible(snapWidget, profName() === DEFAULT_PROFILE);
        setWidgetVisible(orientWidget, false);
        setWidgetVisible(flipBtn, mode !== "raw" && !isInput);
        if (node.computeSize) node.setSize([node.size[0], node.computeSize()[1]]);
        node.setDirtyCanvas?.(true, true);
      }

      // Both sockets connected means "keep the source's ratio", which is exactly what input mode
      // does — so select it, remembering the mode we came from to restore on disconnect. Returns
      // true when the mode changed (the caller relayouts; assigning .value fires no callback).
      function syncInputMode() {
        if (!modeWidget) return false;
        const both = bothLinked();
        if (both && modeWidget.value !== "input") {
          node._resPrevMode = modeWidget.value;
          modeWidget.value = "input";
          return true;
        }
        if (!both && modeWidget.value === "input") {
          modeWidget.value = node._resPrevMode || "megapixel";
          node._resPrevMode = null;
          return true;
        }
        return false;
      }

      // Push the current dims into every node wired to the width/height outputs (slot 0 = width,
      // 1 = height). Returns the number of target nodes touched.
      function pushToTargets() {
        if (!node.graph) return 0;
        // null = input mode, linked, not yet run: the width/height widgets hold the source, not the
        // result, so there is nothing truthful to push. Skip rather than poison a builder's canvas
        // with source dims; onExecuted pushes the real ones as soon as the graph runs.
        const d = outDims();
        if (!d) return 0;
        const links = node.graph.links;
        const getLink = (id) => (links?.get ? links.get(id) : links?.[id]);
        const touched = new Set();
        (node.outputs || []).forEach((out, slot) => {
          const val = slot === 1 ? d.h : d.w;
          for (const id of (out.links || [])) {
            const link = getLink(id);
            if (!link) continue;
            const tgt = node.graph.getNodeById(link.target_id);
            if (!tgt) continue;
            const inp = tgt.inputs?.[link.target_slot];
            const tw = inp && tgt.widgets?.find((x) => x.name === inp.name);
            if (tw) { tw.value = val; tw.callback?.(val); }
            touched.add(tgt);
          }
        });
        for (const t of touched) t.setDirtyCanvas?.(true, true);
        return touched.size;
      }

      // Recompute + relabel; optionally push live to connected canvases.
      function refresh(driver, push) {
        recalcDims(driver);
        updateReadout();
        if (push) pushToTargets();
        node.setDirtyCanvas?.(true, true);
      }

      const flipLabel = () => `⟷ Orientation: ${orient() === "portrait" ? "Portrait" : "Landscape"}`;
      // Flip toggles orientation AND swaps the current width/height, then recomputes: the inverted
      // ratio drives the result, and in auto the swapped width seeds the rotated target. (The button
      // is hidden in raw mode, so the no-aspect case never reaches here.)
      function doFlip() {
        if (orientWidget) orientWidget.value = orient() === "portrait" ? "landscape" : "portrait";
        if (wWidget && hWidget) { const t = wWidget.value; wWidget.value = hWidget.value; hWidget.value = t; }
        if (flipBtn) flipBtn.name = flipLabel();
        refresh("w", true);
      }

      // On load, remap an old/bare aspect value ("16:9", or an old portrait "9:16") to a current
      // landscape label + orientation, so pre-change workflows don't reset to the default.
      function remapAspectOnLoad() {
        if (!arWidget) return;
        const opts = arWidget.options?.values || [];
        const v = String(arWidget.value ?? "");
        if (!opts.includes(v)) {
          const m = /(\d+)\s*:\s*(\d+)/.exec(v);
          if (m) {
            let a = +m[1], b = +m[2], portrait = false;
            if (a < b) { const t = a; a = b; b = t; portrait = true; }   // old portrait -> landscape base
            const preset = ASPECT_PRESETS.find(([r]) => r === `${a}:${b}`);
            if (preset) {
              arWidget.value = aspectLabel(preset[0], preset[1]);
              if (portrait && orientWidget) orientWidget.value = "portrait";
            }
          }
        }
        if (flipBtn) flipBtn.name = flipLabel();
      }

      // ── Read-only output readout (added last → sits under the flip button). ──
      const readoutEl = document.createElement("div");
      readoutEl.style.cssText = "width:100%;box-sizing:border-box;padding:2px 4px;text-align:center;line-height:1.45;";
      const resLine = document.createElement("div");
      resLine.style.cssText = "font:bold 13px monospace;color:#46b4e6;";
      const warnLine = document.createElement("div");
      warnLine.style.cssText = "color:#e0a020;font:10px sans-serif;display:none;";
      readoutEl.append(resLine, warnLine);

      // ── flip button (native, non-serialized) then the readout DOM widget ──
      const flipBtn = node.addWidget("button", flipLabel(), null, doFlip, { serialize: false });
      node.addDOMWidget("output_resolution", "info", readoutEl, { serialize: false });

      // Guarantee snap_multiple always LEAVES the front-end as a valid int in [1,1024]. A ComfyUI
      // number widget can serialize an empty string; int("") then fails the backend's INT validation
      // and rejects the whole node — even under a profile (Ideogram 4) that ignores snap_multiple.
      // serializeValue is graphToPrompt's single serialization point, so coercing here sanitizes
      // every queue path and the saved workflow; sanitizeSnap repairs the DISPLAYED value on
      // load/edit. (Same guard pattern as web/js/prompt_batch.js.)
      const sanitizeSnap = () => { if (snapWidget) snapWidget.value = clampSnap(snapWidget.value); };
      if (snapWidget) snapWidget.serializeValue = () => clampSnap(snapWidget.value);

      // ── wire widget callbacks (auto-push live so a connected canvas tracks edits) ──
      if (profileWidget) chainCallback(profileWidget, "callback", () => { sanitizeSnap(); applyVisibility(); refresh("w", true); });
      if (snapWidget) chainCallback(snapWidget, "callback", () => { sanitizeSnap(); refresh("w", true); });
      if (modeWidget) chainCallback(modeWidget, "callback", () => { applyVisibility(); refresh("w", true); });
      if (arWidget) chainCallback(arWidget, "callback", () => { if (resMode() !== "raw" && resMode() !== "input") refresh("w", true); });
      if (mpWidget) chainCallback(mpWidget, "callback", () => { if (resMode() === "megapixel" || resMode() === "input") refresh(undefined, true); });
      if (wWidget) chainCallback(wWidget, "callback", () => { if (!node._resCalc) refresh("w", true); });
      if (hWidget) chainCallback(hWidget, "callback", () => { if (!node._resCalc) refresh("h", true); });

      // Connection changes: pick up (or drop) input mode, then push to a freshly-connected downstream
      // node so its canvas reflects right away. type 1 = input side, 2 = output side.
      chainCallback(node, "onConnectionsChange", function (type) {
        const inputSide = (type == null || type === 1);
        requestAnimationFrame(() => {
          if (inputSide) node._resLast = null;      // a different source: last run's numbers are stale
          const switched = syncInputMode();
          if (switched) applyVisibility();
          if (switched || inputSide) refresh("w", true);
          else pushToTargets();
        });
      });

      // The backend reports what it resolved, and the source it detected the ratio from. With the
      // width/height sockets linked this is the first moment the editor can know either — so the
      // readout fills in here and the real dims go downstream. (A cached node does not re-execute,
      // so the previous values simply stand.)
      chainCallback(node, "onExecuted", function (message) {
        const dims = message?.dims, src = message?.src;
        if (!Array.isArray(dims) || dims.length < 2) return;
        node._resLast = {
          dims: [parseInt(dims[0], 10) || 0, parseInt(dims[1], 10) || 0],
          src: Array.isArray(src) && src.length >= 2 ? [parseInt(src[0], 10) || 0, parseInt(src[1], 10) || 0] : null,
        };
        updateReadout();
        pushToTargets();
        node.setDirtyCanvas?.(true, true);
      });

      // Apply the current state (remap on load + input-mode sync + visibility + recompute + readout).
      // Reused by onConfigure, so a saved workflow whose links are restored lands in input mode.
      node._resApply = () => {
        sanitizeSnap(); remapAspectOnLoad(); syncInputMode(); applyVisibility(); recalcDims("w"); updateReadout();
      };
      requestAnimationFrame(node._resApply);
    });

    // Re-apply after a saved workflow loads (widget values restored first).
    chainCallback(nodeType.prototype, "onConfigure", function () {
      const node = this;
      requestAnimationFrame(() => { node._resApply?.(); });
    });
  },
});
