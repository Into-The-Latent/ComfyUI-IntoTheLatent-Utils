/*
 * Part of ComfyUI-IntoTheLatent-Utils.
 *
 * Companion to the Ideogram 4 Prompt Builder. GPL-3.0, like the rest of the pack.
 * Upstream lineage: https://github.com/kijai/ComfyUI-KJNodes
 *
 * A "click-together" wizard for the Ideogram 4 style_description fields. A button on the node opens a
 * fullscreen chip-picker (5 categories). On close the picked values are written straight into a
 * connected ITL Ideogram 4 Prompt Builder's style widgets — the node's own data output stays empty
 * so it never overwrites the builder's bounding boxes / palette / high_level_description.
 */
import { chainCallback } from "./utility.js";
const { app } = window.comfyAPI.app;

const BUILDER_TYPE = "ITLIdeogram4PromptBuilder";

// Built-in chip lists (values are exactly what lands in the JSON). Sourced from the research report.
// These are the fallback / "restore defaults" content; the live lists are read from an editable
// preset file (see PRESET_FILE) so users can add or replace styles without touching the code.
// Note: some Photo values contain commas — that's fine; selection uses substring containment, not a
// naive comma split, so a multi-word value is treated as one chip.
const DEFAULT_CATEGORIES = [
  { key: "aesthetics", label: "Aesthetics", chips: [
    "cinematic", "moody", "dreamy", "serene", "vintage", "minimalist", "surreal", "vibrant",
    "cozy", "dramatic", "ethereal", "nostalgic", "dark", "clean", "elegant",
  ] },
  { key: "lighting", label: "Lighting", chips: [
    "golden hour lighting", "blue hour evening light", "sunrise lighting", "sunset lighting",
    "soft diffused lighting", "hard directional lighting", "studio lighting (softbox)", "rim lighting",
    "backlit (silhouette effect)", "volumetric lighting (haze, rays)", "neon lighting", "candlelit",
    "moonlight", "overcast lighting", "high contrast lighting",
  ] },
  { key: "medium", label: "Medium", chips: [
    "photograph", "illustration", "graphic_design", "3d_render", "painting", "pencil_sketch",
    "pixel_art", "line_art", "concept_art", "infographic", "animation",
  ] },
  { key: "photo", label: "Photo (camera / lens)", chips: [
    "35mm lens, f/1.4", "50mm lens, f/1.8", "85mm lens, shallow depth of field", "24mm wide-angle, f/2.8",
    "200mm telephoto, f/2.8", "macro lens, extreme close-up", "drone aerial shot, high angle",
    "instant film photography (Polaroid)", "35mm motion-picture film still", "studio flash lighting, softbox",
    "black-and-white film (e.g. Kodak Tri-X)", "shallow depth of field, bokeh",
  ] },
  { key: "art_style", label: "Art Style", chips: [
    "flat vector illustration", "oil painting", "watercolor illustration", "pencil sketch illustration",
    "comic book style", "isometric illustration", "anime illustration", "low-poly 3D render",
    "hand-drawn doodles style", "retro-futuristic concept art", "flat graphic design style", "art deco style",
  ] },
];

const BLANK = () => ({ aesthetics: "", lighting: "", medium: "", photo: "", art_style: "" });

// ── Editable style presets, stored server-side via ComfyUI's userdata API. This path maps to
// ComfyUI/user/default/itl/ideogram4/WizardStylesDefault.json. Users can edit the file to add or
// replace chips; if it's missing or malformed we fall back to DEFAULT_CATEGORIES and warn in the UI.
const PRESET_FILE = "itl/ideogram4/WizardStylesDefault.json";
const ALLOWED_KEYS = ["aesthetics", "lighting", "medium", "photo", "art_style"];
const DEFAULT_LABELS = Object.fromEntries(DEFAULT_CATEGORIES.map((c) => [c.key, c.label]));
// File body written by "Restore defaults": a documented, hand-editable shape.
const DEFAULT_PRESETS_FILE = {
  _comment: "ITL Ideogram 4 Style Wizard presets. Edit 'chips' freely; 'key' must be one of " +
    ALLOWED_KEYS.join(", ") + ". Delete this file or use the wizard's 'Restore defaults' to reset.",
  version: 1,
  categories: DEFAULT_CATEGORIES.map((c) => ({ key: c.key, label: c.label, chips: c.chips.slice() })),
};

// Validate + normalize a parsed preset file into the internal [{key,label,chips}] shape.
// Returns { cats, error }: cats is the category array on success (error null), or cats null + a
// human-readable error string describing why the file is unusable (→ caller treats as malformed).
function parsePresets(text) {
  let obj;
  try { obj = JSON.parse(text); } catch (e) { return { cats: null, error: "Invalid JSON: " + e.message }; }
  const cats = Array.isArray(obj) ? obj : (obj && Array.isArray(obj.categories) ? obj.categories : null);
  if (!cats) return { cats: null, error: 'Expected an array, or an object with a "categories" array.' };
  if (!cats.length) return { cats: null, error: '"categories" is empty.' };
  const seen = new Set();
  const out = [];
  for (let n = 0; n < cats.length; n++) {
    const c = cats[n], where = "category " + (n + 1);
    if (!c || typeof c !== "object") return { cats: null, error: where + " is not an object." };
    if (!ALLOWED_KEYS.includes(c.key)) return { cats: null, error: where + ': "key" must be one of ' + ALLOWED_KEYS.join(", ") + ' (got ' + JSON.stringify(c.key) + ")." };
    if (seen.has(c.key)) return { cats: null, error: where + ': duplicate key "' + c.key + '".' };
    if (!Array.isArray(c.chips)) return { cats: null, error: where + ' ("' + c.key + '"): "chips" must be an array.' };
    seen.add(c.key);
    const chips = c.chips.filter((x) => typeof x === "string" && x.trim()).map((x) => x.trim());
    const label = typeof c.label === "string" && c.label.trim() ? c.label.trim() : (DEFAULT_LABELS[c.key] || c.key);
    out.push({ key: c.key, label, chips });
  }
  return { cats: out, error: null };
}

// Load the live categories: { cats, state, error } where state is "ok" | "missing" | "malformed".
// Always returns usable cats (the file's when valid, else the built-in defaults).
async function loadCategories() {
  let res;
  try { res = await app.api.getUserData(PRESET_FILE); } catch (e) { res = null; }
  if (!res || res.status !== 200) return { cats: DEFAULT_CATEGORIES, state: "missing", error: null };
  let text = "";
  try { text = await res.text(); } catch (e) { return { cats: DEFAULT_CATEGORIES, state: "malformed", error: "Couldn't read the file." }; }
  const { cats, error } = parsePresets(text);
  return cats ? { cats, state: "ok", error: null } : { cats: DEFAULT_CATEGORIES, state: "malformed", error };
}

// Write the built-in defaults to the preset file (used by the "Restore defaults" button).
async function saveDefaultPresets() {
  await app.api.storeUserData(PRESET_FILE, JSON.stringify(DEFAULT_PRESETS_FILE, null, 2),
    { overwrite: true, stringify: false, throwOnError: true });
}

// Trim + drop empties + de-dupe (case-insensitive) a chip list — used when saving edited presets.
function dedupeTrim(chips) {
  const seen = new Set(), out = [];
  for (const c of chips || []) {
    const t = (c || "").trim();
    if (t && !seen.has(t.toLowerCase())) { seen.add(t.toLowerCase()); out.push(t); }
  }
  return out;
}
// Build the editor's working copy: all five keys in canonical order. A valid file's categories win
// (a key it omits → empty chips); a missing/malformed file starts from the built-in defaults.
function toEditCats(cats, state) {
  const base = state === "ok" ? cats : DEFAULT_CATEGORIES;
  const byKey = Object.fromEntries(base.map((c) => [c.key, c]));
  return ALLOWED_KEYS.map((k) => {
    const src = byKey[k];
    return { key: k, label: (src && src.label) || DEFAULT_LABELS[k] || k, chips: src ? src.chips.slice() : [] };
  });
}

// Normalize a comma-joined field: collapse whitespace around commas, drop empty/duplicate items.
function normField(s) {
  const seen = new Set();
  return (s || "")
    .split(",")
    .map((t) => t.trim())
    .filter((t) => t && !seen.has(t.toLowerCase()) && seen.add(t.toLowerCase()))
    .join(", ");
}
// Is `term` present in `field`? Whole-item match first (handles plain terms), substring fallback
// (handles chip values that themselves contain commas, e.g. "35mm lens, f/1.4").
function hasTerm(field, term) {
  const f = field || "", t = term.trim();
  if (!t) return false;
  if (f.split(",").some((x) => x.trim().toLowerCase() === t.toLowerCase())) return true;
  return f.toLowerCase().includes(t.toLowerCase());
}
// Toggle `term` in `field` (append when absent, remove when present), returning the normalized result.
function toggleTerm(field, term) {
  const t = term.trim();
  if (!t) return normField(field);
  if (hasTerm(field, t)) {
    // Remove the term as a whole comma-item where possible, else as a substring.
    const items = (field || "").split(",").map((x) => x.trim()).filter(Boolean);
    const kept = items.filter((x) => x.toLowerCase() !== t.toLowerCase());
    let out = kept.join(", ");
    if (kept.length === items.length) out = normField(out.replace(t, ""));   // comma-containing chip → substring strip
    return normField(out);
  }
  return normField((field ? field + ", " : "") + t);
}

function injectWizStyle() {
  if (document.getElementById("itl-wiz-style")) return;
  const s = document.createElement("style");
  s.id = "itl-wiz-style";
  s.textContent = `
    .itl-wiz-fs { position:fixed; inset:0; z-index:9200; background:rgba(0,0,0,0.72); display:flex; align-items:center; justify-content:center; }
    .itl-wiz-panel { position:relative; width:min(860px,92vw); max-height:90vh; display:flex; flex-direction:column; background:#1a1a1a; border:1px solid #444; border-radius:8px; box-shadow:0 12px 48px rgba(0,0,0,0.6); }
    .itl-wiz-head { display:flex; align-items:center; gap:8px; padding:10px 14px; border-bottom:1px solid #333; font:13px sans-serif; color:#ddd; }
    .itl-wiz-title { font-weight:600; flex:0 0 auto; }
    .itl-wiz-tabs { display:flex; gap:4px; flex:1 1 auto; }
    .itl-wiz-tab { background:none; border:1px solid transparent; border-radius:5px; color:#aaa; font:12px sans-serif; cursor:pointer; padding:3px 12px; }
    .itl-wiz-tab:hover { color:#fff; }
    .itl-wiz-tab.active { color:#46b4e6; border-color:#46b4e6; background:#2a3a42; }
    .itl-wiz-body { flex:1 1 auto; overflow-y:auto; padding:12px 14px; display:flex; flex-direction:column; gap:14px; }
    .itl-wiz-search { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#ddd; font:12px sans-serif; padding:5px 8px; }
    .itl-wiz-search:focus { border-color:#46b4e6; outline:none; color:#fff; }
    .itl-wiz-status { font:13px sans-serif; color:#8fbfd6; min-height:17px; }
    .itl-wiz-status.off { color:#e05555; font-weight:600; }
    .itl-wiz-edit { display:flex; flex-direction:column; gap:5px; }
    .itl-wiz-editrow { display:flex; align-items:center; gap:6px; }
    .itl-wiz-editrow .itl-wiz-input { font:12px sans-serif; }
    .itl-wiz-del { background:none; border:1px solid #555; border-radius:4px; color:#c77; cursor:pointer; font:12px sans-serif; line-height:1; padding:3px 8px; flex:0 0 auto; }
    .itl-wiz-del:hover { border-color:#a33; color:#fff; background:#a33; }
    .itl-wiz-add { display:flex; align-items:center; gap:6px; margin-top:2px; }
    .itl-wiz-dirty { color:#e6a23c; font:11px sans-serif; flex:1 1 auto; }
    .itl-wiz-toast { position:absolute; left:50%; bottom:16px; transform:translateX(-50%); background:#2a3a42; color:#cfe8f5; border:1px solid #46b4e6; border-radius:6px; padding:5px 14px; font:12px sans-serif; box-shadow:0 4px 14px rgba(0,0,0,0.5); opacity:1; transition:opacity .4s ease; pointer-events:none; z-index:5; }
    .itl-wiz-toast.fade { opacity:0; }
    .itl-wiz-banner { display:flex; align-items:flex-start; gap:10px; padding:8px 10px; border-radius:6px; background:#3a2f1a; border:1px solid #6b5320; color:#e6c98c; font:11px/1.5 sans-serif; }
    .itl-wiz-banner code { color:#f0d9a6; font:11px monospace; }
    .itl-wiz-banner .itl-wiz-btn { flex:0 0 auto; align-self:center; }
    .itl-wiz-banner-msg { flex:1 1 auto; }
    .itl-wiz-cat { display:flex; flex-direction:column; gap:6px; }
    .itl-wiz-catlbl { font:11px sans-serif; color:#8fbfd6; letter-spacing:0.03em; text-transform:uppercase; }
    .itl-wiz-chips { display:flex; flex-wrap:wrap; gap:5px; }
    .itl-wiz-chip { background:#333; border:1px solid #555; border-radius:12px; color:#bbb; font:11px sans-serif; cursor:pointer; padding:2px 10px; line-height:17px; white-space:nowrap; }
    .itl-wiz-chip:hover { border-color:#46b4e6; color:#fff; }
    .itl-wiz-chip.active { border-color:#46b4e6; color:#46b4e6; background:#2a3a42; }
    .itl-wiz-input { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444; border-radius:4px; color:#ddd; font:12px monospace; padding:4px 6px; }
    .itl-wiz-input:focus { border-color:#46b4e6; outline:none; color:#fff; }
    .itl-wiz-foot { flex:0 0 auto; padding:10px 14px; border-top:1px solid #333; display:flex; flex-direction:column; gap:8px; }
    .itl-wiz-warn { color:#e6a23c; font:11px sans-serif; min-height:14px; }
    .itl-wiz-prev { background:#141414; border:1px solid #333; border-radius:4px; color:#9fd29f; font:11px/1.4 monospace; padding:8px; margin:0; max-height:150px; overflow:auto; white-space:pre; }
    .itl-wiz-footrow { display:flex; align-items:center; gap:8px; }
    .itl-wiz-btn { background:#333; border:1px solid #555; border-radius:4px; color:#bbb; font:12px sans-serif; cursor:pointer; padding:4px 12px; }
    .itl-wiz-btn:hover { border-color:#46b4e6; color:#fff; }
    .itl-wiz-btn.primary { border-color:#46b4e6; color:#46b4e6; background:#2a3a42; }
    .itl-wiz-x { background:none; border:none; color:#999; cursor:pointer; font:16px sans-serif; line-height:1; padding:2px 6px; }
    .itl-wiz-x:hover { color:#fff; }
  `;
  document.head.appendChild(s);
}

// Assemble the style_description object in the builder's canonical key order, omitting blank fields.
function buildStyleDescription(wiz) {
  const a = wiz.aesthetics.trim(), l = wiz.lighting.trim(), m = wiz.medium.trim();
  const p = wiz.photo.trim(), ar = wiz.art_style.trim();
  const sd = {};
  if (a) sd.aesthetics = a;
  if (l) sd.lighting = l;
  if (p) {                              // photo mode: ...photo, medium...
    sd.photo = p;
    if (m) sd.medium = m;
  } else {                              // art / none: ...medium, art_style...
    if (m) sd.medium = m;
  }
  if (ar) sd.art_style = ar;            // shown even alongside photo (free-pick); the warning flags it
  return sd;
}

app.registerExtension({
  name: "ITL.Ideogram4StyleWizard",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "ITLIdeogram4StyleWizard") return;
    injectWizStyle();

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const wizWidget = node.widgets?.find((w) => w.name === "wizard_data");
      // Hide the persistence widget but keep it serializable (the proven prompt-builder pattern).
      if (wizWidget) { wizWidget.hidden = true; wizWidget.computeSize = () => [0, -4]; }
      const i = node.inputs?.findIndex((inp) => inp.name === "wizard_data");
      if (i != null && i !== -1) node.removeInput(i);

      node._wiz = BLANK();
      function hydrate() {
        node._wiz = BLANK();
        try {
          const v = JSON.parse(wizWidget?.value || "{}");
          if (v && typeof v === "object") for (const k of Object.keys(node._wiz)) if (typeof v[k] === "string") node._wiz[k] = v[k];
        } catch (e) {}
      }
      hydrate();

      function persist() {
        if (wizWidget) wizWidget.value = JSON.stringify(node._wiz);
        // Nudge ComfyUI to snapshot the change (same trick the prompt builder uses).
        window.dispatchEvent(new MouseEvent("mouseup"));
      }

      // Prompt Builder nodes wired to this node's "style" output (the writeback targets / status).
      function builderTargets() {
        const out = [];
        for (const id of (node.outputs?.[0]?.links || [])) {
          const link = node.graph?.links?.get ? node.graph.links.get(id) : node.graph?.links?.[id];
          if (!link) continue;
          const t = node.graph.getNodeById(link.target_id);
          if (t && t.type === BUILDER_TYPE) out.push(t);
        }
        return out;
      }

      // Write the picked style fields into every connected Prompt Builder. `notify` controls the
      // "nothing connected" alert (shown on an explicit Apply, not on Esc/backdrop close). Returns count.
      function applyToBuilder(notify) {
        const targets = builderTargets();
        for (const target of targets) {
          const set = (name, val) => {
            const w = target.widgets?.find((x) => x.name === name);
            if (w) { w.value = val; w.callback?.(val); }
          };
          // Mirror the wizard exactly, including clears: photo -> "photo", else art_style -> "art_style",
          // else "none". Setting "none" rebuilds the combo without a photo/art_style sub-widget, so a
          // cleared wizard actually clears the builder's photo/art_style (was the "stays filled" bug).
          const p = node._wiz.photo.trim(), ar = node._wiz.art_style.trim();
          const kind = p ? "photo" : (ar ? "art_style" : "none");
          const styleW = target.widgets?.find((x) => x.name === "style");
          if (styleW) styleW.value = kind;                   // DynamicCombo setter rebuilds sub-widgets
          if (kind === "photo") set("style.photo", p);
          else if (kind === "art_style") set("style.art_style", ar);
          set("aesthetics", node._wiz.aesthetics.trim());
          set("lighting", node._wiz.lighting.trim());
          set("medium", node._wiz.medium.trim());
          target.setDirtyCanvas?.(true, true);
        }
        if (!targets.length && notify) {
          window.alert("Wire this node's \"style\" output into an ITL Ideogram 4 Prompt Builder's \"import_json\" input, then reopen the wizard.");
        }
        return targets.length;
      }
      node._wizApply = applyToBuilder;

      // ── modal ──
      const PRESET_PATH_DISPLAY = "user/default/itl/ideogram4/WizardStylesDefault.json";

      async function openModal() {
        if (node._wizOverlay || node._wizOpening) return;
        node._wizOpening = true;                              // guard the async load against double-open
        let loaded;
        try { loaded = await loadCategories(); } finally { node._wizOpening = false; }
        if (node._wizOverlay) return;

        // Shared, mutable across both tabs; refreshed after a Save / Restore.
        let categories = loaded.cats, loadState = loaded.state, loadError = loaded.error;
        let currentTab = "pick", editDirty = false;

        const ov = document.createElement("div"); ov.className = "itl-wiz-fs";
        const panel = document.createElement("div"); panel.className = "itl-wiz-panel";
        ov.appendChild(panel);

        const head = document.createElement("div"); head.className = "itl-wiz-head";
        const title = document.createElement("span"); title.className = "itl-wiz-title"; title.textContent = "Style Wizard";
        const tabs = document.createElement("div"); tabs.className = "itl-wiz-tabs";
        const pickTabBtn = document.createElement("button"); pickTabBtn.className = "itl-wiz-tab"; pickTabBtn.textContent = "Pick styles";
        const editTabBtn = document.createElement("button"); editTabBtn.className = "itl-wiz-tab"; editTabBtn.textContent = "Edit presets";
        tabs.append(pickTabBtn, editTabBtn);
        const xBtn = document.createElement("button"); xBtn.className = "itl-wiz-x"; xBtn.textContent = "✕"; xBtn.title = "Close (applies your picks to the connected builder)";
        head.append(title, tabs, xBtn);

        const body = document.createElement("div"); body.className = "itl-wiz-body";
        const foot = document.createElement("div"); foot.className = "itl-wiz-foot";

        // Transient confirmation pill (Applied / Saved / …).
        function toast(msg) {
          const t = document.createElement("div"); t.className = "itl-wiz-toast"; t.textContent = msg;
          panel.appendChild(t);
          setTimeout(() => { t.classList.add("fade"); setTimeout(() => t.remove(), 450); }, 1300);
        }

        // Overwrite the preset file with the built-in defaults, then refresh both tabs.
        async function doRestore() {
          if (loadState !== "missing" && !window.confirm("Overwrite the preset file with the built-in defaults?")) return;
          try { await saveDefaultPresets(); }
          catch (e) { window.alert("Couldn't write the preset file to the server."); return; }
          const r = await loadCategories();
          categories = r.cats; loadState = r.state; loadError = r.error; editDirty = false;
          showTab(currentTab); toast("Restored defaults ✓");
        }

        let closed = false;
        function close(apply, notify) {
          if (closed) return;
          if (currentTab === "edit" && editDirty && !window.confirm("Discard unsaved preset changes?")) return;
          closed = true;
          if (apply) applyToBuilder(notify);
          document.removeEventListener("keydown", onEsc, true);
          ov.remove(); node._wizOverlay = null;
        }
        function onEsc(e) { if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(true, false); } }

        // ── Pick tab ──
        function renderPickTab() {
          currentTab = "pick"; editDirty = false;
          pickTabBtn.classList.add("active"); editTabBtn.classList.remove("active");
          body.innerHTML = ""; foot.innerHTML = "";
          const refreshers = [];      // re-sync chip highlight + field value from node._wiz
          const allChips = [];        // {term, el} across all categories, for the search filter

          const status = document.createElement("div"); status.className = "itl-wiz-status";
          const warn = document.createElement("div"); warn.className = "itl-wiz-warn";
          const prev = document.createElement("pre"); prev.className = "itl-wiz-prev";
          const footRow = document.createElement("div"); footRow.className = "itl-wiz-footrow";
          const spacer = document.createElement("span"); spacer.style.flex = "1";
          const clearBtn = document.createElement("button"); clearBtn.className = "itl-wiz-btn"; clearBtn.textContent = "Clear all";
          const clearApplyBtn = document.createElement("button"); clearApplyBtn.className = "itl-wiz-btn"; clearApplyBtn.textContent = "Clear & apply & close";
          const applyBtn = document.createElement("button"); applyBtn.className = "itl-wiz-btn"; applyBtn.textContent = "Apply";
          const doneBtn = document.createElement("button"); doneBtn.className = "itl-wiz-btn primary"; doneBtn.textContent = "Apply & close";
          footRow.append(spacer, clearBtn, clearApplyBtn, applyBtn, doneBtn);
          foot.append(status, warn, prev, footRow);

          function updatePreview() {
            prev.textContent = JSON.stringify({ style_description: buildStyleDescription(node._wiz) }, null, 2);
            const both = node._wiz.photo.trim() && node._wiz.art_style.trim();
            warn.textContent = both ? "⚠ Ideogram allows only photo OR art_style — only photo will be applied to the builder." : "";
          }
          function updateStatus() {
            const ts = builderTargets();
            status.classList.toggle("off", !ts.length);
            status.textContent = ts.length
              ? "→ writing to: " + ts.map((t) => (t.title || t.type) + " #" + t.id).join(", ")
              : "⚠ Not connected — wire the \"style\" output into a Prompt Builder's import_json.";
          }
          function buildCatBlock(cat) {
            const block = document.createElement("div"); block.className = "itl-wiz-cat";
            const lbl = document.createElement("div"); lbl.className = "itl-wiz-catlbl"; lbl.textContent = cat.label;
            const chipsWrap = document.createElement("div"); chipsWrap.className = "itl-wiz-chips";
            const input = document.createElement("input");
            input.className = "itl-wiz-input"; input.type = "text";
            input.placeholder = "click chips above, or type your own (comma-separated)";
            input.value = node._wiz[cat.key] || "";
            const chipEls = (cat.chips || []).map((term) => {
              const c = document.createElement("button");
              c.className = "itl-wiz-chip"; c.textContent = term; c.title = term;
              c.addEventListener("click", () => {
                node._wiz[cat.key] = toggleTerm(node._wiz[cat.key], term);
                input.value = node._wiz[cat.key];
                syncChips(); persist(); updatePreview();
              });
              return { term, el: c };
            });
            chipEls.forEach((c) => { chipsWrap.appendChild(c.el); allChips.push(c); });
            input.addEventListener("input", () => {
              node._wiz[cat.key] = input.value;   // keep raw while typing; normalized on chip-toggle/apply
              syncChips(); persist(); updatePreview();
            });
            function syncChips() { for (const c of chipEls) c.el.classList.toggle("active", hasTerm(node._wiz[cat.key], c.term)); }
            function syncCat() { input.value = node._wiz[cat.key] || ""; syncChips(); }
            syncChips();
            refreshers.push(syncCat);
            block.append(lbl, chipsWrap, input);
            return block;
          }

          const search = document.createElement("input");
          search.className = "itl-wiz-search"; search.type = "text"; search.placeholder = "Search chips…";
          search.addEventListener("input", () => {
            const q = search.value.trim().toLowerCase();
            for (const c of allChips) c.el.style.display = (!q || c.term.toLowerCase().includes(q)) ? "" : "none";
          });
          body.appendChild(search);
          if (loadState !== "ok") body.appendChild(buildPickBanner());
          for (const cat of categories) body.appendChild(buildCatBlock(cat));
          updatePreview(); updateStatus();

          function clearWiz() { node._wiz = BLANK(); for (const r of refreshers) r(); persist(); updatePreview(); }
          const applied = (n) => { if (n) toast("Applied to " + n + " builder" + (n > 1 ? "s" : "") + " ✓"); updateStatus(); };
          clearBtn.addEventListener("click", clearWiz);
          clearApplyBtn.addEventListener("click", () => { clearWiz(); close(true, true); });
          applyBtn.addEventListener("click", () => applied(applyToBuilder(true)));
          doneBtn.addEventListener("click", () => close(true, true));
        }

        // Warning banner (Pick tab) for a missing / malformed preset file.
        function buildPickBanner() {
          const banner = document.createElement("div"); banner.className = "itl-wiz-banner";
          const msg = document.createElement("div"); msg.className = "itl-wiz-banner-msg";
          if (loadState === "missing") {
            msg.innerHTML = "⚠ Style preset file not found — using built-in defaults. Use <b>Edit presets → Restore defaults</b> to create it:";
          } else {
            msg.innerHTML = "⚠ Style preset file is malformed — using built-in defaults.";
            if (loadError) { const e = document.createElement("div"); e.style.cssText = "margin-top:4px;color:#f0d9a6;font-family:monospace;"; e.textContent = loadError; msg.appendChild(e); }
            const fix = document.createElement("div"); fix.style.marginTop = "4px"; fix.innerHTML = "Fix the JSON and reopen, or use <b>Edit presets → Restore defaults</b>:"; msg.appendChild(fix);
          }
          const pathEl = document.createElement("div"); pathEl.innerHTML = "<code>" + PRESET_PATH_DISPLAY + "</code>"; msg.appendChild(pathEl);
          const btns = document.createElement("div"); btns.style.cssText = "display:flex;flex-direction:column;gap:4px;";
          const copyBtn = document.createElement("button"); copyBtn.className = "itl-wiz-btn"; copyBtn.textContent = "Copy path";
          copyBtn.addEventListener("click", () => { navigator.clipboard?.writeText?.(PRESET_PATH_DISPLAY).then(() => toast("Path copied")).catch(() => {}); });
          const restoreBtn = document.createElement("button"); restoreBtn.className = "itl-wiz-btn"; restoreBtn.textContent = "Restore defaults";
          restoreBtn.addEventListener("click", () => doRestore());
          btns.append(copyBtn, restoreBtn);
          banner.append(msg, btns);
          return banner;
        }

        // ── Edit presets tab ──
        function renderEditTab() {
          currentTab = "edit"; editDirty = false;
          editTabBtn.classList.add("active"); pickTabBtn.classList.remove("active");
          body.innerHTML = ""; foot.innerHTML = "";
          let editCats = toEditCats(categories, loadState);

          const help = document.createElement("div"); help.className = "itl-wiz-status";
          help.textContent = "Add, rename, or delete chips per category, then Save. New chips append to the end.";
          const footRow = document.createElement("div"); footRow.className = "itl-wiz-footrow";
          const dirtyLbl = document.createElement("div"); dirtyLbl.className = "itl-wiz-dirty";
          const restoreBtn = document.createElement("button"); restoreBtn.className = "itl-wiz-btn"; restoreBtn.textContent = "Restore defaults";
          const discardBtn = document.createElement("button"); discardBtn.className = "itl-wiz-btn"; discardBtn.textContent = "Discard";
          const saveBtn = document.createElement("button"); saveBtn.className = "itl-wiz-btn primary"; saveBtn.textContent = "Save to file";
          footRow.append(dirtyLbl, restoreBtn, discardBtn, saveBtn);
          foot.append(help, footRow);

          function setDirty(v) { editDirty = v; dirtyLbl.textContent = v ? "Unsaved changes" : ""; }

          function buildEditCat(cat) {
            const block = document.createElement("div"); block.className = "itl-wiz-cat";
            const lbl = document.createElement("div"); lbl.className = "itl-wiz-catlbl"; lbl.textContent = cat.label + "  (key: " + cat.key + ")";
            const list = document.createElement("div"); list.className = "itl-wiz-edit";
            cat.chips.forEach((chip, idx) => {
              const row = document.createElement("div"); row.className = "itl-wiz-editrow";
              const inp = document.createElement("input"); inp.className = "itl-wiz-input"; inp.type = "text"; inp.value = chip;
              inp.addEventListener("input", () => { cat.chips[idx] = inp.value; setDirty(true); });
              const del = document.createElement("button"); del.className = "itl-wiz-del"; del.textContent = "✕"; del.title = "Delete chip";
              del.addEventListener("click", () => { cat.chips.splice(idx, 1); setDirty(true); renderEditor(); });
              row.append(inp, del); list.appendChild(row);
            });
            const addWrap = document.createElement("div"); addWrap.className = "itl-wiz-add";
            const addInput = document.createElement("input"); addInput.className = "itl-wiz-input"; addInput.type = "text"; addInput.placeholder = "add a chip…";
            const addBtn = document.createElement("button"); addBtn.className = "itl-wiz-btn"; addBtn.textContent = "+ Add";
            const add = () => { const v = addInput.value.trim(); if (!v) return; cat.chips.push(v); addInput.value = ""; setDirty(true); renderEditor(); };
            addBtn.addEventListener("click", add);
            addInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); add(); } });
            addWrap.append(addInput, addBtn);
            block.append(lbl, list, addWrap);
            return block;
          }
          function renderEditor() { body.innerHTML = ""; for (const cat of editCats) body.appendChild(buildEditCat(cat)); }
          renderEditor();

          saveBtn.addEventListener("click", async () => {
            saveBtn.disabled = true;
            const fileObj = { _comment: DEFAULT_PRESETS_FILE._comment, version: 1,
              categories: editCats.map((c) => ({ key: c.key, label: c.label, chips: dedupeTrim(c.chips) })) };
            try {
              await app.api.storeUserData(PRESET_FILE, JSON.stringify(fileObj, null, 2), { overwrite: true, stringify: false, throwOnError: true });
            } catch (e) { window.alert("Couldn't save the preset file to the server."); saveBtn.disabled = false; return; }
            const r = await loadCategories();
            categories = r.cats; loadState = r.state; loadError = r.error;
            editCats = toEditCats(categories, loadState);
            setDirty(false); saveBtn.disabled = false; renderEditor(); toast("Saved ✓");
          });
          discardBtn.addEventListener("click", async () => {
            const r = await loadCategories();
            categories = r.cats; loadState = r.state; loadError = r.error;
            editCats = toEditCats(categories, loadState);
            setDirty(false); renderEditor(); toast("Reverted");
          });
          restoreBtn.addEventListener("click", () => doRestore());
        }

        function showTab(name) { if (name === "edit") renderEditTab(); else renderPickTab(); }
        pickTabBtn.addEventListener("click", () => { if (currentTab !== "pick" && (!editDirty || window.confirm("Discard unsaved preset changes?"))) showTab("pick"); });
        editTabBtn.addEventListener("click", () => { if (currentTab !== "edit") showTab("edit"); });

        xBtn.addEventListener("click", () => close(true, false));
        ov.addEventListener("mousedown", (e) => { if (e.target === ov) close(true, false); });   // backdrop closes + applies
        document.addEventListener("keydown", onEsc, true);

        panel.append(head, body, foot);
        document.body.appendChild(ov);
        node._wizOverlay = ov;
        showTab("pick");
      }
      node._wizOpen = openModal;

      node.addWidget("button", "🪄 Open Style Wizard", "", openModal);
    });

    // Re-hydrate after a saved workflow loads (widget value is restored before onConfigure fires).
    chainCallback(nodeType.prototype, "onConfigure", function () {
      const wizWidget = this.widgets?.find((w) => w.name === "wizard_data");
      this._wiz = BLANK();
      try {
        const v = JSON.parse(wizWidget?.value || "{}");
        if (v && typeof v === "object") for (const k of Object.keys(this._wiz)) if (typeof v[k] === "string") this._wiz[k] = v[k];
      } catch (e) {}
    });

    // Tidy up an open modal if the node is removed.
    chainCallback(nodeType.prototype, "onRemoved", function () {
      this._wizOverlay?.remove(); this._wizOverlay = null;
    });
  },
});
