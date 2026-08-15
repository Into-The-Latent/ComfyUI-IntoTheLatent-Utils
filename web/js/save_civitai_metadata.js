/*
 * Part of ComfyUI-IntoTheLatent-Utils. GPL-3.0, like the rest of the pack.
 *
 * Front-end for the ITL Save Metadata (Civitai) nodes. Adds a "🔎 Test detection" button that runs
 * the SAME backward graph trace as nodes/civitai_metadata/tracer.py (image -> sampler -> CLIPTextEncode
 * / ITLPromptBatch, and model -> LoraLoader* -> Checkpoint) against the LIVE graph, and shows what
 * save will write. Hashes are computed on save (no file access here). Keep this trace in sync with the
 * Python tracer — the two are a mirror pair, like parsePrompts <-> _parse_prompts.
 */
import { chainCallback } from "./utility.js";
const { app } = window.comfyAPI.app;

const NODE_IDS = ["ITLSaveCivitaiMetadata", "ITLSaveCivitaiMetadataAdvanced"];
const SAMPLER_CLASSES = new Set(["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"]);
const CLIP_CLASSES = new Set(["CLIPTextEncode"]);
// class_type -> widget holding the model filename (mirror of tracer.MODEL_SOURCES; the Python side
// also tracks the folder for hashing, which the preview doesn't need). Covers checkpoints and
// standalone diffusion models (UNETLoader "Load Diffusion Model" — Flux/SD3/Krea/etc.).
const MODEL_SOURCES = {
  CheckpointLoaderSimple: "ckpt_name", CheckpointLoader: "ckpt_name", unCLIPCheckpointLoader: "ckpt_name",
  UNETLoader: "unet_name", UnetLoaderGGUF: "unet_name", UnetLoaderGGUFAdvanced: "unet_name",
};
const LORA_CLASSES = new Set(["LoraLoader", "LoraLoaderModelOnly"]);
// rgthree Power Lora Loader packs many loras into one node; each lora widget's value is a dict
// {on, lora, strength, strengthTwo}. Mirror of tracer.POWER_LORA_CLASSES.
const POWER_LORA_CLASSES = new Set(["Power Lora Loader (rgthree)"]);
// rgthree Lora Loader Stack: flat lora_0N / strength_0N widget pairs. Mirror of tracer.LORA_STACK_CLASSES.
const LORA_STACK_CLASSES = new Set(["Lora Loader Stack (rgthree)"]);
const BATCH_CLASS = "ITLPromptBatch";

// --- Live-graph link helpers (LiteGraph). node.inputs[i].link -> graph.links[id] -> {origin_id, origin_slot}.
function inputLink(node, name) {
  const inp = (node.inputs || []).find((i) => i.name === name);
  if (!inp || inp.link == null) return null;
  const links = node.graph?.links;
  const link = links?.get ? links.get(inp.link) : links?.[inp.link];
  return link || null;
}
function originNode(node, name) {
  const link = inputLink(node, name);
  return link ? node.graph.getNodeById(link.origin_id) : null;
}
function widget(node, name) {
  return node.widgets?.find((w) => w.name === name)?.value;
}
const stem = (s) => String(s).replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "");

// Enabled loras from a live rgthree Power Lora Loader node, in listed order. Each lora row widget
// holds a {on, lora, strength, strengthTwo} value; skip disabled and empty ("None"/blank) slots.
function powerLoraEntries(node) {
  const out = [];
  for (const w of node.widgets || []) {
    const v = w?.value;
    if (!v || typeof v !== "object" || typeof v.lora !== "string") continue;
    if (!v.lora || v.lora === "None" || v.on === false) continue;
    out.push({ name: stem(v.lora), strength: v.strength });
  }
  return out;
}

// Enabled loras from a live rgthree Lora Loader Stack node (flat lora_0N / strength_0N pairs), ordered
// by slot number; skip "None"/blank and zero-strength slots (rgthree won't load those).
function loraStackEntries(node) {
  const slots = [];
  for (const w of node.widgets || []) {
    const m = /^lora_(\d+)$/.exec(w?.name || "");
    if (m) slots.push({ n: parseInt(m[1], 10), num: m[1] });
  }
  slots.sort((a, b) => a.n - b.n);
  const out = [];
  for (const s of slots) {
    const name = widget(node, "lora_" + s.num);
    if (typeof name !== "string" || !name || name === "None") continue;
    const strength = widget(node, "strength_" + s.num);
    if (strength === 0) continue;
    out.push({ name: stem(name), strength });
  }
  return out;
}

function bfsBack(node, classSet) {
  const seen = new Set();
  const queue = [node];
  while (queue.length) {
    const n = queue.shift();
    if (!n || seen.has(n.id)) continue;
    seen.add(n.id);
    if (classSet.has(n.comfyClass || n.type)) return n;
    for (const inp of n.inputs || []) {
      const o = originNode(n, inp.name);
      if (o) queue.push(o);
    }
  }
  return null;
}

// Current line of our ITLPromptBatch node at the given output slot (0=positive, 1=negative).
function batchRow(origin, slot) {
  const idx = parseInt(widget(origin, "index"), 10) || 0;
  let rows = origin._pbRows;
  if (!rows) { try { rows = JSON.parse(widget(origin, "prompts_json") || "[]"); } catch { rows = []; } }
  if (!rows.length) return null;
  const row = rows[Math.max(0, Math.min(idx, rows.length - 1))];
  if (slot === 0) return row.positive || "";
  if (slot === 1) return row.negative || "";
  return null; // any other slot (e.g. the Int `index` output) is not a prompt
}

// Resolve a CLIPTextEncode's text: static widget, our batch node (by output slot), or a plain string node.
function resolveClipText(node) {
  const t = widget(node, "text");
  if (typeof t === "string") return { text: t, ok: true };
  const link = inputLink(node, "text");
  if (!link) return { text: "", ok: false };
  const origin = node.graph.getNodeById(link.origin_id);
  if (!origin) return { text: "", ok: false };
  if ((origin.comfyClass || origin.type) === BATCH_CLASS) {
    const v = batchRow(origin, link.origin_slot);
    return v == null ? { text: "", ok: false } : { text: v, ok: true };
  }
  for (const w of origin.widgets || []) {
    if (typeof w.value === "string" && w.value.trim()) return { text: w.value, ok: true };
  }
  return { text: "", ok: false };
}

// Resolve an Advanced-node override socket for the preview.
// Returns null when the socket is absent or unwired (keep the traced value);
// {ok:false} when wired but not statically resolvable (it WILL be set at run — not "unresolved").
function resolveWiredInput(node, name) {
  const link = inputLink(node, name);
  if (!link) return null;
  const origin = node.graph.getNodeById(link.origin_id);
  if (!origin) return { value: null, ok: false };
  if ((origin.comfyClass || origin.type) === BATCH_CLASS) {
    const v = batchRow(origin, link.origin_slot);
    return v == null ? { value: null, ok: false } : { value: v, ok: true };
  }
  for (const w of origin.widgets || []) {
    if (w.value != null && w.value !== "") return { value: w.value, ok: true };
  }
  return { value: null, ok: false };
}

function traceLive(node) {
  const r = { positive: "", negative: "", steps: null, cfg: null, seed: null, sampler_name: null,
              scheduler: null, model_name: null, loras: [], unresolved: [] };
  let sampler = originNode(node, "images") ? bfsBack(originNode(node, "images"), SAMPLER_CLASSES) : null;
  if (!sampler) {
    const all = (node.graph?._nodes || []).filter((n) => SAMPLER_CLASSES.has(n.comfyClass || n.type));
    sampler = all.length === 1 ? all[0] : null;
  }
  if (!sampler) { r.unresolved.push("sampler"); return r; }

  r.steps = widget(sampler, "steps");
  r.cfg = widget(sampler, "cfg");
  r.seed = widget(sampler, "seed") ?? widget(sampler, "noise_seed");
  r.sampler_name = widget(sampler, "sampler_name");
  r.scheduler = widget(sampler, "scheduler");
  for (const f of ["steps", "cfg", "sampler_name"]) if (r[f] == null) r.unresolved.push(f);

  for (const which of ["positive", "negative"]) {
    const cond = originNode(sampler, which);
    const clip = cond ? bfsBack(cond, CLIP_CLASSES) : null;
    const res = clip ? resolveClipText(clip) : { text: "", ok: false };
    r[which] = res.text;
    if (!res.ok) r.unresolved.push(which);
  }

  // Model chain: walk model links back through LoraLoaders to the checkpoint.
  let cur = originNode(sampler, "model");
  const seen = new Set();
  while (cur && !seen.has(cur.id)) {
    seen.add(cur.id);
    const cls = cur.comfyClass || cur.type;
    if (LORA_CLASSES.has(cls)) {
      const name = widget(cur, "lora_name");
      if (typeof name === "string") r.loras.push({ name: stem(name), strength: widget(cur, "strength_model") });
      cur = originNode(cur, "model");
    } else if (POWER_LORA_CLASSES.has(cls) || LORA_STACK_CLASSES.has(cls)) {
      // Reversed within the node so the final r.loras.reverse() restores listed order.
      const entries = POWER_LORA_CLASSES.has(cls) ? powerLoraEntries(cur) : loraStackEntries(cur);
      for (const lo of entries.reverse()) r.loras.push(lo);
      cur = originNode(cur, "model");
    } else if (MODEL_SOURCES[cls]) {
      const name = widget(cur, MODEL_SOURCES[cls]);
      if (typeof name === "string") r.model_name = stem(name);
      break;
    } else {
      cur = originNode(cur, "model");
    }
  }
  r.loras.reverse();

  // Advanced node: a wired override socket wins over the trace (mirrors _save_with_metadata).
  for (const name of ["positive", "negative", "steps", "cfg", "seed", "sampler_name", "scheduler"]) {
    const res = resolveWiredInput(node, name);
    if (!res) continue;
    const u = r.unresolved.indexOf(name);
    if (u >= 0) r.unresolved.splice(u, 1);
    r[name] = res.ok ? res.value : "(wired — resolved at run)";
  }
  return r;
}

function ensureStyles() {
  if (document.getElementById("itl-scm-style")) return;
  const s = document.createElement("style");
  s.id = "itl-scm-style";
  s.textContent = `
  .itl-scm{font:11.5px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif;color:#d3d3d0;
    background:#1a1a19;border:1px solid #33332f;border-radius:6px;padding:7px 9px;white-space:pre-wrap;
    word-break:break-word}
  .itl-scm .k{color:#8b8b86}.itl-scm .pos{color:#5cae6d}.itl-scm .neg{color:#c86b6b}
  .itl-scm .warn{color:#e0a24e}`;
  document.head.appendChild(s);
}

// positive/negative can be filled at run time (e.g. by the ITL Prompt Batch node walking its rows),
// so "empty in the preview" is not necessarily a problem — say so, rather than only "wire the socket".
const RUNTIME_FIELDS = new Set(["positive", "negative"]);

function renderPreview(el, r) {
  const loras = r.loras.length ? r.loras.map((l) => `<${escapeHtml(l.name)}:${l.strength}>`).join(" ") : "(none)";
  const runtime = r.unresolved.filter((u) => RUNTIME_FIELDS.has(u));
  const hard = r.unresolved.filter((u) => !RUNTIME_FIELDS.has(u));
  let warn = "";
  if (runtime.length) warn += `\n⚠ ${runtime.join(", ")}: empty in preview — resolved at run if driven by the ITL Prompt Batch node (or wire the Advanced socket).`;
  if (hard.length) warn += `\n⚠ unresolved: ${hard.join(", ")} — wire the Advanced node's socket(s).`;
  el.innerHTML =
    `<span class="pos">positive</span>: ${escapeHtml(r.positive) || "(empty)"}\n` +
    `<span class="neg">negative</span>: ${escapeHtml(r.negative) || "(empty)"}\n` +
    `<span class="k">steps</span> ${escapeHtml(r.steps)} · <span class="k">cfg</span> ${escapeHtml(r.cfg)} · ` +
    `<span class="k">sampler</span> ${escapeHtml(r.sampler_name)}/${escapeHtml(r.scheduler)} · <span class="k">seed</span> ${escapeHtml(r.seed)}\n` +
    `<span class="k">model</span> ${escapeHtml(r.model_name ?? "(?)")} · <span class="k">loras</span> ${loras}\n` +
    `<span class="k">size</span> from image at run · hashes computed on save` +
    (warn ? `<span class="warn">${escapeHtml(warn)}</span>` : "");
}
function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

app.registerExtension({
  name: "ITL.SaveCivitaiMetadata",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.includes(nodeData?.name)) return;
    ensureStyles();
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const panel = document.createElement("div");        // widget root; ComfyUI pins its height each frame
      const content = document.createElement("div");      // natural-height content we measure
      content.className = "itl-scm";
      content.textContent = 'Press "🔎 Test detection" to preview what will be saved.';
      panel.appendChild(content);

      const btn = node.addWidget("button", "🔎 Test detection", null, () => {
        try { renderPreview(content, traceLive(node)); }
        catch (e) { content.textContent = "Trace error: " + e.message; }
        fit();
      });
      btn.serialize = false;

      const w = node.addDOMWidget("scm_preview", "info", panel, { serialize: false });

      // Reserve the content's natural height so the node grows to fit its text (measured, like the
      // Prompt Batch rows editor) — otherwise ComfyUI gives the DOM widget a fixed height and the
      // preview overflows below the node border. Measure `content` (not the pinned `panel`) to avoid
      // a grow-by-its-own-padding feedback loop.
      function fit() {
        const h = Math.max(content.scrollHeight, 8);
        w.computeSize = () => [node.size?.[0] || 320, h + 8];
        const want = node.computeSize?.();
        if (want) node.setSize([node.size[0], want[1]]);
        node.setDirtyCanvas?.(true, true);
      }
      w.computeSize = () => [node.size?.[0] || 320, Math.max(content.scrollHeight, 8) + 8];

      // The element attaches after onNodeCreated, so observe it and re-fit the moment it gains (or
      // changes) its natural height — self-corrects with no user interaction.
      let lastH = 0;
      const ro = new ResizeObserver(() => {
        const h = content.scrollHeight;
        if (h && h !== lastH) { lastH = h; fit(); }
      });
      ro.observe(content);
      chainCallback(node, "onRemoved", () => ro.disconnect());
    });
  },
});
