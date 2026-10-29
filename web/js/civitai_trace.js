/*
 * Part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
 *
 * Live-graph (LiteGraph) walk helpers for the ITL Save Metadata (Civitai) preview in save_civitai_metadata.js.
 * Mirror of nodes/civitai_metadata/tracer.py — keep the two in sync. Pure functions (no ComfyUI imports) so they
 * can be checked outside the browser: node tests/js/civitai_trace.check.mjs
 */

export const SAMPLER_CLASSES = new Set(["KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"]);
export const CLIP_CLASSES = new Set(["CLIPTextEncode"]);
export const ZERO_OUT_CLASSES = new Set(["ConditioningZeroOut"]); // mirror of tracer.ZERO_OUT_CLASSES

// node.inputs[i].link -> graph.links[id] -> {origin_id, origin_slot}.
function linkById(graph, id) {
  if (id == null) return null;
  const links = graph?.links;
  return (links?.get ? links.get(id) : links?.[id]) || null;
}
export function inputLink(node, name) {
  const inp = (node.inputs || []).find((i) => i.name === name);
  return inp ? linkById(node.graph, inp.link) : null;
}
// The node feeding `node`'s input `name`, through any legacy Reroute nodes (the API prompt has none).
export function originNode(node, name) {
  let link = inputLink(node, name);
  let n = link ? node.graph.getNodeById(link.origin_id) : null;
  for (let hops = 0; n && (n.comfyClass || n.type) === "Reroute" && hops < 64; hops++) {
    link = linkById(n.graph, n.inputs?.[0]?.link);
    n = link ? n.graph.getNodeById(link.origin_id) : null;
  }
  return n;
}
export function widget(node, name) {
  return node.widgets?.find((w) => w.name === name)?.value;
}
// Mirror of tracer._literal: a widget's value, or null when its input is wired to a real node (the API prompt
// has a link there and the widget only holds a stale value). A virtual source (legacy PrimitiveNode) is
// replaced by its value in the API prompt, and the frontend syncs that value into the widget; a source the
// graph can't find (a subgraph's input) keeps the widget value too.
export function literal(node, name) {
  if (!node) return null;
  const origin = originNode(node, name);
  if (origin && !origin.isVirtualNode) return null;
  return widget(node, name) ?? null;
}

// Mirror of tracer.GUIDER_INPUTS: guider class -> its [positive, negative, cfg] inputs / widgets;
// null = it has none (runs without CFG). Other guiders use the CFGGuider names.
export const GUIDER_INPUTS = {
  CFGGuider: ["positive", "negative", "cfg"],
  DualCFGGuider: ["cond1", "negative", "cfg_cond2_negative"],
  BasicGuider: ["conditioning", null, null],
};
const DEFAULT_GUIDER_INPUTS = ["positive", "negative", "cfg"];
// Mirror of tracer.SAMPLER_NODE_NAMES / SCHEDULER_NODE_NAMES: core nodes that say what runs by their class.
export const SAMPLER_NODE_NAMES = {
  SamplerEulerAncestral: "euler_ancestral", SamplerEulerAncestralCFGPP: "euler_ancestral_cfg_pp",
  SamplerEulerCFGpp: "euler_cfg_pp", SamplerLMS: "lms", SamplerDPMPP_2S_Ancestral: "dpmpp_2s_ancestral",
  SamplerDPMPP_SDE: "dpmpp_sde", SamplerDPMPP_2M_SDE: "dpmpp_2m_sde", SamplerDPMPP_3M_SDE: "dpmpp_3m_sde",
  SamplerDPMAdaptative: "dpm_adaptive", SamplerER_SDE: "er_sde", SamplerSASolver: "sa_solver",
  SamplerSEEDS2: "seeds_2", SamplerLCM: "lcm",
};
export const SCHEDULER_NODE_NAMES = {
  KarrasScheduler: "karras", ExponentialScheduler: "exponential", PolyexponentialScheduler: "polyexponential",
  LaplaceScheduler: "laplace", BetaSamplingScheduler: "beta", AlignYourStepsScheduler: "align_your_steps",
};

// Mirror of tracer._guider: the node holding the prompts, cfg and model (SamplerCustomAdvanced's guider,
// else the sampler itself), with its [positive, negative, cfg] names.
export function guiderOf(sampler) {
  const holder = originNode(sampler, "guider") || sampler;
  return { holder, names: GUIDER_INPUTS[holder.comfyClass || holder.type] || DEFAULT_GUIDER_INPUTS };
}

// Mirror of tracer._settings_node: BFS back from `node`'s input `name` to the nearest node that has a `key`
// widget or input, or is one of `classes`: KSamplerSelect / a dedicated sampler node behind `sampler`, the
// scheduler behind `sigmas` (steps), RandomNoise behind `noise`. Its other fields are read from it too (SplitSigmasDenoise's own denoise is a split point, not the
// scheduler's). Never reads past a sampler.
export function settingsNode(node, name, key, classes = {}) {
  const seen = new Set();
  const queue = [originNode(node, name)];
  while (queue.length) {
    const n = queue.shift();
    if (!n || seen.has(n.id)) continue;
    seen.add(n.id);
    if (SAMPLER_CLASSES.has(n.comfyClass || n.type)) continue;
    if (n.widgets?.some((w) => w.name === key) || (n.inputs || []).some((i) => i.name === key)) return n;
    if (Object.hasOwn(classes, n.comfyClass || n.type)) return n;
    for (const inp of n.inputs || []) queue.push(originNode(n, inp.name));
  }
  return null;
}

// Mirror of tracer._read_sampler_fields for sampler_name / scheduler: the sampler's own widget, else the node
// behind `sampler` / `sigmas` (its widget, or its class for the dedicated nodes).
export function samplerName(sampler) {
  const select = settingsNode(sampler, "sampler", "sampler_name", SAMPLER_NODE_NAMES);
  return literal(sampler, "sampler_name") ?? literal(select, "sampler_name") ??
    (select ? SAMPLER_NODE_NAMES[select.comfyClass || select.type] ?? null : null);
}
export function schedulerName(sampler) {
  const sched = settingsNode(sampler, "sigmas", "steps");
  return literal(sampler, "scheduler") ?? literal(sched, "scheduler") ??
    (sched ? SCHEDULER_NODE_NAMES[sched.comfyClass || sched.type] ?? null : null);
}

const BYPASSED = 4; // LiteGraph node mode: the API prompt drops the node and wires its input through

// Mirror of tracer._find_clip_encode: BFS to the nearest CLIPTextEncode for the sampler input `which`.
// A node with inputs named `which` / `which_*` (WanImageToVideo, PairConditioning*, …) is followed only
// through them, else through every other input but never the other side's; ConditioningZeroOut is a
// dead end unless bypassed. zeroed = no encode found, a zero-out hit and nothing unknown (a text encoder
// we can't read: TextEncode in the class or a wired clip input; a missing node). Other leaves (LoadImage,
// loaders) and earlier samplers say nothing about the text.
export function findClipEncode(node, which) {
  const other = which === "positive" ? "negative" : "positive";
  const named = (name, side) => name === side || name.startsWith(side + "_");
  const seen = new Set();
  const queue = [node];
  let zeroed = false;
  let unknown = false;
  while (queue.length) {
    const n = queue.shift();
    if (!n) { unknown = true; continue; }
    if (seen.has(n.id)) continue;
    seen.add(n.id);
    const cls = n.comfyClass || n.type;
    if (CLIP_CLASSES.has(cls)) return { clip: n, zeroed: false };
    if (ZERO_OUT_CLASSES.has(cls) && n.mode !== BYPASSED) { zeroed = true; continue; }
    if (SAMPLER_CLASSES.has(cls)) continue; // reached through a latent input; conditioning never passes a sampler
    if (String(cls).includes("TextEncode") || inputLink(n, "clip")) { unknown = true; continue; }
    const linked = (n.inputs || []).map((i) => [i.name, linkById(n.graph, i.link)]).filter(([, l]) => l);
    const own = linked.filter(([name]) => named(name, which));
    const links = own.length ? own : linked.filter(([name]) => !named(name, other));
    for (const [, l] of links) queue.push(n.graph.getNodeById(l.origin_id));
  }
  return { clip: null, zeroed: zeroed && !unknown };
}
