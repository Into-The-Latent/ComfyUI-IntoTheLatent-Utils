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
export function originNode(node, name) {
  const link = inputLink(node, name);
  return link ? node.graph.getNodeById(link.origin_id) : null;
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
