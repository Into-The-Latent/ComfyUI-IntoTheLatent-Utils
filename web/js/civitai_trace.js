/*
 * Part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
 *
 * Live-graph (LiteGraph) walk helpers for the ITL Save Metadata (Civitai) preview in save_civitai_metadata.js.
 * Mirror of nodes/civitai_metadata/tracer.py — keep the two in sync. Pure functions (no ComfyUI imports) so they
 * can be checked outside the browser: node tests/js/civitai_trace.check.mjs
 */

export const CLIP_CLASSES = new Set(["CLIPTextEncode"]);
export const ZERO_OUT_CLASSES = new Set(["ConditioningZeroOut"]); // mirror of tracer.ZERO_OUT_CLASSES

// node.inputs[i].link -> graph.links[id] -> {origin_id, origin_slot}.
export function inputLink(node, name) {
  const inp = (node.inputs || []).find((i) => i.name === name);
  if (!inp || inp.link == null) return null;
  const links = node.graph?.links;
  const link = links?.get ? links.get(inp.link) : links?.[inp.link];
  return link || null;
}
export function originNode(node, name) {
  const link = inputLink(node, name);
  return link ? node.graph.getNodeById(link.origin_id) : null;
}

const BYPASSED = 4; // LiteGraph node mode: the API prompt drops the node and wires its input through

// Mirror of tracer._find_clip_encode: BFS to the nearest CLIPTextEncode for the sampler input `which`.
// A node with inputs named `which` / `which_*` (WanImageToVideo, PairConditioning*, …) is followed only
// through them; ConditioningZeroOut is a dead end unless bypassed. zeroed = no encode found and every
// branch ended at a zero-out; any other dead end (an encoder we can't read, a missing node) is unknown.
export function findClipEncode(node, which) {
  const seen = new Set();
  const queue = [node];
  let zeroed = false;
  let deadEnd = false;
  while (queue.length) {
    const n = queue.shift();
    if (!n) { deadEnd = true; continue; }
    if (seen.has(n.id)) continue;
    seen.add(n.id);
    const cls = n.comfyClass || n.type;
    if (CLIP_CLASSES.has(cls)) return { clip: n, zeroed: false };
    if (ZERO_OUT_CLASSES.has(cls) && n.mode !== BYPASSED) { zeroed = true; continue; }
    const linked = (n.inputs || []).map((i) => [i.name, inputLink(n, i.name)]).filter(([, l]) => l);
    const own = linked.filter(([name]) => name === which || name.startsWith(which + "_"));
    const links = own.length ? own : linked;
    if (!links.length) deadEnd = true;
    for (const [, l] of links) queue.push(n.graph.getNodeById(l.origin_id));
  }
  return { clip: null, zeroed: zeroed && !deadEnd };
}
