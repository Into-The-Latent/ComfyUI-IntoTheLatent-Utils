// Checks for web/js/civitai_trace.js (the live-graph prompt walk of the Save Metadata (Civitai) preview, mirror of
// nodes/civitai_metadata/tracer.py). Run: node tests/js/civitai_trace.check.mjs
// (not wired into pytest on purpose: spawning a process from a published .py is something the registry scanner
// can flag - see test_no_python_file_trips_the_registry_env_or_network_rules).
import assert from "node:assert/strict";
import { findClipEncode } from "../../web/js/civitai_trace.js";

// Fake LiteGraph graph. spec: {id: [type, {input: originId | [originId, slot] | null}, extra]}.
function graph(spec) {
  const nodes = {};
  const links = new Map();
  const g = { links, getNodeById: (id) => nodes[id] ?? null };
  let lid = 0;
  for (const [id, [type, ins, extra]] of Object.entries(spec)) {
    const inputs = Object.entries(ins).map(([name, o]) => {
      if (o == null) return { name, link: null };
      const [oid, slot] = Array.isArray(o) ? o : [o, 0];
      links.set(++lid, { origin_id: oid, origin_slot: slot });
      return { name, link: lid };
    });
    nodes[id] = { id, type, graph: g, inputs, mode: 0, ...extra };
  }
  return nodes;
}
// The text a walk from node `start` for sampler input `which` lands on: encoder id, "zeroed" or "unresolved".
const walk = (nodes, start, which) => {
  const { clip, zeroed } = findClipEncode(nodes[start], which);
  return clip ? clip.id : zeroed ? "zeroed" : "unresolved";
};
const ENC = { pos: ["CLIPTextEncode", { clip: "ckpt" }], neg: ["CLIPTextEncode", { clip: "ckpt" }], ckpt: ["CheckpointLoaderSimple", {}] };

let n = 0;
const t = (name, fn) => { fn(); n++; console.log("ok  " + name); };

t("plain negative encode", () => {
  assert.equal(walk(graph(ENC), "neg", "negative"), "neg");
});

t("a zeroed negative is known-empty, not the positive", () => {
  const g = graph({ ...ENC, z: ["ConditioningZeroOut", { conditioning: "pos" }] });
  assert.equal(walk(g, "z", "negative"), "zeroed");
});

t("WanImageToVideo is followed through the matching input only", () => {
  const g = graph({ ...ENC, z: ["ConditioningZeroOut", { conditioning: "pos" }],
    w: ["WanImageToVideo", { positive: "pos", negative: "neg", vae: null }],
    wz: ["WanImageToVideo", { positive: "pos", negative: "z", vae: null }] });
  assert.equal(walk(g, "w", "negative"), "neg");
  assert.equal(walk(g, "w", "positive"), "pos");
  assert.equal(walk(g, "wz", "negative"), "zeroed");
});

t("SD3 template: combine of a zeroed and a real part keeps the real negative", () => {
  const g = graph({ ...ENC, z: ["ConditioningZeroOut", { conditioning: "neg" }],
    t1: ["ConditioningSetTimestepRange", { conditioning: "z" }], t2: ["ConditioningSetTimestepRange", { conditioning: "neg" }],
    c: ["ConditioningCombine", { conditioning_1: "t1", conditioning_2: "t2" }] });
  assert.equal(walk(g, "c", "negative"), "neg");
});

t("a zero-out beside an encoder we can't read stays unresolved", () => {
  const g = graph({ ...ENC, z: ["ConditioningZeroOut", { conditioning: "pos" }],
    sdxl: ["CLIPTextEncodeSDXL", { clip: "ckpt" }], c: ["ConditioningCombine", { conditioning_1: "z", conditioning_2: "sdxl" }] });
  assert.equal(walk(g, "c", "negative"), "unresolved");
});

t("PairConditioning nodes are followed through their suffixed inputs", () => {
  const g = graph({ ...ENC,
    s: ["PairConditioningSetProperties", { positive_NEW: "pos", negative_NEW: "neg" }],
    c: ["PairConditioningCombine", { positive_A: ["s", 0], negative_A: ["s", 1], positive_B: "pos", negative_B: "neg" }] });
  assert.equal(walk(g, "c", "negative"), "neg");
  assert.equal(walk(g, "c", "positive"), "pos");
});

t("a bypassed zero-out passes its input through, like the API prompt", () => {
  const g = graph({ ...ENC, z: ["ConditioningZeroOut", { conditioning: "pos" }, { mode: 4 }] });
  assert.equal(walk(g, "z", "negative"), "pos");
});

t("a dangling `which` link is a dead end, not a reason to walk every input", () => {
  const g = graph({ ...ENC, w: ["WanImageToVideo", { positive: "pos", negative: "gone" }] });
  assert.equal(walk(g, "w", "negative"), "unresolved");
});

t("a zeroed negative through a latent carrier is still known-empty", () => {
  const g = graph({ ...ENC, z: ["ConditioningZeroOut", { conditioning: "pos" }], img: ["LoadImage", {}],
    enc: ["VAEEncode", { pixels: "img", vae: "ckpt" }], ref: ["ReferenceLatent", { conditioning: "z", latent: "enc" }] });
  assert.equal(walk(g, "ref", "negative"), "zeroed");
});

t("a carrier without the matching input never reads the other side", () => {
  const g = graph({ ...ENC, w: ["WanImageToVideo", { positive: "pos", negative: null, vae: null }] });
  assert.equal(walk(g, "w", "negative"), "unresolved");
});

t("the walk stops at an earlier sampler", () => {
  const g = graph({ ...ENC, s1neg: ["CLIPTextEncode", { clip: "ckpt" }],
    k1: ["KSampler", { positive: "pos", negative: "s1neg", model: "ckpt" }], dec: ["VAEDecode", { samples: "k1" }],
    enc: ["VAEEncode", { pixels: "dec" }], z: ["ConditioningZeroOut", { conditioning: "pos" }],
    ref: ["ReferenceLatent", { conditioning: "z", latent: "enc" }] });
  assert.equal(walk(g, "ref", "negative"), "zeroed");
});

t("a zero-out beside a CLIP-fed encoder we can't read stays unresolved", () => {
  const g = graph({ ...ENC, z: ["ConditioningZeroOut", { conditioning: "pos" }],
    wc: ["ImpactWildcardEncode", { clip: "ckpt" }], c: ["ConditioningCombine", { conditioning_1: "z", conditioning_2: "wc" }] });
  assert.equal(walk(g, "c", "negative"), "unresolved");
});

console.log(`${n} checks passed`);
