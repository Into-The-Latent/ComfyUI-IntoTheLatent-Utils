// Checks for web/js/civitai_trace.js (the live-graph prompt walk of the Save Metadata (Civitai) preview, mirror of
// nodes/civitai_metadata/tracer.py). Run: node tests/js/civitai_trace.check.mjs
// (not wired into pytest on purpose: spawning a process from a published .py is something the registry scanner
// can flag - see test_no_python_file_trips_the_registry_env_or_network_rules).
import assert from "node:assert/strict";
import { findClipEncode, guiderOf, settingsNode, literal, samplerName, schedulerName, findGuidance } from "../../web/js/civitai_trace.js";

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

// --- modular samplers (SamplerCustomAdvanced / SamplerCustom) ---
const W = (o) => ({ widgets: Object.entries(o).map(([name, value]) => ({ name, value })) });
const flux = (guider) => graph({ ...ENC, g: guider,
  sel: ["KSamplerSelect", {}, W({ sampler_name: "euler" })],
  sch: ["BasicScheduler", { model: "ckpt" }, W({ scheduler: "simple", steps: 8, denoise: 1 })],
  split: ["SplitSigmas", { sigmas: "sch" }, W({ step: 4 })],
  noise: ["RandomNoise", {}, W({ noise_seed: 42 })],
  k: ["SamplerCustomAdvanced", { noise: "noise", guider: "g", sampler: "sel", sigmas: ["split", 1], latent_image: null }] });

t("SamplerCustomAdvanced: the guider holds the prompts, cfg and model", () => {
  const g = flux(["CFGGuider", { model: "ckpt", positive: "pos", negative: "neg" }, W({ cfg: 3.5 })]);
  const { holder, names } = guiderOf(g.k);
  assert.equal(holder.id, "g");
  assert.deepEqual(names, ["positive", "negative", "cfg"]);
  assert.deepEqual(guiderOf(graph(ENC).pos).names, ["positive", "negative", "cfg"]); // no guider: the sampler
  assert.equal(guiderOf(graph(ENC).pos).holder.id, "pos");
});

t("BasicGuider has no negative and no cfg; DualCFGGuider uses cond1 / cfg_conds", () => {
  assert.deepEqual(guiderOf(flux(["BasicGuider", { model: "ckpt", conditioning: "pos" }]).k).names, ["conditioning", null, null]);
  assert.deepEqual(guiderOf(flux(["DualCFGGuider", { model: "ckpt", cond1: "pos", cond2: "pos", negative: "neg" }]).k).names,
    ["cond1", "negative", "cfg_cond2_negative"]);
});

t("settings are read from the nodes wired into sampler / sigmas / noise", () => {
  const g = flux(["CFGGuider", { model: "ckpt", positive: "pos", negative: "neg" }, W({ cfg: 3.5 })]);
  assert.equal(settingsNode(g.k, "sampler", "sampler_name").id, "sel");
  assert.equal(settingsNode(g.k, "sigmas", "steps").id, "sch");       // through SplitSigmas
  assert.equal(settingsNode(g.k, "noise", "noise_seed").id, "noise");
  assert.equal(settingsNode(g.k, "latent_image", "steps"), null);     // unwired
});

t("SplitSigmasDenoise is passed: its denoise is a split point, the scheduler holds the settings", () => {
  const g = graph({ sch: ["BasicScheduler", {}, W({ scheduler: "simple", steps: 8, denoise: 1 })],
    sd: ["SplitSigmasDenoise", { sigmas: "sch" }, W({ denoise: 0.3 })], k: ["SamplerCustomAdvanced", { sigmas: ["sd", 1] }] });
  assert.equal(settingsNode(g.k, "sigmas", "steps").id, "sch");
});

t("the settings walk never reads past a sampler", () => {
  const g = graph({ k1: ["KSampler", {}, W({ steps: 30 })], x: ["SomeSigmas", { from: "k1" }], k: ["SamplerCustom", { sigmas: "x" }] });
  assert.equal(settingsNode(g.k, "sigmas", "steps"), null);
});

t("a widget whose input is wired is not a literal (the API prompt has a link there)", () => {
  const g = graph({ seed: ["Seed (rgthree)", {}, W({ seed: 7 })], noise: ["RandomNoise", { noise_seed: "seed" }, W({ noise_seed: 0 })] });
  assert.equal(literal(g.noise, "noise_seed"), null);
  assert.equal(literal(g.seed, "seed"), 7);
  // A legacy PrimitiveNode is virtual: the API prompt holds its value, and the frontend syncs it into the widget.
  const p = graph({ prim: ["PrimitiveNode", {}, { isVirtualNode: true, ...W({ value: 9 }) }],
    rr: ["Reroute", { "": "prim" }, { isVirtualNode: true }], k: ["KSampler", { seed: "rr" }, W({ seed: 9 })] });
  assert.equal(literal(p.k, "seed"), 9);
  // Inside a subgraph a promoted widget's link starts at the SubgraphInput, which getNodeById can't find:
  // keep the widget value, as before (the flattened API prompt carries the literal).
  const sg = graph({ k: ["KSampler", { steps: "subgraph-input" }, W({ steps: 20 })] });
  assert.equal(literal(sg.k, "steps"), 20);
});

t("legacy Reroute nodes are passed through, like the API prompt", () => {
  const r = graph({ ...ENC, g: ["CFGGuider", { model: "ckpt", positive: "pos", negative: "neg" }, W({ cfg: 3.5 })],
    rr: ["Reroute", { "": "g" }], rr2: ["Reroute", { "": "rr" }], k: ["SamplerCustomAdvanced", { guider: "rr2" }] });
  assert.equal(guiderOf(r.k).holder.id, "g");
});

t("dedicated sampler / scheduler nodes are named by their class", () => {
  const g = graph({ sel: ["SamplerDPMPP_2M_SDE", {}, W({ eta: 1, noise_device: "gpu" })], sch: ["KarrasScheduler", {}, W({ steps: 30 })],
    wrap: ["SomeSamplerWrapper", { sampler: "sel" }], k: ["SamplerCustomAdvanced", { sampler: "wrap", sigmas: "sch" }] });
  assert.equal(samplerName(g.k), "dpmpp_2m_sde");
  assert.equal(schedulerName(g.k), "karras");
  const ks = graph({ sel: ["KSamplerSelect", {}, W({ sampler_name: "euler" })], sch: ["BasicScheduler", {}, W({ scheduler: "simple", steps: 8 })],
    k: ["SamplerCustomAdvanced", { sampler: "sel", sigmas: "sch" }] });
  assert.equal(samplerName(ks.k), "euler");
  assert.equal(schedulerName(ks.k), "simple");
  const unk = graph({ sel: ["SomeCustomSampler", {}, W({ eta: 1 })], sch: ["Flux2Scheduler", {}, W({ steps: 20 })],
    k: ["SamplerCustomAdvanced", { sampler: "sel", sigmas: "sch" }] });
  assert.equal(samplerName(unk.k), null);
  assert.equal(schedulerName(unk.k), null);
});

t("Flux guidance is read from the positive path (FluxGuidance / CLIPTextEncodeFlux), not past a disable", () => {
  const g = graph({ ...ENC, fg: ["FluxGuidance", { conditioning: "pos" }, W({ guidance: 3.5 })],
    ref: ["ReferenceLatent", { conditioning: "fg", latent: null }],
    off: ["FluxDisableGuidance", { conditioning: "fg" }], flux: ["CLIPTextEncodeFlux", { clip: "ckpt" }, W({ guidance: 4 })],
    negfg: ["FluxGuidance", { conditioning: "neg" }, W({ guidance: 9 })],
    k1: ["KSampler", { positive: "ref", negative: "negfg" }], k2: ["KSampler", { positive: "off" }],
    k3: ["KSampler", { positive: "flux" }], k4: ["KSampler", { positive: "pos", negative: "negfg" }],
    bg: ["BasicGuider", { conditioning: "fg" }] });
  assert.equal(findGuidance(g.k1, "positive"), 3.5);
  assert.equal(findGuidance(g.k2, "positive"), null);
  assert.equal(findGuidance(g.k3, "positive"), 4);
  assert.equal(findGuidance(g.k4, "positive"), null);  // only on the negative
  assert.equal(findGuidance(g.bg, "conditioning"), 3.5);
});

console.log(`${n} checks passed`);
