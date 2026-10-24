// Checks for web/js/moss_style.js (the MOSS-Image Style Wizard write logic). Run: node tests/js/moss_style.check.mjs
// (not wired into pytest on purpose: spawning a process from a published .py is something the registry scanner
// can flag - see test_no_python_file_trips_the_registry_env_or_network_rules).
import assert from "node:assert/strict";
import * as S from "../../web/js/moss_style.js";

const wiz = (o) => ({ aesthetics: "", lighting: "", medium: "", photo: "", art_style: "", ...o });
let n = 0;
const t = (name, fn) => { fn(); n++; console.log("ok  " + name); };

t("medium + art style -> meta tags, underscores to spaces, de-duped", () => {
  assert.deepEqual(S.mossMetaTags(wiz({ medium: "3d_render, painting", art_style: "oil painting, Painting" })),
    ["3d render", "painting", "oil painting"]);
  assert.deepEqual(S.mossMetaTags(wiz({})), []);
});

t("aesthetics + lighting + photo -> one Style sentence (photo chips may contain commas)", () => {
  assert.equal(S.mossStyleSentence(wiz({ aesthetics: "moody, cinematic", lighting: "neon lighting", photo: "35mm lens, f/1.4" })),
    "Style: moody, cinematic, neon lighting, 35mm lens, f/1.4.");
  assert.equal(S.mossStyleSentence(wiz({ lighting: "golden_hour lighting." })), "Style: golden hour lighting.");
  assert.equal(S.mossStyleSentence(wiz({ medium: "painting" })), "");             // medium never goes into the sentence
});

t("caption: appends the sentence, replaces only its own previous sentence", () => {
  assert.equal(S.applyCaptionSentence("A cafe.", "", "Style: neon lighting."), "A cafe. Style: neon lighting.");
  assert.equal(S.applyCaptionSentence("", "", "Style: neon lighting."), "Style: neon lighting.");
  assert.equal(S.applyCaptionSentence("A cafe. Style: neon lighting.", "Style: neon lighting.", "Style: moonlight."), "A cafe. Style: moonlight.");
  assert.equal(S.applyCaptionSentence("A cafe. Style: neon lighting.", "Style: neon lighting.", ""), "A cafe.");
  // the user's own text in between / after is kept; the old sentence is replaced where it stands
  assert.equal(S.applyCaptionSentence("A cafe. Style: neon lighting. More words.", "Style: neon lighting.", "Style: moonlight."),
    "A cafe. Style: moonlight. More words.");
  // the user edited the sentence -> no longer ours: left alone, new one appended
  assert.equal(S.applyCaptionSentence("A cafe. Style: pink neon.", "Style: neon lighting.", "Style: moonlight."),
    "A cafe. Style: pink neon. Style: moonlight.");
  // multi-line caption: the sentence goes on its own line
  assert.equal(S.applyCaptionSentence("Line one.\nLine two.\n", "", "Style: x."), "Line one.\nLine two.\nStyle: x.");
  // removal mid-text and on its own line
  assert.equal(S.applyCaptionSentence("A cafe. Style: neon lighting. More words.", "Style: neon lighting.", ""), "A cafe. More words.");
  assert.equal(S.applyCaptionSentence("Line one.\nStyle: x.", "Style: x.", ""), "Line one.");
  assert.equal(S.applyCaptionSentence("Style: x.", "Style: x.", ""), "");
  // a user sentence that happens to start with "Style:" is never touched
  assert.equal(S.applyCaptionSentence("Style: my own words.", "", ""), "Style: my own words.");
});

t("meta: adds the wizard's tags, removes only tags it added before", () => {
  let r = S.applyMetaTags("poster", [], ["watercolor painting", "illustration"]);
  assert.deepEqual(r, { meta: "poster, watercolor painting, illustration", owned: ["watercolor painting", "illustration"] });
  r = S.applyMetaTags(r.meta, r.owned, ["oil painting"]);
  assert.deepEqual(r, { meta: "poster, oil painting", owned: ["oil painting"] });
  r = S.applyMetaTags(r.meta, r.owned, []);
  assert.deepEqual(r, { meta: "poster", owned: [] });
  // a tag the user already had is not taken over (and so never removed by the wizard)
  r = S.applyMetaTags("Illustration, poster", [], ["illustration"]);
  assert.deepEqual(r, { meta: "Illustration, poster", owned: [] });
  // newline-separated meta is read too, and stays one tag per line
  r = S.applyMetaTags("poster\nphotograph", ["photograph"], ["painting"]);
  assert.deepEqual(r, { meta: "poster\npainting", owned: ["painting"] });
});

t("applyMossStyle: closing with unchanged picks touches nothing (user edits / deletions stay)", () => {
  const picks = wiz({ lighting: "neon lighting", medium: "illustration" });
  const first = S.applyMossStyle({ caption: "A cafe.", meta: "poster" }, null, picks);
  assert.deepEqual([first.caption, first.meta], ["A cafe. Style: neon lighting.", "poster, illustration"]);
  // the user edits the sentence, then opens the wizard and closes it without changing anything
  const edited = S.applyMossStyle({ caption: "A cafe. Style: soft neon lighting.", meta: first.meta }, first.record, picks);
  assert.deepEqual([edited.caption, edited.meta], ["A cafe. Style: soft neon lighting.", "poster, illustration"]);
  // ... or deleted it: it doesn't come back
  const deleted = S.applyMossStyle({ caption: "A cafe.", meta: "poster" }, first.record, picks);
  assert.deepEqual([deleted.caption, deleted.meta], ["A cafe.", "poster"]);
  // a real change of picks still updates
  const changed = S.applyMossStyle({ caption: first.caption, meta: first.meta }, first.record, wiz({ lighting: "moonlight" }));
  assert.deepEqual([changed.caption, changed.meta], ["A cafe. Style: moonlight.", "poster"]);
});

t("applyMetaTags keeps the user's formatting when it adds/removes nothing, and their separator when it does", () => {
  assert.deepEqual(S.applyMetaTags("poster\nvintage", [], []), { meta: "poster\nvintage", owned: [] });
  assert.deepEqual(S.applyMetaTags("poster\nvintage", [], ["illustration"]), { meta: "poster\nvintage\nillustration", owned: ["illustration"] });
  assert.deepEqual(S.applyMetaTags("poster\nillustration", ["illustration"], []), { meta: "poster", owned: [] });
});

t("a malformed record (hand-edited / other tool) is treated as none, never throws", () => {
  for (const bad of [{ added: "oil painting" }, { sentence: 5, tags: "x" }, "junk", 7, [], { added: [1, null, "ok"] }]) {
    const r = S.applyMossStyle({ caption: "A.", meta: "" }, bad, wiz({ lighting: "moonlight" }));
    assert.equal(r.caption, "A. Style: moonlight.");
  }
});

t("an explicit Apply (force) re-inserts a deleted sentence / tags even with unchanged picks", () => {
  const picks = wiz({ lighting: "neon lighting", medium: "illustration" });
  const first = S.applyMossStyle({ caption: "A cafe.", meta: "poster" }, null, picks);
  const r = S.applyMossStyle({ caption: "A cafe.", meta: "poster" }, first.record, picks, { force: true });
  assert.deepEqual([r.caption, r.meta], ["A cafe. Style: neon lighting.", "poster, illustration"]);
});

t("the wizard replaces its own (last) copy of a sentence the user also typed", () => {
  assert.equal(S.applyCaptionSentence("Style: neon lighting. A cafe. Style: neon lighting.", "Style: neon lighting.", "Style: moonlight."),
    "Style: neon lighting. A cafe. Style: moonlight.");
});

t("removing an owned tag keeps a user tag that only differs in case", () => {
  assert.deepEqual(S.applyMetaTags("Illustration, poster, illustration", ["illustration"], []), { meta: "Illustration, poster", owned: [] });
});

t("removing a sentence on its own line leaves no blank line", () => {
  assert.equal(S.applyCaptionSentence("A cafe.\nStyle: x.\nMore.", "Style: x.", ""), "A cafe.\nMore.");
});

console.log(`\n${n} checks passed`);
