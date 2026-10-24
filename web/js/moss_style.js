/*
 * Part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
 *
 * How the Style Wizard's picks land in an ITL MOSS-Image Prompt Builder. MOSS-Image has no style fields: a
 * same-seed test showed a medium is picked up from summary.meta tags (as well as from the caption), but lighting
 * only reliably from the caption text. So Medium + Art Style become meta tags, and Aesthetics + Lighting + Photo
 * one "Style: ..." sentence at the end of the caption. The wizard only ever replaces what it wrote itself.
 * Pure functions (no ComfyUI imports) so they can be checked outside the browser.
 */

// Wizard field -> items. Chip values are comma-joined in a field; underscores (Ideogram enum style, e.g.
// "3d_render") read as words for MOSS. Case-insensitive de-dupe, first spelling wins.
function items(...fields) {
  const seen = new Set(), out = [];
  for (const f of fields) {
    for (const raw of String(f || "").split(",")) {
      const t = raw.replace(/_/g, " ").trim().replace(/[.\s]+$/, "");
      if (t && !seen.has(t.toLowerCase())) { seen.add(t.toLowerCase()); out.push(t); }
    }
  }
  return out;
}

export function mossMetaTags(wiz) {
  return items(wiz.medium, wiz.art_style);
}

export function mossStyleSentence(wiz) {
  // Photo chips may contain commas ("35mm lens, f/1.4"); joined back with ", " they read the same in prose.
  const parts = items(wiz.aesthetics, wiz.lighting, wiz.photo);
  return parts.length ? "Style: " + parts.join(", ") + "." : "";
}

// Last position of `sentence` in `text` as a whole sentence: it starts the text, a line, or follows the end of a
// previous sentence, and is followed by the end / whitespace — so "My Style: x." or "Style: x.y" (the user's own
// words) never count. -1 when absent.
function findSentence(text, sentence) {
  if (!sentence) return -1;
  for (let i = text.lastIndexOf(sentence); i !== -1; i = i ? text.lastIndexOf(sentence, i - 1) : -1) {
    const pre = text.slice(0, i), end = i + sentence.length;
    const startsSentence = !pre.trim() || (/\s$/.test(pre) && /(\n|[.!?…])\s*$/.test(pre));
    if (startsSentence && (end === text.length || /\s/.test(text[end]))) return i;
  }
  return -1;
}

// Put `next` (the wizard's sentence, or "" for none) into `caption`. The previous wizard sentence `prev` is
// replaced where it stands; if it isn't found verbatim (the user edited or removed it) it's no longer the
// wizard's, so it's left alone and `next` is appended. Nothing else in the caption is touched.
export function applyCaptionSentence(caption, prev, next) {
  let text = String(caption || "");
  const at = findSentence(text, prev);                      // the wizard appends, so its copy is the last one
  if (at !== -1) {
    const before = text.slice(0, at), after = text.slice(at + prev.length);
    if (next) return before + next + after;
    // removing: join the two sides with one space (or one line break), no trailing whitespace
    const b = before.replace(/[ \t]+$/, ""), a = after.replace(/^[ \t]+/, "");
    const sep = b && a && !b.endsWith("\n") && !a.startsWith("\n") ? " " : "";
    return (b.endsWith("\n") && a.startsWith("\n") ? b + a.slice(1) : b + sep + a).replace(/\s+$/, "");
  }
  if (!next) return text;
  text = text.replace(/[ \t]+$/, "");
  if (!text.trim()) return next;
  if (text.includes("\n")) return text.replace(/\s+$/, "") + "\n" + next;   // multi-line caption: its own line
  return text + " " + next;
}

// Merge the wizard's tags into the builder's meta field (comma / newline separated). Tags the wizard added on
// the previous apply (`owned`) are removed first — exactly as it wrote them, once each, so a user tag that only
// differs in case stays; a tag the user already has is kept as theirs (not owned). Nothing to remove or add ->
// the field is returned untouched; otherwise a newline-only field stays one tag per line and anything else is
// written comma-joined. Returns { meta, owned } — the new value and the tags this apply added.
export function applyMetaTags(meta, owned, tags) {
  const text = String(meta || "");
  const all = text.split(/[,\n]/).map((t) => t.trim()).filter(Boolean);
  const kept = all.slice();
  for (const t of strings(owned)) { const i = kept.lastIndexOf(t); if (i !== -1) kept.splice(i, 1); }
  const have = new Set(kept.map((t) => t.toLowerCase()));
  const added = [];
  for (const t of strings(tags)) {
    if (!have.has(t.toLowerCase())) { have.add(t.toLowerCase()); added.push(t); }
  }
  if (kept.length === all.length && !added.length) return { meta: text, owned: [] };
  const sep = text.includes("\n") && !text.includes(",") ? "\n" : ", ";
  return { meta: [...kept, ...added].join(sep), owned: added };
}

const strings = (v) => (Array.isArray(v) ? v.filter((t) => typeof t === "string" && t.trim()) : []);

// One apply on one builder. `record` is what the wizard wrote there last time ({ sentence, tags, added }, kept on
// the builder node so it travels with it on copy/paste). If the picks haven't changed since then, nothing is
// written: the user may have edited or deleted the sentence / tags in the meantime, and closing the wizard just
// to look must not undo that. `force` (an explicit Apply click) writes anyway, e.g. to restore a deleted
// sentence. A malformed record counts as none. Returns { caption, meta, record }.
export function applyMossStyle(fields, record, wiz, { force = false } = {}) {
  const r = record && typeof record === "object" && !Array.isArray(record) ? record : {};
  const prev = { sentence: typeof r.sentence === "string" ? r.sentence : "", tags: strings(r.tags), added: strings(r.added) };
  const sentence = mossStyleSentence(wiz), tags = mossMetaTags(wiz);
  const same = (a, b) => a.length === b.length && a.every((t, i) => t.toLowerCase() === b[i].toLowerCase());
  if (!force && record && prev.sentence === sentence && same(prev.tags, tags)) {
    return { caption: fields.caption, meta: fields.meta, record: prev };
  }
  const m = applyMetaTags(fields.meta, prev.added, tags);
  // nothing recorded yet but the caption already has exactly this sentence (a pasted / template caption): adopt it
  const prevSentence = prev.sentence || (findSentence(String(fields.caption || ""), sentence) !== -1 ? sentence : "");
  return { caption: applyCaptionSentence(fields.caption, prevSentence, sentence), meta: m.meta,
           record: { sentence, tags, added: m.owned } };
}
