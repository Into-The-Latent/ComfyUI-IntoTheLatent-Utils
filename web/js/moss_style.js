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

// Put `next` (the wizard's sentence, or "" for none) into `caption`. The previous wizard sentence `prev` is
// replaced where it stands; if it isn't found verbatim (the user edited or removed it) it's no longer the
// wizard's, so it's left alone and `next` is appended. Nothing else in the caption is touched.
export function applyCaptionSentence(caption, prev, next) {
  let text = String(caption || "");
  const at = prev ? text.indexOf(prev) : -1;
  if (at !== -1) {
    const before = text.slice(0, at), after = text.slice(at + prev.length);
    if (next) return before + next + after;
    // removing: join the two sides with one space (or keep their line break), no trailing whitespace
    const b = before.replace(/[ \t]+$/, ""), a = after.replace(/^[ \t]+/, "");
    const sep = b && a && !b.endsWith("\n") && !a.startsWith("\n") ? " " : "";
    return (b + sep + a).replace(/\s+$/, "");
  }
  if (!next) return text;
  text = text.replace(/[ \t]+$/, "");
  if (!text.trim()) return next;
  if (/\n\s*$/.test(text) || text.includes("\n")) return text.replace(/\s+$/, "") + "\n" + next;
  return text + " " + next;
}

// Merge the wizard's tags into the builder's meta field (comma / newline separated). Tags the wizard added on
// the previous apply (`owned`) are removed first; a tag the user already has is kept as theirs (not owned).
// Returns { meta, owned } — the new field value and the tags this apply added.
export function applyMetaTags(meta, owned, tags) {
  const ownedLc = new Set((owned || []).map((t) => t.toLowerCase()));
  const kept = String(meta || "").split(/[,\n]/).map((t) => t.trim()).filter((t) => t && !ownedLc.has(t.toLowerCase()));
  const have = new Set(kept.map((t) => t.toLowerCase()));
  const added = [];
  for (const t of tags || []) {
    if (!have.has(t.toLowerCase())) { have.add(t.toLowerCase()); added.push(t); }
  }
  return { meta: [...kept, ...added].join(", "), owned: added };
}
