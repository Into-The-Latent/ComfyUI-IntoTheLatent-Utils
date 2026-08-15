/*
 * Part of ComfyUI-IntoTheLatent-Utils.
 *
 * Shared front-end for the six Multi Loader nodes (image/audio/video, each simple + Advanced).
 * GPL-3.0, like the rest of the pack.
 *
 * Files dropped/picked are uploaded once to ComfyUI's input/ folder; the hidden `files_json`
 * widget (a JSON array of {name, subfolder, type, enabled}) is the single source of truth for
 * save, restore and execution — the Prompt Batch pattern. A row's `enabled` flag (per-row
 * pill toggle, default true) holds its socket position when off — the file is skipped but the
 * slot isn't removed, so later files don't shift up; see nodes/multi_loader_core.py for why
 * feeding None to a connected OPTIONAL input is safe (it is NOT safe for a REQUIRED input).
 * The Python schema declares MAX_FILES
 * output groups; syncOutputs() trims node.outputs to `count` + a slot count and re-adds up to
 * the ceiling when the slot count grows. That slot count is `output_slots` in "auto" mode (it
 * tracks the file count) or a pinned number, so wires on fixed sockets survive file-list edits.
 * Trimming only ever cuts from the end — ComfyUI validates output types by slot position, so
 * used slots must stay contiguous from slot 0.
 * parseFiles mirrors parse_files in nodes/multi_loader_core.py — keep the two in sync.
 */
import { chainCallback } from "./utility.js";
const { app } = window.comfyAPI.app;

const MAX_FILES = 8;

// group: [prefix, TYPE] per output within one file's group, in schema order.
const NODES = {
  ITLMultiImageLoader:         { kind: "image", group: [["image_", "IMAGE"]] },
  ITLMultiImageLoaderAdvanced: { kind: "image", group: [["image_", "IMAGE"], ["mask_", "MASK"], ["filename_", "STRING"]] },
  ITLMultiAudioLoader:         { kind: "audio", group: [["audio_", "AUDIO"]] },
  ITLMultiAudioLoaderAdvanced: { kind: "audio", group: [["audio_", "AUDIO"], ["filename_", "STRING"]] },
  ITLMultiVideoLoader:         { kind: "video", group: [["video_", "VIDEO"], ["audio_", "AUDIO"]] },
  ITLMultiVideoLoaderAdvanced: { kind: "video", group: [["video_", "VIDEO"], ["audio_", "AUDIO"], ["filename_", "STRING"]] },
};

// ── Mirror of parse_files in nodes/multi_loader_core.py — three intentional deviations:
// (1) empty files_json is OK here (Python raises "No files loaded" only at run time),
// (2) entries beyond MAX_FILES are silently ignored here instead of raising (Python rejects
// the whole list with a "too many files" error), and
// (3) a non-boolean 'enabled' is coerced to true here instead of raising (Python rejects it) —
// the row list should still render from a hand-edited/older files_json. ──
function parseFiles(raw) {
  const text = (raw || "").trim();
  if (!text) return { ok: true, files: [] }; // empty is fine in the UI; Python rejects at run time
  let data;
  try { data = JSON.parse(text); } catch (e) { return { ok: false, error: "Malformed files_json: " + e.message }; }
  if (!Array.isArray(data)) return { ok: false, error: "Expected a JSON array of files." };
  const files = [];
  for (let i = 0; i < data.length && i < MAX_FILES; i++) {
    const e = data[i];
    if (!e || typeof e !== "object" || typeof e.name !== "string" || !e.name.trim()) {
      return { ok: false, error: `File #${i + 1}: 'name' must be a non-empty string.` };
    }
    files.push({
      name: e.name,
      subfolder: typeof e.subfolder === "string" ? e.subfolder : "",
      type: "input",
      enabled: typeof e.enabled === "boolean" ? e.enabled : true,
    });
  }
  return { ok: true, files };
}

const findWidget = (node, name) => node.widgets?.find((w) => w.name === name);

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

function ensureStyles() {
  if (document.getElementById("itl-bl-style")) return;
  const s = document.createElement("style");
  s.id = "itl-bl-style";
  s.textContent = `
  .itl-bl-drop{box-sizing:border-box;width:100%;padding:10px;margin:2px 0;text-align:center;
    font:11.5px -apple-system,"Segoe UI",Roboto,sans-serif;color:#7ab8e6;background:#1d2733;
    border:1px dashed #46b4e6;border-radius:8px;cursor:pointer}
  .itl-bl-drop:hover{background:#24384c}
  .itl-bl-drop.over{background:#24384c;border-style:solid}
  .itl-bl{width:100%;box-sizing:border-box;
    font:12px/1.4 -apple-system,"Segoe UI",Roboto,sans-serif;color:#d3d3d0}
  .itl-bl-content{display:flex;flex-direction:column;gap:5px;width:100%}
  .itl-bl .bl-row{display:flex;align-items:center;gap:7px;background:#262625;
    border:1px solid #3a3a38;border-radius:8px;padding:5px 7px}
  .itl-bl .bl-row.bl-drag{opacity:.45}
  .itl-bl .bl-row.bl-over{border-color:#46b4e6;box-shadow:0 0 0 1px #46b4e6 inset}
  .itl-bl .bl-row.bl-off{opacity:.5}
  .itl-bl .bl-grip{color:#6d6d68;font-size:14px;cursor:grab;user-select:none;flex:none}
  .itl-bl .bl-num{flex:none;width:18px;height:18px;border-radius:50%;background:#333331;
    color:#8b8b86;font:600 10px/18px ui-monospace,Consolas,monospace;text-align:center}
  .itl-bl .bl-thumb{flex:none;width:34px;height:34px;border-radius:4px;object-fit:cover;background:#1a1a19}
  .itl-bl .bl-thumb.bl-off{filter:grayscale(1)}
  .itl-bl .bl-wave{flex:none;width:34px;height:34px;border-radius:4px;background:#13332b;
    color:#46cca8;font-size:15px;line-height:34px;text-align:center}
  .itl-bl .bl-wave.bl-off{background:#2b2b29;color:#6d6d68}
  .itl-bl .bl-clap{flex:none;width:34px;height:34px;border-radius:4px;background:#2a1f3d;
    color:#b48ce6;font-size:15px;line-height:34px;text-align:center}
  .itl-bl .bl-clap.bl-off{background:#2b2b29;color:#6d6d68}
  .itl-bl .bl-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11.5px}
  .itl-bl .bl-meta{flex:none;color:#8b8b86;font:10px ui-monospace,Consolas,monospace}
  .itl-bl .bl-x{flex:none;color:#6d6d68;font-size:13px;cursor:pointer;padding:2px}
  .itl-bl .bl-x:hover{color:#c86b6b}
  .itl-bl .bl-empty{padding:6px;text-align:center;color:#6d6d68;font-size:11px}
  .itl-bl .bl-toggle{flex:none;width:26px;height:14px;border-radius:7px;background:#3a3a38;
    border:1px solid #4a4a47;cursor:pointer;position:relative;box-sizing:border-box;padding:0}
  .itl-bl .bl-toggle .bl-knob{position:absolute;top:1px;left:1px;width:10px;height:10px;
    border-radius:50%;background:#8b8b86;transition:left .12s ease,background .12s ease}
  .itl-bl .bl-toggle.bl-on{background:#1d3644;border-color:#46b4e6}
  .itl-bl .bl-toggle.bl-on .bl-knob{left:13px;background:#46b4e6}
  .itl-bl .bl-header{display:flex;align-items:center;gap:7px;padding:2px 7px 5px}
  .itl-bl .bl-header .bl-toggle-all-label{flex:1;color:#8b8b86;font:600 10.5px -apple-system,"Segoe UI",Roboto,sans-serif;
    text-transform:uppercase;letter-spacing:.03em}
  `;
  document.head.appendChild(s);
}

// Trim node.outputs to count + slotCount groups; re-add (in schema order) up to the ceiling.
// slotCount is the number of sockets to *show* — in "auto" mode that's the file count, but a
// pinned output_slots value overrides it so sockets (and wires) survive file-list edits.
// removeOutput disconnects any links on the removed slot — that is the intended behavior.
function syncOutputs(node, cfg, slotCount) {
  const want = 1 + Math.min(slotCount, MAX_FILES) * cfg.group.length;
  while (node.outputs.length > want) node.removeOutput(node.outputs.length - 1);
  while (node.outputs.length < want) {
    const slot = node.outputs.length;                     // next slot index to create
    const gi = (slot - 1) % cfg.group.length;             // position within the file's group
    const fi = Math.floor((slot - 1) / cfg.group.length) + 1;  // 1-based file number
    const [prefix, type] = cfg.group[gi];
    node.addOutput(prefix + fi, type);
  }
  node.setDirtyCanvas?.(true, true);
}

// Upload one File to ComfyUI's input/ via the same endpoint the stock upload widget uses.
// The endpoint is generic despite its name; the form field is "image" even for audio.
async function uploadFile(file) {
  const form = new FormData();
  form.append("image", file);
  form.append("type", "input");
  const api = window.comfyAPI?.api?.api;
  const res = api?.fetchApi
    ? await api.fetchApi("/upload/image", { method: "POST", body: form })
    : await fetch("/upload/image", { method: "POST", body: form });
  if (res.status !== 200) throw new Error(`upload of ${file.name} failed (HTTP ${res.status})`);
  const data = await res.json();
  return { name: data.name, subfolder: data.subfolder || "", type: "input", enabled: true };
}

app.registerExtension({
  name: "ITL.MultiLoader",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = NODES[nodeData?.name];
    if (!cfg) return;
    ensureStyles();

    const MIRRORED = ["downscale_mode", "max_size", "output_slots", "force_rate"];   // downscale_mode/max_size are image-only, force_rate is video-only; findWidget just misses on the other kinds

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const jsonW = findWidget(node, "files_json");
      hideWidget(jsonW);
      node._blRows = [];

      // INT guard (image nodes): a number widget can serialize '' which fails INT validation.
      const maxSizeW = findWidget(node, "max_size");
      if (maxSizeW) {
        maxSizeW.serializeValue = () => {
          const v = parseInt(maxSizeW.value, 10);
          return Number.isFinite(v) && v >= 8 ? v : 1200;
        };
      }

      // FLOAT guard (video nodes): same '' -> validation-failure trap as max_size above, but
      // for a FLOAT widget; 0 is the sentinel meaning "off" so a bad value coerces to 0, not
      // some non-zero default.
      const forceRateW = findWidget(node, "force_rate");
      if (forceRateW) {
        forceRateW.serializeValue = () => {
          const v = parseFloat(forceRateW.value);
          return Number.isFinite(v) && v >= 0 ? v : 0.0;
        };
      }

      // Status line (read-only DOM widget).
      const statusEl = document.createElement("div");
      statusEl.style.cssText = "width:100%;box-sizing:border-box;padding:3px 6px;text-align:center;line-height:1.4;font:12px sans-serif;";
      const setStatus = (text, color) => { statusEl.textContent = text; statusEl.style.color = color; node.setDirtyCanvas?.(true, true); };
      node._blSetStatus = setStatus;

      function syncJson() {
        if (jsonW) jsonW.value = JSON.stringify(node._blRows);
      }
      node._blSyncJson = syncJson;

      // Resolve how many sockets to show: "auto" (or a missing/unparsable value — e.g. a
      // restored workflow handing back null) follows the file count; otherwise the pinned
      // number, clamped to 1..MAX_FILES.
      function slotCount() {
        const w = findWidget(node, "output_slots");
        const raw = w?.value;
        if (!raw || raw === "auto") return node._blRows.length;
        const n = parseInt(raw, 10);
        return Number.isFinite(n) ? Math.max(1, Math.min(MAX_FILES, n)) : node._blRows.length;
      }
      node._blSyncOutputs = () => syncOutputs(node, cfg, slotCount());

      // In manual mode the socket count no longer tracks the file count; say so plainly, because a
      // wired-but-empty socket outputs nothing and fails at run time.
      function slotNote() {
        const w = findWidget(node, "output_slots");
        const raw = w?.value;
        if (!raw || raw === "auto") return "";
        const slots = Math.max(1, Math.min(MAX_FILES, parseInt(raw, 10) || 0));
        const n = node._blRows.length;
        // Format a range as "a" for a single item, "a-b" for a span.
        const formatRange = (start, end) => (start === end ? String(start) : `${start}-${end}`);
        if (n < slots) {
          const range = formatRange(n + 1, slots);
          const one = n + 1 === slots;
          return ` — socket${one ? "" : "s"} ${range} ${one ? "has" : "have"} no file yet; leave ${one ? "it" : "them"} unwired until you add more.`;
        }
        if (n > slots) {
          const range = formatRange(slots + 1, n);
          return ` — only ${slots} socket${slots === 1 ? "" : "s"} shown, so file${n - slots === 1 ? "" : "s"} ${range} can't be reached.`;
        }
        return "";
      }
      node._blSlotNote = slotNote;

      // Terse reminder appended to add/remove status lines when some rows are switched off —
      // those rows still occupy a socket but emit nothing, so it's worth flagging inline.
      function offSuffix() {
        const n = node._blRows.filter((f) => !f.enabled).length;
        return n ? ` (${n} off)` : "";
      }
      node._blOffSuffix = offSuffix;

      // Changing the pinned slot count re-syncs sockets immediately and reports the new
      // file/socket mismatch (if any) in the status line.
      const slotsW = findWidget(node, "output_slots");
      if (slotsW) {
        slotsW.callback = () => {
          node._blSyncOutputs();
          node._blRender?.();
          setStatus(`Output slots: ${slotsW.value}.${slotNote()}`, "#8a8a8a");
        };
      }

      // Audio containers often self-report as video/* (e.g. .ogg) or with no MIME at all on
      // Windows for unregistered extensions; the backend decoder handles both fine, so accept
      // them here too. Video containers can likewise report no MIME on Windows for
      // unregistered extensions. Images: only the empty-MIME case needs the same allowance.
      const acceptFile = cfg.kind === "audio"
        ? (f) => f.type.startsWith("audio/") || f.type.startsWith("video/") || f.type === ""
        : cfg.kind === "video"
        ? (f) => f.type.startsWith("video/") || f.type === ""
        : (f) => f.type.startsWith("image/") || f.type === "";

      async function addFiles(fileList) {
        const files = [...fileList].filter(acceptFile);
        const rejected = fileList.length - files.length;
        const free = MAX_FILES - node._blRows.length;
        const taking = files.slice(0, free);
        let skipped = files.length - taking.length;
        let failed = null;
        // taking.length is capped against a snapshot of node._blRows.length taken above; if
        // another drop is uploading concurrently (interleaved awaits), that snapshot goes
        // stale. Re-check right before each push so two overlapping drops can't push the
        // node past MAX_FILES between them.
        for (let idx = 0; idx < taking.length; idx++) {
          const f = taking[idx];
          let uploaded;
          try {
            uploaded = await uploadFile(f);
          } catch (e) {
            failed = e.message;
            break;
          }
          if (node._blRows.length >= MAX_FILES) {
            skipped += taking.length - idx; // safety net: a concurrent drop filled the node while this file was uploading
            break;
          }
          node._blRows.push(uploaded);
        }
        syncJson();
        node._blSyncOutputs();
        node._blRender?.();
        if (failed) {
          setStatus(`❌ ${failed} — ${node._blRows.length} file${node._blRows.length === 1 ? "" : "s"} loaded before the error.${offSuffix()}`, "#e0555a");
        } else {
          const parts = [`${node._blRows.length} file${node._blRows.length === 1 ? "" : "s"} loaded`];
          if (skipped) parts.push(`only ${MAX_FILES} fit — ${skipped} skipped`);
          if (rejected) parts.push(`${rejected} not ${cfg.kind} — ignored`);
          setStatus((skipped || rejected ? "⚠ " : "✅ ") + parts.join("; ") + offSuffix() + slotNote(), skipped || rejected ? "#e0a03c" : "#46b4e6");
        }
      }
      node._blAddFiles = addFiles;

      // ── Drop zone (DOM widget): also doubles as the file picker (click to browse), so there is
      // one field for both gestures instead of a drop zone plus a separate ＋ Add button.
      // stopPropagation beats ComfyUI's global drop handler, which would otherwise try to load
      // the files as a workflow. ──
      const dropEl = document.createElement("div");
      dropEl.className = "itl-bl-drop";
      dropEl.textContent = cfg.kind === "image" ? "Drop images here — or click to browse"
        : cfg.kind === "audio" ? "Drop audio here — or click to browse"
        : "Drop videos here — or click to browse";
      for (const ev of ["dragenter", "dragover"]) {
        dropEl.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropEl.classList.add("over"); });
      }
      dropEl.addEventListener("dragleave", () => dropEl.classList.remove("over"));
      dropEl.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation();
        dropEl.classList.remove("over");
        if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
      });
      dropEl.addEventListener("click", () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.multiple = true;
        picker.accept = cfg.kind === "audio" ? "audio/*,video/*" : cfg.kind === "video" ? "video/*" : "image/*";
        picker.onchange = () => picker.files?.length && addFiles(picker.files);
        picker.click();
      });
      node._blDropEl = dropEl;
      node.addDOMWidget("multi_loader_drop", "drop", dropEl, { serialize: false });

      // ── Rows list (DOM widget): one row per file, in socket order. listEl is the widget root
      // (ComfyUI pins its height each frame); the rows live in contentEl, whose *natural* height
      // we measure so the node can both grow and shrink. Measuring listEl instead would report
      // the pinned height back to itself, so the node could never learn it needs to grow (or
      // shrink) — the prompt_batch.js editor/content split, applied here. ──
      const listEl = document.createElement("div");
      listEl.className = "itl-bl";
      const contentEl = document.createElement("div");
      contentEl.className = "itl-bl-content";
      listEl.append(contentEl);
      const rowsWidget = node.addDOMWidget("multi_loader_rows", "rows", listEl, { serialize: false });
      let dragIndex = -1;

      // Auto-fit node height to the rows (measured; the prompt_batch pattern). Measure contentEl
      // (natural height), NOT listEl (height pinned by ComfyUI) — otherwise a removed row
      // couldn't shrink the node, because the pinned element keeps its old, larger height.
      function fitToContent() {
        const h = Math.max(contentEl.scrollHeight, 8);
        rowsWidget.computeSize = () => [node.size?.[0] || 300, h + 8];
        const want = node.computeSize?.();
        if (want) node.setSize([node.size[0], want[1]]);
        node.setDirtyCanvas?.(true, true);
      }
      let lastFitH = 0;
      const ro = new ResizeObserver(() => {
        const h = contentEl.scrollHeight;
        if (h && h !== lastFitH) { lastFitH = h; fitToContent(); }
      });
      ro.observe(contentEl);
      chainCallback(node, "onRemoved", () => ro.disconnect());

      const viewUrl = (f) =>
        `/view?filename=${encodeURIComponent(f.name)}&type=input&subfolder=${encodeURIComponent(f.subfolder)}`;

      // Any wire on any group socket means a reorder/delete changes what flows where.
      const anyGroupWired = () =>
        (node.outputs || []).slice(1).some((o) => o.links && o.links.length);

      // Pill toggle (Power Lora Loader style): rounded track + sliding knob, on = accent
      // colour, off = muted. Click stops propagation so it can never arm the row's drag
      // (the grip is the only drag affordance) or bubble into a row/header click.
      function makeToggle(checked, title, onToggle) {
        const el = document.createElement("span");
        el.className = "bl-toggle" + (checked ? " bl-on" : "");
        el.title = title;
        const knob = document.createElement("span");
        knob.className = "bl-knob";
        el.appendChild(knob);
        el.addEventListener("mousedown", (e) => e.stopPropagation());
        el.addEventListener("click", (e) => {
          e.stopPropagation();
          onToggle();
        });
        return el;
      }

      // Flip one row's enabled flag. A disabled row keeps its socket position — the Python
      // side emits None for it instead of shifting later rows up (hold-position semantics).
      function toggleRow(k) {
        const f = node._blRows[k];
        f.enabled = !f.enabled;
        node._blSyncJson(); render();
        node._blSetStatus(
          `${f.enabled ? "Enabled" : "Disabled"} ${f.name}${f.enabled ? "" : " — its socket now outputs nothing"}.${offSuffix()}`,
          "#8a8a8a",
        );
      }

      // Toggle All: on when every row is enabled; clicking flips ALL rows to the opposite of
      // that state (any row off -> turn everything on; all on -> turn everything off).
      function toggleAllRows() {
        const allOn = node._blRows.length > 0 && node._blRows.every((r) => r.enabled);
        const next = !allOn;
        node._blRows.forEach((r) => { r.enabled = next; });
        node._blSyncJson(); render();
        const n = node._blRows.length;
        node._blSetStatus(`${next ? "Enabled" : "Disabled"} all ${n} row${n === 1 ? "" : "s"}.${offSuffix()}`, "#8a8a8a");
      }

      function removeAt(k) {
        const removed = node._blRows[k].name;
        const moved = node._blRows.slice(k + 1).map((f) => f.name);   // mirror of core remove_file
        const start = 1 + k * cfg.group.length;
        const disturbed = (node.outputs || []).slice(start).some((o) => o.links && o.links.length);
        node._blRows.splice(k, 1);
        node._blSyncJson(); node._blSyncOutputs(); render();
        if (disturbed && moved.length) {
          node._blSetStatus(`⚠ Removed ${removed} — ${moved.join(", ")} moved up a slot. Check your wires.${offSuffix()}${slotNote()}`, "#e0a03c");
        } else if (disturbed) {
          node._blSetStatus(`⚠ Removed ${removed} — its socket was wired; that connection is gone.${offSuffix()}${slotNote()}`, "#e0a03c");
        } else {
          node._blSetStatus(`Removed ${removed}.${offSuffix()}${slotNote()}`, "#8a8a8a");
        }
      }

      function render() {
        contentEl.replaceChildren();
        if (!node._blRows.length) {
          const empty = document.createElement("div");
          empty.className = "bl-empty";
          empty.textContent = "No files loaded.";
          contentEl.appendChild(empty);
          return;
        }

        // Toggle All header — reads the aggregate state of the rows below it, not its own.
        const allOn = node._blRows.every((r) => r.enabled);
        const header = document.createElement("div");
        header.className = "bl-header";
        const headerToggle = makeToggle(allOn, allOn ? "Turn all off" : "Turn all on", toggleAllRows);
        const headerLabel = document.createElement("span");
        headerLabel.className = "bl-toggle-all-label";
        headerLabel.textContent = "Toggle All";
        header.append(headerToggle, headerLabel);
        contentEl.appendChild(header);

        node._blRows.forEach((f, k) => {
          const row = document.createElement("div");
          row.className = "bl-row" + (f.enabled ? "" : " bl-off");

          const toggle = makeToggle(f.enabled, f.enabled ? "Disable this row" : "Enable this row", () => toggleRow(k));

          const grip = document.createElement("span");
          grip.className = "bl-grip";
          grip.textContent = "⠿";
          grip.title = "Drag to reorder";
          grip.addEventListener("mousedown", () => { row.draggable = true; });
          row.addEventListener("mouseup", () => { row.draggable = false; });
          row.addEventListener("dragstart", (e) => { dragIndex = k; e.dataTransfer.effectAllowed = "move"; row.classList.add("bl-drag"); });
          row.addEventListener("dragend", () => { row.draggable = false; dragIndex = -1; row.classList.remove("bl-drag"); contentEl.querySelectorAll(".bl-over").forEach((n) => n.classList.remove("bl-over")); });
          row.addEventListener("dragover", (e) => { e.preventDefault(); if (dragIndex > -1 && dragIndex !== k) row.classList.add("bl-over"); });
          row.addEventListener("dragleave", () => row.classList.remove("bl-over"));
          row.addEventListener("drop", (e) => {
            e.preventDefault(); e.stopPropagation();
            row.classList.remove("bl-over");
            if (dragIndex > -1 && dragIndex !== k) {
              const wired = anyGroupWired();
              const [movedRow] = node._blRows.splice(dragIndex, 1);
              node._blRows.splice(k, 0, movedRow);
              node._blSyncJson(); render();
              if (wired) node._blSetStatus("⚠ Reordered — sockets now carry different files. Check your wires.", "#e0a03c");
            }
          });

          const num = document.createElement("span");
          num.className = "bl-num";
          num.textContent = String(k + 1);

          let preview;
          const meta = document.createElement("span");
          meta.className = "bl-meta";
          if (cfg.kind === "image") {
            preview = document.createElement("img");
            preview.className = "bl-thumb" + (f.enabled ? "" : " bl-off");
            preview.src = viewUrl(f);
            preview.addEventListener("load", () => { meta.textContent = `${preview.naturalWidth}×${preview.naturalHeight}`; });
          } else if (cfg.kind === "video") {
            preview = document.createElement("span");
            preview.className = "bl-clap" + (f.enabled ? "" : " bl-off");
            preview.textContent = "🎬";
            // Like the Audio() probe below, this element is never appended to the DOM —
            // loadedmetadata fires from the network fetch alone, no layout needed.
            const probe = document.createElement("video");
            probe.preload = "metadata";
            probe.src = viewUrl(f);
            probe.addEventListener("loadedmetadata", () => {
              const s = Math.round(probe.duration);
              const time = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
              meta.textContent = `${time} · ${probe.videoWidth}×${probe.videoHeight}`;
            });
          } else {
            preview = document.createElement("span");
            preview.className = "bl-wave" + (f.enabled ? "" : " bl-off");
            preview.textContent = "♪";
            const probe = new Audio();
            probe.preload = "metadata";
            probe.src = viewUrl(f);
            probe.addEventListener("loadedmetadata", () => {
              const s = Math.round(probe.duration);
              meta.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
            });
          }

          const name = document.createElement("span");
          name.className = "bl-name";
          name.textContent = f.name;
          name.title = (f.subfolder ? f.subfolder + "/" : "") + f.name;

          const x = document.createElement("span");
          x.className = "bl-x";
          x.textContent = "✕";
          x.title = "Remove this file";
          x.addEventListener("click", () => removeAt(k));

          row.append(toggle, grip, num, preview, name, meta, x);
          contentEl.appendChild(row);
        });
      }
      node._blRender = render;

      const sortBtn = node.addWidget("button", "Sort by name", null, () => {
        if (node._blRows.length < 2) return;
        const wired = anyGroupWired();
        node._blRows.sort((a, b) => a.name.localeCompare(b.name));
        node._blSyncJson(); render();
        node._blSetStatus(wired ? "⚠ Sorted — sockets now carry different files. Check your wires." : "Sorted by name.", wired ? "#e0a03c" : "#8a8a8a");
      });
      sortBtn.serialize = false;

      render();

      node.addDOMWidget("multi_loader_status", "info", statusEl, { serialize: false });
      setStatus(`Drop ${cfg.kind} files here, or click the field to browse.`, "#8a8a8a");

      // Fresh node: no files yet -> trim the declared ceiling down to just `count`.
      node._blSyncOutputs();
    });

    // After a workflow loads: rebuild rows from the restored files_json, then re-trim.
    // (Serialized nodes save their trimmed outputs, so this is normally a no-op — it heals
    // hand-edited or older workflows.)
    chainCallback(nodeType.prototype, "onConfigure", function (info) {
      const node = this;
      requestAnimationFrame(() => {
        const mirror = info?.properties?.itl_bl;
        if (mirror && typeof mirror === "object") {
          for (const name of MIRRORED) {
            const w = findWidget(node, name);
            if (w && mirror[name] !== undefined) w.value = mirror[name];
          }
        }
        const res = parseFiles(findWidget(node, "files_json")?.value);
        node._blRows = res.ok ? res.files : [];
        if (res.ok) {
          node._blSyncJson?.();
        } else {
          // Don't overwrite a possibly hand-edited files_json with "[]" — leave the widget
          // value alone so the user can see and fix what's actually there.
          node._blSetStatus?.(`❌ ${res.error}`, "#e0555a");
        }
        node._blSyncOutputs?.();
        node._blRender?.();
      });
    });

    chainCallback(nodeType.prototype, "onSerialize", function (o) {
      const mirror = {};
      for (const name of MIRRORED) {
        const w = findWidget(this, name);
        if (w) mirror[name] = w.value;
      }
      o.properties = o.properties || {};
      o.properties.itl_bl = mirror;
    });
  },
});
