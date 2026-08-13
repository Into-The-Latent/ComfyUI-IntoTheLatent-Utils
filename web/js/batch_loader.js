/*
 * Part of ComfyUI-AI2Go-Utils.
 *
 * Shared front-end for the four Batch Loader nodes. GPL-3.0, like the rest of the pack.
 *
 * Files dropped/picked are uploaded once to ComfyUI's input/ folder; the hidden `files_json`
 * widget (a JSON array of {name, subfolder, type}) is the single source of truth for save,
 * restore and execution — the Prompt Batch pattern. The Python schema declares MAX_FILES
 * output groups; syncOutputs() trims node.outputs to `count` + the loaded groups and re-adds
 * up to the ceiling when files come back. Trimming only ever cuts from the end — ComfyUI
 * validates output types by slot position, so used slots must stay contiguous from slot 0.
 * parseFiles mirrors parse_files in nodes/batch_loader_core.py — keep the two in sync.
 */
import { chainCallback } from "./utility.js";
const { app } = window.comfyAPI.app;

const MAX_FILES = 8;

// group: [prefix, TYPE] per output within one file's group, in schema order.
const NODES = {
  AI2GoBatchImageLoader:         { kind: "image", group: [["image_", "IMAGE"]] },
  AI2GoBatchImageLoaderAdvanced: { kind: "image", group: [["image_", "IMAGE"], ["mask_", "MASK"], ["filename_", "STRING"]] },
  AI2GoBatchAudioLoader:         { kind: "audio", group: [["audio_", "AUDIO"]] },
  AI2GoBatchAudioLoaderAdvanced: { kind: "audio", group: [["audio_", "AUDIO"], ["filename_", "STRING"]] },
};

// ── Mirror of parse_files in nodes/batch_loader_core.py ──
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
    files.push({ name: e.name, subfolder: typeof e.subfolder === "string" ? e.subfolder : "", type: "input" });
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
  if (document.getElementById("ai2go-bl-style")) return;
  const s = document.createElement("style");
  s.id = "ai2go-bl-style";
  s.textContent = `
  .ai2go-bl-drop{box-sizing:border-box;width:100%;padding:10px;margin:2px 0;text-align:center;
    font:11.5px -apple-system,"Segoe UI",Roboto,sans-serif;color:#7ab8e6;background:#1d2733;
    border:1px dashed #46b4e6;border-radius:8px;cursor:copy}
  .ai2go-bl-drop.over{background:#24384c;border-style:solid}
  .ai2go-bl{display:flex;flex-direction:column;gap:5px;width:100%;box-sizing:border-box;
    font:12px/1.4 -apple-system,"Segoe UI",Roboto,sans-serif;color:#d3d3d0}
  .ai2go-bl .bl-row{display:flex;align-items:center;gap:7px;background:#262625;
    border:1px solid #3a3a38;border-radius:8px;padding:5px 7px}
  .ai2go-bl .bl-row.bl-drag{opacity:.45}
  .ai2go-bl .bl-row.bl-over{border-color:#46b4e6;box-shadow:0 0 0 1px #46b4e6 inset}
  .ai2go-bl .bl-grip{color:#6d6d68;font-size:14px;cursor:grab;user-select:none;flex:none}
  .ai2go-bl .bl-num{flex:none;width:18px;height:18px;border-radius:50%;background:#333331;
    color:#8b8b86;font:600 10px/18px ui-monospace,Consolas,monospace;text-align:center}
  .ai2go-bl .bl-thumb{flex:none;width:34px;height:34px;border-radius:4px;object-fit:cover;background:#1a1a19}
  .ai2go-bl .bl-wave{flex:none;width:34px;height:34px;border-radius:4px;background:#13332b;
    color:#46cca8;font-size:15px;line-height:34px;text-align:center}
  .ai2go-bl .bl-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11.5px}
  .ai2go-bl .bl-meta{flex:none;color:#8b8b86;font:10px ui-monospace,Consolas,monospace}
  .ai2go-bl .bl-x{flex:none;color:#6d6d68;font-size:13px;cursor:pointer;padding:2px}
  .ai2go-bl .bl-x:hover{color:#c86b6b}
  .ai2go-bl .bl-empty{padding:6px;text-align:center;color:#6d6d68;font-size:11px}
  `;
  document.head.appendChild(s);
}

// Trim node.outputs to count + fileCount groups; re-add (in schema order) up to the ceiling.
// removeOutput disconnects any links on the removed slot — that is the intended behavior.
function syncOutputs(node, cfg, fileCount) {
  const want = 1 + Math.min(fileCount, MAX_FILES) * cfg.group.length;
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
  return { name: data.name, subfolder: data.subfolder || "", type: "input" };
}

app.registerExtension({
  name: "AI2Go.BatchLoader",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = NODES[nodeData?.name];
    if (!cfg) return;
    ensureStyles();

    const MIRRORED = ["downscale_mode", "max_size"];   // present on image nodes only; findWidget just misses on audio

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

      // Status line (read-only DOM widget).
      const statusEl = document.createElement("div");
      statusEl.style.cssText = "width:100%;box-sizing:border-box;padding:3px 6px;text-align:center;line-height:1.4;font:12px sans-serif;";
      const setStatus = (text, color) => { statusEl.textContent = text; statusEl.style.color = color; node.setDirtyCanvas?.(true, true); };
      node._blSetStatus = setStatus;

      function syncJson() {
        if (jsonW) jsonW.value = JSON.stringify(node._blRows);
      }
      node._blSyncJson = syncJson;
      node._blSyncOutputs = () => syncOutputs(node, cfg, node._blRows.length);

      async function addFiles(fileList) {
        const files = [...fileList].filter((f) => f.type.startsWith(cfg.kind + "/"));
        const rejected = fileList.length - files.length;
        const free = MAX_FILES - node._blRows.length;
        const taking = files.slice(0, free);
        const skipped = files.length - taking.length;
        let failed = null;
        for (const f of taking) {
          try {
            node._blRows.push(await uploadFile(f));
          } catch (e) {
            failed = e.message;
            break;
          }
        }
        syncJson();
        node._blSyncOutputs();
        node._blRender?.();
        if (failed) {
          setStatus(`❌ ${failed} — ${node._blRows.length} file${node._blRows.length === 1 ? "" : "s"} loaded before the error.`, "#e0555a");
        } else {
          const parts = [`${node._blRows.length} file${node._blRows.length === 1 ? "" : "s"} loaded`];
          if (skipped) parts.push(`only ${MAX_FILES} fit — ${skipped} skipped`);
          if (rejected) parts.push(`${rejected} not ${cfg.kind} — ignored`);
          setStatus((skipped || rejected ? "⚠ " : "✅ ") + parts.join("; "), skipped || rejected ? "#e0a03c" : "#46b4e6");
        }
      }
      node._blAddFiles = addFiles;

      // ── Drop zone (DOM widget). stopPropagation beats ComfyUI's global drop handler,
      // which would otherwise try to load the files as a workflow. ──
      const dropEl = document.createElement("div");
      dropEl.className = "ai2go-bl-drop";
      dropEl.textContent = cfg.kind === "image" ? "Drop images here" : "Drop audio here";
      for (const ev of ["dragenter", "dragover"]) {
        dropEl.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropEl.classList.add("over"); });
      }
      dropEl.addEventListener("dragleave", () => dropEl.classList.remove("over"));
      dropEl.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation();
        dropEl.classList.remove("over");
        if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
      });
      node._blDropEl = dropEl;
      node.addDOMWidget("batch_loader_drop", "drop", dropEl, { serialize: false });

      // ── Rows list (DOM widget): one row per file, in socket order. ──
      const listEl = document.createElement("div");
      listEl.className = "ai2go-bl";
      const rowsWidget = node.addDOMWidget("batch_loader_rows", "rows", listEl, { serialize: false });
      let dragIndex = -1;

      // Auto-fit node height to the rows (measured; the prompt_batch pattern).
      function fitToContent() {
        const h = Math.max(listEl.scrollHeight, 8);
        rowsWidget.computeSize = () => [node.size?.[0] || 300, h + 8];
        const want = node.computeSize?.();
        if (want) node.setSize([node.size[0], want[1]]);
        node.setDirtyCanvas?.(true, true);
      }
      let lastFitH = 0;
      const ro = new ResizeObserver(() => {
        const h = listEl.scrollHeight;
        if (h && h !== lastFitH) { lastFitH = h; fitToContent(); }
      });
      ro.observe(listEl);
      chainCallback(node, "onRemoved", () => ro.disconnect());

      const viewUrl = (f) =>
        `/view?filename=${encodeURIComponent(f.name)}&type=input&subfolder=${encodeURIComponent(f.subfolder)}`;

      // Any wire on any group socket means a reorder/delete changes what flows where.
      const anyGroupWired = () =>
        (node.outputs || []).slice(1).some((o) => o.links && o.links.length);

      function removeAt(k) {
        const removed = node._blRows[k].name;
        const moved = node._blRows.slice(k + 1).map((f) => f.name);   // mirror of core remove_file
        const start = 1 + k * cfg.group.length;
        const disturbed = (node.outputs || []).slice(start).some((o) => o.links && o.links.length);
        node._blRows.splice(k, 1);
        node._blSyncJson(); node._blSyncOutputs(); render();
        if (disturbed && moved.length) {
          node._blSetStatus(`⚠ Removed ${removed} — ${moved.join(", ")} moved up a slot. Check your wires.`, "#e0a03c");
        } else {
          node._blSetStatus(`Removed ${removed}.`, "#8a8a8a");
        }
      }

      function render() {
        listEl.replaceChildren();
        if (!node._blRows.length) {
          const empty = document.createElement("div");
          empty.className = "bl-empty";
          empty.textContent = "No files loaded.";
          listEl.appendChild(empty);
          return;
        }
        node._blRows.forEach((f, k) => {
          const row = document.createElement("div");
          row.className = "bl-row";

          const grip = document.createElement("span");
          grip.className = "bl-grip";
          grip.textContent = "⠿";
          grip.title = "Drag to reorder";
          grip.addEventListener("mousedown", () => { row.draggable = true; });
          row.addEventListener("mouseup", () => { row.draggable = false; });
          row.addEventListener("dragstart", (e) => { dragIndex = k; e.dataTransfer.effectAllowed = "move"; row.classList.add("bl-drag"); });
          row.addEventListener("dragend", () => { row.draggable = false; dragIndex = -1; row.classList.remove("bl-drag"); listEl.querySelectorAll(".bl-over").forEach((n) => n.classList.remove("bl-over")); });
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
            preview.className = "bl-thumb";
            preview.src = viewUrl(f);
            preview.addEventListener("load", () => { meta.textContent = `${preview.naturalWidth}×${preview.naturalHeight}`; });
          } else {
            preview = document.createElement("span");
            preview.className = "bl-wave";
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

          row.append(grip, num, preview, name, meta, x);
          listEl.appendChild(row);
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

      const addBtn = node.addWidget("button", cfg.kind === "image" ? "＋ Add images" : "＋ Add audio", null, () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.multiple = true;
        picker.accept = cfg.kind + "/*";
        picker.onchange = () => picker.files?.length && addFiles(picker.files);
        picker.click();
      });
      addBtn.serialize = false;

      node.addDOMWidget("batch_loader_status", "info", statusEl, { serialize: false });
      setStatus(`Drop ${cfg.kind} files here or press ＋ Add.`, "#8a8a8a");

      // Fresh node: no files yet -> trim the declared ceiling down to just `count`.
      node._blSyncOutputs();
    });

    // After a workflow loads: rebuild rows from the restored files_json, then re-trim.
    // (Serialized nodes save their trimmed outputs, so this is normally a no-op — it heals
    // hand-edited or older workflows.)
    chainCallback(nodeType.prototype, "onConfigure", function (info) {
      const node = this;
      requestAnimationFrame(() => {
        const mirror = info?.properties?.ai2go_bl;
        if (mirror && typeof mirror === "object") {
          for (const name of MIRRORED) {
            const w = findWidget(node, name);
            if (w && mirror[name] !== undefined) w.value = mirror[name];
          }
        }
        const res = parseFiles(findWidget(node, "files_json")?.value);
        node._blRows = res.ok ? res.files : [];
        node._blSyncJson?.();
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
      o.properties.ai2go_bl = mirror;
    });
  },
});
