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
    chainCallback(nodeType.prototype, "onConfigure", function () {
      const node = this;
      requestAnimationFrame(() => {
        const res = parseFiles(findWidget(node, "files_json")?.value);
        node._blRows = res.ok ? res.files : [];
        node._blSyncJson?.();
        node._blSyncOutputs?.();
        node._blRender?.();
      });
    });
  },
});
