/*
 * Part of ComfyUI-IntoTheLatent-Utils.
 *
 * Front-end for the ITL Prompt Batch node. GPL-3.0, like the rest of the pack.
 *
 * A dynamic row editor in the spirit of rgthree's Power Lora Loader: each prompt is a row with a
 * positive and a negative text box, drag-to-reorder (⠿) and per-row remove (🗑). Buttons:
 *   • 📥 Read from JSON — parse the node wired into the optional `json_in` socket and APPEND its
 *     prompts to the rows (pulled to the top, next to the input).
 *   • 🗑 Clear All — wipe the list; sits directly under Read from JSON (the two list-level ops
 *     belong together). Two-step confirm (arm, then click again) — never a blocking confirm().
 *   • ➕ Add Prompt — grow the list; the node auto-grows AND auto-shrinks to fit.
 *   • 🔍 Check for prompts — validate + count, reset index to 0, report how many runs to queue.
 * The "delete_empty_prompts" toggle (default on) drops rows with an empty positive on Check and just
 * before a batch is queued, so blank rows never break a run or skew the count.
 *
 * The rows are the source of truth. They are serialized into the hidden `prompts_json` widget, which
 * is what the Python `execute` reads — so save/load and execution both flow through that one field.
 * The index is walked by the node itself: an `afterQueued` hook bumps `index` by 1 after each queued
 * run (immune to the user's "Widget Value Control Mode" setting), and a global queue hook zeroes it
 * at batch start for nodes whose "reset_index_at_batch_start" toggle is on. The parse logic mirrors
 * _parse_prompts in nodes/prompt_batch.py — keep the two in sync.
 */
import { chainCallback } from "./utility.js";
const { app } = window.comfyAPI.app;

const NODE_ID = "ITLPromptBatch";

// ── Mirror of _parse_prompts in nodes/prompt_batch.py. Returns { ok, prompts } or { ok:false, error }.
function parsePrompts(raw) {
  const text = (raw || "").trim();
  if (!text) return { ok: false, error: "Prompt JSON is empty — add a prompt or import one." };
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    return { ok: false, error: "Malformed JSON: " + e.message };
  }
  if (data && typeof data === "object" && !Array.isArray(data) && Array.isArray(data.prompts)) {
    data = data.prompts; // tolerate a { "prompts": [...] } wrapper
  }
  if (!Array.isArray(data)) return { ok: false, error: 'Expected a JSON array of prompts, e.g. [{"positive": "..."}, ...].' };
  if (data.length === 0) return { ok: false, error: "Prompt list is empty — it needs at least one prompt." };

  const prompts = [];
  for (let i = 0; i < data.length; i++) {
    const entry = data[i];
    const where = `Prompt #${i + 1}`;
    let positive, negative;
    if (typeof entry === "string") {
      positive = entry;
      negative = "";
    } else if (entry && typeof entry === "object" && !Array.isArray(entry)) {
      positive = entry.positive ?? entry.prompt ?? "";
      negative = entry.negative ?? "";
      if (negative == null) negative = "";
      if (typeof positive !== "string") return { ok: false, error: `${where}: 'positive' must be a string.` };
      if (typeof negative !== "string") return { ok: false, error: `${where}: 'negative' must be a string.` };
    } else {
      return { ok: false, error: `${where}: each entry must be an object with a 'positive' field (or a plain string).` };
    }
    if (!String(positive).trim()) return { ok: false, error: `${where}: 'positive' is empty.` };
    prompts.push({ positive, negative });
  }
  return { ok: true, prompts };
}

const findWidget = (node, name) => node.widgets?.find((w) => w.name === name);

// Set a node's index widget (and fire its callback so any display updates).
function setIndex(node, value) {
  const idxW = findWidget(node, "index");
  if (!idxW) return;
  idxW.value = value;
  idxW.callback?.(value);
  node.setDirtyCanvas?.(true, true);
}

// Collapse a native widget so it takes no space (the Resolution Selector trick). Used to hide the
// authoritative-but-internal prompts_json field.
function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

// Read the JSON string from whatever plain text/primitive node is wired into `json_in`.
function readConnectedJson(node) {
  const inp = (node.inputs || []).find((i) => i.name === "json_in");
  if (!inp) return { ok: false, error: "No json_in input on this node." };
  if (inp.link == null) return { ok: false, error: "Nothing connected — wire a text/JSON node into json_in." };
  const links = node.graph?.links;
  const link = links?.get ? links.get(inp.link) : links?.[inp.link];
  if (!link) return { ok: false, error: "json_in link could not be resolved." };
  const origin = node.graph?.getNodeById(link.origin_id);
  if (!origin) return { ok: false, error: "Source node for json_in not found." };
  // Grab the first non-empty string widget value on the source (covers primitives, text nodes, etc.).
  for (const w of origin.widgets || []) {
    if (typeof w.value === "string" && w.value.trim()) return { ok: true, raw: w.value };
  }
  return { ok: false, error: "Source node has no readable text value (only static text/primitive nodes can be imported)." };
}

// One shared stylesheet for every Prompt Batch editor on the graph.
function ensureStyles() {
  if (document.getElementById("itl-pb-style")) return;
  const s = document.createElement("style");
  s.id = "itl-pb-style";
  s.textContent = `
  .itl-pb{display:flex;flex-direction:column;gap:6px;width:100%;box-sizing:border-box;
    font:12px/1.4 -apple-system,"Segoe UI",Roboto,sans-serif;color:#d3d3d0;padding:1px 0}
  .itl-pb-content{display:flex;flex-direction:column;gap:6px;width:100%}
  .itl-pb .pb-head{display:flex;justify-content:space-between;padding:0 2px;font-size:10.5px;color:#8b8b86}
  .itl-pb .pb-head .pos{color:#5cae6d}.itl-pb .pb-head .neg{color:#c86b6b}
  .itl-pb .pb-empty{padding:8px;text-align:center;color:#6d6d68;font-size:11px;
    border:1px dashed #3a3a38;border-radius:7px}
  .itl-pb .pb-row{display:flex;align-items:stretch;gap:7px;background:#262625;
    border:1px solid #3a3a38;border-radius:8px;padding:7px}
  .itl-pb .pb-row.pb-drag{opacity:.45}
  .itl-pb .pb-row.pb-over{border-color:#46b4e6;box-shadow:0 0 0 1px #46b4e6 inset}
  .itl-pb .pb-grip{align-self:center;color:#6d6d68;font-size:15px;cursor:grab;user-select:none;line-height:1}
  .itl-pb .pb-num{align-self:center;flex:none;width:20px;height:20px;border-radius:50%;
    background:#333331;color:#8b8b86;font:600 11px/20px ui-monospace,Consolas,monospace;text-align:center}
  .itl-pb .pb-fields{display:flex;gap:7px;flex:1;min-width:0}
  .itl-pb .pb-field{flex:1;min-width:0;display:flex;flex-direction:column;gap:3px}
  .itl-pb .pb-lbl{font-size:9px;letter-spacing:.06em;text-transform:uppercase;font-weight:600}
  .itl-pb .pb-field.pos .pb-lbl{color:#5cae6d}.itl-pb .pb-field.neg .pb-lbl{color:#c86b6b}
  .itl-pb textarea{width:100%;box-sizing:border-box;resize:vertical;min-height:88px;
    background:#1a1a19;border:1px solid #33332f;border-radius:5px;padding:5px 7px;
    color:#d3d3d0;font:11.5px/1.4 -apple-system,"Segoe UI",Roboto,sans-serif;outline:none}
  .itl-pb textarea:focus{border-color:#46b4e6}
  .itl-pb .pb-field.pos textarea{border-left:2px solid #5cae6d}
  .itl-pb .pb-field.neg textarea{border-left:2px solid #c86b6b}
  .itl-pb .pb-trash{align-self:center;flex:none;color:#6d6d68;font-size:13px;cursor:pointer;padding:2px}
  .itl-pb .pb-trash:hover{color:#c86b6b}
  `;
  document.head.appendChild(s);
}

app.registerExtension({
  name: "ITL.PromptBatch",

  // Wrap the queue action once: before a batch is queued, zero the index of every Prompt Batch node
  // whose "reset_index_at_batch_start" is on. The per-run afterQueued increment (wired in
  // onNodeCreated) then walks 0,1,2… across the batch.
  async setup() {
    const orig = app.queuePrompt;
    if (typeof orig !== "function" || orig._itlWrapped) return;
    const wrapped = async function (...args) {
      try {
        for (const node of app.graph?._nodes || []) {
          if (node?.comfyClass !== NODE_ID && node?.type !== NODE_ID) continue;
          if (node._pbPruneEmpties && findWidget(node, "delete_empty_prompts")?.value) node._pbPruneEmpties();
          const resetW = findWidget(node, "reset_index_at_batch_start");
          // Default-on: reset unless the toggle is explicitly false. A fresh-restore null/undefined
          // value must NOT be read as "skip reset" (that would leave a null index heading into the run).
          if (!resetW || resetW.value !== false) setIndex(node, 0);
        }
      } catch (e) {
        console.error("[ITL PromptBatch] batch-start index reset failed:", e);
      }
      return orig.apply(this, args);
    };
    wrapped._itlWrapped = true;
    app.queuePrompt = wrapped;
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    ensureStyles();

    chainCallback(nodeType.prototype, "onNodeCreated", function () {
      const node = this;
      const jsonW = findWidget(node, "prompts_json");
      hideWidget(jsonW);

      // Rows are the source of truth (array of {positive, negative}); jsonW is the serialized mirror.
      node._pbRows = [];
      let dragIndex = -1;

      // Own the per-run walk: after each queued run, advance the index by 1 (always after
      // serialization, so the run that fired used the pre-increment value).
      const idxW = findWidget(node, "index");
      if (idxW) {
        idxW.afterQueued = () => setIndex(node, (parseInt(idxW.value, 10) || 0) + 1);
        // Guarantee `index` always leaves the front-end as a real non-negative int. On a fresh session
        // the restored widget value can arrive null/undefined; that serializes as JSON null and makes
        // ComfyUI's INT validation reject the run ("int() argument … not 'NoneType'"). serializeValue
        // is graphToPrompt's single serialization point for every queue path, so coercing here sanitizes
        // both the value sent to Python and the value written back into the saved workflow.
        idxW.serializeValue = () => { const v = parseInt(idxW.value, 10); return Number.isFinite(v) && v >= 0 ? v : 0; };
      }
      const delEmptyW = findWidget(node, "delete_empty_prompts");

      // ── DOM row editor. editorEl is the widget root (ComfyUI pins its height each frame); the rows
      // live in contentEl, whose *natural* height we measure so the node can both grow and shrink. ──
      const editorEl = document.createElement("div");
      editorEl.className = "itl-pb";
      const contentEl = document.createElement("div");
      contentEl.className = "itl-pb-content";
      editorEl.append(contentEl);
      const rowsWidget = node.addDOMWidget("prompt_batch_rows", "rows", editorEl, { serialize: false });

      // Serialize rows -> hidden prompts_json (the field Python reads and the workflow saves).
      function syncJson() {
        if (jsonW) jsonW.value = JSON.stringify(node._pbRows.map((r) => ({ positive: r.positive, negative: r.negative })));
      }

      // Resize the node to fit the current rows (measured, so no magic row-height constant). We measure
      // contentEl (natural height), NOT editorEl (height pinned by ComfyUI) — otherwise a removed row
      // couldn't shrink the node, because the pinned element keeps its old, larger height.
      function fitToContent() {
        const h = Math.max(contentEl.scrollHeight, 8);
        rowsWidget.computeSize = () => [node.size?.[0] || 320, h + 8];
        const want = node.computeSize?.();
        if (want) node.setSize([node.size[0], want[1]]);
        node.setDirtyCanvas?.(true, true);
      }
      function syncSize() {
        requestAnimationFrame(fitToContent);
      }

      // On a fresh page load / UI refresh, ComfyUI attaches the DOM widget's element *after* our initial
      // syncSize runs, so scrollHeight measures 0 and the node never grows to fit — the rows then render
      // on top of the buttons below (previously only unstuck by editing/removing a row, which re-fires
      // syncSize). Observe the editor's real layout and re-fit the moment it gets, or changes, its natural
      // height, so it self-corrects with no user interaction. fitToContent doesn't alter contentEl's own
      // height (only the node/reserved area), so this converges instead of looping.
      let lastFitH = 0;
      const rowsResizeObserver = new ResizeObserver(() => {
        const h = contentEl.scrollHeight;
        if (h && h !== lastFitH) { lastFitH = h; fitToContent(); }
      });
      rowsResizeObserver.observe(contentEl);
      chainCallback(node, "onRemoved", () => rowsResizeObserver.disconnect());

      function render() {
        contentEl.replaceChildren();

        const head = document.createElement("div");
        head.className = "pb-head";
        head.innerHTML = `<span>Prompts</span><span><span class="pos">positive</span> / <span class="neg">negative</span></span>`;
        contentEl.appendChild(head);

        if (!node._pbRows.length) {
          const empty = document.createElement("div");
          empty.className = "pb-empty";
          empty.textContent = "No prompts — press ➕ Add Prompt or 📥 Read from JSON.";
          contentEl.appendChild(empty);
          return;
        }

        node._pbRows.forEach((row, k) => {
          const el = document.createElement("div");
          el.className = "pb-row";

          const grip = document.createElement("span");
          grip.className = "pb-grip";
          grip.textContent = "⠿";
          grip.title = "Drag to reorder";
          // Only make the row draggable while the grip is held, so textarea selection still works.
          grip.addEventListener("mousedown", () => { el.draggable = true; });
          el.addEventListener("mouseup", () => { el.draggable = false; }); // grip pressed but not dragged
          el.addEventListener("dragstart", (e) => { dragIndex = k; e.dataTransfer.effectAllowed = "move"; el.classList.add("pb-drag"); });
          el.addEventListener("dragend", () => { el.draggable = false; dragIndex = -1; el.classList.remove("pb-drag"); contentEl.querySelectorAll(".pb-over").forEach((n) => n.classList.remove("pb-over")); });
          el.addEventListener("dragover", (e) => { e.preventDefault(); if (dragIndex > -1 && dragIndex !== k) el.classList.add("pb-over"); });
          el.addEventListener("dragleave", () => el.classList.remove("pb-over"));
          el.addEventListener("drop", (e) => {
            e.preventDefault();
            el.classList.remove("pb-over");
            if (dragIndex > -1 && dragIndex !== k) {
              const [moved] = node._pbRows.splice(dragIndex, 1);
              node._pbRows.splice(k, 0, moved);
              render(); syncJson();
            }
          });

          const num = document.createElement("span");
          num.className = "pb-num";
          num.textContent = String(k + 1);

          const fields = document.createElement("div");
          fields.className = "pb-fields";
          for (const key of ["positive", "negative"]) {
            const f = document.createElement("div");
            f.className = "pb-field " + (key === "positive" ? "pos" : "neg");
            const lbl = document.createElement("span");
            lbl.className = "pb-lbl";
            lbl.textContent = key;
            const ta = document.createElement("textarea");
            ta.value = row[key] || "";
            ta.placeholder = key === "positive" ? "prompt…" : "(optional)";
            ta.rows = 5;
            ta.addEventListener("input", () => { row[key] = ta.value; syncJson(); });
            // Don't let canvas hotkeys / drag steal keyboard focus while typing.
            ta.addEventListener("pointerdown", (e) => e.stopPropagation());
            f.append(lbl, ta);
            fields.appendChild(f);
          }

          const trash = document.createElement("span");
          trash.className = "pb-trash";
          trash.textContent = "🗑";
          trash.title = "Remove this prompt";
          trash.addEventListener("click", () => { node._pbRows.splice(k, 1); render(); syncJson(); syncSize(); });

          el.append(grip, num, fields, trash);
          contentEl.appendChild(el);
        });
      }

      // Rebuild the rows from the hidden JSON (on create from defaults, and after a workflow loads).
      node._pbRebuild = () => {
        const res = parsePrompts(jsonW?.value);
        node._pbRows = res.ok ? res.prompts.map((p) => ({ positive: p.positive, negative: p.negative })) : [];
        render(); syncJson(); syncSize();
      };

      // Drop rows whose positive is empty. Used by "Check for prompts" and by the queue hook when the
      // "delete_empty_prompts" toggle is on, so blank rows never break a run or shift the count.
      node._pbPruneEmpties = () => {
        const before = node._pbRows.length;
        node._pbRows = node._pbRows.filter((r) => String(r.positive || "").trim() !== "");
        const removed = before - node._pbRows.length;
        if (removed) { render(); syncJson(); syncSize(); }
        return removed;
      };

      // ── Buttons (native, so they look/behave like ComfyUI). Created in display order; Read is then
      // hoisted above the index/reset widgets. ──
      const readBtn = node.addWidget("button", "📥 Read from JSON", null, () => {
        const res = readConnectedJson(node);
        if (!res.ok) { setStatus("❌ " + res.error, "#e0555a"); return; }
        const parsed = parsePrompts(res.raw);
        if (!parsed.ok) { setStatus("❌ " + parsed.error, "#e0555a"); return; }
        for (const p of parsed.prompts) node._pbRows.push({ positive: p.positive, negative: p.negative });
        render(); syncJson(); syncSize();
        setStatus(`✅ Imported ${parsed.prompts.length} — ${node._pbRows.length} prompt${node._pbRows.length === 1 ? "" : "s"} total.`, "#46b4e6");
      });
      readBtn.serialize = false;

      const addBtn = node.addWidget("button", "➕ Add Prompt", null, () => {
        node._pbRows.push({ positive: "", negative: "" });
        render(); syncJson(); syncSize();
      });
      addBtn.serialize = false;

      // No window-blocking confirm() here: a native dialog fired from inside the canvas's
      // pointer state machine swallows the matching pointerup, which leaves pan/drag state stuck
      // (the "canvas flies away" corruption) and re-fires the prompt when the cursor is parked
      // over the button. Two-step confirm instead: first press arms, second press clears.
      const CLEAR_LABEL = "🗑 Clear All";
      let clearArmTimer = null;
      const disarmClear = () => {
        if (clearArmTimer != null) { clearTimeout(clearArmTimer); clearArmTimer = null; }
        clearBtn.label = CLEAR_LABEL;
        node.setDirtyCanvas?.(true, true);
      };
      const clearBtn = node.addWidget("button", CLEAR_LABEL, null, (v, canvas, n, pos, event) => {
        // Pan/drag sequences with a second mouse button can replay a widget click at a stale
        // cursor position — only a plain left-button press may operate this button.
        if (event && typeof event.button === "number" && event.button !== 0) return;
        if (!node._pbRows.length) { setStatus("Nothing to clear.", "#8a8a8a"); return; }
        if (clearArmTimer == null) {
          clearBtn.label = "⚠ Sure? Click again";
          clearArmTimer = setTimeout(disarmClear, 4000);
          setStatus(`Removes all ${node._pbRows.length} prompt${node._pbRows.length === 1 ? "" : "s"} — click again within 4s to confirm.`, "#e0a03c");
          node.setDirtyCanvas?.(true, true);
          return;
        }
        disarmClear();
        node._pbRows = [];
        render(); syncJson(); syncSize();
        setStatus('Cleared. Press "Add Prompt" or "Read from JSON" to start again.', "#8a8a8a");
      });
      clearBtn.serialize = false;

      // Status line (read-only DOM widget), then the Check button above it.
      const statusEl = document.createElement("div");
      statusEl.style.cssText = "width:100%;box-sizing:border-box;padding:3px 6px;text-align:center;line-height:1.4;font:12px sans-serif;";
      const setStatus = (text, color) => { statusEl.textContent = text; statusEl.style.color = color; node.setDirtyCanvas?.(true, true); };

      const checkBtn = node.addWidget("button", "🔍 Check for prompts", null, () => {
        const removed = delEmptyW?.value ? node._pbPruneEmpties() : 0;
        const res = parsePrompts(jsonW?.value);
        if (!res.ok) { setStatus("❌ " + res.error, "#e0555a"); return; }
        const n = res.prompts.length;
        setIndex(node, 0);
        const dropped = removed ? ` (removed ${removed} empty)` : "";
        setStatus(`✅ ${n} prompt${n === 1 ? "" : "s"} detected${dropped} — set ComfyUI runs to ${n} (index reset to 0).`, "#46b4e6");
      });
      checkBtn.serialize = false;

      node.addDOMWidget("prompt_batch_status", "info", statusEl, { serialize: false });
      setStatus('Add prompts, or import JSON — then press "Check for prompts".', "#8a8a8a");

      // Hoist "Read from JSON" (with "Clear All" right under it — the two list-level operations
      // belong together) to just above the index widget, and drop the rows editor right below
      // reset (it was appended after the buttons). Final order: [json(hidden), read, clear, index,
      // reset, rows, add, check, status]. Reordering the buttons is safe because restore skips
      // serialize=false widgets and the scalar widgets are mirrored by name (see onSerialize below).
      const move = (w, beforeName) => {
        const arr = node.widgets, cur = arr.indexOf(w);
        if (cur > -1) arr.splice(cur, 1);
        const at = arr.findIndex((x) => x.name === beforeName);
        arr.splice(at > -1 ? at : arr.length, 0, w);
      };
      move(readBtn, "index");
      move(clearBtn, "index");
      move(rowsWidget, "➕ Add Prompt");

      node._pbRebuild();
    });

    // The frontend saves widgets_values at *absolute* widget indices (serialize=false buttons
    // leave null holes) but restores them *compacted* (skipping those buttons) — so every scalar
    // that sits after a button shifts by one slot on load. That is why a restored `index` could
    // arrive null. Mirror the small scalars by name into node properties on save and restore them
    // from there, immune to widget order. prompts_json stays positional: it is widget #0 (aligned
    // under both schemes) and far too large to duplicate into properties.
    const MIRRORED = ["index", "reset_index_at_batch_start", "delete_empty_prompts"];
    chainCallback(nodeType.prototype, "onSerialize", function (o) {
      const mirror = {};
      for (const name of MIRRORED) {
        const w = findWidget(this, name);
        if (w) mirror[name] = w.value;
      }
      o.properties = o.properties || {};
      o.properties.itl_pb = mirror;
    });

    // After a saved workflow loads: prefer the by-name mirror over whatever the positional restore
    // produced (old workflows without the mirror keep the legacy behavior, which the existing
    // guards — serializeValue coercion, "!== false" reset check — already handle). Then rebuild
    // the rows from the restored prompts_json.
    chainCallback(nodeType.prototype, "onConfigure", function (info) {
      const node = this;
      const mirror = info?.properties?.itl_pb;
      if (mirror && typeof mirror === "object") {
        for (const name of MIRRORED) {
          const w = findWidget(node, name);
          if (w && mirror[name] !== undefined) w.value = mirror[name];
        }
      }
      requestAnimationFrame(() => node._pbRebuild?.());
    });
  },
});
