# Prompt Batch node — part of ComfyUI-IntoTheLatent-Utils.
#
# A standalone companion to the batch-upscale style of workflow. GPL-3.0, like the rest of the pack.
#
"""JSON prompt batcher.

ComfyUI has no real for-loop: to process a set you queue N runs by hand and let an *index* walk the
set one item per run. The Batch Upscale example does this for a folder of images (WAS "Load Image
Batch" + an increment primitive). This node is the text analog — paste a JSON list of prompts and it
emits one positive/negative pair per run, walking the list by the same increment-index mechanism.

Flow (mirrors the example's ritual):
  1. Build the prompt list in the front-end row editor (positive/negative text boxes, "➕ Add Prompt"),
     or "📥 Read from JSON" to import a list wired into the ``json_in`` socket. The rows are synced
     into the hidden ``prompts_json`` field, which is the authoritative execution source.
  2. Click "Check for prompts" (front-end) — validates the JSON, counts the prompts, resets index → 0,
     and tells you how many ComfyUI runs to queue.
  3. Set the ComfyUI batch/queue count to that number and run.

The front-end (web/js/prompt_batch.js) advances ``index`` by 1 after each queued run so it walks the
list 0,1,2… across the batch (the same effect as the example's increment primitive), and the "reset
index at batch start" toggle (default on) zeroes it when a new batch is queued so you never forget
step 2's reset. The walk is driven from JS — via the node's own ``afterQueued`` hook rather than the
built-in ``control_after_generate`` — so the increment timing is immune to the user's "Widget Value
Control Mode" setting. The current index is also emitted as an ``index`` output for use as a filename
suffix. Parsing here is authoritative and mirrored in the JS — keep the two in sync.
"""

from comfy_api.latest import io

from .prompt_batch_core import DEFAULT_PROMPTS_JSON, parse_prompts as _parse_prompts, select_prompt as _select


class ITLPromptBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLPromptBatch",
            display_name="ITL Prompt Batch",
            category="Into The Latent/text",
            search_aliases=["batch", "prompt", "json", "queue", "index", "list", "positive", "negative"],
            is_experimental=True,
            description="""
Walk a pasted JSON list of prompts, one per queued run — the text analog of the Batch Upscale
example's "Load Image Batch + increment index" trick.

- prompts_json: a JSON array of prompts. Each entry is an object with a required 'positive' and an
  optional 'negative' (a plain string entry is treated as positive-only). Example:
  [{"positive": "a red fox, 8k", "negative": "blurry"}, {"positive": "a neon city"}].
- Rows editor (front-end): edit prompts as positive/negative text boxes; "➕ Add Prompt" / "🗑 Clear
  All" manage the list. The rows are kept in sync with the hidden prompts_json field, which is what
  actually runs.
- json_in: optional STRING socket. "📥 Read from JSON" imports a connected node's JSON list into the
  rows. Ignored at run time — the rows are authoritative.
- index: which prompt to emit this run (0-based). The front-end advances it by 1 after each queued
  run, so it walks 0,1,2… across a queued batch.
- reset_index_at_batch_start: (front-end) zero the index when a new batch is queued so every batch
  starts from the first prompt.

Click "Check for prompts" to validate the JSON, count the prompts, and reset the index to 0 — then set
the ComfyUI queue count to that number. Outputs the current prompt's positive and negative strings and
the 0-based index used this run (handy as a SaveImage filename suffix); if the index overshoots the
list it clamps to the last prompt.""",
            inputs=[
                io.String.Input(
                    "json_in", optional=True, force_input=True,
                    tooltip="Optional STRING socket. Wire a text/primitive node holding a JSON prompt list here, "
                            "then press \"Read from JSON\" to import it into the rows. Ignored at run time — the "
                            "rows you edit are what actually run.",
                ),
                io.String.Input(
                    "prompts_json", default=DEFAULT_PROMPTS_JSON,
                    tooltip="Authoritative prompt list as JSON. Hidden in the UI and kept in sync with the row "
                            "editor by the front-end — edit the rows, not this.",
                ),
                io.Int.Input(
                    "index", default=0, min=0, max=1_000_000, step=1,
                    tooltip="Which prompt to emit this run (0-based). The front-end advances it by 1 after each "
                            "queued run so the batch walks the list. \"Check for prompts\" resets it to 0.",
                ),
                io.Boolean.Input(
                    "reset_index_at_batch_start", default=True,
                    tooltip="When on, the index is reset to 0 at the start of every queued batch, so each batch "
                            "starts from the first prompt. Off = the index keeps its value between batches.",
                ),
                io.Boolean.Input(
                    "delete_empty_prompts", default=True,
                    tooltip="When on, prompts with an empty 'positive' are dropped when you press \"Check for "
                            "prompts\" and just before a batch is queued, so blank rows never break a run or "
                            "throw off the count.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="positive"),
                io.String.Output(display_name="negative"),
                io.Int.Output(display_name="index"),
            ],
        )

    @classmethod
    def execute(cls, json_in=None, prompts_json=DEFAULT_PROMPTS_JSON, index=0,
                reset_index_at_batch_start=True, delete_empty_prompts=True) -> io.NodeOutput:
        # json_in, reset_index_at_batch_start and delete_empty_prompts are front-end-only (see
        # web/js/prompt_batch.js): json_in is an import source for the "Read from JSON" button, and the
        # two toggles are queue-time behaviors. All three are accepted here only so they serialize /
        # the socket exists. The rows the user edits are synced into prompts_json, which is the
        # authoritative execution source.
        prompts = _parse_prompts(prompts_json)
        positive, negative, used_index = _select(prompts, index)
        return io.NodeOutput(positive, negative, used_index)
