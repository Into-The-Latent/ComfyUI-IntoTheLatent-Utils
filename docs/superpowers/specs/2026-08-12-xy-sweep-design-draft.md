# XY Parameter Sweep — design draft (brainstorm in progress)

> **Status: DRAFT.** Snapshot of the brainstorming session on 2026-08-12 so the decisions aren't
> lost. Open questions below are still undecided — this is not yet an approved spec.

## Goal

Make A1111-style X/Y plotting easy in ComfyUI: a user tests different KSampler settings
(cfg, steps, sampler, scheduler), LoRA strengths, and the same prompt across multiple
checkpoints — without building a complicated graph. Ranges should be expressible simply,
e.g. "cfg from 1 to 8, increasing by 0.5 each render".

## Decisions so far

1. **Deliverable: a composited XY grid image** — one big stitched image with axis labels
   (like A1111's X/Y plot), assembled by a collector node when the last run finishes.
2. **Axes: exactly 2 (X and Y) in v1**, but architected so a third (Z → one grid image per
   Z value) can be added later without redesign.
3. **Integration: value emitter + companion loader nodes** ("Idea 3" below) — chosen over a
   pure value emitter (can't switch models) and an all-in-one mega sampler (locks the
   pipeline, high maintenance).
4. **Execution model: reuse the Prompt Batch mechanism** already proven in this pack —
   ComfyUI has no for-loop, so the node expands the sweep into N combinations, tells the
   user how many runs to queue, and walks an index one combination per queued run via the
   front-end `afterQueued` hook (immune to the widget-control-mode setting).
5. **Model-load ordering**: ComfyUI caches loaded models between runs, so the controller
   orders combinations with the model axis outermost — each checkpoint loads once per
   sweep, regardless of which axis (X or Y) the user assigned models to.

## Planned node set

| Node | Role |
|------|------|
| **AI2Go XY Sweep** (controller) | Holds both axis definitions. Per run, outputs the current combination as typed sockets — cfg (FLOAT), steps (INT), sampler/scheduler name, checkpoint name (STRING), LoRA strength (FLOAT) — plus a human-readable `label` (e.g. `cfg=3.5 \| juggernaut_v9`), the run index, and grid metadata for the collector. |
| **AI2Go Checkpoint by Name** | Thin loader: checkpoint name arrives via socket instead of the stock dropdown (dropdowns can't accept connections — the wall that blocks model sweeps with stock nodes). Outputs MODEL/CLIP/VAE. |
| **AI2Go LoRA by Name** | Same trick for LoRAs: name + strength as inputs; supports "no LoRA this run" for baseline rows/columns, and enables comparing different LoRAs, not just strengths. |
| **AI2Go XY Grid Collector** | Receives each run's image; on the last run stitches the labeled grid image (axis labels from the controller's metadata). |

The user's own KSampler and the rest of the workflow stay untouched — sweep outputs are
wired into existing nodes; the helpers replace only the loaders, and only when the sweep
involves models/LoRAs.

## Rejected alternatives (for the record)

- **Idea 1 — pure value emitter:** smallest build, works with any pipeline, but the stock
  Checkpoint Loader's dropdown can't take a text connection, so model comparison — a
  headline feature — is impossible with stock nodes.
- **Idea 2 — all-in-one "XY Sampler":** easiest wiring (model list, LoRA, KSampler all
  inside one node; Efficiency-Nodes approach), but it replaces the user's sampling
  pipeline (breaks Flux / refiner / custom-sampler setups) and means maintaining our own
  copy of loading + LoRA + sampling internals.

## Open questions (not yet discussed)

- **Range entry UX**: text mini-syntax ("1-8 step 0.5") vs. structured widgets
  (from/to/step fields) vs. both; how value lists (samplers, model names) are picked.
- **Seed handling**: same fixed seed for every cell (usual choice for comparability) vs.
  per-cell seeds; where the seed comes from.
- **Grid collector mechanics**: how images persist across runs (in-memory vs. temp files),
  label rendering (Pillow is available), cell size normalization, and what happens if the
  user aborts mid-sweep.
- **Queue-count UX**: "Check sweep" button that validates ranges, reports N runs, resets
  the index (mirroring Prompt Batch), and whether we can auto-set the queue count.
- **Which KSampler params in v1**: cfg, steps, denoise, sampler, scheduler confirmed
  candidates; seed-as-axis maybe later.
- **Known pack pitfalls to honor** (from memory notes): append new widgets at schema end
  (old-workflow compat), guard INT widget serialization (`''` → validation failure),
  widgets_values save/restore index mismatch (mirror scalars by name into properties), no
  blocking dialogs in widget callbacks (two-step arm/confirm instead).

## Next steps

1. Resume brainstorming: settle the open questions above (one at a time).
2. Present the full design in sections for approval.
3. Finalize this doc as the approved spec (drop DRAFT status).
4. Write the implementation plan (`docs/superpowers/plans/`).
