# Save Civitai Metadata writes the positive prompt as "Negative prompt"

## Problem
The Save Civitai Metadata nodes can write the **positive prompt** into the `Negative prompt:` line of the A1111 text. Civitai then shows that wrong text on the image page, and every tool that reads the PNG shows it too. This happens in two common kinds of workflow.

### Case 1: the negative is zeroed out
This is usual for turbo models and anything run at CFG 1:

```
CLIPTextEncode "a cat" ──► ConditioningZeroOut ──► KSampler.negative
                     └─────────────────────────► KSampler.positive
```

The workflow has **no negative prompt**, but the metadata says `Negative prompt: a cat`.

Real example: <https://civitai.com/images/143992601> (Krea 2 Turbo, CFG 1). It was saved with `SaveImageWithMetaData`, a different node pack that walks the graph the same way.

### Case 2: the negative goes through a node that carries both conditionings
The nodes are WanImageToVideo, ControlNetApplyAdvanced and similar ones, with outputs `positive`(0), `negative`(1) and so on:

```
CLIPTextEncode "a cat"  ──► WanImageToVideo.positive
CLIPTextEncode "blurry" ──► WanImageToVideo.negative
WanImageToVideo[0] ──► KSamplerAdvanced.positive
WanImageToVideo[1] ──► KSamplerAdvanced.negative
```

The negative is really "blurry", but the metadata says `Negative prompt: a cat`. Every Wan 2.2 image-to-video workflow is affected, at any CFG.

## Cause
In `nodes/civitai_metadata/tracer.py`, `_find_clip_encode` does a breadth-first search back from the sampler's `negative` link through **every** link input of every node:

- **Case 1:** `ConditioningZeroOut.conditioning` leads straight to the positive `CLIPTextEncode`, which it returns.
- **Case 2:** `WanImageToVideo` lists `positive` before `negative`, so the search reaches the positive encode first.

`_resolve_conditioning_text` then reports that text as the negative and marks it as resolved.

## Fix
In `_find_clip_encode`, take the input being resolved as a parameter (`which` = `"positive"` / `"negative"`), and while walking:

1. **Stop at `ConditioningZeroOut`.** Do not walk through it, and remember that a zero-out was reached.
2. **Follow a node that has inputs named `which` or `which_*` only through those inputs**, if any of them is a link. The `which_*` form covers PairConditioningSetProperties (`negative_NEW`) and PairConditioningCombine (`negative_A`, `negative_B`). Otherwise follow every other link input, but **never the other side's** (`positive` / `positive_*` when resolving the negative), so a carrier whose optional negative is unwired can't fall back to the positive.
3. **Stop at a sampler.** The walk only reaches one through a latent input (two-stage workflows). Conditioning never passes through a sampler, so its prompts must not be reported.
4. **Stop at a text encoder we can't read**, and mark the prompt **unknown**. That is a class name containing `TextEncode` (CLIPTextEncodeSDXL, TextEncodeQwenImage, …) or a node with a wired `clip` input (ImpactWildcardEncode, WAS Text to Conditioning, …). A missing node is unknown too.
5. Return a third outcome besides "found text" and "not found": **zeroed**. That means no encoder was found, the walk reached a zero-out, and nothing unknown was reached. Other leaves (LoadImage, VAE / ControlNet loaders behind latent or image inputs, as in Flux Kontext's ReferenceLatent) say nothing about the text and don't count.

`_resolve_conditioning_text` reports a zeroed prompt as an **empty**, resolved prompt, so `_resolve_prompts` does **not** add it to `unresolved`. A combine of a zeroed part and a real part, as in ComfyUI's SD3 template, still finds the real encode through the other branch, so it keeps its real negative. A combine of a zeroed part and an unknown encoder stays unresolved.

The live-graph preview's walk is in `web/js/civitai_trace.js` (`findClipEncode`). It applies the same rules, and also walks through a **bypassed** (mode 4) zero-out, because the API prompt drops bypassed nodes.

DiffusionNexus has the first version of this fix (rules 1, 2 without the `which_*` and other-side parts, and zeroed); the review refinements above (rules 2–4, unknown encoders) may still be missing there. It is in its C# port of this file, `DiffusionNexus.UI/Services/Distiller/ComfyUiPromptTracer.cs` (`FindEncoder`, Little-God1983/DiffusionNexus PR #671, commits `a6a75aee` and `013fd419`).

## Tests to add (`tests/test_tracer_prompts.py`)
- **KSampler with a zeroed negative:** the negative is `""`, the positive is "a cat", and `"negative"` is not in `unresolved`.
- **Wan I2V, real negative:** the negative is "blurry" and the positive is "a cat".
- **Wan I2V with a zeroed negative** (`WanImageToVideo.negative` ← `ConditioningZeroOut`): the negative is `""`.
- **SD3 template:** `ConditioningCombine` of a `SetTimestepRange(ZeroOut(enc))` and a `SetTimestepRange(enc)`, where `enc` is "blurry". The negative is "blurry".
- **A plain workflow with a real negative encode** still gives that text (regression).

## Notes
- Images saved before the fix keep their wrong line. DiffusionNexus already detects the zero-out from the embedded workflow when it reads such a PNG.
- The Advanced node's `negative` override socket is not affected. When it is wired, it wins as before.
