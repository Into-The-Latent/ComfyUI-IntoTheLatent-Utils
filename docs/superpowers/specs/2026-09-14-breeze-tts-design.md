# Breeze TTS 2 nodes — loader with auto-download, Clone / Design / Direction (Normal + Advanced)

**Date:** 2026-09-14
**Nodes:** `ITLBreezeTTSLoader`, `ITLBreezeTTSVoiceClone`, `ITLBreezeTTSVoiceCloneAdvanced`,
`ITLBreezeTTSVoiceDesign`, `ITLBreezeTTSVoiceDesignAdvanced`, `ITLBreezeTTSVoiceDirection`,
`ITLBreezeTTSVoiceDirectionAdvanced`
(`nodes/breeze_tts_core.py`, `nodes/breeze_tts_loader.py`, `nodes/breeze_tts_generate.py`)
**Upstream model:** Breeze TTS 2 — BreezeBlue, 3B params, English + Chinese, released 2026-08-25.
Weights: Hugging Face `BreezeBlue/Breeze-TTS-2` (7.7 GB, *research and non-commercial* license).
Code: Apache 2.0, https://github.com/breezeblue-ai/breeze-tts
**Our fork:** https://github.com/Into-The-Latent/breeze-tts, branch `feature/comfyui`, local clone
`E:\Repos\breeze-tts`.

## Problem

Breeze TTS 2 has no ComfyUI integration. Upstream ships a CLI and an HTTP API only, is not
pip-installable, and hard-pins `torch==2.9.1`, `transformers==4.57.3` and `qwen-tts==0.1.1` —
pins that would downgrade a current ComfyUI install (the reference machine runs
torch 2.11.0+cu130, transformers 5.15.0, Python 3.12, RTX 5090).

## Goals

1. A **loader** node that fetches the weights from Hugging Face the first time a workflow runs,
   loads them, and caches the loaded runtime across queue runs.
2. **Six generate nodes** — Voice Clone, Voice Design, Voice Direction, each as *Normal* and
   *Advanced* — that take a `BREEZE_TTS` model and return a ComfyUI `AUDIO`.
3. Install through ComfyUI Manager with no manual steps and **no change to the user's torch or
   transformers version**.
4. The utils pack keeps loading (all other nodes usable) when the Breeze dependency is missing.

## Non-goals (v1)

- Batching several texts in one run; streaming playback; dual CFG (`guidance_scale_ref` /
  `guidance_scale_ins` — the fast runtime rejects it anyway).
- Alternative checkpoints / finetunes. The loader loads the official snapshot only. A `model`
  dropdown can be added later without breaking saved workflows (new widget with a default).
- CPU or MPS inference. Upstream's runtime raises without CUDA; we surface that, we do not
  work around it.
- A language selector. The model is bilingual and infers language from the text; there is no
  such parameter in the runtime.

---

## 1. Dependency strategy

### 1.1 What the fork already does (done, pushed)

- Top-level package `models` renamed to `breeze_models` (a package literally named `models`
  would collide with ComfyUI's own `models/` directory on `sys.path`).
- transformers 5 compatibility: `no_init_weights` import falls back between
  `transformers.initialization` (5.x) and `transformers.modeling_utils` (4.x); the T5Gemma
  text-encoder shims are registered with `exist_ok=True` because transformers 5 ships native
  `t5_gemma_module` / `t5gemma2_text` entries.
- `pyproject.toml` (packages `breeze_models`, `breeze_infer`), dependency ranges
  `torch>=2.9`, `transformers>=4.57,<6`, `numpy>=2`, `soundfile>=0.13`.
- Verified: all 47 upstream unit tests pass under the reference ComfyUI venv; wheel builds.

### 1.2 Remaining fork work: vendor the audio tokenizer

`qwen-tts==0.1.1` cannot be a dependency: it hard-pins `transformers==4.57.3` and
`accelerate==1.12.0`, drags in `gradio`, `sox`, `onnxruntime`, and its 12 Hz tokenizer breaks on
transformers 5 (`@check_model_inputs()` — the decorator is no longer a factory; verified in the
reference venv). Breeze uses exactly three files from it:

| qwen-tts file | lines | used by |
|---|---|---|
| `core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py` | 172 | model config |
| `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` | 1027 | `breeze_models/stream_runtime/core/compat.py` |
| `inference/qwen3_tts_tokenizer.py` (`Qwen3TTSTokenizer`) | 410 | `breeze_infer/runtime.py`, `compat.py` |

These are copied into the fork as `breeze_models/qwen_tokenizer/` (Apache 2.0 headers and the
Alibaba copyright kept; a `NOTICE` line added), with two changes:

- `@check_model_inputs()` → a small shim: use `merge_with_config_defaults` when present
  (transformers 5), else `check_model_inputs` called as a factory (4.57).
- Relative imports rewired to the new location; the `qwen_tts` fallback error text in
  `compat.py` removed.

`librosa` becomes a fork dependency (the wrapper resamples reference audio with it). Verified:
with the decorator fix, both modules import and `breeze_models.stream_runtime` +
`breeze_infer.api` import under transformers 5.15.

After this the whole dependency is **one pure-Python package** (the fork) whose only
non-ComfyUI-standard requirement is `librosa`.

### 1.3 How the utils pack installs it

`requirements.txt` gains:

```
librosa>=0.10
breeze-tts @ git+https://github.com/Into-The-Latent/breeze-tts@<tag>
```

pinned to a fork tag (`comfyui-v1`, moved forward deliberately), never a branch. ComfyUI
Manager runs `pip install -r requirements.txt` on install and update, so a plain Manager
install is enough. No `install.py`.

Because pip git installs need `git` on `PATH` (portable Windows builds sometimes lack it), the
Breeze modules are imported **lazily** inside the node functions. `__init__.py` always registers
the seven nodes; if the import fails at execution time the node raises
`ImportError("Breeze TTS is not installed: pip install ... — see README")`. The rest of the
pack is unaffected.

---

## 2. Loader — `ITL Breeze TTS Loader`

**Widgets**

| name | type | default | notes |
|---|---|---|---|
| `attention` | combo `eager`, `sdpa` | `sdpa` | passed as `attn_implementation`; `eager` is upstream's tested default, `sdpa` is faster |
| `fast_path` | bool | `False` | `FastStreamingConfig(fast_all=True)`: CUDA-graph capture for every stage. Upstream: ~14.4 GiB VRAM vs ~7.7 GiB eager |

**Output:** `BREEZE_TTS` (`io.Custom("BREEZE_TTS")`) — an opaque handle wrapping
`tokenizer`, `model`, `audio_tokenizer`, `runtime` (`FastBreezeStreamingRuntime`) and
`sample_rate`.

**Behaviour on execute**

1. `models/breeze_tts/` is registered with `folder_paths.add_model_folder_path` at import
   time; the snapshot lives at `models/breeze_tts/Breeze-TTS-2/`.
2. If that directory is missing the required files (both safetensors shards, `config.json`,
   `tokenizer.json`, `audio_tokenizer/`), run
   `huggingface_hub.snapshot_download("BreezeBlue/Breeze-TTS-2", local_dir=...)`. Progress
   goes to the console via `tqdm` (huggingface_hub's default) plus a one-line "Breeze TTS:
   downloading 7.7 GB to …" before and "done" after. A partially downloaded folder is resumed,
   not re-fetched — `snapshot_download` is idempotent per file.
3. Load through the fork's `load_runtime(ckpt_dir, device, attn_implementation)` then
   `update_generation_config_for_breeze(model)` and construct
   `FastBreezeStreamingRuntime(model, audio_tokenizer, FastStreamingConfig(...), tokenizer)`.
4. Cache the handle in a module-level dict keyed by `(ckpt_dir, attention, fast_path)`. A new
   key evicts every other entry first (one 7.7 GB model resident at a time), calling
   `comfy.model_management.soft_empty_cache()` after dropping references.
5. `IS_CHANGED` (`fingerprint_inputs`) returns the cache key, so ComfyUI re-runs the loader only
   when a widget changes; a cache hit returns the existing handle without touching the GPU.

**Errors**

- No CUDA device → `RuntimeError("Breeze TTS needs an NVIDIA GPU (CUDA); the upstream runtime
  has no CPU path.")` raised *before* any download starts.
- Download failure → the `huggingface_hub` exception is re-raised with the target path
  prepended, so the user can see where a partial download lives.
- The weights' non-commercial license line is printed once per process at first load and is
  stated in the README.

---

## 3. Generate nodes

Three modes × two tiers. Category `IntoTheLatent/Breeze TTS`. Every node has one output,
`AUDIO`, and one required input `model: BREEZE_TTS`.

### 3.1 Inputs by mode

| input | Clone | Design | Direction | type / default |
|---|---|---|---|---|
| `text` | ✓ | ✓ | ✓ | STRING, multiline, required |
| `reference_audio` | ✓ | | ✓ | AUDIO, required |
| `reference_text` | ✓ | | ✓ | STRING, multiline, required — exact transcript of the clip |
| `instruction` | | ✓ | ✓ | STRING, multiline, required — voice description (Design) / tone, pace, emotion (Direction) |
| `seed` | ✓ | ✓ | ✓ | INT, default 42, `control_after_generate` |
| `cfg_scale` | ✓ | ✓ | ✓ | FLOAT; default **1.0** Clone, **4.0** Design & Direction (upstream README guidance); 0.1–10, step 0.1 |

Upstream's `select_template_name` picks the template from which request keys are present;
each node builds exactly the request for its mode, so the mode is explicit — never inferred.

### 3.2 Advanced adds

| input | type | default | bounds |
|---|---|---|---|
| `temperature` | FLOAT | 0.9 | 0.05–2.0 |
| `top_k` | INT | 50 | 0–500 (0 = off) |
| `top_p` | FLOAT | 1.0 | 0.0–1.0 |
| `repetition_penalty` | FLOAT | 1.1 | 1.0–2.0 |
| `max_new_tokens` | INT | 750 | 50–1500 — ~12.5 codec frames/s of audio |

Defaults equal `update_generation_config_for_breeze` / `FastStreamingConfig` defaults, so a
Normal node and an Advanced node at defaults produce the same audio for the same seed.

The Advanced sampling values go through `FastStreamingConfig` (`temperature`, `top_k`,
`top_p`, `repetition_penalty`, `max_new_tokens`). Because the runtime is built by the loader
with one config, the generate node **rebuilds the runtime object** when its sampling config
differs from the cached one — `FastBreezeStreamingRuntime` construction is cheap relative to
model load (it wraps the already-loaded model; in eager mode no graph capture happens). The
loader handle keeps `model`, `tokenizer`, `audio_tokenizer` so this never reloads weights.
With `fast_path=True` a config change re-captures CUDA graphs; documented in the README.

### 3.3 Execution (shared, `breeze_tts_core.py` + a single `_generate()` in the node module)

1. Validate: `text` non-empty; for Clone/Direction `reference_text` non-empty and
   `reference_audio` present; for Design/Direction `instruction` non-empty. Violations raise
   `ValueError` naming the input — never silently fall back to another mode.
2. `reference_audio` (`{"waveform": [B, C, N], "sample_rate": sr}`): take batch item 0,
   downmix to mono by channel mean, write float32 PCM_16 WAV to
   `folder_paths.get_temp_directory()/breeze_ref_<uuid>.wav` with `soundfile`. The runtime
   takes a **path** (`ref_audio_path`); it resamples internally. The temp file is deleted in a
   `finally`.
3. Build the request dict (`id`, `text`, `speaker: "S0"`, plus mode keys), call
   `prepare_inputs(tokenizer, audio_tokenizer, model, [request], get_template(name),
   guidance_scale=cfg_scale)`.
4. `set_all_seeds(seed)`, then iterate `runtime.iter_audio_chunks(inputs, request_id, seed)`
   collecting `chunk.audio` (1-D float32 numpy). Concatenate → `torch.from_numpy(...)[None,
   None, :]` → `{"waveform": Tensor[1, 1, N], "sample_rate": runtime.sample_rate}` (24 000 Hz
   from the codec config; never hard-coded).
5. Run under `torch.inference_mode()`; on exception the temp file is still removed.

### 3.4 Pure helpers in `breeze_tts_core.py` (no ComfyUI, no torch model)

- `build_request(mode, text, reference_text=None, instruction=None, has_ref_audio=False)` →
  dict or raises `ValueError` — the validation table above lives here.
- `audio_to_mono_numpy(waveform_tensor)` → 1-D float32 array (batch 0, channel mean).
- `chunks_to_audio(chunks, sample_rate)` → AUDIO dict; raises on zero chunks.
- `cache_key(ckpt_dir, attention, fast_path)` and `sampling_config(...)` → frozen tuples used
  for equality checks in the loader / generate cache logic.
- `snapshot_is_complete(ckpt_dir)` → bool, the file-list check from §2.

---

## 4. Registration and files

```
nodes/breeze_tts_core.py       pure helpers (tested without ComfyUI)
nodes/breeze_tts_loader.py     ITLBreezeTTSLoader
nodes/breeze_tts_generate.py   six generate nodes built from one schema factory + one _generate()
tests/test_breeze_tts_core.py  unit tests for the helpers
tests/test_breeze_tts_nodes.py smoke tests: nodes registered, schemas valid, _generate() against a stub runtime
```

`__init__.py` imports the three modules and adds seven entries to `NODE_CLASS_MAPPINGS` /
`NODE_DISPLAY_NAME_MAPPINGS`:

| node_id | display name |
|---|---|
| `ITLBreezeTTSLoader` | ITL Breeze TTS Loader |
| `ITLBreezeTTSVoiceClone` | ITL Breeze TTS Voice Clone |
| `ITLBreezeTTSVoiceCloneAdvanced` | ITL Breeze TTS Voice Clone Advanced |
| `ITLBreezeTTSVoiceDesign` | ITL Breeze TTS Voice Design |
| `ITLBreezeTTSVoiceDesignAdvanced` | ITL Breeze TTS Voice Design Advanced |
| `ITLBreezeTTSVoiceDirection` | ITL Breeze TTS Voice Direction |
| `ITLBreezeTTSVoiceDirectionAdvanced` | ITL Breeze TTS Voice Direction Advanced |

The six generate classes are produced by a small factory (`_make_node(mode, advanced)`) so
the input table in §3 exists in exactly one place. No `web/` JS is needed.

`pyproject.toml`: version bump to 1.8.0 (new nodes); `requirements.txt` per §1.3; README gets a
"Breeze TTS" section: what the three modes do, the inline vocal-event syntax
(`(laugh)`, `(sigh)`, `[笑]`), VRAM figures, the first-run download, and the non-commercial
weights license.

---

## 5. Testing

- **Unit (no GPU, no model):** `build_request` validation matrix (all mode × missing-input
  combinations), `audio_to_mono_numpy` shapes/dtypes for mono/stereo/batched input,
  `chunks_to_audio` concatenation and empty-chunk error, `snapshot_is_complete` against a temp
  dir with/without each required file, `sampling_config` equality.
- **Node smoke (no GPU):** all seven node_ids registered and match their schema; every
  generate node's `execute` runs end-to-end with a stub handle whose `runtime.iter_audio_chunks`
  yields two fixed numpy chunks; temp WAV is created with the mono content and removed
  afterwards; missing-input errors surface with the input name.
- **Manual on the reference machine (documented in the PR, not automated):** first-run
  download into `models/breeze_tts/`, one generation per mode in eager mode, one Clone with
  `fast_path=True`, a second queue run hitting the loader cache without reload, and a
  `transformers 5.15` / `torch 2.11` environment left unchanged (`pip freeze` diff before/after
  install).

---

## 6. Order of work

1. Fork: vendor the tokenizer (§1.2), bump fork version, tag `comfyui-v1`, push.
2. Utils pack, branch `feature/breeze-tts`: core helpers + tests → loader → generate nodes +
   smoke tests → `__init__`, requirements, README, version bump.
3. Manual verification on the reference machine, PR to `main`.
