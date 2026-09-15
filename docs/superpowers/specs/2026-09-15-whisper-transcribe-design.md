# Whisper transcription nodes — loader with auto-download + Audio-to-Text

**Date:** 2026-09-15
**Nodes:** `ITLWhisperLoader`, `ITLWhisperTranscribe`
(`nodes/whisper_core.py`, `nodes/whisper_loader.py`, `nodes/whisper_transcribe.py`)
**Upstream model:** OpenAI Whisper, run through `transformers` (`WhisperForConditionalGeneration` +
`WhisperProcessor`). Weights: Hugging Face `openai/whisper-*` (Apache 2.0; the Whisper code is MIT).
**Motivation:** the Breeze TTS Clone / Direction nodes need the exact transcript of the reference
clip (`reference_text`). Typing it by hand is error-prone; the third-party Whisper node we tried
(comfy-mtb) failed to download its model on the reference machine.

## Problem

There is no reliable, zero-extra-dependency way in this pack to turn a ComfyUI `AUDIO` into its
transcript. Third-party Whisper packs bring their own runtimes (`openai-whisper` needs ffmpeg +
tiktoken, `faster-whisper` needs CTranslate2 and its own CUDA libraries) and their own download
paths, which is where the mtb node broke.

## Goals

1. A **loader** node that fetches a Whisper checkpoint from Hugging Face on first use into
   `models/whisper/<name>/`, loads it, and keeps it resident across queue runs.
2. A **transcribe** node: `AUDIO` in, `STRING` out, ready to wire into Breeze's `reference_text`.
3. **No new dependency.** Everything runs on `transformers` (already required by the Breeze
   fork and by ComfyUI itself), `librosa` (resampling; already required) and `huggingface_hub`.
4. Works on transformers 4.57 **and** 5.x, CUDA **and** CPU.
5. The pack keeps loading when transformers is unexpectedly missing (lazy imports, like Breeze).

## Non-goals (v1)

- Word / segment timestamps, SRT output, speaker diarization, translation to English.
- Beam search, temperature fallback, initial prompts. Greedy decoding is what the
  `reference_text` use case needs; knobs can be added later as new widgets with defaults.
- Distilled / community checkpoints or a free-text repo id. The dropdown lists the six official
  sizes; more can be appended later without breaking saved workflows.
- Forcing simplified vs. traditional Chinese output. Whisper has no switch for it; it follows the
  audio. Documented in the README as a thing to check before feeding Breeze.
- A separate Unload node. The transcribe node has an `unload_after` toggle; the largest model is
  ~3.5 GiB, so a pass-through node like Breeze's is not worth its own entry.

---

## 1. Dependency strategy

`transformers` ships Whisper natively (verified: `WhisperForConditionalGeneration`,
`WhisperProcessor` import on the reference venv, transformers 5.15 / torch 2.11 / cu130). No line is
added to `requirements.txt`: ComfyUI core already depends on transformers, and the Breeze fork
pins `>=4.57,<6`. The Whisper modules import transformers lazily inside the node functions; a
failed import raises `ImportError(INSTALL_HINT)` naming `pip install transformers>=4.57`.

Resampling uses `librosa.resample` (already a dependency through Breeze and listed in
`requirements.txt`). Nothing else is needed: the audio arrives as a tensor, so no ffmpeg.

---

## 2. Loader — `ITL Whisper Loader`

**Widgets**

| name | type | default | notes |
|---|---|---|---|
| `model` | combo `large-v3-turbo`, `large-v3`, `medium`, `small`, `base`, `tiny` | `large-v3-turbo` | maps to `openai/whisper-<name>` |
| `device` | combo `auto`, `cuda`, `cpu` | `auto` | `auto` = cuda if available else cpu |

**Output:** `WHISPER` (`io.Custom("WHISPER")`). Like `BREEZE_TTS`, the graph carries only the
immutable **cache key** `(ckpt_dir, device)`, not the model: ComfyUI's output cache then holds no
second reference to the tensors, so `unload()` really frees them (lesson from PR #4).

**Model table** (`whisper_core.MODELS`)

| name | HF repo | params | `model.safetensors` | fp16 VRAM (approx.) |
|---|---|---|---|---|
| `large-v3-turbo` | `openai/whisper-large-v3-turbo` | 809 M | 1.6 GB | ~2 GiB |
| `large-v3` | `openai/whisper-large-v3` | 1.55 B | 3.1 GB | ~3.5 GiB |
| `medium` | `openai/whisper-medium` | 769 M | 3.1 GB (fp32) | ~2 GiB |
| `small` | `openai/whisper-small` | 244 M | 967 MB (fp32) | ~1 GiB |
| `base` | `openai/whisper-base` | 74 M | 290 MB (fp32) | <1 GiB |
| `tiny` | `openai/whisper-tiny` | 39 M | 151 MB (fp32) | <1 GiB |

**Behaviour on execute**

1. `models/whisper/` is registered with `folder_paths.add_model_folder_path` at import time;
   checkpoint `<name>` lives at `models/whisper/whisper-<name>/`.
2. If any required file is missing or empty, run `huggingface_hub.snapshot_download(repo,
   local_dir=..., allow_patterns=DOWNLOAD_PATTERNS)`. The allow-list is essential: the `large-v3`
   repo also carries Flax, TF, `.bin` and fp32 duplicates (~12 GB extra). Required files:
   `config.json`, `generation_config.json`, `model.safetensors`, `preprocessor_config.json`,
   `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`,
   `merges.txt`, `vocab.json`, `normalizer.json`. One console line before ("downloading X to …")
   and after; partial downloads resume.
3. Load `WhisperProcessor.from_pretrained(dir)` and
   `WhisperForConditionalGeneration.from_pretrained(dir, torch_dtype=fp16 on cuda / fp32 on cpu)`,
   `.to(device).eval()`.
4. Cache in a module-level dict keyed by `(ckpt_dir, device)`; a new key evicts every other
   entry first (one Whisper resident at a time), then `gc.collect()` +
   `comfy.model_management.soft_empty_cache()`. Independent of the Breeze cache: both models can
   be resident together (turbo + Breeze ≈ 10 GiB).
5. `resolve_handle(key)` (used by the transcribe node) returns the cached handle or reloads.

**Errors:** `device="cuda"` without CUDA → `RuntimeError` before any download. Download failure
→ re-raised with the target path. Files still missing after download → `RuntimeError` listing
them.

---

## 3. Transcribe — `ITL Whisper Transcribe`

**Inputs**

| name | type | default | notes |
|---|---|---|---|
| `model` | `WHISPER` | required | from the loader |
| `audio` | `AUDIO` | required | batch item 0; stereo is downmixed |
| `language` | combo `auto` + the curated list below | `auto` | `auto` lets Whisper detect; a fixed code is passed as `language=` to `generate()` |
| `unload_after` | bool | `False` | free the Whisper model after this node |

Curated language list (ISO 639-1 codes, shown as-is in the combo): `en`, `zh`, `de`, `fr`, `es`,
`it`, `pt`, `nl`, `pl`, `ru`, `uk`, `tr`, `ar`, `hi`, `ja`, `ko`, `vi`, `id`, `th`, `sv`, `da`, `no`,
`fi`, `cs`, `el`, `he`, `hu`, `ro`. Stored as a tuple in `whisper_core.LANGUAGES` so the schema
does not need transformers at import time.

**Output:** `STRING` — the transcript, whitespace-normalised (`" ".join(text.split())`).

**Execution (`whisper_core.transcribe(handle, audio, language)`)**

1. `audio_to_mono_numpy(waveform)` (batch 0, channel mean, float32) — same rule as Breeze.
   Empty waveform → `ValueError("audio is empty")`.
2. Resample to 16 000 Hz with `librosa.resample` when `sample_rate != 16000`.
3. `processor(samples, sampling_rate=16000, return_tensors="pt", truncation=False,
   padding="longest", return_attention_mask=True)` → `input_features` (+ `attention_mask`).
   With ≤ 30 s of audio this is the ordinary `[1, n_mels, 3000]` features and `generate()` runs
   short-form; longer audio yields wider features and `generate()` runs transformers' sequential
   long-form decoding (`return_timestamps=True` is required for it, `condition_on_prev_tokens=False`
   keeps it from looping on repeated phrases). One code path for both.
4. `model.generate(input_features.to(device, dtype), attention_mask=..., task="transcribe",
   language=None|code, return_timestamps=True, condition_on_prev_tokens=False, num_beams=1)`
   under `torch.inference_mode()`.
5. `processor.batch_decode(ids, skip_special_tokens=True)[0]` → normalise → return.
6. If `unload_after`: `unload()` in the node after the text is produced.

Whisper's model-side language detection (`language=None`) is what `auto` means; no separate
detection call.

**Pure helpers in `whisper_core.py`** (no ComfyUI): `MODELS`, `LANGUAGES`,
`REQUIRED_FILES`, `DOWNLOAD_PATTERNS`, `missing_files(dir)`, `cache_key(dir, device)`,
`resolve_device(choice, cuda_available) -> "cuda"|"cpu"` (raises for `cuda` without CUDA),
`prepare_samples(audio, resample=librosa_resample) -> np.ndarray @16 kHz`, `normalise_text(str)`,
`WhisperHandle(key, processor, model, device, dtype)`, and `transcribe(handle, audio, language)`
which only touches the handle's `processor` / `model` through the calls in steps 3–5 so a stub
model works under test. `audio_to_mono_numpy` is imported from `breeze_tts_core` rather than
duplicated.

---

## 4. Registration and files

```
nodes/whisper_core.py          pure helpers + transcribe() (tested without ComfyUI)
nodes/whisper_loader.py        ITLWhisperLoader, WHISPER type, cache / unload / resolve_handle
nodes/whisper_transcribe.py    ITLWhisperTranscribe
tests/test_whisper_core.py     unit tests
tests/test_whisper_nodes.py    node smoke tests (skip without comfy_api)
```

`__init__.py` gains two entries:

| node_id | display name |
|---|---|
| `ITLWhisperLoader` | ITL Whisper Loader |
| `ITLWhisperTranscribe` | ITL Whisper Transcribe |

Category `Into The Latent/audio`, `is_experimental=True`, search aliases `whisper`,
`transcribe`, `speech to text`, `audio to text`, `stt`.

`pyproject.toml` version → 1.9.0. README: a "Whisper Transcribe" section (what it is for, model
sizes / VRAM, first-run download location, the Breeze `reference_text` wiring, the Chinese
script caveat), and a pointer from the Breeze section. Breeze's Clone / Direction
`reference_text` tooltip mentions the Whisper node.

---

## 5. Testing

- **Unit (no GPU, no weights):** `missing_files` on a temp dir with/without each file and with a
  0-byte stub; `cache_key` normalisation; `resolve_device` for each choice with CUDA present /
  absent; `prepare_samples` for mono / stereo / batched input, no-op at 16 kHz, calls the
  injected resampler otherwise, raises on empty; `normalise_text`; `transcribe()` against a stub
  processor/model that records the kwargs it received (`task`, `language`, `return_timestamps`,
  `truncation`, `padding`) and returns fixed ids.
- **Node smoke (ComfyUI checkout):** both node_ids registered and schemas match the input
  tables; models folder registered; `ensure_snapshot` skips / downloads / reports still-missing
  (download stubbed, `allow_patterns` asserted); `load_handle` caches, evicts on a new key;
  loader `execute` returns the key; transcribe `execute` returns the stub's text and calls
  `unload()` only when `unload_after` is on; install-hint `ImportError` when transformers is
  poisoned in `sys.modules`.
- **Manual on the reference machine (documented in the PR):** first-run download of
  `large-v3-turbo` into `models/whisper/`, one transcription of a < 30 s English clip and one of a
  > 30 s clip on transformers 5.15 (venv) and 4.57 (system python, CPU), `language=zh` on a
  Chinese clip, a second queue run hitting the cache, `unload_after` freeing VRAM, and the
  transcript wired into Breeze Voice Clone producing speech.

---

## 6. Order of work

Branch `feature/whisper-transcribe` (from `main`, which now contains the merged Breeze nodes):
core helpers + tests → loader → transcribe node + smoke tests → `__init__`, README, version
bump → manual verification → PR to `main`.
