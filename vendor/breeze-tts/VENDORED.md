# Vendored: Breeze TTS 2 model code

This directory is a verbatim copy of the Python packages `breeze_models` and `breeze_infer`
from **Into-The-Latent/breeze-tts**, our fork of
[breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) (Breeze TTS 2 by
BreezeBlue).

| | |
|---|---|
| Source repo | https://github.com/Into-The-Latent/breeze-tts |
| Tag | `comfyui-v1.7` |
| Commit | `ebe5ca2069c818c62524a42a4a617b1516ca1a92` |
| License | Apache-2.0 (see `LICENSE` here; `breeze_models/qwen_tokenizer/NOTICE` covers the vendored Qwen tokenizer) |
| Left out | `breeze_infer/api.py` (the fork's FastAPI server; not used by the nodes, needs fastapi/uvicorn) |

## Why it is vendored instead of pip-installed

The Comfy registry scans every published version. A dependency written as
`breeze-tts @ git+https://github.com/...` is reported as
`contains_custom_url_dependency` ("Detects custom wheel or URL dependencies") and the version
is flagged, which hides it from ComfyUI Manager. Pack versions 1.8.0 to 1.9.1 were lost that way.
Shipping the code inside the pack removes the URL, and it also means end users need neither
`git` on PATH nor a working GitHub connection at install time.

## How it is loaded

`nodes/breeze_vendor.py` prepends this directory to `sys.path` (called from the pack's
`__init__.py` and before each lazy import in the Breeze nodes), so the copy is imported under
the fork's own top-level names `breeze_models` / `breeze_infer` with no source changes.

## Rules that still apply

- Nothing in here may declare a torch version floor (`tests/test_requirements.py` checks).
  `breeze_models/__init__.py` enforces torch >= 2.7 at import time instead; a floor in pip
  metadata would make pip replace a Windows user's CUDA torch with the CPU-only wheel.
- Nothing in here may pass `device_map=` to `from_pretrained` or import `accelerate`
  (`tests/test_requirements.py` checks): transformers then demands the optional `accelerate`
  package, which a fresh ComfyUI venv does not have. Load on CPU and `.to(device)` instead.
- Nothing in here may read or write environment variables or make network requests
  (`tests/test_requirements.py` checks). The registry scanner matches plain text in every
  published file, and pack 1.10.0 was flagged for exactly that: rules
  `python_environment_manipulation` (rank lookups, determinism / tokenizer settings) and
  `python_network_operations` (the qwen tokenizer downloading audio URLs). Fixed in the fork at
  `comfyui-v1.7`: rank comes from `torch.distributed`, and `load_audio` refuses URLs. To see why a
  version was flagged: `GET https://api.comfy.org/nodes/comfyui-intothelatent-utils/versions?include_status_reason=true`
- Do not edit files in this directory by hand. Fix things in the fork, tag it, and re-sync.

## Re-syncing from the fork

```bash
cd ComfyUI-IntoTheLatent-Utils
rm -rf vendor/breeze-tts/breeze_models vendor/breeze-tts/breeze_infer
git -C ../breeze-tts archive <new-tag> breeze_models breeze_infer LICENSE | tar -x -C vendor/breeze-tts
rm vendor/breeze-tts/breeze_infer/api.py
```

Then update the tag and commit in the table above, run the pack's tests, and run the real-model
check (see the project notes) before publishing.
