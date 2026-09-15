# Breeze TTS 2 Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Breeze TTS 2 loader (auto-downloading the weights on first run) and six generate nodes (Voice Clone / Design / Direction × Normal / Advanced) to ComfyUI-IntoTheLatent-Utils, installable through ComfyUI Manager without touching the user's torch or transformers version.

**Architecture:** Two repos. The fork `Into-The-Latent/breeze-tts` (local `E:\Repos\breeze-tts`, branch `feature/comfyui`) becomes the single pip dependency: Task 1 vendors the three qwen-tts tokenizer files into it so `qwen-tts` is no longer needed. The utils pack gets a pure, ComfyUI-free engine module `nodes/breeze_tts_core.py` (request building, audio conversion, snapshot check, handle/runtime caching, `generate_audio`) that is unit-tested with stubs, plus two thin ComfyUI node modules (`breeze_tts_loader.py`, `breeze_tts_generate.py`) whose Breeze imports are lazy so the pack always loads.

**Tech Stack:** Python 3.10+, ComfyUI v3 node API (`comfy_api.latest.io`), torch, numpy, soundfile, huggingface_hub, the fork's `breeze_infer` / `breeze_models` packages. Tests: pytest from the repo root (`addopts = --confcutdir=tests` is already configured).

**Spec:** `docs/superpowers/specs/2026-09-14-breeze-tts-design.md`

## Global Constraints

- Never `pip install` into `E:\AI\ComfyUI\venv`. Test with `PYTHONPATH` and the scratch library folder; run fork tests with `PYTHONPATH="E:\Repos\breeze-tts;<scratch>\pylib" E:\AI\ComfyUI\venv\Scripts\python.exe -W ignore -m pytest ...`. Scratch: `C:\Users\LITTLE~1\AppData\Local\Temp\claude\e--Repos-ComfyUI-IntoTheLatent-Utils\1bfb643d-b625-476a-a86b-3086bf6a2c9b\scratchpad` (contains `pylib/` with pytest, qwen-tts, sox).
- Fork dependency ranges stay: `torch>=2.9`, `transformers>=4.57,<6`, `numpy>=2.0`, `soundfile>=0.13`; Task 1 adds `librosa>=0.10`. No exact pins anywhere.
- Utils pack `requirements.txt` pins the fork to the **tag** `comfyui-v1`, never a branch.
- Node IDs and display names exactly as in spec §4 (prefix `ITL`, category `Into The Latent/audio` — the existing nodes use `Into The Latent/<group>`, so this replaces the spec's `IntoTheLatent/Breeze TTS`).
- `cfg_scale` default: 1.0 for Clone, 4.0 for Design and Direction. Sampling defaults: temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.1, max_new_tokens 750.
- Weights folder: `<ComfyUI>/models/breeze_tts/Breeze-TTS-2/`, HF repo `BreezeBlue/Breeze-TTS-2`. Output sample rate comes from `runtime.sample_rate`, never hard-coded.
- Breeze imports inside the utils pack are lazy (inside functions). `__init__.py` must import cleanly with the fork absent.
- Git: commit in the fork as `Into The Latent <beyondmatrixdevelopments@gmail.com>` (repo-local config already set); pushing needs `gh auth switch --user Into-The-Latent` before and `gh auth switch --user Little-God1983` after. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Spec deviation, agreed: the loader does not implement `fingerprint_inputs`; ComfyUI's default input-based caching already skips re-execution when the two widgets are unchanged, and the module-level cache covers process-wide reuse.
- License header on new utils-pack files: `# <Name> — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.` (matches existing modules).

---

## File structure

**Fork (`E:\Repos\breeze-tts`)**
- Create `breeze_models/qwen_tokenizer/__init__.py`, `configuration_qwen3_tts_tokenizer_v2.py`, `modeling_qwen3_tts_tokenizer_v2.py`, `tokenizer.py` (vendored from qwen-tts 0.1.1, Apache 2.0), `NOTICE`.
- Modify `breeze_models/stream_runtime/core/compat.py`, `breeze_infer/runtime.py`, `pyproject.toml`, `requirements.txt`, `README.md`.
- Create `tests/test_qwen_tokenizer_vendored.py`.

**Utils pack (`E:\Repos\ComfyUI-IntoTheLatent-Utils`)**
- Create `nodes/breeze_tts_core.py` — engine, no ComfyUI imports.
- Create `nodes/breeze_tts_loader.py` — `ITLBreezeTTSLoader`.
- Create `nodes/breeze_tts_generate.py` — the six generate nodes via a factory.
- Create `tests/test_breeze_tts_core.py`, `tests/test_breeze_tts_nodes.py`.
- Modify `__init__.py`, `requirements.txt`, `pyproject.toml`, `README.md`.

---

### Task 1: Vendor the qwen-tts 12 Hz tokenizer into the fork

**Files:**
- Create: `E:\Repos\breeze-tts\breeze_models\qwen_tokenizer\{__init__.py, configuration_qwen3_tts_tokenizer_v2.py, modeling_qwen3_tts_tokenizer_v2.py, tokenizer.py, NOTICE}`
- Modify: `E:\Repos\breeze-tts\breeze_models\stream_runtime\core\compat.py`, `E:\Repos\breeze-tts\breeze_infer\runtime.py:97`, `E:\Repos\breeze-tts\pyproject.toml`, `E:\Repos\breeze-tts\requirements.txt`, `E:\Repos\breeze-tts\README.md`
- Test: `E:\Repos\breeze-tts\tests\test_qwen_tokenizer_vendored.py`

**Interfaces:**
- Consumes: qwen-tts 0.1.1 source in `<scratch>\pylib\qwen_tts\` (already downloaded; its `modeling_qwen3_tts_tokenizer_v2.py` there has the decorator already patched — copy from a *fresh* `pip download` below, not from that patched file, so the diff against upstream is exactly what this task describes).
- Produces: `breeze_models.qwen_tokenizer.Qwen3TTSTokenizer` (class with `from_pretrained(path, **kwargs)`, `encode`, `decode`, `get_output_sample_rate()`), `Qwen3TTSTokenizerV2Config`, `Qwen3TTSTokenizerV2Model`, and the six `Qwen3TTSTokenizerV2*` layer classes that `compat.py` re-exports. Git tag `comfyui-v1`.

- [ ] **Step 1: Get a pristine copy of the three source files**

```bash
S="/c/Users/LITTLE~1/AppData/Local/Temp/claude/e--Repos-ComfyUI-IntoTheLatent-Utils/1bfb643d-b625-476a-a86b-3086bf6a2c9b/scratchpad"
mkdir -p "$S/qwen_src" && cd "$S/qwen_src"
/e/AI/ComfyUI/venv/Scripts/python.exe -m pip download qwen-tts==0.1.1 --no-deps -q -d . && unzip -o -q qwen_tts-0.1.1-*.whl -d wheel
D=/e/Repos/breeze-tts/breeze_models/qwen_tokenizer; mkdir -p "$D"
cp wheel/qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py "$D/"
cp wheel/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py "$D/"
cp wheel/qwen_tts/inference/qwen3_tts_tokenizer.py "$D/tokenizer.py"
grep -n 'check_model_inputs' "$D/modeling_qwen3_tts_tokenizer_v2.py"   # expect: line 42 import, line ~499 "@check_model_inputs()"
```

- [ ] **Step 2: Write the failing import test**

`E:\Repos\breeze-tts\tests\test_qwen_tokenizer_vendored.py`:

```python
"""The vendored Qwen3-TTS 12 Hz tokenizer must import without the qwen_tts package."""
import sys


def test_vendored_tokenizer_imports_without_qwen_tts(monkeypatch):
    monkeypatch.setitem(sys.modules, "qwen_tts", None)  # make `import qwen_tts` raise ImportError
    for name in [m for m in list(sys.modules) if m.startswith("breeze_models.qwen_tokenizer")]:
        monkeypatch.delitem(sys.modules, name)
    from breeze_models.qwen_tokenizer import (
        Qwen3TTSTokenizer,
        Qwen3TTSTokenizerV2Config,
        Qwen3TTSTokenizerV2Model,
    )
    assert callable(Qwen3TTSTokenizer.from_pretrained)
    assert Qwen3TTSTokenizerV2Config.model_type == "qwen3_tts_tokenizer_12hz"
    assert Qwen3TTSTokenizerV2Model.config_class is Qwen3TTSTokenizerV2Config


def test_compat_reexports_from_vendored_package():
    from breeze_models.stream_runtime.core import compat
    from breeze_models import qwen_tokenizer

    assert compat.Qwen3TTSTokenizer is qwen_tokenizer.Qwen3TTSTokenizer
    assert compat.Qwen3TTSTokenizerV2Decoder is qwen_tokenizer.Qwen3TTSTokenizerV2Decoder


def test_runtime_does_not_reference_qwen_tts_package():
    import inspect
    from breeze_infer import runtime

    assert "from qwen_tts" not in inspect.getsource(runtime)
```

- [ ] **Step 3: Run it to verify it fails**

```bash
cd /e/Repos/breeze-tts && PYTHONPATH="E:\Repos\breeze-tts;$(cygpath -w "$S/pylib")" /e/AI/ComfyUI/venv/Scripts/python.exe -W ignore -m pytest tests/test_qwen_tokenizer_vendored.py -q -p no:cacheprovider
```
Expected: 3 failures (`ModuleNotFoundError: breeze_models.qwen_tokenizer`, and the `from qwen_tts` assertion).

- [ ] **Step 4: Patch the vendored modeling file (decorator shim)**

In `modeling_qwen3_tts_tokenizer_v2.py` replace line 42
`from transformers.utils.generic import check_model_inputs` with:

```python
def check_model_inputs(func=None):  # noqa: D401 — name kept so the diff vs upstream stays minimal
    """Compatibility shim. transformers >= 5 replaced the `check_model_inputs()` decorator
    factory with the plain `merge_with_config_defaults` decorator; 4.57 still has the factory."""
    try:
        from transformers.utils.generic import merge_with_config_defaults as _decorate
    except ImportError:  # transformers 4.57.x
        from transformers.utils.generic import check_model_inputs as _factory

        def _decorate(f):
            return _factory()(f)

    if func is None:
        return _decorate
    return _decorate(func)
```

Leave the `@check_model_inputs()` call site untouched — the shim accepts both `@check_model_inputs()` and `@check_model_inputs`.

- [ ] **Step 5: Patch the vendored wrapper (`tokenizer.py`)**

Replace the `from ..core import (...)` block (lines 29–34) with:

```python
from .configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
```

In `from_pretrained` delete the two V1 lines:

```python
        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)
```

and change the two V2 registrations to `exist_ok=True` (the loader may be called more than once per process):

```python
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config, exist_ok=True)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model, exist_ok=True)
```

Then `grep -n 'V1' tokenizer.py` — any remaining `Qwen3TTSTokenizerV1*` reference (e.g. an `isinstance` branch in `get_model_type`) is replaced by the V2-only equivalent or removed. Add below the Apache header: `# Vendored from qwen-tts 0.1.1 (Alibaba Qwen team, Apache-2.0) for Breeze TTS; V1 (25 Hz) tokenizer support removed.`

- [ ] **Step 6: Package init and NOTICE**

`breeze_models/qwen_tokenizer/__init__.py`:

```python
"""Qwen3-TTS 12 Hz audio tokenizer, vendored from qwen-tts 0.1.1 (Apache-2.0).

Only the parts Breeze TTS 2 needs. The weights live in the Breeze checkpoint's
``audio_tokenizer/`` folder; this is just the code that runs them.
"""
from .configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderDecoderBlock,
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
    Qwen3TTSTokenizerV2Model,
)
from .tokenizer import Qwen3TTSTokenizer

__all__ = [
    "Qwen3TTSTokenizer",
    "Qwen3TTSTokenizerV2CausalConvNet",
    "Qwen3TTSTokenizerV2CausalTransConvNet",
    "Qwen3TTSTokenizerV2Config",
    "Qwen3TTSTokenizerV2ConvNeXtBlock",
    "Qwen3TTSTokenizerV2Decoder",
    "Qwen3TTSTokenizerV2DecoderDecoderBlock",
    "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit",
    "Qwen3TTSTokenizerV2Model",
]
```

`breeze_models/qwen_tokenizer/NOTICE`:

```
This directory contains code copied from qwen-tts 0.1.1
(https://pypi.org/project/qwen-tts/, Copyright 2026 The Alibaba Qwen team),
licensed under the Apache License, Version 2.0. Modifications: transformers 5
decorator shim, removal of the 25 Hz (V1) tokenizer, relative imports.
```

- [ ] **Step 7: Rewire compat.py and runtime.py**

Replace the whole of `breeze_models/stream_runtime/core/compat.py` with:

```python
from __future__ import annotations

from breeze_models.qwen_tokenizer import (
    Qwen3TTSTokenizer,
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderDecoderBlock,
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
)

BACKEND_NAME = "vendored"

__all__ = [
    "BACKEND_NAME",
    "Qwen3TTSTokenizer",
    "Qwen3TTSTokenizerV2CausalConvNet",
    "Qwen3TTSTokenizerV2CausalTransConvNet",
    "Qwen3TTSTokenizerV2ConvNeXtBlock",
    "Qwen3TTSTokenizerV2Decoder",
    "Qwen3TTSTokenizerV2DecoderDecoderBlock",
    "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit",
]
```

In `breeze_infer/runtime.py` replace `    from qwen_tts import Qwen3TTSTokenizer` (line 97) with `    from breeze_models.qwen_tokenizer import Qwen3TTSTokenizer`.

`grep -rn 'qwen_tts\|QWEN_TTS_STREAM_BACKEND' --include=*.py .` must now return nothing outside `breeze_models/qwen_tokenizer/` (fix any hit, e.g. in `tests/` or `docker/`).

- [ ] **Step 8: Dependencies and docs**

`pyproject.toml`: add `"librosa>=0.10",` to `dependencies`; delete the `qwen = [...]` extra and its comment; bump `version = "2.0.0.post2"`. `requirements.txt`: add `librosa>=0.10`, delete the two qwen-tts comment lines. `README.md` fork note: replace the `qwen-tts` bullet with `- the Qwen3-TTS 12 Hz audio tokenizer code is vendored (\`breeze_models/qwen_tokenizer\`), so \`qwen-tts\` and its transformers 4.57.3 pin are not needed`.

- [ ] **Step 9: Run the full fork test suite**

```bash
cd /e/Repos/breeze-tts && PYTHONPATH="E:\Repos\breeze-tts;$(cygpath -w "$S/pylib")" /e/AI/ComfyUI/venv/Scripts/python.exe -W ignore -m pytest tests -q -p no:cacheprovider
```
Expected: `50 passed` (47 upstream + 3 new). Then prove the package works with qwen-tts *absent*: run the same command with `PYTHONPATH="E:\Repos\breeze-tts"` only — still `50 passed` (librosa is in the venv; `pylib` only contributed pytest before, so add `$(cygpath -w "$S/pylib")` back if pytest is not found — the point is that `qwen_tts` must not be importable; confirm with `python -c "import qwen_tts"` failing under that PYTHONPATH... note `pylib` contains qwen_tts, so for this check temporarily use `pip install --target "$S/pytest_only" pytest` and PYTHONPATH that folder instead).

- [ ] **Step 10: Commit, tag, push**

```bash
cd /e/Repos/breeze-tts && git add -A && git commit -F - <<'EOF'
feat: vendor the Qwen3-TTS 12 Hz tokenizer, drop the qwen-tts dependency

qwen-tts 0.1.1 hard-pins transformers==4.57.3 / accelerate==1.12.0 and its
tokenizer fails to import on transformers 5 (`check_model_inputs()` is no
longer a factory). Breeze needs three of its files; they now live in
breeze_models/qwen_tokenizer (Apache-2.0, NOTICE added) with a decorator
shim that works on 4.57 and 5.x, and without the 25 Hz V1 tokenizer (sox).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
git tag -a comfyui-v1 -m "First tag consumed by ComfyUI-IntoTheLatent-Utils"
gh auth switch --user Into-The-Latent && git push origin feature/comfyui --follow-tags; gh auth switch --user Little-God1983
```

---

### Task 2: Core — request building and sampling config

**Files:**
- Create: `nodes/breeze_tts_core.py`
- Test: `tests/test_breeze_tts_core.py`

**Interfaces:**
- Produces:
  - `MODES = ("clone", "design", "direction")`
  - `SamplingConfig` frozen dataclass: `temperature: float = 0.9`, `top_k: int = 50`, `top_p: float = 1.0`, `repetition_penalty: float = 1.1`, `max_new_tokens: int = 750`; method `fast_config_kwargs() -> dict` (keys `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens`, `max_seq_len`).
  - `DEFAULT_SAMPLING = SamplingConfig()`
  - `build_request(mode, text, *, reference_text=None, instruction=None, has_reference_audio=False) -> dict` — keys `id`, `text`, `speaker`, plus `ref_text` (clone/direction) and `instruction` (design/direction). Raises `ValueError` naming the missing input.

- [ ] **Step 1: Write the failing tests**

`tests/test_breeze_tts_core.py`:

```python
# Tests for the Breeze TTS engine helpers — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import pytest

from nodes.breeze_tts_core import (
    DEFAULT_SAMPLING,
    MODES,
    SamplingConfig,
    build_request,
)


def test_modes():
    assert MODES == ("clone", "design", "direction")


def test_sampling_defaults_match_upstream():
    s = DEFAULT_SAMPLING
    assert (s.temperature, s.top_k, s.top_p, s.repetition_penalty, s.max_new_tokens) == (0.9, 50, 1.0, 1.1, 750)


def test_sampling_fast_config_kwargs_and_seq_len():
    kw = SamplingConfig(max_new_tokens=1500).fast_config_kwargs()
    assert kw["max_new_tokens"] == 1500
    assert kw["max_seq_len"] == 2048          # max_new_tokens + 512, at least 1024
    assert SamplingConfig().fast_config_kwargs()["max_seq_len"] == 1262
    assert SamplingConfig(max_new_tokens=100).fast_config_kwargs()["max_seq_len"] == 1024
    assert set(kw) == {"temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens", "max_seq_len"}


def test_sampling_is_hashable_and_comparable():
    assert SamplingConfig() == SamplingConfig()
    assert hash(SamplingConfig()) == hash(DEFAULT_SAMPLING)
    assert SamplingConfig(top_k=0) != DEFAULT_SAMPLING


def test_build_request_clone():
    r = build_request("clone", "hello", reference_text="ref words", has_reference_audio=True)
    assert r == {"id": "comfyui", "text": "hello", "speaker": "S0", "ref_text": "ref words"}


def test_build_request_design():
    r = build_request("design", "hello", instruction="a calm deep voice")
    assert r == {"id": "comfyui", "text": "hello", "speaker": "S0", "instruction": "a calm deep voice"}


def test_build_request_direction():
    r = build_request("direction", "hello", reference_text="ref", instruction="whisper", has_reference_audio=True)
    assert r == {"id": "comfyui", "text": "hello", "speaker": "S0", "ref_text": "ref", "instruction": "whisper"}


def test_build_request_strips_and_ignores_unused_inputs():
    r = build_request("design", "  hi  ", instruction=" x ", reference_text="ignored")
    assert r["text"] == "hi" and r["instruction"] == "x" and "ref_text" not in r


@pytest.mark.parametrize("mode,kwargs,missing", [
    ("clone", {"text": "", "reference_text": "r", "has_reference_audio": True}, "text"),
    ("clone", {"text": "t", "reference_text": " ", "has_reference_audio": True}, "reference_text"),
    ("clone", {"text": "t", "reference_text": "r", "has_reference_audio": False}, "reference_audio"),
    ("design", {"text": "t", "instruction": ""}, "instruction"),
    ("direction", {"text": "t", "reference_text": "r", "has_reference_audio": True, "instruction": None}, "instruction"),
    ("direction", {"text": "t", "instruction": "i", "reference_text": None, "has_reference_audio": True}, "reference_text"),
])
def test_build_request_missing_inputs(mode, kwargs, missing):
    with pytest.raises(ValueError, match=missing):
        build_request(mode, **kwargs)


def test_build_request_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        build_request("sing", "t")
```

- [ ] **Step 2: Run to verify failure**

Run (from the repo root): `python -m pytest tests/test_breeze_tts_core.py -q`
Expected: `ModuleNotFoundError: No module named 'nodes.breeze_tts_core'`.

- [ ] **Step 3: Implement**

`nodes/breeze_tts_core.py`:

```python
# Breeze TTS engine — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# ComfyUI-free helpers behind the Breeze TTS 2 nodes (design:
# docs/superpowers/specs/2026-09-14-breeze-tts-design.md). Everything here is testable with
# stubs: no comfy_api, no folder_paths, no model weights. The Breeze runtime itself (our fork
# of breezeblue-ai/breeze-tts) is only ever touched through the `api` object handed to
# generate_audio() and the runtime_factory handed to BreezeHandle.
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

MODES = ("clone", "design", "direction")
SPEAKER = "S0"          # Breeze's single built-in speaker slot
REQUEST_ID = "comfyui"


@dataclass(frozen=True)
class SamplingConfig:
    """Per-generation sampling knobs; defaults equal the fork's update_generation_config_for_breeze()
    and FastStreamingConfig defaults, so Normal and Advanced nodes agree at defaults."""
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    max_new_tokens: int = 750

    def fast_config_kwargs(self) -> dict:
        # max_seq_len must hold prompt + output; upstream's infer.py pairs 1500 tokens with 2048.
        return {
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "top_p": float(self.top_p),
            "repetition_penalty": float(self.repetition_penalty),
            "max_new_tokens": int(self.max_new_tokens),
            "max_seq_len": max(1024, int(self.max_new_tokens) + 512),
        }


DEFAULT_SAMPLING = SamplingConfig()


def _clean(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_request(mode: str, text: str, *, reference_text: str | None = None,
                  instruction: str | None = None, has_reference_audio: bool = False) -> dict:
    """Build the request dict the fork's templates expect for one mode.

    The mode is explicit — the request only ever carries the keys its mode uses, so upstream's
    select_template_name() cannot pick a different template than the node the user placed.
    Raises ValueError naming the missing input.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown Breeze TTS mode {mode!r}; expected one of {MODES}")
    text = _clean(text)
    if not text:
        raise ValueError("text is empty")
    request = {"id": REQUEST_ID, "text": text, "speaker": SPEAKER}
    if mode in ("clone", "direction"):
        if not has_reference_audio:
            raise ValueError("reference_audio is required for voice clone / direction")
        ref = _clean(reference_text)
        if not ref:
            raise ValueError("reference_text is empty — it must be the exact transcript of the reference audio")
        request["ref_text"] = ref
    if mode in ("design", "direction"):
        ins = _clean(instruction)
        if not ins:
            raise ValueError("instruction is empty")
        request["instruction"] = ins
    return request
```

- [ ] **Step 4: Run tests**

`python -m pytest tests/test_breeze_tts_core.py -q` — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nodes/breeze_tts_core.py tests/test_breeze_tts_core.py
git commit -m "feat(breeze): request builder and sampling config for the Breeze TTS engine

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Core — audio conversion (reference WAV in, AUDIO out)

**Files:**
- Modify: `nodes/breeze_tts_core.py`
- Test: `tests/test_breeze_tts_core.py`

**Interfaces:**
- Produces:
  - `audio_to_mono_numpy(waveform) -> np.ndarray` — accepts torch `[B, C, N]` or `[C, N]`; batch 0, channel mean, float32 1-D.
  - `write_reference_wav(audio: dict, directory: str) -> str` — writes `breeze_ref_<uuid>.wav` (PCM_16, mono, the AUDIO dict's own sample rate) and returns its path.
  - `chunks_to_audio(chunks, sample_rate: int) -> dict` — `chunks` iterable of 1-D float arrays; returns `{"waveform": Tensor[1, 1, N] float32, "sample_rate": int}`; `ValueError` on no samples.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_breeze_tts_core.py`)

```python
import numpy as np
import soundfile as sf
import torch

from nodes.breeze_tts_core import audio_to_mono_numpy, chunks_to_audio, write_reference_wav


def test_audio_to_mono_numpy_batched_stereo_means_channels():
    wav = torch.zeros((2, 2, 4))
    wav[0, 0] = 1.0   # batch 0 left = 1, right = 0 -> mean 0.5
    wav[1] = 9.0      # batch 1 must be ignored
    out = audio_to_mono_numpy(wav)
    assert out.shape == (4,) and out.dtype == np.float32
    assert np.allclose(out, 0.5)


def test_audio_to_mono_numpy_accepts_2d():
    out = audio_to_mono_numpy(torch.ones((1, 3)))
    assert out.shape == (3,)


def test_audio_to_mono_numpy_rejects_other_ranks():
    with pytest.raises(ValueError, match="waveform"):
        audio_to_mono_numpy(torch.ones(5))


def test_write_reference_wav_roundtrip(tmp_path):
    sr = 16000
    t = torch.linspace(0, 1, sr)
    audio = {"waveform": torch.stack([t, -t])[None], "sample_rate": sr}   # [1, 2, N]
    path = write_reference_wav(audio, str(tmp_path))
    assert os.path.dirname(path) == str(tmp_path) and os.path.basename(path).startswith("breeze_ref_")
    data, rate = sf.read(path, dtype="float32")
    assert rate == sr and data.ndim == 1 and len(data) == sr
    assert np.allclose(data, 0.0, atol=1e-4)   # channels cancel out


def test_chunks_to_audio_concatenates():
    out = chunks_to_audio([np.array([1, 2], np.float32), np.array([3], np.float32)], 24000)
    assert out["sample_rate"] == 24000
    assert out["waveform"].shape == (1, 1, 3) and out["waveform"].dtype == torch.float32
    assert out["waveform"][0, 0].tolist() == [1.0, 2.0, 3.0]


def test_chunks_to_audio_rejects_empty():
    with pytest.raises(ValueError, match="no audio"):
        chunks_to_audio([], 24000)
    with pytest.raises(ValueError, match="no audio"):
        chunks_to_audio([np.zeros(0, np.float32)], 24000)
```

Add `import os` at the top of the test file.

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_breeze_tts_core.py -q` → ImportError on the three names.

- [ ] **Step 3: Implement** (append to `nodes/breeze_tts_core.py`)

```python
def audio_to_mono_numpy(waveform) -> np.ndarray:
    """ComfyUI AUDIO waveform ([B, C, N] or [C, N]) -> 1-D float32 mono of batch item 0."""
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be [B, C, N] or [C, N], got shape {tuple(waveform.shape)}")
    return waveform.detach().float().mean(dim=0).cpu().numpy().astype(np.float32, copy=False)


def write_reference_wav(audio: dict, directory: str) -> str:
    """Write an AUDIO dict as a mono PCM_16 WAV the Breeze runtime can load by path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"breeze_ref_{uuid.uuid4().hex}.wav")
    sf.write(path, audio_to_mono_numpy(audio["waveform"]), int(audio["sample_rate"]), subtype="PCM_16")
    return path


def chunks_to_audio(chunks, sample_rate: int) -> dict:
    """Concatenate the runtime's 1-D float chunks into a ComfyUI AUDIO dict [1, 1, N]."""
    parts = [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]
    parts = [p for p in parts if p.size]
    if not parts:
        raise ValueError("Breeze TTS produced no audio")
    wave = torch.from_numpy(np.concatenate(parts))
    return {"waveform": wave[None, None, :], "sample_rate": int(sample_rate)}
```

- [ ] **Step 4: Run tests** — Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add nodes/breeze_tts_core.py tests/test_breeze_tts_core.py
git commit -m "feat(breeze): AUDIO <-> reference WAV / chunk conversion helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Core — snapshot check, cache key, model handle

**Files:**
- Modify: `nodes/breeze_tts_core.py`
- Test: `tests/test_breeze_tts_core.py`

**Interfaces:**
- Produces:
  - `REPO_ID = "BreezeBlue/Breeze-TTS-2"`, `SNAPSHOT_DIRNAME = "Breeze-TTS-2"`, `REQUIRED_SNAPSHOT_FILES` tuple (11 relative paths below), `DOWNLOAD_IGNORE = ("assets/*",)`.
  - `missing_snapshot_files(ckpt_dir) -> list[str]`, `snapshot_is_complete(ckpt_dir) -> bool`.
  - `cache_key(ckpt_dir, attention, fast_path) -> tuple[str, str, bool]` (normalised absolute path).
  - `class BreezeHandle` — `__init__(self, key, tokenizer, model, audio_tokenizer, runtime_factory)`; attrs `key`, `tokenizer`, `model`, `audio_tokenizer`; `runtime_for(sampling: SamplingConfig)` builds via `runtime_factory(model, audio_tokenizer, tokenizer, sampling.fast_config_kwargs())` and caches the last runtime by `sampling`; `fast_path` property = `key[2]`.

- [ ] **Step 1: Write the failing tests** (append)

```python
from nodes.breeze_tts_core import (
    REQUIRED_SNAPSHOT_FILES, BreezeHandle, cache_key, missing_snapshot_files, snapshot_is_complete,
)


def _touch_all(root):
    for rel in REQUIRED_SNAPSHOT_FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")


def test_required_snapshot_files_cover_model_tokenizer_and_codec():
    assert "model-00001-of-00002.safetensors" in REQUIRED_SNAPSHOT_FILES
    assert "model-00002-of-00002.safetensors" in REQUIRED_SNAPSHOT_FILES
    assert "audio_tokenizer/model.safetensors" in REQUIRED_SNAPSHOT_FILES
    assert "tokenizer.json" in REQUIRED_SNAPSHOT_FILES


def test_snapshot_complete_and_missing(tmp_path):
    assert missing_snapshot_files(str(tmp_path)) == list(REQUIRED_SNAPSHOT_FILES)
    assert snapshot_is_complete(str(tmp_path)) is False
    _touch_all(tmp_path)
    assert snapshot_is_complete(str(tmp_path)) is True
    (tmp_path / "audio_tokenizer" / "model.safetensors").unlink()
    assert missing_snapshot_files(str(tmp_path)) == ["audio_tokenizer/model.safetensors"]


def test_snapshot_ignores_zero_byte_files(tmp_path):
    _touch_all(tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(b"")
    assert missing_snapshot_files(str(tmp_path)) == ["tokenizer.json"]


def test_cache_key_normalises_path(tmp_path):
    a = cache_key(str(tmp_path / "x" / ".."), "sdpa", True)
    b = cache_key(str(tmp_path), "sdpa", True)
    assert a == b and a[1:] == ("sdpa", True)
    assert cache_key(str(tmp_path), "eager", True) != b


def test_handle_runtime_for_caches_by_sampling():
    calls = []

    def factory(model, audio_tokenizer, tokenizer, kwargs):
        calls.append(kwargs)
        return object()

    h = BreezeHandle(("p", "sdpa", False), "tok", "model", "atok", factory)
    r1 = h.runtime_for(SamplingConfig())
    r2 = h.runtime_for(SamplingConfig())
    assert r1 is r2 and len(calls) == 1 and calls[0]["max_new_tokens"] == 750
    r3 = h.runtime_for(SamplingConfig(top_k=10))
    assert r3 is not r1 and len(calls) == 2 and calls[1]["top_k"] == 10
    r4 = h.runtime_for(SamplingConfig())          # previous config again -> rebuilt (only last is kept)
    assert r4 is not r1 and len(calls) == 3
    assert h.fast_path is False and h.tokenizer == "tok"
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement** (append)

```python
REPO_ID = "BreezeBlue/Breeze-TTS-2"
SNAPSHOT_DIRNAME = "Breeze-TTS-2"
DOWNLOAD_IGNORE = ("assets/*",)   # logos and the leaderboard SVG are not needed
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/preprocessor_config.json",
)


def missing_snapshot_files(ckpt_dir: str) -> list[str]:
    """Required files that are absent or empty (a killed download leaves 0-byte stubs)."""
    out = []
    for rel in REQUIRED_SNAPSHOT_FILES:
        p = os.path.join(ckpt_dir, *rel.split("/"))
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            out.append(rel)
    return out


def snapshot_is_complete(ckpt_dir: str) -> bool:
    return not missing_snapshot_files(ckpt_dir)


def cache_key(ckpt_dir: str, attention: str, fast_path: bool) -> tuple:
    return (os.path.normcase(os.path.abspath(ckpt_dir)), str(attention), bool(fast_path))


class BreezeHandle:
    """What the loader node emits as BREEZE_TTS: the loaded pieces plus a lazily built runtime.

    runtime_factory(model, audio_tokenizer, tokenizer, fast_config_kwargs) -> runtime. Building a
    runtime is cheap next to loading weights (eager mode: no graph capture), so only the most
    recent sampling config's runtime is kept; a change rebuilds it without reloading anything.
    """

    def __init__(self, key, tokenizer, model, audio_tokenizer, runtime_factory):
        self.key = key
        self.tokenizer = tokenizer
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        self._runtime_factory = runtime_factory
        self._runtime = None
        self._runtime_sampling = None

    @property
    def fast_path(self) -> bool:
        return bool(self.key[2])

    def runtime_for(self, sampling: SamplingConfig):
        if self._runtime is None or sampling != self._runtime_sampling:
            self._runtime = self._runtime_factory(self.model, self.audio_tokenizer, self.tokenizer,
                                                  sampling.fast_config_kwargs())
            self._runtime_sampling = sampling
        return self._runtime
```

- [ ] **Step 4: Run tests** — all pass.

- [ ] **Step 5: Commit**

```bash
git add nodes/breeze_tts_core.py tests/test_breeze_tts_core.py
git commit -m "feat(breeze): snapshot completeness check, cache key and model handle

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Core — `generate_audio` end to end against a stub API

**Files:**
- Modify: `nodes/breeze_tts_core.py`
- Test: `tests/test_breeze_tts_core.py`

**Interfaces:**
- Consumes: `build_request`, `write_reference_wav`, `chunks_to_audio`, `BreezeHandle.runtime_for`.
- Produces: `generate_audio(handle, api, *, mode, text, seed, cfg_scale, sampling=DEFAULT_SAMPLING, reference_audio=None, reference_text=None, instruction=None, temp_dir) -> dict` (AUDIO). `api` is any object with `prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, *, guidance_scale, guidance_scale_ref, guidance_scale_ins)`, `select_template_name(request) -> str`, `get_template(name)`, `set_all_seeds(seed)` — exactly the fork's `breeze_infer.templates` / `breeze_infer.runtime` functions.

- [ ] **Step 1: Write the failing tests** (append)

```python
from types import SimpleNamespace

from nodes.breeze_tts_core import generate_audio


class _Runtime:
    sample_rate = 24000

    def __init__(self, log):
        self.log = log

    def iter_audio_chunks(self, inputs, *, request_id=None, seed=None):
        self.log.append(("iter", inputs, request_id, seed))
        yield SimpleNamespace(audio=np.array([0.1, 0.2], np.float32), is_final=False)
        yield SimpleNamespace(audio=np.array([0.3], np.float32), is_final=True)


def _stub(log):
    handle = BreezeHandle(("p", "sdpa", False), "tok", "model", "atok",
                          lambda m, a, t, kw: _Runtime(log))

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, *, guidance_scale,
                       guidance_scale_ref, guidance_scale_ins):
        log.append(("prepare", requests, template, guidance_scale, guidance_scale_ref, guidance_scale_ins))
        return {"prepared": True}

    api = SimpleNamespace(
        prepare_inputs=prepare_inputs,
        select_template_name=lambda r: "tpl:" + ",".join(sorted(k for k in r if k in ("ref_text", "instruction"))),
        get_template=lambda name: ("template", name),
        set_all_seeds=lambda s: log.append(("seed", s)),
    )
    return handle, api


def test_generate_audio_design_mode(tmp_path):
    log = []
    handle, api = _stub(log)
    out = generate_audio(handle, api, mode="design", text="hi", seed=7, cfg_scale=4.0,
                         instruction="deep voice", temp_dir=str(tmp_path))
    assert out["sample_rate"] == 24000 and out["waveform"].shape == (1, 1, 3)
    kinds = [e[0] for e in log]
    assert kinds == ["prepare", "seed", "iter"]
    _, requests, template, cfg, ref, ins = log[0]
    assert requests == [{"id": "comfyui", "text": "hi", "speaker": "S0", "instruction": "deep voice"}]
    assert template == ("template", "tpl:instruction") and (cfg, ref, ins) == (4.0, None, None)
    assert log[1] == ("seed", 7) and log[2][2:] == ("comfyui", 7)
    assert not list(tmp_path.iterdir())     # no temp file for design mode


def test_generate_audio_clone_writes_and_removes_temp_wav(tmp_path):
    log = []
    handle, api = _stub(log)
    seen = {}

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **kw):
        path = requests[0]["ref_audio_path"]
        seen["exists_during"] = os.path.isfile(path)
        seen["path"] = path
        return {}
    api.prepare_inputs = prepare_inputs
    ref = {"waveform": torch.zeros((1, 1, 800)), "sample_rate": 8000}
    generate_audio(handle, api, mode="clone", text="hi", seed=1, cfg_scale=1.0,
                   reference_audio=ref, reference_text="hi", temp_dir=str(tmp_path))
    assert seen["exists_during"] is True and seen["path"].startswith(str(tmp_path))
    assert not os.path.exists(seen["path"])


def test_generate_audio_removes_temp_wav_on_failure(tmp_path):
    log = []
    handle, api = _stub(log)

    def boom(*a, **k):
        raise RuntimeError("cuda oom")
    api.prepare_inputs = boom
    ref = {"waveform": torch.zeros((1, 1, 800)), "sample_rate": 8000}
    with pytest.raises(RuntimeError, match="cuda oom"):
        generate_audio(handle, api, mode="direction", text="hi", seed=1, cfg_scale=4.0,
                       reference_audio=ref, reference_text="hi", instruction="fast", temp_dir=str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_generate_audio_validates_before_touching_runtime(tmp_path):
    log = []
    handle, api = _stub(log)
    with pytest.raises(ValueError, match="reference_audio"):
        generate_audio(handle, api, mode="clone", text="hi", seed=1, cfg_scale=1.0,
                       reference_text="hi", temp_dir=str(tmp_path))
    assert log == []


def test_generate_audio_uses_sampling_for_runtime(tmp_path):
    log = []
    kwargs_seen = []
    handle = BreezeHandle(("p", "sdpa", False), "tok", "model", "atok",
                          lambda m, a, t, kw: (kwargs_seen.append(kw), _Runtime(log))[1])
    _, api = _stub(log)
    generate_audio(handle, api, mode="design", text="hi", seed=1, cfg_scale=4.0, instruction="x",
                   sampling=SamplingConfig(max_new_tokens=300, top_p=0.8), temp_dir=str(tmp_path))
    assert kwargs_seen[0]["max_new_tokens"] == 300 and kwargs_seen[0]["top_p"] == 0.8
```

- [ ] **Step 2: Run to verify failure** — ImportError on `generate_audio`.

- [ ] **Step 3: Implement** (append)

```python
def generate_audio(handle: BreezeHandle, api, *, mode: str, text: str, seed: int, cfg_scale: float,
                   sampling: SamplingConfig = DEFAULT_SAMPLING, reference_audio: dict | None = None,
                   reference_text: str | None = None, instruction: str | None = None,
                   temp_dir: str) -> dict:
    """One synthesis. `api` exposes the fork's prepare_inputs / select_template_name /
    get_template / set_all_seeds (injected so this runs under test without the fork)."""
    request = build_request(mode, text, reference_text=reference_text, instruction=instruction,
                            has_reference_audio=reference_audio is not None)
    ref_path = None
    try:
        if mode in ("clone", "direction"):
            ref_path = write_reference_wav(reference_audio, temp_dir)
            request["ref_audio_path"] = ref_path
        template = api.get_template(api.select_template_name(request))
        inputs = api.prepare_inputs(handle.tokenizer, handle.audio_tokenizer, handle.model, [request],
                                    template, guidance_scale=float(cfg_scale),
                                    guidance_scale_ref=None, guidance_scale_ins=None)
        api.set_all_seeds(int(seed))
        runtime = handle.runtime_for(sampling)
        with torch.inference_mode():
            chunks = [c.audio for c in runtime.iter_audio_chunks(inputs, request_id=REQUEST_ID, seed=int(seed))]
        return chunks_to_audio(chunks, runtime.sample_rate)
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
```

- [ ] **Step 4: Run the whole core test file** — all pass.

- [ ] **Step 5: Commit**

```bash
git add nodes/breeze_tts_core.py tests/test_breeze_tts_core.py
git commit -m "feat(breeze): generate_audio orchestration with injected runtime API

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Loader node with auto-download

**Files:**
- Create: `nodes/breeze_tts_loader.py`
- Test: `tests/test_breeze_tts_nodes.py`

**Interfaces:**
- Consumes: `REPO_ID`, `SNAPSHOT_DIRNAME`, `DOWNLOAD_IGNORE`, `missing_snapshot_files`, `cache_key`, `BreezeHandle` from core.
- Produces: `ITLBreezeTTSLoader` (v3 `io.ComfyNode`, node_id `ITLBreezeTTSLoader`, output type `BREEZE_TTS`); module-level `BREEZE_TTS = io.Custom("BREEZE_TTS")` (the generate nodes import it); `ensure_snapshot(ckpt_dir) -> str`; `load_handle(attention, fast_path) -> BreezeHandle`; `_CACHE: dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_breeze_tts_nodes.py`:

```python
# Node-level tests for the Breeze TTS nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
# Everything that needs weights or a GPU is stubbed; these only run inside a ComfyUI checkout.
import os
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("comfy_api")  # only runs inside a ComfyUI environment
import numpy as np  # noqa: E402
import torch  # noqa: E402

from nodes import breeze_tts_core as core  # noqa: E402
from nodes import breeze_tts_loader as loader  # noqa: E402


def test_loader_schema():
    s = loader.ITLBreezeTTSLoader.define_schema()
    assert s.node_id == "ITLBreezeTTSLoader" and s.display_name == "ITL Breeze TTS Loader"
    names = [i.id for i in s.inputs]
    assert names == ["attention", "fast_path"]
    assert [o.io_type for o in s.outputs] == ["BREEZE_TTS"]


def test_models_folder_registered():
    import folder_paths
    paths = folder_paths.get_folder_paths("breeze_tts")
    assert any(p.replace("\\", "/").endswith("models/breeze_tts") for p in paths)


def test_ensure_snapshot_skips_download_when_complete(tmp_path, monkeypatch):
    for rel in core.REQUIRED_SNAPSHOT_FILES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    monkeypatch.setattr(loader, "_snapshot_download", lambda **kw: pytest.fail("must not download"))
    assert loader.ensure_snapshot(str(tmp_path)) == str(tmp_path)


def test_ensure_snapshot_downloads_when_incomplete(tmp_path, monkeypatch):
    seen = {}

    def fake_download(**kw):
        seen.update(kw)
        for rel in core.REQUIRED_SNAPSHOT_FILES:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    monkeypatch.setattr(loader, "_snapshot_download", fake_download)
    loader.ensure_snapshot(str(tmp_path))
    assert seen["repo_id"] == core.REPO_ID and seen["local_dir"] == str(tmp_path)
    assert list(seen["ignore_patterns"]) == list(core.DOWNLOAD_IGNORE)


def test_ensure_snapshot_reports_still_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_snapshot_download", lambda **kw: None)
    with pytest.raises(RuntimeError, match="tokenizer.json"):
        loader.ensure_snapshot(str(tmp_path))


def test_load_handle_caches_and_evicts(monkeypatch, tmp_path):
    loader._CACHE.clear()
    loads = []
    monkeypatch.setattr(loader, "_require_cuda", lambda: None)
    monkeypatch.setattr(loader, "ensure_snapshot", lambda d: d)
    monkeypatch.setattr(loader, "_snapshot_dir", lambda: str(tmp_path))
    monkeypatch.setattr(loader, "_load_pieces", lambda d, attention: (loads.append(attention), ("tok", "model", "atok"))[1])
    monkeypatch.setattr(loader, "_runtime_factory", lambda fast_path: (lambda m, a, t, kw: "rt"))

    h1 = loader.load_handle("sdpa", False)
    h2 = loader.load_handle("sdpa", False)
    assert h1 is h2 and loads == ["sdpa"] and isinstance(h1, core.BreezeHandle)
    h3 = loader.load_handle("eager", False)
    assert h3 is not h1 and loads == ["sdpa", "eager"]
    assert list(loader._CACHE) == [h3.key]          # old entry evicted
    loader._CACHE.clear()


def test_loader_execute_returns_handle(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(loader, "load_handle", lambda attention, fast_path: sentinel)
    out = loader.ITLBreezeTTSLoader.execute(attention="sdpa", fast_path=False)
    assert out.args[0] is sentinel
```

- [ ] **Step 2: Run to verify failure**

Run from the ComfyUI venv so `comfy_api` resolves: `PYTHONPATH="E:\AI\ComfyUI" E:\AI\ComfyUI\venv\Scripts\python.exe -W ignore -m pytest tests/test_breeze_tts_nodes.py -q -p no:cacheprovider` (add `;<scratch>\pylib` to PYTHONPATH for pytest). Expected: `ModuleNotFoundError: nodes.breeze_tts_loader`.

- [ ] **Step 3: Implement**

`nodes/breeze_tts_loader.py`:

```python
# Breeze TTS Loader node — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Loads Breeze TTS 2 (BreezeBlue, 3B, English + Chinese) and hands it to the generate nodes as
# a BREEZE_TTS handle. The first run downloads the Hugging Face snapshot (~7.2 GB) into
# models/breeze_tts/Breeze-TTS-2/; later runs find it there. Design:
# docs/superpowers/specs/2026-09-14-breeze-tts-design.md. All Breeze imports are lazy so the
# pack loads even when the `breeze-tts` fork is not installed.
import gc
import os

import folder_paths
from comfy_api.latest import io

from .breeze_tts_core import (
    DOWNLOAD_IGNORE, REPO_ID, SNAPSHOT_DIRNAME, BreezeHandle, cache_key, missing_snapshot_files,
)

BREEZE_TTS = io.Custom("BREEZE_TTS")

_MODELS_SUBDIR = "breeze_tts"
folder_paths.add_model_folder_path(_MODELS_SUBDIR, os.path.join(folder_paths.models_dir, _MODELS_SUBDIR))

_CACHE: dict = {}          # cache_key -> BreezeHandle; at most one entry (one 7 GB model resident)
_LICENSE_PRINTED = False
INSTALL_HINT = ("Breeze TTS is not installed. Run ComfyUI Manager's 'Try fix' for "
                "ComfyUI-IntoTheLatent-Utils, or: pip install -r custom_nodes/ComfyUI-IntoTheLatent-Utils/requirements.txt")


def _snapshot_dir() -> str:
    return os.path.join(folder_paths.get_folder_paths(_MODELS_SUBDIR)[0], SNAPSHOT_DIRNAME)


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download
    snapshot_download(**kwargs)


def ensure_snapshot(ckpt_dir: str) -> str:
    """Download the weights if any required file is missing; resumes partial downloads."""
    missing = missing_snapshot_files(ckpt_dir)
    if not missing:
        return ckpt_dir
    print(f"[Breeze TTS] downloading {REPO_ID} (~7.2 GB) to {ckpt_dir} ...")
    try:
        _snapshot_download(repo_id=REPO_ID, local_dir=ckpt_dir, ignore_patterns=list(DOWNLOAD_IGNORE))
    except Exception as e:
        raise RuntimeError(f"Breeze TTS download to {ckpt_dir} failed: {e}") from e
    still = missing_snapshot_files(ckpt_dir)
    if still:
        raise RuntimeError(f"Breeze TTS download finished but files are missing in {ckpt_dir}: {', '.join(still)}")
    print("[Breeze TTS] download done")
    return ckpt_dir


def _require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Breeze TTS needs an NVIDIA GPU (CUDA); the upstream runtime has no CPU path.")


def _load_pieces(ckpt_dir: str, attention: str):
    try:
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    from pathlib import Path
    tokenizer, model, audio_tokenizer = load_runtime(Path(ckpt_dir), device=resolve_device(),
                                                     attn_implementation=attention)
    update_generation_config_for_breeze(model)
    return tokenizer, model, audio_tokenizer


def _runtime_factory(fast_path: bool):
    from breeze_models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    def build(model, audio_tokenizer, tokenizer, kwargs):
        config = FastStreamingConfig(fast_all=True if fast_path else None, **kwargs)
        return FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)
    return build


def _evict_all():
    _CACHE.clear()
    gc.collect()
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass


def load_handle(attention: str, fast_path: bool) -> BreezeHandle:
    global _LICENSE_PRINTED
    _require_cuda()
    ckpt_dir = ensure_snapshot(_snapshot_dir())
    key = cache_key(ckpt_dir, attention, fast_path)
    handle = _CACHE.get(key)
    if handle is not None:
        return handle
    _evict_all()
    if not _LICENSE_PRINTED:
        print("[Breeze TTS] weights are licensed for research and non-commercial use only "
              "(BreezeBlue Research and Non-Commercial License).")
        _LICENSE_PRINTED = True
    tokenizer, model, audio_tokenizer = _load_pieces(ckpt_dir, attention)
    handle = BreezeHandle(key, tokenizer, model, audio_tokenizer, _runtime_factory(fast_path))
    _CACHE[key] = handle
    return handle


class ITLBreezeTTSLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLBreezeTTSLoader",
            display_name="ITL Breeze TTS Loader",
            category="Into The Latent/audio",
            search_aliases=["breeze", "tts", "text to speech", "voice clone"],
            is_experimental=True,
            description="""
Loads Breeze TTS 2 (BreezeBlue — English + Chinese text-to-speech, voice clone / design / direction).

First run downloads the weights (~7.2 GB) from Hugging Face into models/breeze_tts/Breeze-TTS-2/.
Needs an NVIDIA GPU: ~7.7 GiB VRAM (fast_path off) or ~14.4 GiB (fast_path on).
Weights are research / non-commercial (BreezeBlue license).""",
            inputs=[
                io.Combo.Input("attention", options=["sdpa", "eager"], default="sdpa",
                               tooltip="Attention kernel. 'sdpa' is faster; 'eager' is upstream's reference path."),
                io.Boolean.Input("fast_path", default=False,
                                 tooltip="Capture CUDA graphs for every stage (upstream --fast-all). Faster "
                                         "generation, ~2x VRAM, longer first run. Changing sampling settings "
                                         "re-captures."),
            ],
            outputs=[BREEZE_TTS.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, attention="sdpa", fast_path=False) -> io.NodeOutput:
        return io.NodeOutput(load_handle(attention, fast_path))
```

- [ ] **Step 4: Run the node tests** — all loader tests pass.

- [ ] **Step 5: Commit**

```bash
git add nodes/breeze_tts_loader.py tests/test_breeze_tts_nodes.py
git commit -m "feat(breeze): loader node with first-run Hugging Face download and handle cache

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The six generate nodes

**Files:**
- Create: `nodes/breeze_tts_generate.py`
- Test: `tests/test_breeze_tts_nodes.py`

**Interfaces:**
- Consumes: `BREEZE_TTS` from loader; `generate_audio`, `SamplingConfig`, `DEFAULT_SAMPLING`, `MODES` from core.
- Produces: `GENERATE_NODES: dict[node_id, class]` with the six classes, also exported as module attributes `ITLBreezeTTSVoiceClone`, `ITLBreezeTTSVoiceCloneAdvanced`, `ITLBreezeTTSVoiceDesign`, `ITLBreezeTTSVoiceDesignAdvanced`, `ITLBreezeTTSVoiceDirection`, `ITLBreezeTTSVoiceDirectionAdvanced`; `_api()` (lazy namespace of fork functions, monkeypatch target).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_breeze_tts_nodes.py`)

```python
from nodes import breeze_tts_generate as gen  # noqa: E402

EXPECTED = {
    "ITLBreezeTTSVoiceClone": ("ITL Breeze TTS Voice Clone", ["model", "text", "reference_audio", "reference_text", "seed", "cfg_scale"]),
    "ITLBreezeTTSVoiceCloneAdvanced": ("ITL Breeze TTS Voice Clone Advanced", ["model", "text", "reference_audio", "reference_text", "seed", "cfg_scale", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]),
    "ITLBreezeTTSVoiceDesign": ("ITL Breeze TTS Voice Design", ["model", "text", "instruction", "seed", "cfg_scale"]),
    "ITLBreezeTTSVoiceDesignAdvanced": ("ITL Breeze TTS Voice Design Advanced", ["model", "text", "instruction", "seed", "cfg_scale", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]),
    "ITLBreezeTTSVoiceDirection": ("ITL Breeze TTS Voice Direction", ["model", "text", "reference_audio", "reference_text", "instruction", "seed", "cfg_scale"]),
    "ITLBreezeTTSVoiceDirectionAdvanced": ("ITL Breeze TTS Voice Direction Advanced", ["model", "text", "reference_audio", "reference_text", "instruction", "seed", "cfg_scale", "temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]),
}


@pytest.mark.parametrize("node_id", list(EXPECTED))
def test_generate_schemas(node_id):
    cls = gen.GENERATE_NODES[node_id]
    assert getattr(gen, node_id) is cls
    s = cls.define_schema()
    display, inputs = EXPECTED[node_id]
    assert s.node_id == node_id and s.display_name == display
    assert [i.id for i in s.inputs] == inputs
    assert [o.io_type for o in s.outputs] == ["AUDIO"]
    assert s.inputs[0].io_type == "BREEZE_TTS"
    cfg = next(i for i in s.inputs if i.id == "cfg_scale")
    assert cfg.default == (1.0 if "Clone" in node_id else 4.0)
    if "Advanced" in node_id:
        by_id = {i.id: i for i in s.inputs}
        assert (by_id["temperature"].default, by_id["top_k"].default, by_id["top_p"].default,
                by_id["repetition_penalty"].default, by_id["max_new_tokens"].default) == (0.9, 50, 1.0, 1.1, 750)


def _capture(monkeypatch, tmp_path):
    calls = {}

    def fake_generate(handle, api, **kw):
        calls["handle"], calls["api"], calls["kw"] = handle, api, kw
        return {"waveform": torch.zeros((1, 1, 10)), "sample_rate": 24000}
    monkeypatch.setattr(gen, "generate_audio", fake_generate)
    monkeypatch.setattr(gen, "_api", lambda: "API")
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))
    return calls


def test_design_normal_execute(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    out = gen.ITLBreezeTTSVoiceDesign.execute(model="H", text="hi", instruction="deep", seed=3, cfg_scale=4.0)
    assert out.args[0]["sample_rate"] == 24000
    kw = calls["kw"]
    assert calls["handle"] == "H" and calls["api"] == "API"
    assert kw["mode"] == "design" and kw["text"] == "hi" and kw["instruction"] == "deep"
    assert kw["seed"] == 3 and kw["cfg_scale"] == 4.0 and kw["temp_dir"] == str(tmp_path)
    assert kw["sampling"] == core.DEFAULT_SAMPLING
    assert kw["reference_audio"] is None and kw["reference_text"] is None


def test_direction_advanced_execute_passes_sampling(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    ref = {"waveform": torch.zeros((1, 1, 5)), "sample_rate": 8000}
    gen.ITLBreezeTTSVoiceDirectionAdvanced.execute(
        model="H", text="hi", reference_audio=ref, reference_text="hi", instruction="fast",
        seed=1, cfg_scale=2.5, temperature=0.5, top_k=10, top_p=0.9, repetition_penalty=1.3, max_new_tokens=300)
    kw = calls["kw"]
    assert kw["mode"] == "direction" and kw["reference_audio"] is ref and kw["reference_text"] == "hi"
    assert kw["sampling"] == core.SamplingConfig(0.5, 10, 0.9, 1.3, 300)


def test_clone_normal_execute(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, tmp_path)
    ref = {"waveform": torch.zeros((1, 1, 5)), "sample_rate": 8000}
    gen.ITLBreezeTTSVoiceClone.execute(model="H", text="hi", reference_audio=ref, reference_text="hi", seed=1, cfg_scale=1.0)
    assert calls["kw"]["mode"] == "clone" and calls["kw"]["instruction"] is None


def test_api_import_error_names_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "breeze_infer", None)
    monkeypatch.setitem(sys.modules, "breeze_infer.templates", None)
    with pytest.raises(ImportError, match="not installed"):
        gen._api()
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: nodes.breeze_tts_generate`.

- [ ] **Step 3: Implement**

`nodes/breeze_tts_generate.py`:

```python
# Breeze TTS generate nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Voice Clone / Voice Design / Voice Direction, each as Normal and Advanced — six classes built
# by one factory so the input table exists once (design:
# docs/superpowers/specs/2026-09-14-breeze-tts-design.md §3). The engine is
# breeze_tts_core.generate_audio; the fork is imported lazily in _api().
from types import SimpleNamespace

import folder_paths
from comfy_api.latest import io

from .breeze_tts_core import DEFAULT_SAMPLING, SamplingConfig, generate_audio
from .breeze_tts_loader import BREEZE_TTS, INSTALL_HINT

_MODE_INFO = {
    "clone": ("Voice Clone", 1.0,
              "Speak `text` in the voice of `reference_audio`. `reference_text` must be that clip's exact transcript."),
    "design": ("Voice Design", 4.0,
               "Speak `text` in a voice described by `instruction` (e.g. 'a warm, low male voice, slow'). No reference audio."),
    "direction": ("Voice Direction", 4.0,
                  "Speak `text` in the voice of `reference_audio` (+ its exact `reference_text`), steered by "
                  "`instruction` (tone, pace, emotion)."),
}
_EVENTS_HELP = ("Inline vocal events: (laugh), (sigh), (clears throat) in English; [笑], [叹气] in Chinese. "
                "The model is bilingual (English / Chinese) and detects the language from the text.")


def _api():
    """The fork's functions, imported on first use."""
    try:
        from breeze_infer import templates as t
        from breeze_infer.runtime import set_all_seeds
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    return SimpleNamespace(prepare_inputs=t.prepare_inputs, select_template_name=t.select_template_name,
                           get_template=t.get_template, set_all_seeds=set_all_seeds)


def _inputs(mode: str, advanced: bool):
    title, cfg_default, _ = _MODE_INFO[mode]
    ins = [
        BREEZE_TTS.Input("model", tooltip="From ITL Breeze TTS Loader."),
        io.String.Input("text", multiline=True, default="", tooltip="What to say. " + _EVENTS_HELP),
    ]
    if mode in ("clone", "direction"):
        ins += [
            io.Audio.Input("reference_audio", tooltip="A few seconds of the voice to imitate (mono is fine; stereo is downmixed)."),
            io.String.Input("reference_text", multiline=True, default="",
                            tooltip="Exact transcript of reference_audio. Wrong text = wrong voice."),
        ]
    if mode in ("design", "direction"):
        ins.append(io.String.Input("instruction", multiline=True, default="",
                                   tooltip="Voice description (Design) or delivery direction (Direction): tone, pace, emotion."))
    ins += [
        io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
        io.Float.Input("cfg_scale", default=cfg_default, min=0.1, max=10.0, step=0.1,
                       tooltip="Classifier-free guidance. Upstream suggests 1.0 for clone, ~4 for design / direction."),
    ]
    if advanced:
        d = DEFAULT_SAMPLING
        ins += [
            io.Float.Input("temperature", default=d.temperature, min=0.05, max=2.0, step=0.05),
            io.Int.Input("top_k", default=d.top_k, min=0, max=500, tooltip="0 disables top-k."),
            io.Float.Input("top_p", default=d.top_p, min=0.0, max=1.0, step=0.01),
            io.Float.Input("repetition_penalty", default=d.repetition_penalty, min=1.0, max=2.0, step=0.01),
            io.Int.Input("max_new_tokens", default=d.max_new_tokens, min=50, max=1500,
                         tooltip="Caps output length; ~12.5 codec frames per second of audio."),
        ]
    return ins


def _make_node(mode: str, advanced: bool):
    title, _, blurb = _MODE_INFO[mode]
    suffix = " Advanced" if advanced else ""
    node_id = "ITLBreezeTTS" + title.replace(" ", "") + ("Advanced" if advanced else "")
    display = f"ITL Breeze TTS {title}{suffix}"

    class _Node(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=node_id,
                display_name=display,
                category="Into The Latent/audio",
                search_aliases=["breeze", "tts", "text to speech", mode],
                is_experimental=True,
                description=blurb + ("\n\nAdvanced: exposes the sampling settings; defaults equal the Normal node."
                                     if advanced else ""),
                inputs=_inputs(mode, advanced),
                outputs=[io.Audio.Output(display_name="audio")],
            )

        @classmethod
        def execute(cls, model, text, seed, cfg_scale, reference_audio=None, reference_text=None,
                    instruction=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None,
                    max_new_tokens=None) -> io.NodeOutput:
            sampling = DEFAULT_SAMPLING
            if advanced:
                sampling = SamplingConfig(temperature=temperature, top_k=top_k, top_p=top_p,
                                          repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens)
            audio = generate_audio(model, _api(), mode=mode, text=text, seed=seed, cfg_scale=cfg_scale,
                                   sampling=sampling, reference_audio=reference_audio,
                                   reference_text=reference_text, instruction=instruction,
                                   temp_dir=folder_paths.get_temp_directory())
            return io.NodeOutput(audio)

    _Node.__name__ = _Node.__qualname__ = node_id
    return node_id, _Node


GENERATE_NODES = dict(_make_node(m, adv) for m in ("clone", "design", "direction") for adv in (False, True))
globals().update(GENERATE_NODES)

ITLBreezeTTSVoiceClone = GENERATE_NODES["ITLBreezeTTSVoiceClone"]
ITLBreezeTTSVoiceCloneAdvanced = GENERATE_NODES["ITLBreezeTTSVoiceCloneAdvanced"]
ITLBreezeTTSVoiceDesign = GENERATE_NODES["ITLBreezeTTSVoiceDesign"]
ITLBreezeTTSVoiceDesignAdvanced = GENERATE_NODES["ITLBreezeTTSVoiceDesignAdvanced"]
ITLBreezeTTSVoiceDirection = GENERATE_NODES["ITLBreezeTTSVoiceDirection"]
ITLBreezeTTSVoiceDirectionAdvanced = GENERATE_NODES["ITLBreezeTTSVoiceDirectionAdvanced"]
```

If `io.Int.Input` rejects `control_after_generate` or `io.Audio.Input` lacks a `tooltip` kwarg in the installed ComfyUI, check `comfy_api/latest/_io.py` for the exact kwarg names and adjust — the schema test pins ids, defaults and order, not tooltips.

- [ ] **Step 4: Run the node tests** — all pass. Then run the *entire* suite: `python -m pytest -q` (plain venv: node tests skip; ComfyUI venv with `PYTHONPATH=E:\AI\ComfyUI`: everything runs).

- [ ] **Step 5: Commit**

```bash
git add nodes/breeze_tts_generate.py tests/test_breeze_tts_nodes.py
git commit -m "feat(breeze): Voice Clone / Design / Direction nodes, Normal and Advanced

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Registration, requirements, README, version

**Files:**
- Modify: `__init__.py`, `requirements.txt`, `pyproject.toml`, `README.md`
- Test: `tests/test_breeze_tts_nodes.py`

- [ ] **Step 1: Write the failing test** (append)

```python
def test_pack_registers_all_seven_breeze_nodes():
    import importlib
    pack = importlib.import_module("nodes").__name__  # sanity: repo root importable
    import __init__ as root  # the pack's __init__ (repo root is on sys.path via conftest)
    ids = ["ITLBreezeTTSLoader", *EXPECTED]
    for node_id in ids:
        assert node_id in root.NODE_CLASS_MAPPINGS, node_id
        assert root.NODE_CLASS_MAPPINGS[node_id].define_schema().node_id == node_id
        assert node_id in root.NODE_DISPLAY_NAME_MAPPINGS
    assert root.NODE_DISPLAY_NAME_MAPPINGS["ITLBreezeTTSLoader"] == "ITL Breeze TTS Loader"
```

If importing the root `__init__` as a bare module fails because of its relative imports (`from .nodes...`), load it the way ComfyUI does instead: `importlib.import_module("ComfyUI-IntoTheLatent-Utils")` after inserting the parent of the repo root on `sys.path` — mirror whatever `tests/test_nodes_integration.py` does for other nodes if such a pattern exists there; otherwise use `importlib.util.spec_from_file_location("itl_pack", "<repo>/__init__.py", submodule_search_locations=["<repo>"])`.

- [ ] **Step 2: Run to verify failure** — KeyError on `ITLBreezeTTSLoader`.

- [ ] **Step 3: Register**

`__init__.py`: after the `multi_video_loader` import add

```python
from .nodes.breeze_tts_loader import ITLBreezeTTSLoader
from .nodes.breeze_tts_generate import (
    ITLBreezeTTSVoiceClone, ITLBreezeTTSVoiceCloneAdvanced,
    ITLBreezeTTSVoiceDesign, ITLBreezeTTSVoiceDesignAdvanced,
    ITLBreezeTTSVoiceDirection, ITLBreezeTTSVoiceDirectionAdvanced,
)
```

and the seven entries to both mappings (display names per spec §4: "ITL Breeze TTS Loader", "ITL Breeze TTS Voice Clone", "... Voice Clone Advanced", "... Voice Design", "... Voice Design Advanced", "... Voice Direction", "... Voice Direction Advanced").

- [ ] **Step 4: Requirements and version**

`requirements.txt`:

```
pillow>=10.3.0
# Breeze TTS 2 nodes (ITL Breeze TTS *). Our fork of breezeblue-ai/breeze-tts: pip-installable,
# transformers 4.57–5.x, no qwen-tts pin. Pinned to a tag on purpose.
breeze-tts @ git+https://github.com/Into-The-Latent/breeze-tts@comfyui-v1
librosa>=0.10
soundfile>=0.13
huggingface_hub>=0.25

# dev/test only (not needed at runtime):
# pytest>=8.0
```

`pyproject.toml`: `version = "1.8.0"`; `dependencies` gets the same four entries (`"breeze-tts @ git+https://github.com/Into-The-Latent/breeze-tts@comfyui-v1"`, `"librosa>=0.10"`, `"soundfile>=0.13"`, `"huggingface_hub>=0.25"`).

- [ ] **Step 5: README section** — add under the node list, matching the existing per-node style:

```markdown
### ITL Breeze TTS (Loader, Voice Clone / Design / Direction)

Text-to-speech with [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) (BreezeBlue, 3B,
English + Chinese). Three modes, each as a Normal and an Advanced node:

- **Voice Clone** — `reference_audio` + its exact `reference_text` → speak `text` in that voice.
- **Voice Design** — describe the voice in `instruction` ("a calm, deep male voice"); no reference.
- **Voice Direction** — reference audio + transcript + an `instruction` for tone, pace, emotion.

Inline vocal events work in the text: `(laugh)`, `(sigh)`, `(clears throat)`; Chinese `[笑]`, `[叹气]`.
Advanced nodes add `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_new_tokens`
(defaults equal the Normal nodes). `cfg_scale` defaults to 1.0 for Clone and 4.0 for Design / Direction.

**First run** downloads the weights (~7.2 GB) from Hugging Face into `models/breeze_tts/Breeze-TTS-2/`.
Needs an NVIDIA GPU: ~7.7 GiB VRAM, or ~14.4 GiB with the loader's `fast_path` (CUDA graphs).
Changing Advanced sampling settings with `fast_path` on re-captures the graphs (a few seconds).

The model code is installed from our fork (`Into-The-Latent/breeze-tts`, tag `comfyui-v1`), which
works with transformers 4.57–5.x and does not change your torch install. If the nodes report
"Breeze TTS is not installed", run ComfyUI Manager's *Try fix* on this pack (pip needs `git`).

**License:** the node code is GPL-3.0 like the rest of this pack; the Breeze weights are
*research and non-commercial* (BreezeBlue Research and Non-Commercial License).
```

- [ ] **Step 6: Run the full suite in both environments**

```bash
python -m pytest -q                                   # plain: core tests pass, node tests skip
PYTHONPATH="E:\AI\ComfyUI;<scratch>\pylib" E:\AI\ComfyUI\venv\Scripts\python.exe -W ignore -m pytest -q -p no:cacheprovider
```
Expected: no failures. Also `E:\AI\ComfyUI\venv\Scripts\python.exe -c "import ast,sys; ast.parse(open('__init__.py').read())"` and, with `breeze_infer` deliberately absent from `sys.path`, `python -c "import nodes.breeze_tts_loader, nodes.breeze_tts_generate"` from the ComfyUI venv with `PYTHONPATH=E:\AI\ComfyUI` — must import (lazy imports).

- [ ] **Step 7: Commit and push**

```bash
git add __init__.py requirements.txt pyproject.toml README.md tests/test_breeze_tts_nodes.py
git commit -m "feat(breeze): register the seven Breeze TTS nodes, deps, README; bump to 1.8.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
gh auth switch --user Into-The-Latent && git push -u origin feature/breeze-tts; gh auth switch --user Little-God1983
```

---

### Task 9: Manual verification on the reference machine and PR

**Files:** none new. Output: a checklist in the PR description.

- [ ] **Step 1: Install the fork into the ComfyUI venv** — this is the one step that touches the venv, and it is what Manager will do for users. Ask the user for a go before running it (it is reversible: `pip uninstall breeze-tts`):

```bash
E:\AI\ComfyUI\venv\Scripts\python.exe -m pip freeze > "<scratch>/freeze_before.txt"
E:\AI\ComfyUI\venv\Scripts\python.exe -m pip install -r requirements.txt
E:\AI\ComfyUI\venv\Scripts\python.exe -m pip freeze > "<scratch>/freeze_after.txt"
diff "<scratch>/freeze_before.txt" "<scratch>/freeze_after.txt"
```
Expected diff: only `breeze-tts` added (librosa, soundfile, huggingface_hub, pillow already present). torch and transformers lines unchanged — paste the diff into the PR.

- [ ] **Step 2: Link the pack into ComfyUI if not already** (`custom_nodes/ComfyUI-IntoTheLatent-Utils` → junction to `E:\Repos\ComfyUI-IntoTheLatent-Utils`, or confirm it is already there), start ComfyUI, confirm the seven nodes appear under *Into The Latent/audio* and the console shows no import error for the pack.

- [ ] **Step 3: First-run download** — queue Loader → Voice Design ("Hello from Breeze.", instruction "a calm deep male voice") → Save Audio. Console shows the download line, `models/breeze_tts/Breeze-TTS-2/` ends up with the 11 required files, audio plays. Note wall time.

- [ ] **Step 4: Other modes** — Voice Clone with a short mono clip + exact transcript; Voice Direction with the same clip + "whisper, slowly". Then Voice Clone Advanced with `max_new_tokens=200` (audio truncates) and `fast_path=True` on the loader (VRAM ~14 GiB, second run faster). Queue the same graph twice: second run must not print the download or load lines (cache hit).

- [ ] **Step 5: Error paths** — Clone with empty `reference_text` → node error names `reference_text`; Design with empty `instruction` → names `instruction`.

- [ ] **Step 6: Open the PR** to `main` with the spec link, the checklist results, VRAM/time figures and the pip freeze diff; description ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Use `gh auth switch --user Into-The-Latent` before `gh pr create`, switch back after.
