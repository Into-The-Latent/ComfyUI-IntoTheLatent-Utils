"""Breeze TTS 2 model code.

The package metadata deliberately declares `torch` without a version floor: a floor makes pip
replace the host's torch build (a ComfyUI venv on Windows then gets the CPU-only wheel from PyPI,
which breaks ComfyUI itself). The floor is enforced here instead, at the first import of any
model code, with a message that says what to do. This module is the common import of both entry
points (breeze_infer.runtime and breeze_models.fast_streaming).

Floor: torch 2.7. The fast/CUDA-graph paths write torch._dynamo.config.recompile_limit and
accumulated_recompile_limit (stream_runtime/stream/runtime.py, cudagraph/depth_decoder_graph.py),
names introduced by torch 2.7's rename of cache_size_limit; on older torch they fail mid-generation
with an AttributeError. (The eager path only needs torch.compiler.is_compiling, torch 2.3.)
"""
import re as _re

import torch as _torch

_MIN_TORCH = (2, 7)


def _torch_major_minor(version: str) -> tuple[int, int]:
    # Compare major.minor only: "2.7.0a0+git1234" (NGC container builds) is a pre-release of 2.7.0
    # under PEP 440 and would otherwise fall below the floor although it is a 2.7 torch.
    m = _re.match(r"(\d+)\.(\d+)", str(version))
    if m is None:
        return _MIN_TORCH  # unparseable version string: do not block the import
    return int(m.group(1)), int(m.group(2))


if _torch_major_minor(_torch.__version__) < _MIN_TORCH:
    # RuntimeError on purpose: callers wrap ImportError as "package not installed", which this is not.
    raise RuntimeError(
        f"breeze-tts needs torch >= {_MIN_TORCH[0]}.{_MIN_TORCH[1]}, found {_torch.__version__}. The "
        "package does not pin torch, so pip never replaces your torch build; upgrade torch yourself with "
        "a wheel that matches your GPU (see https://pytorch.org/get-started/locally/)."
    )
