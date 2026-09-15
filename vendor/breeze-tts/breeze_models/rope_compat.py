"""Rotary-embedding init compatibility between transformers 4.57 and 5.x.

transformers 5 removed the ``"default"`` entry from ``ROPE_INIT_FUNCTIONS``; models are expected
to compute the plain RoPE inverse frequencies themselves. This module restores that entry for the
Breeze and vendored Qwen3-TTS rotary classes, using the exact 4.57 formula.
"""
from __future__ import annotations

import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
    """Plain RoPE inverse frequencies (transformers 4.57 `_compute_default_rope_parameters`)."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def resolve_rope_init_fn(rope_type: str):
    """Return the init function for ``rope_type`` on any supported transformers version."""
    if rope_type in ROPE_INIT_FUNCTIONS:
        return ROPE_INIT_FUNCTIONS[rope_type]
    if rope_type == "default":
        return compute_default_rope_parameters
    raise KeyError(f"Unknown rope_type {rope_type!r}; available: {sorted(ROPE_INIT_FUNCTIONS)} + ['default']")
