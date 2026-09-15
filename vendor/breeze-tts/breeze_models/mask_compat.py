"""Causal-mask helpers compatible with transformers 4.57 and 5.x.

transformers 5 renamed ``create_causal_mask(input_embeds=...)`` to ``inputs_embeds`` and dropped the
``cache_position`` argument: the key/value offset now comes from the cache object's *current* fill
level. Breeze's CUDA-graph code prebuilds masks for *future* positions against an empty
``StaticCache`` (``depth_decoder_graph._build_attention_masks``), which 4.57 supported through the
explicit ``cache_position``; under 5.x the library would build those masks for offset 0 and the
depth decoder would attend to a single slot. When the installed library lacks ``cache_position``
and a caller gives one for a static cache without a padding mask, the mask is built here exactly
as 4.57 did: bool (True = attend) for SDPA-style backends, additive float for eager.
"""
from __future__ import annotations

import inspect

import torch
from transformers import masking_utils as _mu


def _static_kv_len(past_key_values) -> int | None:
    """Fixed key/value length of a *static* (compileable) cache; ``None`` for every other cache.

    Only a compileable cache (``StaticCache`` and friends) has a fixed key/value extent a mask may
    be prebuilt against, so the explicit branch must not fire for a growing cache: on 5.x
    ``Cache.max_cache_len`` is a deprecated base-class property that resolves for ``DynamicCache``
    too and answers ``-1``, and on a cache with no layers yet it raises. ``get_max_length()`` is the
    non-deprecated 5.x spelling (``max_cache_len`` / ``get_max_cache_shape`` go away in 5.16);
    4.57 has only the latter two.
    """
    if past_key_values is None:
        return None
    try:
        if not getattr(past_key_values, "is_compileable", False):
            return None
        n = None
        get_max_length = getattr(past_key_values, "get_max_length", None)  # 5.x, not deprecated
        if callable(get_max_length):
            n = get_max_length()
        if n is None:
            n = getattr(past_key_values, "max_cache_len", None)  # 4.57
        if n is None and hasattr(past_key_values, "get_max_cache_shape"):
            try:
                n = past_key_values.get_max_cache_shape()
            except TypeError:  # 5.x takes a layer index
                n = past_key_values.get_max_cache_shape(0)
    except (AttributeError, TypeError, ValueError):
        return None
    return int(n) if n is not None and int(n) > 0 else None


def _explicit_position_mask(config, input_embeds, cache_position, kv_len: int, sliding_window: int | None = None):
    """4.57-equivalent causal mask for queries at absolute positions ``cache_position``.

    Returns ``None`` for backends that take no dense mask: 4.57 also returned ``None`` for
    ``flash_attention_2``, and ``flex_attention`` expects a ``BlockMask`` (Breeze does not
    advertise flex support), so a bool tensor would be wrong for both.
    """
    impl = getattr(config, "_attn_implementation", "eager")
    if impl not in ("eager", "sdpa"):
        return None
    batch, q_len = input_embeds.shape[0], input_embeds.shape[1]
    positions = cache_position.to(input_embeds.device).reshape(-1)
    if positions.numel() != q_len:
        # Deliberate divergence: 4.57 derives q_length from cache_position, so it never sees this
        # case. Breeze's graph builders pass one absolute position for a multi-query block, which we
        # read as the position of the *last* query and extrapolate backwards.
        positions = positions[-1] - torch.arange(q_len - 1, -1, -1, device=positions.device)
    kv = torch.arange(kv_len, device=positions.device)
    allowed = kv[None, :] <= positions[:, None]
    if sliding_window:
        allowed &= kv[None, :] > positions[:, None] - sliding_window
    mask = allowed[None, None].expand(batch, 1, q_len, kv_len)
    if impl == "eager":
        dtype = input_embeds.dtype if input_embeds.is_floating_point() else torch.float32
        return torch.where(mask, torch.zeros((), dtype=dtype, device=mask.device),
                           torch.full((), torch.finfo(dtype).min, dtype=dtype, device=mask.device))
    return mask


def _wrap(fn, sliding: bool = False):
    params = inspect.signature(fn).parameters
    rename_embeds = "inputs_embeds" in params and "input_embeds" not in params
    drop_cache_position = "cache_position" not in params

    def compat(*args, **kwargs):
        if drop_cache_position:
            cache_position = kwargs.pop("cache_position", None)
            # The explicit branch reads config/embeds/cache by keyword; a positional caller gets a
            # plain passthrough rather than a misread of args.
            if not args:
                embeds = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
                kv_len = _static_kv_len(kwargs.get("past_key_values"))
                if (cache_position is not None and kwargs.get("attention_mask") is None
                        and embeds is not None and kv_len is not None):
                    window = getattr(kwargs.get("config"), "sliding_window", None) if sliding else None
                    return _explicit_position_mask(kwargs["config"], embeds, cache_position, kv_len, window)
        if rename_embeds and "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        return fn(*args, **kwargs)

    compat.__name__ = fn.__name__
    compat.__wrapped__ = fn
    return compat


create_causal_mask = _wrap(_mu.create_causal_mask)
create_sliding_window_causal_mask = _wrap(_mu.create_sliding_window_causal_mask, sliding=True)
