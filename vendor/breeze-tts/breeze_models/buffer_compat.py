"""Re-materialise computed, non-persistent buffers after ``from_pretrained``.

transformers 5 builds models on the meta device and replaces every non-persistent buffer with
``torch.empty_like`` at load time; only modules it recognises (``*RotaryEmbedding`` with an
``original_inv_freq`` attribute) get recomputed. Breeze has several computed buffers that are not
recognised -- the audio token offsets, the vendored Qwen3-TTS decoder rotary (its class is spelled
``RotatoryEmbedding``), the T5Gemma 2 per-layer rotary buffers and its embedding scale -- and they
came back as uninitialised memory. This pass recomputes all of them deterministically. On
transformers 4.57 the buffers were never lost, so it only rewrites them with the same values.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def reinit_computed_buffers(root) -> list[str]:
    """Recompute the computed buffers this module knows about (the ones listed above) under
    ``root``. Returns the qualified names touched."""
    touched: list[str] = []
    if not isinstance(root, torch.nn.Module):  # test doubles
        return touched
    for name, module in root.named_modules():
        # Audio token offsets: arange(num_codebooks) * vocab_size (BreezeBackboneModelEmbeddings).
        offsets = getattr(module, "audio_tokens_offsets", None)
        if isinstance(offsets, torch.Tensor) and hasattr(module, "num_codebooks") and hasattr(module, "vocab_size"):
            offsets.copy_(torch.arange(module.num_codebooks, device=offsets.device) * module.vocab_size)
            touched.append(f"{name}.audio_tokens_offsets")
        # Rotary embeddings that carry their own init function (Breeze, vendored Qwen3-TTS decoder).
        rope_init_fn = getattr(module, "rope_init_fn", None)
        inv_freq = getattr(module, "inv_freq", None)
        if callable(rope_init_fn) and isinstance(inv_freq, torch.Tensor):
            value, scaling = rope_init_fn(module.config, inv_freq.device)
            inv_freq.copy_(value.to(inv_freq.dtype))
            module.attention_scaling = scaling
            original = getattr(module, "original_inv_freq", None)
            if isinstance(original, torch.Tensor) and original.data_ptr() != inv_freq.data_ptr():
                # On 5.x this is a plain attribute the loader leaves on the meta device; copy_ into
                # meta storage is a silent no-op, so rebind it to a real tensor instead.
                module.original_inv_freq = value.to(device=inv_freq.device, dtype=original.dtype)
            touched.append(f"{name}.inv_freq")
        # T5Gemma 2 shim: one inv_freq buffer per layer type.
        compute_inv_freq = getattr(module, "_compute_inv_freq", None)
        layer_types = getattr(module, "layer_types", None)
        if callable(compute_inv_freq) and layer_types:
            for layer_type in layer_types:
                buf = getattr(module, f"{layer_type}_inv_freq", None)
                if isinstance(buf, torch.Tensor):
                    value, scaling = compute_inv_freq(module.config, layer_type, buf.device)
                    buf.copy_(value.to(buf.dtype))
                    setattr(module, f"{layer_type}_attention_scaling", scaling)
                    touched.append(f"{name}.{layer_type}_inv_freq")
        # T5Gemma 2 shim: scaled word embedding.
        embed_scale = getattr(module, "embed_scale", None)
        if isinstance(embed_scale, torch.Tensor) and hasattr(module, "scalar_embed_scale"):
            embed_scale.fill_(module.scalar_embed_scale)
            touched.append(f"{name}.embed_scale")
    return touched
