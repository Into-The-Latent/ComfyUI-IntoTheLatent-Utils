"""StaticCache compatibility between transformers 4.57 and 5.x.

transformers 5 changed ``StaticLayer.lazy_initialization(key_states)`` to
``lazy_initialization(key_states, value_states)``. Breeze's CUDA-graph code pre-initialises the
cache layers with a dummy key tensor before capture; this helper passes the same tensor for both
when the installed version asks for two.
"""
from __future__ import annotations

import inspect


def lazy_init_static_layer(layer, dummy_states) -> None:
    """Initialise one StaticCache layer with ``dummy_states`` on either transformers API."""
    params = inspect.signature(layer.lazy_initialization).parameters
    if "value_states" in params:
        layer.lazy_initialization(dummy_states, dummy_states)
    else:
        layer.lazy_initialization(dummy_states)


def cache_layer_kv(past_key_values, layer_idx: int):
    """(keys, values) of one cache layer.

    Both 4.57.1 and 5.x expose ``.layers``, so that is the normal path; the indexing fallback is
    kept for older/other cache objects (4.57 caches are still indexable, plain tuple lists are not
    ``Cache`` instances at all).
    """
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        layer = layers[layer_idx]
        return layer.keys, layer.values
    return past_key_values[layer_idx]
