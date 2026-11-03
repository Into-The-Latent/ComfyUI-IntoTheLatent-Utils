"""A do-nothing stand-in for transformers' ``auto_docstring`` decorator.

That decorator only generates ``__doc__`` text, which nothing here reads. Recent transformers also
validates the signature while it does so and prints ``[ERROR] `cache_position` is part of ...'s
signature, but not documented`` for these vendored classes, which looks like a crash in the
ComfyUI console although nothing is wrong. It is an internal helper with no stability promise, so
the vendored code does not depend on it.
"""
from __future__ import annotations


def auto_docstring(obj=None, **_):
    """Supports both ``@auto_docstring`` and ``@auto_docstring(custom_intro=...)``."""
    if obj is not None:
        return obj
    return lambda decorated: decorated
