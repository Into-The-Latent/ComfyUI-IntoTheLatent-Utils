"""Puts the vendored Breeze TTS 2 model code on ``sys.path``.

The Breeze TTS nodes need the ``breeze_models`` / ``breeze_infer`` packages. They ship inside this
pack under ``vendor/breeze-tts`` (our fork of breezeblue-ai/breeze-tts, Apache-2.0; provenance and
re-sync steps in ``vendor/breeze-tts/VENDORED.md``) instead of being pip-installed from GitHub,
because the Comfy registry scanner flags URL dependencies and hides such versions from ComfyUI
Manager. The copy is unchanged, so it is imported under the fork's own top-level package names.

The directory goes at the *front* of ``sys.path`` so a stale ``breeze-tts`` that an older version
of this pack pip-installed into the venv cannot shadow it.
"""
import os
import sys

VENDOR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "vendor", "breeze-tts")
)


def ensure_on_path() -> str:
    """Idempotently prepend the vendored fork to ``sys.path``; returns the directory."""
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    return VENDOR_DIR
