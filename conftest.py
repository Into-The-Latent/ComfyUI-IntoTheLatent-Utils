# Pytest bootstrap for ComfyUI-IntoTheLatent-Utils. Puts the repo root on sys.path so tests can
# import the comfy-free engine modules (`nodes.civitai_metadata.*`, `nodes.prompt_batch_core`)
# without a ComfyUI install. GPL-3.0, like the rest of the pack.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
