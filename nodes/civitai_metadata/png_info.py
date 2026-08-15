# PNG text-chunk builder — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Always writes the A1111 `parameters` chunk (what Civitai reads). Optionally also writes ComfyUI's
# native `prompt` + `workflow` chunks (when the node's save_workflow toggle is on) so the PNG stays
# drag-and-drop reproducible in ComfyUI.
import json

from PIL.PngImagePlugin import PngInfo


def build_pnginfo(parameters, *, extra_pnginfo=None, prompt=None):
    info = PngInfo()
    info.add_text("parameters", parameters)
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            info.add_text(key, json.dumps(value))
    return info
