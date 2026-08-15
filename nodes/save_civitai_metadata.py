# Save Metadata (Civitai) nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Two output nodes that save PNG(s) with the A1111 `parameters` chunk Civitai reads (see the design at
# docs/superpowers/specs/2026-07-12-save-civitai-metadata-design.md). Basic auto-traces the graph;
# Advanced adds override sockets for the volatile fields. The graph tracer is mirrored in
# web/js/save_civitai_metadata.js (the Test button) — keep the two in sync.
import os
import logging
from datetime import datetime

import numpy as np
from PIL import Image

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io, ui

try:
    from server import PromptServer
except Exception:  # pragma: no cover - server always present in ComfyUI, guarded for safety
    PromptServer = None

from .civitai_metadata.tracer import trace
from .civitai_metadata.sampler_names import to_a1111_sampler
from .civitai_metadata.a1111 import format_parameters
from .civitai_metadata.hashing import HashCache
from .civitai_metadata.png_info import build_pnginfo
from .civitai_metadata.filenames import expand_date_tokens

logger = logging.getLogger(__name__)

# Persistent AutoV2 hash cache lives in the pack root (git-ignored).
_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".hash_cache.json")

# Override keys the Advanced node can supply; each maps to a TraceResult attribute.
_OVERRIDE_KEYS = ("positive", "negative", "steps", "cfg", "seed", "sampler_name", "scheduler")


def _save_with_metadata(cls, images, filename_prefix, save_workflow, overrides):
    """Shared engine call + PNG save for both nodes. Returns an io.NodeOutput."""
    prompt = cls.hidden.prompt or {}
    result = trace(prompt, cls.hidden.unique_id)

    # Advanced overrides: a supplied (non-None) socket value wins over the trace.
    for key in _OVERRIDE_KEYS:
        val = (overrides or {}).get(key)
        if val is not None and val != "":
            setattr(result, key, val)
            if key in result.unresolved:
                result.unresolved.remove(key)

    # Resolve model + LoRA file paths to AutoV2 hashes (cached).
    cache = HashCache(_CACHE_PATH)
    model_hash = cache.get(
        folder_paths.get_full_path(result.model_folder or "checkpoints", result.model_file)
    ) if result.model_file else None
    for lo in result.loras:
        path = folder_paths.get_full_path("loras", lo["file"]) if lo.get("file") else None
        lo["hash"] = cache.get(path) if path else None

    height, width = int(images[0].shape[0]), int(images[0].shape[1])
    sampler_label = to_a1111_sampler(result.sampler_name or "", result.scheduler or "")
    parameters = format_parameters(
        positive=result.positive, negative=result.negative, steps=result.steps, cfg=result.cfg,
        seed=result.seed, sampler=sampler_label, width=width, height=height,
        model_name=result.model_name, model_hash=model_hash, loras=result.loras,
        denoise=result.denoise, clip_skip=result.clip_skip,
    )

    # Expand %date:FORMAT% (frontend feature that doesn't reach a custom node's widget) before the
    # rest of the tokens (%year%/subfolders) that get_save_image_path itself handles.
    filename_prefix = expand_date_tokens(filename_prefix, datetime.now())
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory(), width, height)

    results = []
    for batch_number, image in enumerate(images):
        arr = 255.0 * image.cpu().numpy()
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        info = None
        if not args.disable_metadata:
            info = build_pnginfo(
                parameters,
                extra_pnginfo=(cls.hidden.extra_pnginfo if save_workflow else None),
                prompt=(cls.hidden.prompt if save_workflow else None),
            )
        fname = filename.replace("%batch_num%", str(batch_number))
        file = f"{fname}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), pnginfo=info, compress_level=4)
        results.append(ui.SavedResult(filename=file, subfolder=subfolder, type=io.FolderType.output))
        counter += 1

    if result.unresolved:
        msg = "ITL Save Metadata: unresolved fields -> %s (use the Advanced node's sockets)"
        logger.warning(msg, ", ".join(result.unresolved))
        if PromptServer is not None and cls.hidden.unique_id is not None:
            try:
                PromptServer.instance.send_progress_text(
                    "⚠ unresolved: " + ", ".join(result.unresolved), cls.hidden.unique_id)
            except Exception:
                pass

    return io.NodeOutput(images, ui={"images": results})


_METADATA_DESC = (
    "Save PNG(s) with A1111-style metadata Civitai can read (positive/negative, steps, CFG, sampler, "
    "seed, size, model + LoRA AutoV2 hashes). Auto-traces the KSampler from the saved image — including "
    "our Prompt Batch node's actual line for this run. save_workflow (off) additionally embeds the "
    "ComfyUI workflow like the stock Save Image."
)


class ITLSaveCivitaiMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLSaveCivitaiMetadata",
            display_name="ITL Save Metadata (Civitai)",
            category="Into The Latent/image",
            search_aliases=["save", "civitai", "metadata", "a1111", "parameters", "png"],
            is_output_node=True,
            is_experimental=True,
            description=_METADATA_DESC,
            inputs=[
                io.Image.Input("images", tooltip="Images to save."),
                io.String.Input("filename_prefix", default="ITL",
                                tooltip="Filename prefix (supports %date:...% tokens like Save Image)."),
                io.Boolean.Input("save_workflow", default=False,
                                 tooltip="Also embed the ComfyUI workflow/prompt chunks (off = only the "
                                         "Civitai `parameters` chunk)."),
            ],
            outputs=[io.Image.Output(display_name="images")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images, filename_prefix="ITL", save_workflow=False) -> io.NodeOutput:
        return _save_with_metadata(cls, images, filename_prefix, save_workflow, overrides=None)


class ITLSaveCivitaiMetadataAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLSaveCivitaiMetadataAdvanced",
            display_name="ITL Save Metadata (Civitai) Advanced",
            category="Into The Latent/image",
            search_aliases=["save", "civitai", "metadata", "a1111", "parameters", "png", "advanced"],
            is_output_node=True,
            is_experimental=True,
            description=_METADATA_DESC + " Advanced: wire positive/negative (and optionally the sampler "
                        "settings) to override the trace — the reliable path for dynamic prompts.",
            inputs=[
                io.Image.Input("images", tooltip="Images to save."),
                io.String.Input("filename_prefix", default="ITL",
                                tooltip="Filename prefix (supports %date:...% tokens like Save Image)."),
                io.Boolean.Input("save_workflow", default=False,
                                 tooltip="Also embed the ComfyUI workflow/prompt chunks (off = only the "
                                         "Civitai `parameters` chunk)."),
                io.String.Input("positive", optional=True, force_input=True,
                                tooltip="Override positive prompt (wire the Prompt Batch 'positive' output)."),
                io.String.Input("negative", optional=True, force_input=True,
                                tooltip="Override negative prompt (wire the Prompt Batch 'negative' output)."),
                io.Int.Input("steps", optional=True, force_input=True, tooltip="Override steps."),
                io.Float.Input("cfg", optional=True, force_input=True, tooltip="Override CFG scale."),
                io.Int.Input("seed", optional=True, force_input=True, tooltip="Override seed."),
                io.String.Input("sampler_name", optional=True, force_input=True, tooltip="Override sampler name."),
                io.String.Input("scheduler", optional=True, force_input=True, tooltip="Override scheduler."),
            ],
            outputs=[io.Image.Output(display_name="images")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, images, filename_prefix="ITL", save_workflow=False, positive=None, negative=None,
                steps=None, cfg=None, seed=None, sampler_name=None, scheduler=None) -> io.NodeOutput:
        overrides = {"positive": positive, "negative": negative, "steps": steps, "cfg": cfg,
                     "seed": seed, "sampler_name": sampler_name, "scheduler": scheduler}
        return _save_with_metadata(cls, images, filename_prefix, save_workflow, overrides=overrides)
