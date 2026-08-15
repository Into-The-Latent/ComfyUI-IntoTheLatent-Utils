"""ComfyUI-IntoTheLatent-Utils — a ComfyUI custom node pack by Into The Latent.

Node classes are registered in NODE_CLASS_MAPPINGS; their friendly labels in
NODE_DISPLAY_NAME_MAPPINGS. Front-end (JS) extensions live under ``web/`` and are
served to the ComfyUI frontend via WEB_DIRECTORY.
"""

from .nodes.ideogram4_nodes import ITLIdeogram4PromptBuilder
from .nodes.ideogram4_style_wizard import ITLIdeogram4StyleWizard
from .nodes.resolution_selector import ITLResolutionSelector
from .nodes.prompt_batch import ITLPromptBatch
from .nodes.save_civitai_metadata import ITLSaveCivitaiMetadata, ITLSaveCivitaiMetadataAdvanced
from .nodes.multi_image_loader import ITLMultiImageLoader, ITLMultiImageLoaderAdvanced
from .nodes.multi_audio_loader import ITLMultiAudioLoader, ITLMultiAudioLoaderAdvanced
from .nodes.multi_video_loader import ITLMultiVideoLoader, ITLMultiVideoLoaderAdvanced

# Key MUST match each node's schema node_id.
NODE_CLASS_MAPPINGS = {
    "ITLIdeogram4PromptBuilder": ITLIdeogram4PromptBuilder,
    "ITLIdeogram4StyleWizard": ITLIdeogram4StyleWizard,
    "ITLResolutionSelector": ITLResolutionSelector,
    "ITLPromptBatch": ITLPromptBatch,
    "ITLSaveCivitaiMetadata": ITLSaveCivitaiMetadata,
    "ITLSaveCivitaiMetadataAdvanced": ITLSaveCivitaiMetadataAdvanced,
    "ITLMultiImageLoader": ITLMultiImageLoader,
    "ITLMultiImageLoaderAdvanced": ITLMultiImageLoaderAdvanced,
    "ITLMultiAudioLoader": ITLMultiAudioLoader,
    "ITLMultiAudioLoaderAdvanced": ITLMultiAudioLoaderAdvanced,
    "ITLMultiVideoLoader": ITLMultiVideoLoader,
    "ITLMultiVideoLoaderAdvanced": ITLMultiVideoLoaderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ITLIdeogram4PromptBuilder": "ITL Ideogram 4 Prompt Builder",
    "ITLIdeogram4StyleWizard": "ITL Ideogram 4 Style Wizard",
    "ITLResolutionSelector": "ITL Resolution Selector",
    "ITLPromptBatch": "ITL Prompt Batch",
    "ITLSaveCivitaiMetadata": "ITL Save Metadata (Civitai)",
    "ITLSaveCivitaiMetadataAdvanced": "ITL Save Metadata (Civitai) Advanced",
    "ITLMultiImageLoader": "ITL Multi Image Loader",
    "ITLMultiImageLoaderAdvanced": "ITL Multi Image Loader Advanced",
    "ITLMultiAudioLoader": "ITL Multi Audio Loader",
    "ITLMultiAudioLoaderAdvanced": "ITL Multi Audio Loader Advanced",
    "ITLMultiVideoLoader": "ITL Multi Video Loader",
    "ITLMultiVideoLoaderAdvanced": "ITL Multi Video Loader Advanced",
}

# Folder of front-end JavaScript served to the ComfyUI client.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
