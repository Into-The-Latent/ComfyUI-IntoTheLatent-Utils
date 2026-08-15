"""ComfyUI-AI2Go-Utils — a ComfyUI custom node pack by AIKnowledge2Go.

Node classes are registered in NODE_CLASS_MAPPINGS; their friendly labels in
NODE_DISPLAY_NAME_MAPPINGS. Front-end (JS) extensions live under ``web/`` and are
served to the ComfyUI frontend via WEB_DIRECTORY.
"""

from .nodes.ideogram4_nodes import AI2GoIdeogram4PromptBuilder
from .nodes.ideogram4_style_wizard import AI2GoIdeogram4StyleWizard
from .nodes.resolution_selector import AI2GoResolutionSelector
from .nodes.prompt_batch import AI2GoPromptBatch
from .nodes.save_civitai_metadata import AI2GoSaveCivitaiMetadata, AI2GoSaveCivitaiMetadataAdvanced
from .nodes.multi_image_loader import AI2GoMultiImageLoader, AI2GoMultiImageLoaderAdvanced
from .nodes.multi_audio_loader import AI2GoMultiAudioLoader, AI2GoMultiAudioLoaderAdvanced
from .nodes.multi_video_loader import AI2GoMultiVideoLoader, AI2GoMultiVideoLoaderAdvanced

# Key MUST match each node's schema node_id.
NODE_CLASS_MAPPINGS = {
    "AI2GoIdeogram4PromptBuilder": AI2GoIdeogram4PromptBuilder,
    "AI2GoIdeogram4StyleWizard": AI2GoIdeogram4StyleWizard,
    "AI2GoResolutionSelector": AI2GoResolutionSelector,
    "AI2GoPromptBatch": AI2GoPromptBatch,
    "AI2GoSaveCivitaiMetadata": AI2GoSaveCivitaiMetadata,
    "AI2GoSaveCivitaiMetadataAdvanced": AI2GoSaveCivitaiMetadataAdvanced,
    "AI2GoMultiImageLoader": AI2GoMultiImageLoader,
    "AI2GoMultiImageLoaderAdvanced": AI2GoMultiImageLoaderAdvanced,
    "AI2GoMultiAudioLoader": AI2GoMultiAudioLoader,
    "AI2GoMultiAudioLoaderAdvanced": AI2GoMultiAudioLoaderAdvanced,
    "AI2GoMultiVideoLoader": AI2GoMultiVideoLoader,
    "AI2GoMultiVideoLoaderAdvanced": AI2GoMultiVideoLoaderAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AI2GoIdeogram4PromptBuilder": "AI2Go Ideogram 4 Prompt Builder",
    "AI2GoIdeogram4StyleWizard": "AI2Go Ideogram 4 Style Wizard",
    "AI2GoResolutionSelector": "AI2Go Resolution Selector",
    "AI2GoPromptBatch": "AI2Go Prompt Batch",
    "AI2GoSaveCivitaiMetadata": "AI2Go Save Metadata (Civitai)",
    "AI2GoSaveCivitaiMetadataAdvanced": "AI2Go Save Metadata (Civitai) Advanced",
    "AI2GoMultiImageLoader": "AI2Go Multi Image Loader",
    "AI2GoMultiImageLoaderAdvanced": "AI2Go Multi Image Loader Advanced",
    "AI2GoMultiAudioLoader": "AI2Go Multi Audio Loader",
    "AI2GoMultiAudioLoaderAdvanced": "AI2Go Multi Audio Loader Advanced",
    "AI2GoMultiVideoLoader": "AI2Go Multi Video Loader",
    "AI2GoMultiVideoLoaderAdvanced": "AI2Go Multi Video Loader Advanced",
}

# Folder of front-end JavaScript served to the ComfyUI client.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
