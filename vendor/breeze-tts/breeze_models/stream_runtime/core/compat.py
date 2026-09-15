from __future__ import annotations

from breeze_models.qwen_tokenizer import (
    Qwen3TTSTokenizer,
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderDecoderBlock,
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
)

BACKEND_NAME = "vendored"

__all__ = [
    "BACKEND_NAME",
    "Qwen3TTSTokenizer",
    "Qwen3TTSTokenizerV2CausalConvNet",
    "Qwen3TTSTokenizerV2CausalTransConvNet",
    "Qwen3TTSTokenizerV2ConvNeXtBlock",
    "Qwen3TTSTokenizerV2Decoder",
    "Qwen3TTSTokenizerV2DecoderDecoderBlock",
    "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit",
]
