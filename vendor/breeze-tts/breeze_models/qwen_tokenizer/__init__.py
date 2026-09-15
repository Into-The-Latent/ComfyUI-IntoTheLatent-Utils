"""Qwen3-TTS 12 Hz audio tokenizer, vendored from qwen-tts 0.1.1 (Apache-2.0).

Only the parts Breeze TTS 2 needs. The weights live in the Breeze checkpoint's
``audio_tokenizer/`` folder; this is just the code that runs them.
"""
from .configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
from .modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderDecoderBlock,
    Qwen3TTSTokenizerV2DecoderDecoderResidualUnit,
    Qwen3TTSTokenizerV2Model,
)
from .tokenizer import Qwen3TTSTokenizer

__all__ = [
    "Qwen3TTSTokenizer",
    "Qwen3TTSTokenizerV2CausalConvNet",
    "Qwen3TTSTokenizerV2CausalTransConvNet",
    "Qwen3TTSTokenizerV2Config",
    "Qwen3TTSTokenizerV2ConvNeXtBlock",
    "Qwen3TTSTokenizerV2Decoder",
    "Qwen3TTSTokenizerV2DecoderDecoderBlock",
    "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit",
    "Qwen3TTSTokenizerV2Model",
]
