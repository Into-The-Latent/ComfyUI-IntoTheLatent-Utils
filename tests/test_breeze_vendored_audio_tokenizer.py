"""The vendored audio tokenizer must load through from_pretrained on the installed transformers.

Why this exists: transformers 5.21 reorganised its Mimi model and every Breeze load crashed there,
while all stubbed tests stayed green because none of them built this model. A tiny random model on
the CPU runs the real load path without the checkpoint or a GPU. Run the pack's tests after each
re-sync of vendor/breeze-tts, as VENDORED.md says, and against the newest transformers before a release.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "breeze-tts"


@pytest.fixture
def qwen(monkeypatch):
    """The tokenizer package from the vendored tree, even if a stale pip copy is installed."""
    monkeypatch.syspath_prepend(str(VENDOR))
    for name in [m for m in sys.modules if m == "breeze_models" or m.startswith("breeze_models.")]:
        monkeypatch.delitem(sys.modules, name)
    from breeze_models.qwen_tokenizer import modeling_qwen3_tts_tokenizer_v2 as modeling
    from breeze_models.qwen_tokenizer import configuration_qwen3_tts_tokenizer_v2 as configuration

    assert Path(modeling.__file__).is_relative_to(VENDOR)
    return modeling, configuration


def test_audio_tokenizer_round_trip(qwen, tmp_path):
    modeling, configuration = qwen
    config = configuration.Qwen3TTSTokenizerV2Config(
        encoder_config={
            "hidden_size": 32, "intermediate_size": 64, "num_filters": 4, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16, "codebook_size": 16,
            "codebook_dim": 8, "vector_quantization_hidden_dimension": 8, "num_quantizers": 4,
            "sliding_window": 5, "_frame_rate": 12.5,
        },
        decoder_config={
            "attention_bias": False, "attention_dropout": 0.0, "latent_dim": 64, "codebook_dim": 16,
            "codebook_size": 16, "decoder_dim": 32, "hidden_act": "silu", "hidden_size": 32,
            "intermediate_size": 64, "layer_scale_initial_scale": 0.01, "max_position_embeddings": 128,
            "head_dim": 16, "num_attention_heads": 2, "num_hidden_layers": 1, "num_key_value_heads": 2,
            "num_quantizers": 4, "num_semantic_quantizers": 1, "rms_norm_eps": 1e-05, "rope_theta": 10000,
            "semantic_codebook_size": 16, "sliding_window": 5, "upsample_rates": [8, 5, 4, 3],
            "upsampling_ratios": [2, 2], "vector_quantization_hidden_dimension": 16,
        },
        encoder_valid_num_quantizers=4,
    )
    torch.manual_seed(1)
    model = modeling.Qwen3TTSTokenizerV2Model(config).eval()
    for name, buffer in model.named_buffers():
        if name.startswith("encoder.") and name.endswith("embed_sum"):
            torch.nn.init.normal_(buffer)
    model.save_pretrained(tmp_path)

    loaded = modeling.Qwen3TTSTokenizerV2Model.from_pretrained(tmp_path).eval()

    saved = model.state_dict()
    for name, tensor in loaded.encoder.state_dict().items():
        assert torch.equal(tensor, saved[f"encoder.{name}"]), name
    audio = torch.randn(1, 4800, generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        codes = loaded.encode(audio, torch.ones_like(audio, dtype=torch.long), return_dict=True).audio_codes[0]
        decoded = loaded.decode(codes[None], return_dict=True).audio_values[0]
    assert codes.shape == (3, 4)  # ceil(4800 / 1920) frames, 4 quantizers
    assert decoded.shape[-1] == 3 * 1920
    assert torch.isfinite(decoded).all()
