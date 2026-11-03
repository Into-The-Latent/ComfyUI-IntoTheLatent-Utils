"""The encoder half of the Qwen3-TTS audio tokenizer, as self-contained PyTorch.

Voice cloning turns the reference clip into codes with this. The checkpoint stores it in the layout
of transformers' Mimi model, and upstream subclassed that model. transformers is free to reorganise
its model classes between releases and did so in 5.21: the old subclass then crashed at load, so
this file owns the code instead. Nothing here imports a transformers model, so no release can break
it. Only the path the tokenizer uses is kept: whole-clip encoding, without streaming or caches.

The arithmetic follows transformers' ``modeling_mimi.py`` (Apache-2.0, The Kyutai and HuggingFace
teams) and the parameter names match the checkpoint. ``tests/test_qwen_audio_encoder.py`` compares
the codes against transformers' model whenever it can still be taken apart.

Computed values live in plain attributes or are built per call rather than in non-persistent
buffers: transformers 5 leaves those uninitialised after ``from_pretrained`` (see buffer_compat).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _CausalConv1d(nn.Module):
    """Conv1d with Mimi's padding: everything on the left, plus what completes the last frame."""

    def __init__(self, config, in_channels, out_channels, kernel_size, stride=1, dilation=1, pad_mode=None, bias=True):
        super().__init__()
        if not config.use_causal_conv:
            raise ValueError("The Qwen3-TTS audio encoder only supports causal convolutions.")
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation, bias=bias)
        self.stride = stride
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        self.padding_total = self.effective_kernel_size - stride

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.shape[-1]
        frames = math.ceil((length - self.effective_kernel_size + self.padding_total) / self.stride + 1) - 1
        ideal_length = frames * self.stride + self.effective_kernel_size - self.padding_total
        hidden_states = F.pad(hidden_states, (self.padding_total, ideal_length - length), mode=self.pad_mode)
        return self.conv(hidden_states)


class _ResnetBlock(nn.Module):
    def __init__(self, config, dim: int, dilations: list[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        hidden = dim // config.compress
        block: list[nn.Module] = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_channels = dim if i == 0 else hidden
            out_channels = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU(), _CausalConv1d(config, in_channels, out_channels, kernel_size, dilation=dilation)]
        self.block = nn.ModuleList(block)
        self.shortcut = _CausalConv1d(config, dim, dim, 1) if config.use_conv_shortcut else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(hidden_states)
        for layer in self.block:
            hidden_states = layer(hidden_states)
        return residual + hidden_states


class _ConvEncoder(nn.Module):
    """The SEANet convolution stack: waveform in, latent frames out."""

    def __init__(self, config):
        super().__init__()
        layers: list[nn.Module] = [_CausalConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1
        for ratio in reversed(config.upsampling_ratios):
            dim = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                layers.append(_ResnetBlock(config, dim, [config.dilation_growth_rate**j, 1]))
            layers += [nn.ELU(), _CausalConv1d(config, dim, dim * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2
        layers += [nn.ELU(), _CausalConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _LayerScale(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = nn.Parameter(torch.full((config.hidden_size,), config.layer_scale_initial_scale))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.scale * hidden_states


class _Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != "gelu":
            raise ValueError(f"Unsupported audio encoder activation: {config.hidden_act}")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=bias)

    def _rope(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = hidden_states.device
        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=False):
            exponents = torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=device).float() / self.head_dim
            inv_freq = 1.0 / (self.rope_theta**exponents)
            positions = torch.arange(hidden_states.shape[1], device=device).float()
            freqs = positions[:, None] @ inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(hidden_states.dtype), emb.sin().to(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(hidden_states)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        repeats = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)

        output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, -1))


class _TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = _Attention(config)
        self.mlp = _Mlp(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn_layer_scale = _LayerScale(config)
        self.mlp_layer_scale = _LayerScale(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended = self.self_attn(self.input_layernorm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.self_attn_layer_scale(attended)
        fed_forward = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.mlp_layer_scale(fed_forward)


class _Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sliding_window = config.sliding_window
        self.layers = nn.ModuleList(_TransformerLayer(config) for _ in range(config.num_hidden_layers))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each frame sees itself and the sliding_window - 1 frames before it (True = may attend).
        length = hidden_states.shape[1]
        positions = torch.arange(length, device=hidden_states.device)
        distance = positions[:, None] - positions[None, :]
        attention_mask = (distance >= 0) & (distance < self.sliding_window)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class _Codebook(nn.Module):
    def __init__(self, config, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size))
        self.register_buffer("embed_sum", torch.zeros(config.codebook_size, config.codebook_dim))

    @property
    def embed(self) -> torch.Tensor:
        return self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]


class _VectorQuantization(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook = _Codebook(config)


class _ResidualVectorQuantizer(nn.Module):
    def __init__(self, config, num_quantizers: int):
        super().__init__()
        self.layers = nn.ModuleList(_VectorQuantization(config) for _ in range(num_quantizers))
        # output_proj only serves decoding, which the separate decoder does. It stays because the
        # checkpoint stores it and an unexpected key would look like a broken load.
        hidden, quantized = config.hidden_size, config.vector_quantization_hidden_dimension
        self.input_proj = nn.Conv1d(hidden, quantized, 1, bias=False) if quantized != hidden else None
        self.output_proj = nn.Conv1d(quantized, hidden, 1, bias=False) if quantized != hidden else None

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(batch, dim, frames) in, (quantizers, batch, frames) of codebook indices out."""
        residual = self.input_proj(embeddings) if self.input_proj is not None else embeddings
        all_indices = []
        for layer in self.layers:
            embed = layer.codebook.embed
            vectors = residual.permute(0, 2, 1)
            flat = vectors.reshape(-1, vectors.shape[-1])
            indices = torch.cdist(flat[None].float(), embed[None].float(), p=2)[0].argmin(dim=-1)
            indices = indices.view(*vectors.shape[:-1])
            residual = residual - F.embedding(indices, embed).permute(0, 2, 1)
            all_indices.append(indices)
        return torch.stack(all_indices)


class _SplitResidualVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        semantic = config.num_semantic_quantizers
        self.semantic_residual_vector_quantizer = _ResidualVectorQuantizer(config, semantic)
        self.acoustic_residual_vector_quantizer = _ResidualVectorQuantizer(config, config.num_quantizers - semantic)

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Both quantizers start from the same embeddings: that is how the codec was trained.
        return torch.cat(
            [
                self.semantic_residual_vector_quantizer.encode(embeddings),
                self.acoustic_residual_vector_quantizer.encode(embeddings),
            ],
            dim=0,
        )


class Qwen3TTSAudioEncoder(nn.Module):
    """Waveform to codes. ``state_dict`` names are the checkpoint's, after its ``encoder.`` prefix."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = _ConvEncoder(config)
        self.encoder_transformer = _Transformer(config)
        self.downsample = _CausalConv1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
            stride=2,
            pad_mode="replicate",
            bias=False,
        )
        self.quantizer = _SplitResidualVectorQuantizer(config)

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        """(batch, channels, samples) in, (batch, quantizers, frames) of codebook indices out."""
        embeddings = self.encoder(input_values)
        embeddings = self.encoder_transformer(embeddings.transpose(1, 2)).transpose(1, 2)
        embeddings = self.downsample(embeddings)
        return self.quantizer.encode(embeddings).transpose(0, 1)
