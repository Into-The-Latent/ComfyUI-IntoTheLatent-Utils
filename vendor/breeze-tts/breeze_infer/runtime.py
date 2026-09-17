from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from breeze_models.breeze import BreezeForConditionalGeneration
from breeze_models.buffer_compat import reinit_computed_buffers
from breeze_models.dist_info import get_rank


def get_dist_info() -> tuple[int, int, int]:
    # Asks torch.distributed instead of reading RANK / WORLD_SIZE / LOCAL_RANK: this package is
    # vendored into a ComfyUI pack and the Comfy registry flags any environment variable access
    # (tests/test_registry_scanner_clean.py). Single process, as in ComfyUI: (0, 1, 0).
    rank = get_rank()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    return rank, world_size, local_rank


def resolve_device(explicit_device: str | None = None) -> str:
    if explicit_device:
        return explicit_device

    _, _, local_rank = get_dist_info()
    if torch.cuda.is_available():
        return f"cuda:{local_rank}"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_generation_config_for_breeze(
    model: torch.nn.Module,
    generation_config: dict[str, Any] | None = None,
) -> None:
    generation_config = generation_config or {
        "depth_decoder_do_sample": True,
        "depth_decoder_temperature": 0.9,
        "depth_decoder_top_p": 1.0,
        "depth_decoder_top_k": 50,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "max_new_tokens": 750,
        "temperature": 0.9,
    }

    prefix = "depth_decoder_"
    depth_decoder_attrs = {
        attr[len(prefix) :]: value
        for attr, value in generation_config.items()
        if attr.startswith(prefix)
    }
    vars(model.depth_decoder.generation_config).update(
        {"_from_model_config": False, **depth_decoder_attrs}
    )
    vars(model.generation_config).update(generation_config)


def load_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    attn_implementation: str,
) -> tuple[AutoTokenizer, BreezeForConditionalGeneration, Any]:

    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            rank, world_size, local_rank = get_dist_info()
            raise RuntimeError(
                "Failed to set CUDA device "
                f"device={device} rank={rank} world_size={world_size} local_rank={local_rank} "
                f"device_count={torch.cuda.device_count()}"
            ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        fix_mistral_regex=False,
    )
    model = BreezeForConditionalGeneration.from_pretrained(
        ckpt_dir,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    model.to(device).eval()
    # transformers 5 leaves computed non-persistent buffers uninitialised after from_pretrained.
    reinit_computed_buffers(model)

    from breeze_models.qwen_tokenizer import Qwen3TTSTokenizer

    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    # Load on CPU and move afterwards, like the main model above. Passing `device_map=` here makes
    # transformers demand the optional `accelerate` package, which a fresh ComfyUI venv does not
    # have (and accelerate carries a torch floor, which this package must never pull in).
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(str(bundled_audio_tokenizer))
    audio_tokenizer.model.to(device).eval()
    audio_tokenizer.device = audio_tokenizer.model.device
    reinit_computed_buffers(getattr(audio_tokenizer, "model", None))
    return tokenizer, model, audio_tokenizer
