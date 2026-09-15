# Breeze TTS engine — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# ComfyUI-free helpers behind the Breeze TTS 2 nodes (design:
# docs/superpowers/specs/2026-09-14-breeze-tts-design.md). Everything here is testable with
# stubs: no comfy_api, no folder_paths, no model weights. The Breeze runtime itself (our fork
# of breezeblue-ai/breeze-tts) is only ever touched through the `api` object handed to
# generate_audio() and the runtime_factory handed to BreezeHandle.
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

MODES = ("clone", "design", "direction")
SPEAKER = "S0"          # Breeze's single built-in speaker slot
REQUEST_ID = "comfyui"


@dataclass(frozen=True)
class SamplingConfig:
    """Per-generation sampling knobs; defaults equal the fork's update_generation_config_for_breeze()
    and FastStreamingConfig defaults, so Normal and Advanced nodes agree at defaults."""
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    max_new_tokens: int = 750

    def fast_config_kwargs(self) -> dict:
        # max_seq_len must hold prompt + output; upstream's infer.py pairs 1500 tokens with 2048.
        return {
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "top_p": float(self.top_p),
            "repetition_penalty": float(self.repetition_penalty),
            "max_new_tokens": int(self.max_new_tokens),
            "max_seq_len": max(1024, int(self.max_new_tokens) + 512),
        }


DEFAULT_SAMPLING = SamplingConfig()


def _clean(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_request(mode: str, text: str, *, reference_text: str | None = None,
                  instruction: str | None = None, has_reference_audio: bool = False) -> dict:
    """Build the request dict the fork's templates expect for one mode.

    The mode is explicit — the request only ever carries the keys its mode uses, so upstream's
    select_template_name() cannot pick a different template than the node the user placed.
    Raises ValueError naming the missing input.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown Breeze TTS mode {mode!r}; expected one of {MODES}")
    text = _clean(text)
    if not text:
        raise ValueError("text is empty")
    request = {"id": REQUEST_ID, "text": text, "speaker": SPEAKER}
    if mode in ("clone", "direction"):
        if not has_reference_audio:
            raise ValueError("reference_audio is required for voice clone / direction")
        ref = _clean(reference_text)
        if not ref:
            raise ValueError("reference_text is empty — it must be the exact transcript of the reference audio")
        request["ref_text"] = ref
    if mode in ("design", "direction"):
        ins = _clean(instruction)
        if not ins:
            raise ValueError("instruction is empty")
        request["instruction"] = ins
    return request


def audio_to_mono_numpy(waveform) -> np.ndarray:
    """ComfyUI AUDIO waveform ([B, C, N] or [C, N]) -> 1-D float32 mono of batch item 0."""
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() != 2:
        raise ValueError(f"waveform must be [B, C, N] or [C, N], got shape {tuple(waveform.shape)}")
    return waveform.detach().float().mean(dim=0).cpu().numpy().astype(np.float32, copy=False)


def write_reference_wav(audio: dict, directory: str) -> str:
    """Write an AUDIO dict as a mono PCM_16 WAV the Breeze runtime can load by path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"breeze_ref_{uuid.uuid4().hex}.wav")
    sf.write(path, audio_to_mono_numpy(audio["waveform"]), int(audio["sample_rate"]), subtype="PCM_16")
    return path


def chunks_to_audio(chunks, sample_rate: int) -> dict:
    """Concatenate the runtime's 1-D float chunks into a ComfyUI AUDIO dict [1, 1, N]."""
    parts = [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]
    parts = [p for p in parts if p.size]
    if not parts:
        raise ValueError("Breeze TTS produced no audio")
    wave = torch.from_numpy(np.concatenate(parts))
    return {"waveform": wave[None, None, :], "sample_rate": int(sample_rate)}
