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


REPO_ID = "BreezeBlue/Breeze-TTS-2"
SNAPSHOT_DIRNAME = "Breeze-TTS-2"
DOWNLOAD_IGNORE = ("assets/*",)   # logos and the leaderboard SVG are not needed
REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
    "audio_tokenizer/preprocessor_config.json",
)


def missing_snapshot_files(ckpt_dir: str) -> list[str]:
    """Required files that are absent or empty (a killed download leaves 0-byte stubs)."""
    out = []
    for rel in REQUIRED_SNAPSHOT_FILES:
        p = os.path.join(ckpt_dir, *rel.split("/"))
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            out.append(rel)
    return out


def snapshot_is_complete(ckpt_dir: str) -> bool:
    return not missing_snapshot_files(ckpt_dir)


def cache_key(ckpt_dir: str, attention: str, fast_path: bool) -> tuple:
    return (os.path.normcase(os.path.abspath(ckpt_dir)), str(attention), bool(fast_path))


class BreezeHandle:
    """What the loader node emits as BREEZE_TTS: the loaded pieces plus a lazily built runtime.

    runtime_factory(model, audio_tokenizer, tokenizer, fast_config_kwargs) -> runtime. Building a
    runtime is cheap next to loading weights (eager mode: no graph capture), so only the most
    recent sampling config's runtime is kept; a change rebuilds it without reloading anything.
    """

    def __init__(self, key, tokenizer, model, audio_tokenizer, runtime_factory):
        self.key = key
        self.tokenizer = tokenizer
        self.model = model
        self.audio_tokenizer = audio_tokenizer
        self._runtime_factory = runtime_factory
        self._runtime = None
        self._runtime_sampling = None

    @property
    def fast_path(self) -> bool:
        return bool(self.key[2])

    def runtime_for(self, sampling: SamplingConfig):
        if self._runtime is None or sampling != self._runtime_sampling:
            self._runtime = self._runtime_factory(self.model, self.audio_tokenizer, self.tokenizer,
                                                  sampling.fast_config_kwargs())
            self._runtime_sampling = sampling
        return self._runtime


def generate_audio(handle: BreezeHandle, api, *, mode: str, text: str, seed: int, cfg_scale: float,
                   sampling: SamplingConfig = DEFAULT_SAMPLING, reference_audio: dict | None = None,
                   reference_text: str | None = None, instruction: str | None = None,
                   temp_dir: str) -> dict:
    """One synthesis. `api` exposes the fork's prepare_inputs / select_template_name /
    get_template / set_all_seeds (injected so this runs under test without the fork)."""
    request = build_request(mode, text, reference_text=reference_text, instruction=instruction,
                            has_reference_audio=reference_audio is not None)
    ref_path = None
    try:
        if mode in ("clone", "direction"):
            ref_path = write_reference_wav(reference_audio, temp_dir)
            request["ref_audio_path"] = ref_path
        template = api.get_template(api.select_template_name(request))
        inputs = api.prepare_inputs(handle.tokenizer, handle.audio_tokenizer, handle.model, [request],
                                    template, guidance_scale=float(cfg_scale),
                                    guidance_scale_ref=None, guidance_scale_ins=None)
        api.set_all_seeds(int(seed))
        runtime = handle.runtime_for(sampling)
        with torch.inference_mode():
            chunks = [c.audio for c in runtime.iter_audio_chunks(inputs, request_id=REQUEST_ID, seed=int(seed))]
        return chunks_to_audio(chunks, runtime.sample_rate)
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
