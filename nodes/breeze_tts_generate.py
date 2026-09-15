# Breeze TTS generate nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Voice Clone / Voice Design / Voice Direction, each as Normal and Advanced — six classes built
# by one factory so the input table exists once (design:
# docs/superpowers/specs/2026-09-14-breeze-tts-design.md §3). The engine is
# breeze_tts_core.generate_audio; the vendored fork (vendor/breeze-tts) is imported lazily in _api().
from types import SimpleNamespace

import folder_paths
from comfy_api.latest import io

from .breeze_tts_core import DEFAULT_SAMPLING, SamplingConfig, generate_audio
from .breeze_tts_loader import BREEZE_TTS, INSTALL_HINT, resolve_handle, unload
from .breeze_vendor import ensure_on_path

_MODE_INFO = {
    # (title, cfg_scale default, blurb). Clone's cfg default is unused — the clone template has no
    # negative prompt, so Clone nodes don't get a cfg_scale input at all (core rejects != 1.0).
    "clone": ("Voice Clone", None,
              "Speak `text` in the voice of `reference_audio`. `reference_text` must be that clip's exact transcript."),
    "design": ("Voice Design", 4.0,
               "Speak `text` in a voice described by `instruction` (e.g. 'a warm, low male voice, slow'). No reference audio."),
    "direction": ("Voice Direction", 4.0,
                  "Speak `text` in the voice of `reference_audio` (+ its exact `reference_text`), steered by "
                  "`instruction` (tone, pace, emotion)."),
}
_EVENTS_HELP = ("Inline vocal events: (laugh), (sigh), (clears throat) in English; [笑], [叹气] in Chinese. "
                "The model is bilingual (English / Chinese) and detects the language from the text.")


def _api():
    """The fork's functions, imported on first use."""
    ensure_on_path()
    try:
        from breeze_infer import templates as t
        from breeze_infer.runtime import set_all_seeds
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    return SimpleNamespace(prepare_inputs=t.prepare_inputs, select_template_name=t.select_template_name,
                           get_template=t.get_template, set_all_seeds=set_all_seeds)


def _inputs(mode: str, advanced: bool):
    title, cfg_default, _ = _MODE_INFO[mode]
    ins = [
        BREEZE_TTS.Input("model", tooltip="From ITL Breeze TTS Loader."),
        io.String.Input("text", multiline=True, default="", tooltip="What to say. " + _EVENTS_HELP),
    ]
    if mode in ("clone", "direction"):
        ins += [
            io.Audio.Input("reference_audio", tooltip="A few seconds of the voice to imitate (mono is fine; stereo is downmixed)."),
            io.String.Input("reference_text", multiline=True, default="",
                            tooltip="Exact transcript of reference_audio. Wrong text = wrong voice. "
                                    "ITL Whisper Transcribe can produce it from the clip."),
        ]
    if mode in ("design", "direction"):
        ins.append(io.String.Input("instruction", multiline=True, default="",
                                   tooltip="Voice description (Design) or delivery direction (Direction): tone, pace, emotion."))
    ins.append(io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFF, control_after_generate=True))
    if mode != "clone":
        ins.append(io.Float.Input("cfg_scale", default=cfg_default, min=0.1, max=10.0, step=0.1,
                                  tooltip="Classifier-free guidance. Upstream suggests ~4 for design / direction."))
    if advanced:
        d = DEFAULT_SAMPLING
        ins += [
            io.Float.Input("temperature", default=d.temperature, min=0.05, max=2.0, step=0.05),
            io.Int.Input("top_k", default=d.top_k, min=0, max=500, tooltip="0 disables top-k."),
            io.Float.Input("top_p", default=d.top_p, min=0.0, max=1.0, step=0.01),
            io.Float.Input("repetition_penalty", default=d.repetition_penalty, min=1.0, max=2.0, step=0.01),
            io.Int.Input("max_new_tokens", default=d.max_new_tokens, min=50, max=1500,
                         tooltip="Caps output length; ~12.5 codec frames per second of audio."),
        ]
    ins.append(io.Boolean.Input("unload_after", default=False,
                                tooltip="Free the Breeze model (~7.7 GiB VRAM) after this node runs. Turn on when "
                                        "an image/video model runs later in the same workflow; the next Breeze "
                                        "node reloads the weights."))
    return ins


def _make_node(mode: str, advanced: bool):
    title, _, blurb = _MODE_INFO[mode]
    suffix = " Advanced" if advanced else ""
    node_id = "ITLBreezeTTS" + title.replace(" ", "") + ("Advanced" if advanced else "")
    display = f"ITL Breeze TTS {title}{suffix}"

    class _Node(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id=node_id,
                display_name=display,
                category="Into The Latent/audio",
                search_aliases=["breeze", "tts", "text to speech", mode],
                is_experimental=True,
                description=blurb + ("\n\nAdvanced: exposes the sampling settings; defaults equal the Normal node."
                                     if advanced else ""),
                inputs=_inputs(mode, advanced),
                outputs=[io.Audio.Output(display_name="audio")],
            )

        @classmethod
        def execute(cls, model, text, seed, reference_audio=None, reference_text=None, instruction=None,
                    cfg_scale=1.0, temperature=None, top_k=None, top_p=None, repetition_penalty=None,
                    max_new_tokens=None, unload_after=False) -> io.NodeOutput:
            sampling = DEFAULT_SAMPLING
            if advanced:
                sampling = SamplingConfig(temperature=temperature, top_k=top_k, top_p=top_p,
                                          repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens)
            audio = generate_audio(resolve_handle(model), _api(), mode=mode, text=text, seed=seed,
                                   cfg_scale=cfg_scale, sampling=sampling, reference_audio=reference_audio,
                                   reference_text=reference_text, instruction=instruction,
                                   temp_dir=folder_paths.get_temp_directory())
            if unload_after:
                unload()
            return io.NodeOutput(audio)

    _Node.__name__ = _Node.__qualname__ = node_id
    return node_id, _Node


GENERATE_NODES = dict(_make_node(m, adv) for m in ("clone", "design", "direction") for adv in (False, True))
globals().update(GENERATE_NODES)

ITLBreezeTTSVoiceClone = GENERATE_NODES["ITLBreezeTTSVoiceClone"]
ITLBreezeTTSVoiceCloneAdvanced = GENERATE_NODES["ITLBreezeTTSVoiceCloneAdvanced"]
ITLBreezeTTSVoiceDesign = GENERATE_NODES["ITLBreezeTTSVoiceDesign"]
ITLBreezeTTSVoiceDesignAdvanced = GENERATE_NODES["ITLBreezeTTSVoiceDesignAdvanced"]
ITLBreezeTTSVoiceDirection = GENERATE_NODES["ITLBreezeTTSVoiceDirection"]
ITLBreezeTTSVoiceDirectionAdvanced = GENERATE_NODES["ITLBreezeTTSVoiceDirectionAdvanced"]
