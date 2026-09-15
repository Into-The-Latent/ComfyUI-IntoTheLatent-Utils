# Whisper Transcribe node — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# AUDIO -> transcript STRING through a WHISPER model from ITL Whisper Loader. Built for feeding
# Breeze TTS Voice Clone / Direction their `reference_text`, but it is a general speech-to-text
# node. Engine: whisper_core.transcribe (design:
# docs/superpowers/specs/2026-09-15-whisper-transcribe-design.md §3).
from comfy_api.latest import io

from .whisper_core import LANGUAGES, transcribe
from .whisper_loader import WHISPER, resolve_handle, unload


class ITLWhisperTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLWhisperTranscribe",
            display_name="ITL Whisper Transcribe",
            category="Into The Latent/audio",
            search_aliases=["whisper", "transcribe", "speech to text", "audio to text", "stt"],
            is_experimental=True,
            description="""
Transcribes audio to text with Whisper (speech-to-text).

Wire the output into a Breeze TTS Voice Clone / Direction node's `reference_text`. Stereo is
downmixed, any sample rate is accepted, clips longer than 30 s are transcribed in full.
Punctuation and casing come from the model; check Chinese output for simplified vs. traditional
characters before using it as a Breeze transcript.""",
            inputs=[
                WHISPER.Input("model", tooltip="From ITL Whisper Loader."),
                io.Audio.Input("audio", tooltip="Batch item 0 is transcribed; stereo is downmixed."),
                io.Combo.Input("language", options=["auto", *LANGUAGES], default="auto",
                               tooltip="auto lets Whisper detect the language; pick a code when it guesses wrong."),
                io.Boolean.Input("unload_after", default=False,
                                 tooltip="Free the Whisper model after this node runs (the next Whisper node "
                                         "reloads it). Turn on when VRAM is tight."),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, model, audio, language="auto", unload_after=False) -> io.NodeOutput:
        # The handle is deliberately never bound to a local: the temporary dies when transcribe()
        # returns, so by the time unload() runs only whisper_loader._CACHE references the model and
        # its VRAM really comes back. Binding `handle = resolve_handle(...)` would silently defeat
        # unload_after (the node tests stub both calls and cannot catch that).
        text = transcribe(resolve_handle(model), audio, language=language)
        if unload_after:
            unload()
        return io.NodeOutput(text)
