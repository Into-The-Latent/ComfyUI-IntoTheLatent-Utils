# Multi Audio Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
#
# Two loaders that emit one output socket group per dropped file (design:
# docs/superpowers/specs/2026-08-12-batch-loaders-design.md). Simple = audio_N + count;
# Advanced adds filename_N. Audio is never resampled — each AUDIO output keeps its file's
# native sample rate, exactly like stock LoadAudio. The decoder below mirrors
# comfy_extras/nodes_audio.py load() (ComfyUI, GPL-3.0 — same license as this pack).
import os

import av
import torch

import folder_paths
from comfy_api.latest import io

from .multi_loader_core import MAX_FILES, parse_files


def _input_path(f):
    """Resolve a files_json entry to a path inside the input directory (traversal-safe)."""
    base = os.path.abspath(folder_paths.get_input_directory())
    path = os.path.abspath(os.path.join(base, f["subfolder"], f["name"]))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"File path escapes the input folder: {f['name']!r}")
    return path


def _f32_pcm(wav):
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def _load_audio(path, pos):
    """Decode one file -> {"waveform": Tensor[1,C,S], "sample_rate": native_rate}.

    ``pos`` is the file's 1-based position in files_json, included in error messages so a
    failure can be traced back to a specific row in the UI.
    """
    try:
        with av.open(path) as af:
            if not af.streams.audio:
                raise ValueError("no audio stream in the file")
            stream = af.streams.audio[0]
            sr = stream.codec_context.sample_rate
            frames = []
            for frame in af.decode(streams=stream.index):
                buf = torch.from_numpy(frame.to_ndarray())
                if buf.shape[0] != frame.layout.nb_channels:
                    buf = buf.view(-1, frame.layout.nb_channels).t()
                frames.append(buf)
            if not frames:
                raise ValueError("file decoded to zero samples")
            wav = _f32_pcm(torch.cat(frames, dim=1))
    except Exception as e:
        raise ValueError(f"File #{pos}: Could not read audio {os.path.basename(path)!r}: {e}") from e
    return {"waveform": wav.unsqueeze(0), "sample_rate": sr}


def _run_audio(files_json, advanced):
    """Shared engine for both nodes. Returns the padded output list (count first)."""
    files = parse_files(files_json)
    group = 2 if advanced else 1
    outputs = [None] * (1 + MAX_FILES * group)
    outputs[0] = len(files)
    for i, f in enumerate(files):
        path = _input_path(f)
        if not os.path.isfile(path):
            raise ValueError(f"File #{i + 1} ({f['name']!r}): not found in the input folder — re-add it to the node.")
        base = 1 + i * group
        outputs[base] = _load_audio(path, i + 1)
        if advanced:
            outputs[base + 1] = f["name"]
    return outputs


def _fingerprint(files_json):
    try:
        files = parse_files(files_json)
    except ValueError:
        return files_json
    sig = []
    for f in files:
        try:
            st = os.stat(_input_path(f))
            sig.append(f"{f['subfolder']}/{f['name']}:{st.st_size}:{st.st_mtime_ns}")
        except (OSError, ValueError):
            sig.append(f"{f['subfolder']}/{f['name']}:missing")
    return "|".join(sig)


def _audio_inputs():
    return [
        io.String.Input(
            "files_json", default="[]",
            tooltip="Authoritative file list as JSON. Hidden in the UI and kept in sync by the "
                    "front-end — drop files onto the node instead of editing this.",
        ),
    ]


def _audio_outputs(advanced):
    outs = [io.Int.Output(display_name="count")]
    for i in range(1, MAX_FILES + 1):
        outs.append(io.Audio.Output(display_name=f"audio_{i}"))
        if advanced:
            outs.append(io.String.Output(display_name=f"filename_{i}"))
    return outs


class AI2GoMultiAudioLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoMultiAudioLoader",
            display_name="AI2Go Multi Audio Loader",
            category="AI2Go/audio",
            search_aliases=["batch", "load", "audio", "multi", "drop", "upload", "wav", "mp3"],
            description="Drop up to 8 audio files onto the node; each gets its own audio_N "
                        "output socket (sockets appear/disappear with the list). Audio is never "
                        "resampled — each output keeps its file's native sample rate. count = "
                        "number of files loaded.",
            inputs=_audio_inputs(),
            outputs=_audio_outputs(advanced=False),
        )

    @classmethod
    def execute(cls, files_json="[]") -> io.NodeOutput:
        return io.NodeOutput(*_run_audio(files_json, advanced=False))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]"):
        return _fingerprint(files_json)


class AI2GoMultiAudioLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AI2GoMultiAudioLoaderAdvanced",
            display_name="AI2Go Multi Audio Loader Advanced",
            category="AI2Go/audio",
            search_aliases=["batch", "load", "audio", "multi", "drop", "upload", "filename"],
            description="Multi Audio Loader plus a filename_N output per file.",
            inputs=_audio_inputs(),
            outputs=_audio_outputs(advanced=True),
        )

    @classmethod
    def execute(cls, files_json="[]") -> io.NodeOutput:
        return io.NodeOutput(*_run_audio(files_json, advanced=True))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]"):
        return _fingerprint(files_json)
