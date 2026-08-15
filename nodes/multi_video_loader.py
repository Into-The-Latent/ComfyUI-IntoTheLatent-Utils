# Multi Video Loader nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Two loaders that emit one output socket group per dropped file (design:
# docs/superpowers/specs/2026-08-12-batch-loaders-design.md). Simple = video_N + audio_N +
# count; Advanced adds filename_N. video_N is lazy (VideoFromFile — decodes nothing) unless
# force_rate retimes the clip, which requires decoding its frames. audio_N is decoded
# separately with PyAV so audio is available even on the cheap (force_rate == 0) path; a clip
# with no (decodable) audio track yields None there, same as an unplugged OPTIONAL input. The
# audio decoder mirrors nodes/multi_audio_loader.py's _load_audio (ComfyUI, GPL-3.0 — same
# license as this pack); it is duplicated here rather than imported for the same comfy-free
# reason _input_path is duplicated across the loaders.
import os
from fractions import Fraction

import av
import torch

import folder_paths
from comfy_api.latest import InputImpl, Types, io

from .multi_loader_core import MAX_FILES, parse_files, resample_frame_indices


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


def _decode_audio_track(path):
    """Decode only the audio stream -> {"waveform": Tensor[1,C,S], "sample_rate": native_rate},
    or None if the file has no (decodable) audio track. Raises only if the file itself can't
    be opened.

    Stream selection mirrors ComfyUI's own last_decodable_audio_stream
    (comfy_api.latest._input_impl.video_types): pick the LAST audio stream FFmpeg can actually
    decode, not just streams.audio[0]. A stream FFmpeg has no decoder for (e.g. an iPhone APAC
    spatial-audio track) has codec_context = None; forcing index 0 on such a file used to make
    this node fail even though the force_rate>0 path — which goes through ComfyUI's own
    get_components() and applies this same selection — tolerated it fine.
    """
    with av.open(path) as af:
        stream = next(
            (s for s in reversed(af.streams.audio)
             if s.codec_context is not None and s.codec_context.sample_rate),
            None,
        )
        if stream is None:
            return None
        sr = stream.codec_context.sample_rate
        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != frame.layout.nb_channels:
                buf = buf.view(-1, frame.layout.nb_channels).t()
            frames.append(buf)
        if not frames:
            return None
        wav = _f32_pcm(torch.cat(frames, dim=1))
    return {"waveform": wav.unsqueeze(0), "sample_rate": sr}


def _load_video(path, force_rate, pos):
    """Load one file -> (video_input, audio_dict_or_None).

    ``pos`` is the file's 1-based position in files_json, included in error messages so a
    failure can be traced back to a specific row in the UI. ``force_rate == 0`` is the cheap
    path: video_N is a lazy VideoFromFile (decodes nothing) and audio_N is decoded on its own
    via PyAV. ``force_rate > 0`` normally decodes the video's components once, resamples the
    frames, and reuses the already-decoded audio instead of decoding it a second time — unless
    the clip is already at the requested rate, in which case it takes the same cheap path as
    force_rate == 0 (see the get_frame_rate() check below).
    """
    try:
        video = InputImpl.VideoFromFile(path)
        if force_rate <= 0:
            audio = _decode_audio_track(path)
            return video, audio

        if abs(float(video.get_frame_rate()) - force_rate) < 1e-6:
            # Cheap metadata-only check (get_frame_rate() reads container metadata, no frame
            # decoding): the clip is already at the requested rate, so take the same free path
            # as force_rate == 0 instead of paying for get_components() just to discard the
            # result. audio_N still has to come from _decode_audio_track — nothing above filled
            # it in on this branch.
            audio = _decode_audio_track(path)
            return video, audio

        components = video.get_components()
        src_count = components.images.shape[0]
        indices = resample_frame_indices(src_count, float(components.frame_rate), force_rate)
        if indices is None or float(components.frame_rate) == force_rate:
            return video, components.audio

        resampled_images = components.images[indices]
        resampled = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=resampled_images,
                audio=components.audio,
                # limit_denominator(1001) turns NTSC-style rates (e.g. 29.97) into the clean
                # 2997/100 rational instead of Fraction(force_rate)'s exact binary-float
                # fraction (8437190785180631/281474976710656).
                frame_rate=Fraction(force_rate).limit_denominator(1001),
            ),
            bit_depth=video.get_bit_depth(),
        )
        return resampled, components.audio
    except Exception as e:
        raise ValueError(f"File #{pos}: Could not read video {os.path.basename(path)!r}: {e}") from e


def _run_video(files_json, force_rate, advanced):
    """Shared engine for both nodes. Returns the padded output list (count first).

    A row with ``enabled: False`` keeps its socket position — its group's slots stay
    None — and is not counted; it is never opened/decoded. ``count`` is the number of
    files actually emitted (enabled rows), not the number of rows.
    """
    files = parse_files(files_json)
    group = 3 if advanced else 2
    outputs = [None] * (1 + MAX_FILES * group)
    emitted = 0
    for i, f in enumerate(files):
        base = 1 + i * group  # slot base stays tied to the loop index — disabled rows don't shift later ones
        if not f["enabled"]:
            continue
        path = _input_path(f)
        if not os.path.isfile(path):
            raise ValueError(f"File #{i + 1} ({f['name']!r}): not found in the input folder — re-add it to the node.")
        video, audio = _load_video(path, force_rate, i + 1)
        outputs[base] = video
        outputs[base + 1] = audio
        if advanced:
            outputs[base + 2] = f["name"]
        emitted += 1
    outputs[0] = emitted
    return outputs


def _fingerprint(files_json, force_rate):
    try:
        files = parse_files(files_json)
    except ValueError:
        return files_json
    sig = [str(force_rate)]
    for f in files:
        try:
            st = os.stat(_input_path(f))
            sig.append(f"{f['subfolder']}/{f['name']}:{st.st_size}:{st.st_mtime_ns}:{f['enabled']}")
        except (OSError, ValueError):
            sig.append(f"{f['subfolder']}/{f['name']}:missing:{f['enabled']}")
    return "|".join(sig)


def _video_inputs():
    return [
        io.String.Input(
            "files_json", default="[]",
            tooltip="Authoritative file list as JSON. Hidden in the UI and kept in sync by the "
                    "front-end — drop files onto the node instead of editing this.",
        ),
        io.Float.Input(
            "force_rate", default=0.0, min=0.0, max=240.0, step=1.0,
            tooltip="0 = off — every clip keeps its own frame rate, and video_N is returned "
                    "lazily without decoding a single frame (free). Any other value drops or "
                    "duplicates frames so all clips run at that rate while keeping their real "
                    "running time (a 10s clip stays 10s). Forcing a rate has to decode the "
                    "clip's frames to do this, so it costs time and memory — 0 stays free.",
        ),
        io.Combo.Input(
            "output_slots", options=["auto", "1", "2", "3", "4", "5", "6", "7", "8"], default="auto",
            tooltip="How many output sockets to show. 'auto' follows the number of loaded files, so "
                    "sockets appear and disappear as you edit the list. Pick a fixed number to keep the "
                    "sockets (and your wires) in place while you swap files around — extra sockets with "
                    "no file behind them output nothing, so don't wire more than you load.",
        ),
    ]


def _video_outputs(advanced):
    outs = [io.Int.Output(display_name="count")]
    for i in range(1, MAX_FILES + 1):
        outs.append(io.Video.Output(display_name=f"video_{i}"))
        outs.append(io.Audio.Output(display_name=f"audio_{i}"))
        if advanced:
            outs.append(io.String.Output(display_name=f"filename_{i}"))
    return outs


class ITLMultiVideoLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLMultiVideoLoader",
            display_name="ITL Multi Video Loader",
            category="Into The Latent/video",
            search_aliases=["batch", "load", "video", "multi", "drop", "upload", "mp4", "mov"],
            description="Drop up to 8 video files onto the node; each gets its own video_N and "
                        "audio_N output socket (sockets appear/disappear with the list, or pin "
                        "output_slots to a fixed count so wires survive file edits). Each row has "
                        "an on/off toggle: a switched-off row keeps its socket position but "
                        "outputs None for video_N/audio_N (and filename_N on Advanced) — "
                        "equivalent to leaving an unplugged OPTIONAL input, so don't switch off a "
                        "row feeding a REQUIRED input. force_rate at 0 (default) keeps every "
                        "clip's native rate and costs nothing to load; any other value retimes "
                        "every clip to that rate (dropping or duplicating frames, real duration "
                        "preserved) and requires decoding. A clip with no audio track outputs "
                        "None for audio_N. count = number of files actually emitted (enabled "
                        "rows), not the row count.",
            inputs=_video_inputs(),
            outputs=_video_outputs(advanced=False),
        )

    @classmethod
    def execute(cls, files_json="[]", force_rate=0.0, output_slots="auto") -> io.NodeOutput:
        # output_slots is front-end-only (see web/js/multi_loader.js): it only picks how many
        # sockets are shown, not what the sockets carry. Accepted here only so it serializes /
        # the socket exists.
        return io.NodeOutput(*_run_video(files_json, force_rate, advanced=False))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", force_rate=0.0, output_slots="auto"):
        return _fingerprint(files_json, force_rate)


class ITLMultiVideoLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLMultiVideoLoaderAdvanced",
            display_name="ITL Multi Video Loader Advanced",
            category="Into The Latent/video",
            search_aliases=["batch", "load", "video", "multi", "drop", "upload", "filename"],
            description="Multi Video Loader plus a filename_N output per file.",
            inputs=_video_inputs(),
            outputs=_video_outputs(advanced=True),
        )

    @classmethod
    def execute(cls, files_json="[]", force_rate=0.0, output_slots="auto") -> io.NodeOutput:
        # output_slots is front-end-only (see web/js/multi_loader.js): it only picks how many
        # sockets are shown, not what the sockets carry. Accepted here only so it serializes /
        # the socket exists.
        return io.NodeOutput(*_run_video(files_json, force_rate, advanced=True))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", force_rate=0.0, output_slots="auto"):
        return _fingerprint(files_json, force_rate)
