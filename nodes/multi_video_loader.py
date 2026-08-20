# Multi Video Loader nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Two loaders that emit one output socket group per dropped file (design:
# docs/superpowers/specs/2026-08-12-batch-loaders-design.md, amended for frames_N/extract_frames
# — see the Amendments section). Simple = video_N + frames_N + audio_N + count; Advanced adds
# filename_N. video_N is lazy (VideoFromFile — decodes nothing) unless force_rate retimes the
# clip, which requires decoding its frames. frames_N is the same decoded IMAGE batch video_N
# would show, but only populated when the extract_frames widget is on (front-end auto-enables it
# the moment a frames_N socket gets wired) — VideoHelperSuite-style workflows and nodes like
# MiniMax H3's ref_video_0 want raw frames, not ComfyUI's native VIDEO object, and this is the
# only way to get them out of this loader. audio_N is decoded separately with PyAV so audio is
# available even on the cheap (force_rate == 0, extract_frames off) path; a clip with no
# (decodable) audio track yields None there, same as an unplugged OPTIONAL input. The audio
# decoder mirrors nodes/multi_audio_loader.py's _load_audio (ComfyUI, GPL-3.0 — same license as
# this pack); it is duplicated here rather than imported for the same comfy-free reason
# _input_path is duplicated across the loaders.
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


def _load_video(path, force_rate, extract_frames, pos):
    """Load one file -> (video_input, frames_or_None, audio_dict_or_None).

    ``pos`` is the file's 1-based position in files_json, included in error messages so a
    failure can be traced back to a specific row in the UI.

    Decode decision:
    - ``need_retime`` = force_rate > 0 AND the clip's metadata rate (cheap get_frame_rate(),
      reads container metadata, no frame decoding) differs from force_rate by >= 1e-6.
    - ``need_decode`` = need_retime OR extract_frames.
    - If not need_decode: video_N is a lazy VideoFromFile (nothing decoded) and audio_N comes
      from _decode_audio_track on its own — the fully free path.
    - If need_decode: components = video.get_components() is paid once. When need_retime, the
      frames are resampled and video_N becomes a VideoFromComponents at the forced rate (the
      limit_denominator(1001) Fraction keeps NTSC-style rates like 29.97 clean); otherwise
      video_N stays the lazy VideoFromFile (already-decoded frames are only used for frames_N in
      that case). Either way audio_N reuses components.audio instead of decoding a second time,
      and frames_N — when extract_frames is on — is exactly the frames video_N would show, so a
      retimed clip's frames_N always matches its (retimed) video_N.
    """
    try:
        video = InputImpl.VideoFromFile(path)
        need_retime = force_rate > 0 and abs(float(video.get_frame_rate()) - force_rate) >= 1e-6
        need_decode = need_retime or extract_frames

        if not need_decode:
            audio = _decode_audio_track(path)
            return video, None, audio

        components = video.get_components()

        if need_retime:
            src_count = components.images.shape[0]
            indices = resample_frame_indices(src_count, float(components.frame_rate), force_rate)
            if indices is None or float(components.frame_rate) == force_rate:
                # Safety net for float-precision edge cases where the decoded rate turns out to
                # exactly match force_rate despite the metadata-only check above disagreeing:
                # nothing to resample, so fall back to the lazy video like the no-retime branch.
                images = components.images
                out_video = video
            else:
                images = components.images[indices]
                out_video = InputImpl.VideoFromComponents(
                    Types.VideoComponents(
                        images=images,
                        audio=components.audio,
                        # limit_denominator(1001) turns NTSC-style rates (e.g. 29.97) into the
                        # clean 2997/100 rational instead of Fraction(force_rate)'s exact
                        # binary-float fraction (8437190785180631/281474976710656).
                        frame_rate=Fraction(force_rate).limit_denominator(1001),
                    ),
                    bit_depth=video.get_bit_depth(),
                )
        else:
            images = components.images
            out_video = video

        frames = images if extract_frames else None
        return out_video, frames, components.audio
    except Exception as e:
        raise ValueError(f"File #{pos}: Could not read video {os.path.basename(path)!r}: {e}") from e


def _run_video(files_json, force_rate, extract_frames, advanced):
    """Shared engine for both nodes. Returns the padded output list (count first).

    A row with ``enabled: False`` keeps its socket position — its group's slots (now including
    frames_N) stay None — and is not counted; it is never opened/decoded. ``count`` is the
    number of files actually emitted (enabled rows), not the number of rows.
    """
    files = parse_files(files_json)
    group = 4 if advanced else 3
    outputs = [None] * (1 + MAX_FILES * group)
    emitted = 0
    for i, f in enumerate(files):
        base = 1 + i * group  # slot base stays tied to the loop index — disabled rows don't shift later ones
        if not f["enabled"]:
            continue
        path = _input_path(f)
        if not os.path.isfile(path):
            raise ValueError(f"File #{i + 1} ({f['name']!r}): not found in the input folder — re-add it to the node.")
        video, frames, audio = _load_video(path, force_rate, extract_frames, i + 1)
        outputs[base] = video
        outputs[base + 1] = frames
        outputs[base + 2] = audio
        if advanced:
            outputs[base + 3] = f["name"]
        emitted += 1
    outputs[0] = emitted
    return outputs


def _fingerprint(files_json, force_rate, extract_frames):
    try:
        files = parse_files(files_json)
    except ValueError:
        return files_json
    sig = [str(force_rate), str(extract_frames)]
    for f in files:
        try:
            st = os.stat(_input_path(f))
            sig.append(f"{f['subfolder']}/{f['name']}:{st.st_size}:{st.st_mtime_ns}:{f['enabled']}")
        except (OSError, ValueError):
            sig.append(f"{f['subfolder']}/{f['name']}:missing:{f['enabled']}")
    return "|".join(sig)


def _frames_positions(advanced):
    """Map each frames_N output's absolute slot position -> its 1-based file number."""
    group = 4 if advanced else 3
    return {1 + i * group + 1: i + 1 for i in range(MAX_FILES)}


def _check_frames_guard(prompt, unique_id, advanced):
    """Raise if a frames_N socket is wired downstream while extract_frames is off.

    The front-end auto-enables extract_frames the instant a frames_N socket gets connected (see
    web/js/multi_loader.js), so this only fires for prompts built without going through that
    front-end (e.g. hand-built API calls) — a silent None on a wired frames_N would otherwise be
    confusing to debug. ``prompt`` is the full API-format prompt dict (node_id -> {"class_type",
    "inputs"}); a link is an ``[source_node_id, source_slot]`` pair on some other node's input.
    IDs/slots are compared as str/int since prompt keys and link ids can arrive as either.
    """
    if not prompt or unique_id is None:
        return
    positions = _frames_positions(advanced)
    uid = str(unique_id)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for val in inputs.values():
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                continue
            src_id, src_slot = val
            try:
                slot = int(src_slot)
            except (TypeError, ValueError):
                continue
            if str(src_id) == uid and slot in positions:
                n = positions[slot]
                raise ValueError(
                    f"frames_{n} is connected but extract_frames is off — switch extract_frames "
                    "on (the front-end normally does this for you; this prompt was built without it)."
                )


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
        # Appended last (pack rule: new widgets go at the end so old workflows' positional
        # widgets_values still line up with the right widget on restore).
        io.Boolean.Input(
            "extract_frames", default=False,
            tooltip="Off (default): frames_N stays empty and loading is as free as possible — the "
                    "same fast path as force_rate=0. On: frames_N carries this file's decoded frames "
                    "as an IMAGE batch (matching whatever video_N would show), for nodes that want raw "
                    "frames instead of ComfyUI's VIDEO object — VideoHelperSuite-style inputs, or "
                    "MiniMax H3's ref_video_0. Decoding costs time and memory, same as forcing a rate. "
                    "Wiring a frames_N socket switches this on for you automatically.",
        ),
    ]


def _video_outputs(advanced):
    outs = [io.Int.Output(display_name="count")]
    for i in range(1, MAX_FILES + 1):
        outs.append(io.Video.Output(display_name=f"video_{i}"))
        outs.append(io.Image.Output(display_name=f"frames_{i}"))
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
            description="Drop up to 8 video files onto the node; each gets its own video_N, "
                        "frames_N and audio_N output socket (sockets appear/disappear with the "
                        "list, or pin output_slots to a fixed count so wires survive file edits). "
                        "frames_N stays empty unless extract_frames is on (auto-enabled the "
                        "moment a frames_N socket is wired) — it carries this file's decoded "
                        "frames as an IMAGE batch, for nodes that want raw frames instead of the "
                        "VIDEO object (VideoHelperSuite-style inputs, MiniMax H3's ref_video_0). "
                        "Each row has an on/off toggle: a switched-off row keeps its socket "
                        "position but outputs None for video_N/frames_N/audio_N (and filename_N "
                        "on Advanced) — equivalent to leaving an unplugged OPTIONAL input, so "
                        "don't switch off a row feeding a REQUIRED input. force_rate at 0 "
                        "(default) keeps every clip's native rate and costs nothing to load "
                        "unless extract_frames is also on; any other value retimes every clip to "
                        "that rate (dropping or duplicating frames, real duration preserved) and "
                        "requires decoding. A clip with no audio track outputs None for audio_N. "
                        "count = number of files actually emitted (enabled rows), not the row "
                        "count. NOTE: this output layout (video_N, frames_N, audio_N per file) "
                        "is a breaking socket-order change from earlier versions of this node — "
                        "workflows saved before frames_N was added need their wires reconnected.",
            inputs=_video_inputs(),
            outputs=_video_outputs(advanced=False),
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, files_json="[]", force_rate=0.0, output_slots="auto", extract_frames=False) -> io.NodeOutput:
        # output_slots is front-end-only (see web/js/multi_loader.js): it only picks how many
        # sockets are shown, not what the sockets carry. Accepted here only so it serializes /
        # the socket exists.
        if not extract_frames:
            _check_frames_guard(cls.hidden.prompt, cls.hidden.unique_id, advanced=False)
        return io.NodeOutput(*_run_video(files_json, force_rate, extract_frames, advanced=False))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", force_rate=0.0, output_slots="auto", extract_frames=False):
        return _fingerprint(files_json, force_rate, extract_frames)


class ITLMultiVideoLoaderAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ITLMultiVideoLoaderAdvanced",
            display_name="ITL Multi Video Loader Advanced",
            category="Into The Latent/video",
            search_aliases=["batch", "load", "video", "multi", "drop", "upload", "filename"],
            description="Multi Video Loader plus a filename_N output per file. NOTE: this output "
                        "layout (video_N, frames_N, audio_N, filename_N per file) is a breaking "
                        "socket-order change from earlier versions of this node — workflows saved "
                        "before frames_N was added need their wires reconnected.",
            inputs=_video_inputs(),
            outputs=_video_outputs(advanced=True),
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, files_json="[]", force_rate=0.0, output_slots="auto", extract_frames=False) -> io.NodeOutput:
        # output_slots is front-end-only (see web/js/multi_loader.js): it only picks how many
        # sockets are shown, not what the sockets carry. Accepted here only so it serializes /
        # the socket exists.
        if not extract_frames:
            _check_frames_guard(cls.hidden.prompt, cls.hidden.unique_id, advanced=True)
        return io.NodeOutput(*_run_video(files_json, force_rate, extract_frames, advanced=True))

    @classmethod
    def fingerprint_inputs(cls, files_json="[]", force_rate=0.0, output_slots="auto", extract_frames=False):
        return _fingerprint(files_json, force_rate, extract_frames)
