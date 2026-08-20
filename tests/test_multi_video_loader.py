# Integration test for the Multi Video Loader nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import json
import types

import pytest

pytest.importorskip("comfy_api")
pytest.importorskip("av")

import torch  # noqa: E402


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    return tmp_path


def _write_video(path, fps=24, seconds=1, color=(200, 100, 50), audio_hz=None, audio_rate=44100):
    """Write a tiny synthetic solid-color video with PyAV.

    ``audio_hz`` is optional: when given, a mono ``audio_rate``-Hz sine tone of the same
    duration is muxed in as an AAC track (the whole waveform is handed to
    ``AudioStream.encode()`` in one call — it internally chunks to the codec's frame size, same
    pattern ComfyUI's own VideoFromComponents.save_to uses). Left ``None``, the file has no
    audio track at all, as before.
    """
    import av
    import numpy as np

    n_frames = int(round(fps * seconds))
    container = av.open(str(path), mode="w")
    # Both streams must be registered before any packet is muxed — the container writes its
    # header on the first mux() call, so an audio stream added only after the video track is
    # fully flushed produces a broken/undersized header ("Cannot rebase to zero time" on mux).
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    audio_stream = None
    if audio_hz is not None:
        audio_stream = container.add_stream("aac", rate=audio_rate, layout="mono")

    frame_data = np.full((32, 32, 3), color, dtype=np.uint8)
    for _ in range(n_frames):
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

    if audio_stream is not None:
        n_samples = int(round(audio_rate * seconds))
        t = np.arange(n_samples, dtype=np.float32) / audio_rate
        tone = (0.2 * np.sin(2 * np.pi * audio_hz * t)).reshape(1, n_samples)
        aframe = av.AudioFrame.from_ndarray(np.ascontiguousarray(tone), format="fltp", layout="mono")
        aframe.sample_rate = audio_rate
        aframe.pts = 0
        for packet in audio_stream.encode(aframe):
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

    container.close()


# ── Layout note: each file's group is now video_N/frames_N/audio_N (+ filename_N on Advanced) —
# group size 3 (simple) / 4 (Advanced). Slot indices below are computed as
# base = 1 + (file_index - 1) * group; video=base, frames=base+1, audio=base+2, filename=base+3. ──


def test_simple_layout(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "one.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([{"name": "one.mp4"}]), force_rate=0.0, extract_frames=False, advanced=False)

    assert len(out) == 25 and out[0] == 1          # 1 + 8*3
    assert out[1] is not None                       # video_1
    assert out[2] is None                            # frames_1 - extract_frames off
    assert out[3] is None or isinstance(out[3], dict)  # audio_1 - no track -> None
    assert out[4] is None                             # padding (video_2)


def test_advanced_layout_two_clips(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "a.mp4", fps=24, seconds=1)
    _write_video(input_dir / "b.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([{"name": "a.mp4"}, {"name": "b.mp4"}]),
                      force_rate=0.0, extract_frames=False, advanced=True)

    assert len(out) == 33 and out[0] == 2          # 1 + 8*4
    assert out[1] is not None and out[2] is None and out[4] == "a.mp4"    # video_1, frames_1 off, filename_1
    assert out[5] is not None and out[6] is None and out[8] == "b.mp4"    # video_2, frames_2 off, filename_2


def test_disabled_row_holds_position(input_dir):
    # Middle file disabled: its slots stay None (not skipped-and-shifted), the third file
    # still lands in its own video_3/frames_3/audio_3/filename_3 group, and count only reflects
    # the two enabled files.
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "a.mp4", fps=24, seconds=1)
    _write_video(input_dir / "b.mp4", fps=24, seconds=1)
    _write_video(input_dir / "c.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([
        {"name": "a.mp4"},
        {"name": "b.mp4", "enabled": False},
        {"name": "c.mp4"},
    ]), force_rate=0.0, extract_frames=False, advanced=True)

    assert out[0] == 2                          # count = enabled files, not rows
    assert out[1] is not None                    # video_1
    # video_2/frames_2/audio_2/filename_2 all None (file #2, base=5)
    assert out[5] is None and out[6] is None and out[7] is None and out[8] is None
    video3, name3 = out[9], out[12]                # video_3/filename_3 — its own slots, not moved up
    assert video3 is not None and name3 == "c.mp4"


def test_all_disabled_yields_count_zero(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "a.mp4", fps=24, seconds=1)
    _write_video(input_dir / "b.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([
        {"name": "a.mp4", "enabled": False},
        {"name": "b.mp4", "enabled": False},
    ]), force_rate=0.0, extract_frames=False, advanced=True)

    assert out[0] == 0
    assert all(v is None for v in out[1:])


def test_force_rate_retimes_and_halves_frames(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "clip.mp4", fps=24, seconds=1)  # 24 frames @ 24fps
    out = _run_video(json.dumps([{"name": "clip.mp4"}]), force_rate=12.0, extract_frames=False, advanced=False)

    video, frames = out[1], out[2]
    assert frames is None                       # extract_frames off -> frames_1 stays None
    components = video.get_components()
    assert components.images.shape[0] == 12          # half the frames
    assert float(components.frame_rate) == 12.0


def test_audio_track_decoded_on_both_rate_paths(input_dir):
    # force_rate=0 exercises _decode_audio_track's success path directly; force_rate=12.0 (half
    # the clip's native 24fps) forces get_components() and proves audio_N still comes through —
    # reused from components.audio rather than decoded a second time.
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "clip.mp4", fps=24, seconds=1, audio_hz=440)

    out_cheap = _run_video(json.dumps([{"name": "clip.mp4"}]), force_rate=0.0, extract_frames=False, advanced=False)
    audio_cheap = out_cheap[3]        # audio_1 (base=1, +2)
    assert isinstance(audio_cheap, dict)
    assert audio_cheap["waveform"].numel() > 0
    assert audio_cheap["sample_rate"] > 0

    out_forced = _run_video(json.dumps([{"name": "clip.mp4"}]), force_rate=12.0, extract_frames=False, advanced=False)
    audio_forced = out_forced[3]
    assert isinstance(audio_forced, dict)
    assert audio_forced["waveform"].numel() > 0
    assert audio_forced["sample_rate"] > 0


def test_missing_file_raises(input_dir):
    from nodes.multi_video_loader import _run_video
    with pytest.raises(ValueError, match="gone.mp4"):
        _run_video(json.dumps([{"name": "gone.mp4"}]), force_rate=0.0, extract_frames=False, advanced=False)


def test_extract_frames_true_no_retime_yields_full_frame_batch(input_dir):
    # extract_frames on, force_rate off: frames_1 is a float IMAGE batch matching the clip's own
    # frame count, and video_1 stays the lazy VideoFromFile (no retiming needed).
    from comfy_api.latest import InputImpl
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "clip.mp4", fps=24, seconds=1)  # 24 frames
    out = _run_video(json.dumps([{"name": "clip.mp4"}]), force_rate=0.0, extract_frames=True, advanced=False)

    video, frames = out[1], out[2]
    assert isinstance(video, InputImpl.VideoFromFile)     # still lazy — not rebuilt from components
    assert isinstance(frames, torch.Tensor)
    assert frames.dtype.is_floating_point
    assert frames.ndim == 4 and frames.shape[0] == 24 and frames.shape[-1] == 3


def test_extract_frames_true_with_retime_matches_video_components(input_dir):
    # extract_frames on AND force_rate retimes: frames_1's frame count must match the retimed
    # count, and must equal video_1's own components' frame count (frames_N mirrors what
    # video_N would show).
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "clip.mp4", fps=24, seconds=1)  # 24 frames @ 24fps
    out = _run_video(json.dumps([{"name": "clip.mp4"}]), force_rate=12.0, extract_frames=True, advanced=False)

    video, frames = out[1], out[2]
    assert isinstance(frames, torch.Tensor)
    assert frames.shape[0] == 12
    components = video.get_components()
    assert components.images.shape[0] == frames.shape[0]


def test_wired_but_off_guard_raises(monkeypatch):
    # An API-built prompt links some other node's input directly to our node's frames_1 output
    # (slot 2 in the simple layout: video_1=1, frames_1=2, audio_1=3) while extract_frames is
    # off. The front-end normally prevents this by auto-enabling extract_frames, so this guard
    # only matters for hand-built/API prompts — it should fail loudly instead of silently
    # handing that downstream node None.
    from nodes.multi_video_loader import ITLMultiVideoLoader as Node

    prompt = {
        "1": {"class_type": "ITLMultiVideoLoader", "inputs": {"files_json": "[]"}},
        "2": {"class_type": "SomeDownstreamNode", "inputs": {"images": ["1", 2]}},
    }
    fake_hidden = types.SimpleNamespace(prompt=prompt, unique_id="1")
    monkeypatch.setattr(Node, "hidden", fake_hidden, raising=False)

    with pytest.raises(ValueError, match="extract_frames"):
        Node.execute(files_json="[]", force_rate=0.0, extract_frames=False)


def test_wired_but_off_guard_tolerates_str_int_id_mismatch(monkeypatch):
    # Same as above but unique_id is an int and the link's source id is a str — prompt keys and
    # link ids can arrive as either, per the design's str()-normalization requirement.
    from nodes.multi_video_loader import ITLMultiVideoLoaderAdvanced as Node

    prompt = {
        "3": {"class_type": "ITLMultiVideoLoaderAdvanced", "inputs": {"files_json": "[]"}},
        "4": {"class_type": "SomeDownstreamNode", "inputs": {"images": ["3", 2]}},  # frames_1 slot
    }
    fake_hidden = types.SimpleNamespace(prompt=prompt, unique_id=3)
    monkeypatch.setattr(Node, "hidden", fake_hidden, raising=False)

    with pytest.raises(ValueError, match="extract_frames"):
        Node.execute(files_json="[]", force_rate=0.0, extract_frames=False)


def test_extract_frames_on_skips_guard(monkeypatch, input_dir):
    # Same wiring as the guard test, but extract_frames is on — no guard should fire, and the
    # run should complete normally.
    from nodes.multi_video_loader import ITLMultiVideoLoader as Node
    _write_video(input_dir / "clip.mp4", fps=24, seconds=1)

    prompt = {
        "1": {"class_type": "ITLMultiVideoLoader", "inputs": {}},
        "2": {"class_type": "SomeDownstreamNode", "inputs": {"images": ["1", 2]}},
    }
    fake_hidden = types.SimpleNamespace(prompt=prompt, unique_id="1")
    monkeypatch.setattr(Node, "hidden", fake_hidden, raising=False)

    result = Node.execute(files_json=json.dumps([{"name": "clip.mp4"}]), force_rate=0.0, extract_frames=True)
    assert result.args[0] == 1  # count
