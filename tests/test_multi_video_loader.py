# Integration test for the Multi Video Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
import json

import pytest

pytest.importorskip("comfy_api")
pytest.importorskip("av")


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    return tmp_path


def _write_video(path, fps=24, seconds=1, color=(200, 100, 50)):
    """Write a tiny synthetic solid-color video (no audio track) with PyAV."""
    import av
    import numpy as np

    n_frames = int(round(fps * seconds))
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 32
    stream.height = 32
    stream.pix_fmt = "yuv420p"
    frame_data = np.full((32, 32, 3), color, dtype=np.uint8)
    for _ in range(n_frames):
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_simple_layout(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "one.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([{"name": "one.mp4"}]), force_rate=0.0, advanced=False)

    assert len(out) == 17 and out[0] == 1
    assert out[1] is not None                 # video_1
    assert out[2] is None or isinstance(out[2], dict)   # audio_1 - no track -> None
    assert out[3] is None                      # padding (video_2)


def test_advanced_layout_two_clips(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "a.mp4", fps=24, seconds=1)
    _write_video(input_dir / "b.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([{"name": "a.mp4"}, {"name": "b.mp4"}]), force_rate=0.0, advanced=True)

    assert len(out) == 25 and out[0] == 2
    assert out[1] is not None and out[3] == "a.mp4"
    assert out[4] is not None and out[6] == "b.mp4"


def test_disabled_row_holds_position(input_dir):
    # Middle file disabled: its slots stay None (not skipped-and-shifted), the third file
    # still lands in its own video_3/audio_3/filename_3 group, and count only reflects the
    # two enabled files.
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "a.mp4", fps=24, seconds=1)
    _write_video(input_dir / "b.mp4", fps=24, seconds=1)
    _write_video(input_dir / "c.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([
        {"name": "a.mp4"},
        {"name": "b.mp4", "enabled": False},
        {"name": "c.mp4"},
    ]), force_rate=0.0, advanced=True)

    assert out[0] == 2                          # count = enabled files, not rows
    assert out[1] is not None                    # video_1
    assert out[4] is None and out[5] is None and out[6] is None   # video_2/audio_2/filename_2 all None
    video3, name3 = out[7], out[9]                # video_3/filename_3 — its own slots, not moved up
    assert video3 is not None and name3 == "c.mp4"


def test_all_disabled_yields_count_zero(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "a.mp4", fps=24, seconds=1)
    _write_video(input_dir / "b.mp4", fps=24, seconds=1)
    out = _run_video(json.dumps([
        {"name": "a.mp4", "enabled": False},
        {"name": "b.mp4", "enabled": False},
    ]), force_rate=0.0, advanced=True)

    assert out[0] == 0
    assert all(v is None for v in out[1:])


def test_force_rate_retimes_and_halves_frames(input_dir):
    from nodes.multi_video_loader import _run_video
    _write_video(input_dir / "clip.mp4", fps=24, seconds=1)  # 24 frames @ 24fps
    out = _run_video(json.dumps([{"name": "clip.mp4"}]), force_rate=12.0, advanced=False)

    video = out[1]
    components = video.get_components()
    assert components.images.shape[0] == 12          # half the frames
    assert float(components.frame_rate) == 12.0


def test_missing_file_raises(input_dir):
    from nodes.multi_video_loader import _run_video
    with pytest.raises(ValueError, match="gone.mp4"):
        _run_video(json.dumps([{"name": "gone.mp4"}]), force_rate=0.0, advanced=False)
