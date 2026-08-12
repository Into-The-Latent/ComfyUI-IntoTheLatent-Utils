# Integration test for the Batch Audio Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
import json
import wave

import pytest

pytest.importorskip("comfy_api")
pytest.importorskip("av")


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    return tmp_path


def _write_wav(path, sample_rate=22050, n=2205):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x10" * n)


def test_advanced_audio_and_filenames(input_dir):
    from nodes.batch_audio_loader import _run_audio
    _write_wav(input_dir / "vo.wav", sample_rate=22050)
    _write_wav(input_dir / "bed.wav", sample_rate=44100)
    out = _run_audio(json.dumps([{"name": "vo.wav"}, {"name": "bed.wav"}]), advanced=True)

    assert len(out) == 17 and out[0] == 2
    a1, name1, a2 = out[1], out[2], out[3]
    assert a1["sample_rate"] == 22050          # native rate, no resampling
    assert a2["sample_rate"] == 44100
    assert a1["waveform"].ndim == 3 and a1["waveform"].shape[0] == 1   # [1,C,S]
    assert name1 == "vo.wav"
    assert out[5] is None                       # padding after file 2's group


def test_simple_layout(input_dir):
    from nodes.batch_audio_loader import _run_audio
    _write_wav(input_dir / "one.wav")
    out = _run_audio(json.dumps([{"name": "one.wav"}]), advanced=False)
    assert len(out) == 9 and out[0] == 1
    assert out[1]["sample_rate"] == 22050 and out[2] is None


def test_missing_file_raises(input_dir):
    from nodes.batch_audio_loader import _run_audio
    with pytest.raises(ValueError, match="gone.wav"):
        _run_audio(json.dumps([{"name": "gone.wav"}]), advanced=False)
