# Integration test for the Multi Audio Loader nodes — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import json
import wave

import pytest

pytest.importorskip("comfy_api")
pytest.importorskip("av")

from nodes.multi_loader_core import MAX_FILES, OUTPUT_SLOT_OPTIONS  # noqa: E402  (comfy-free)


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
    import torch
    from nodes.multi_audio_loader import _run_audio
    _write_wav(input_dir / "vo.wav", sample_rate=22050)
    _write_wav(input_dir / "bed.wav", sample_rate=44100)
    out = _run_audio(json.dumps([{"name": "vo.wav"}, {"name": "bed.wav"}]), advanced=True)

    assert len(out) == 1 + 2 * MAX_FILES and out[0] == 2
    a1, name1, a2 = out[1], out[2], out[3]
    assert a1["sample_rate"] == 22050          # native rate, no resampling
    assert a2["sample_rate"] == 44100
    assert a1["waveform"].shape == (1, 1, 2205)   # batch=1, mono, 2205 samples
    assert a1["waveform"].dtype == torch.float32
    assert name1 == "vo.wav"
    assert out[5] is None                       # padding after file 2's group


def test_simple_layout(input_dir):
    from nodes.multi_audio_loader import _run_audio
    _write_wav(input_dir / "one.wav")
    out = _run_audio(json.dumps([{"name": "one.wav"}]), advanced=False)
    assert len(out) == 1 + MAX_FILES and out[0] == 1
    assert out[1]["sample_rate"] == 22050 and out[2] is None


def test_missing_file_raises(input_dir):
    from nodes.multi_audio_loader import _run_audio
    with pytest.raises(ValueError, match="gone.wav"):
        _run_audio(json.dumps([{"name": "gone.wav"}]), advanced=False)


def test_corrupt_file_raises(input_dir):
    from nodes.multi_audio_loader import _run_audio
    # Write a text file with .wav extension
    (input_dir / "bad.wav").write_text("not audio data")
    with pytest.raises(ValueError, match="bad.wav"):
        _run_audio(json.dumps([{"name": "bad.wav"}]), advanced=False)


def test_disabled_row_holds_position(input_dir):
    # Middle file disabled: its slots stay None (not skipped-and-shifted), the third file
    # still lands in its own audio_3/filename_3 group, and count only reflects the two
    # enabled files.
    from nodes.multi_audio_loader import _run_audio
    _write_wav(input_dir / "a.wav")
    _write_wav(input_dir / "b.wav")
    _write_wav(input_dir / "c.wav")
    out = _run_audio(json.dumps([
        {"name": "a.wav"},
        {"name": "b.wav", "enabled": False},
        {"name": "c.wav"},
    ]), advanced=True)

    assert out[0] == 2                          # count = enabled files, not rows
    assert out[1] is not None                    # audio_1
    assert out[3] is None and out[4] is None      # audio_2/filename_2 both None
    audio3, name3 = out[5], out[6]                # audio_3/filename_3 — its own slots, not moved up
    assert audio3 is not None and name3 == "c.wav"


def test_all_disabled_yields_count_zero(input_dir):
    # All rows disabled (Toggle All off): count == 0, all output slots are None, no exception.
    from nodes.multi_audio_loader import _run_audio
    _write_wav(input_dir / "a.wav")
    _write_wav(input_dir / "b.wav")
    out = _run_audio(json.dumps([
        {"name": "a.wav", "enabled": False},
        {"name": "b.wav", "enabled": False},
    ]), advanced=True)

    assert out[0] == 0                          # count == 0, not > 0 (no exception)
    assert all(v is None for v in out[1:])      # all output slots None


def test_ten_files_fill_the_last_group(input_dir):
    from nodes.multi_audio_loader import _run_audio
    names = [f"c{i}.wav" for i in range(1, MAX_FILES + 1)]
    for n in names:
        _write_wav(input_dir / n)
    out = _run_audio(json.dumps([{"name": n} for n in names]), advanced=True)
    assert out[0] == MAX_FILES and len(out) == 1 + 2 * MAX_FILES
    base = 1 + (MAX_FILES - 1) * 2                    # file 10's group: audio_10/filename_10
    assert out[base]["sample_rate"] == 22050 and out[base + 1] == names[-1]
    assert all(o is not None for o in out)


@pytest.mark.parametrize("advanced", [False, True])
def test_schema_outputs_match_padded_return(input_dir, advanced):
    from nodes.multi_audio_loader import ITLMultiAudioLoader, ITLMultiAudioLoaderAdvanced, _run_audio
    _write_wav(input_dir / "a.wav")
    schema = (ITLMultiAudioLoaderAdvanced if advanced else ITLMultiAudioLoader).define_schema()
    out = _run_audio(json.dumps([{"name": "a.wav"}]), advanced=advanced)
    assert len(schema.outputs) == len(out)
    assert schema.outputs[-1].display_name == (f"filename_{MAX_FILES}" if advanced else f"audio_{MAX_FILES}")
    slots = next(i for i in schema.inputs if i.id == "output_slots")
    assert list(slots.options) == list(OUTPUT_SLOT_OPTIONS)
