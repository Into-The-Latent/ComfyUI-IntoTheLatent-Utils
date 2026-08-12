# Batch Loader core tests — part of ComfyUI-AI2Go-Utils. GPL-3.0.
import json

import pytest

from nodes.batch_loader_core import MAX_FILES, parse_files


def test_parse_valid_list():
    raw = json.dumps([
        {"name": "fox.png", "subfolder": "", "type": "input"},
        {"name": "city.jpg"},  # subfolder/type optional
    ])
    files = parse_files(raw)
    assert files == [
        {"name": "fox.png", "subfolder": "", "type": "input"},
        {"name": "city.jpg", "subfolder": "", "type": "input"},
    ]


def test_parse_keeps_subfolder():
    files = parse_files(json.dumps([{"name": "a.png", "subfolder": "refs"}]))
    assert files[0]["subfolder"] == "refs"


@pytest.mark.parametrize("raw", ["", "   ", "[]"])
def test_parse_empty_raises(raw):
    with pytest.raises(ValueError, match="No files loaded"):
        parse_files(raw)


def test_parse_malformed_json_raises():
    with pytest.raises(ValueError, match="Malformed files_json"):
        parse_files("[{oops")


def test_parse_non_list_raises():
    with pytest.raises(ValueError, match="JSON array"):
        parse_files('{"name": "a.png"}')


@pytest.mark.parametrize("entry", [42, "a.png", {}, {"name": ""}, {"name": 3}])
def test_parse_bad_entry_raises(entry):
    with pytest.raises(ValueError, match="File #1"):
        parse_files(json.dumps([entry]))


def test_parse_over_ceiling_raises():
    raw = json.dumps([{"name": f"f{i}.png"} for i in range(MAX_FILES + 1)])
    with pytest.raises(ValueError, match="at most 8"):
        parse_files(raw)


from nodes.batch_loader_core import downscale_size


def test_downscale_off_untouched():
    assert downscale_size(2400, 1200, "off", 1200) == ((2400, 1200), None)


def test_downscale_below_threshold_untouched():
    for mode in ("fit", "crop", "stretch"):
        assert downscale_size(800, 600, mode, 1200) == ((800, 600), None)


def test_downscale_exactly_max_untouched():
    assert downscale_size(1200, 1200, "fit", 1200) == ((1200, 1200), None)


def test_fit_caps_longest_edge():
    assert downscale_size(2400, 1200, "fit", 1200) == ((1200, 600), None)
    assert downscale_size(1200, 2400, "fit", 1200) == ((600, 1200), None)


def test_fit_never_returns_zero():
    (w, h), box = downscale_size(5000, 2, "fit", 1200)
    assert (w, h) == (1200, 1) and box is None


def test_crop_centers_square_then_caps():
    # shorter edge 1200 -> centered 1200x1200 box, already <= max
    assert downscale_size(2400, 1200, "crop", 1200) == ((1200, 1200), (600, 0, 1800, 1200))
    # shorter edge 800 < max -> square of 800, no upscale
    assert downscale_size(2400, 800, "crop", 1200) == ((800, 800), (800, 0, 1600, 800))
    # shorter edge 2000 > max -> crop 2000-box, downscale to 1200
    assert downscale_size(3000, 2000, "crop", 1200) == ((1200, 1200), (500, 0, 2500, 2000))


def test_stretch_forces_square():
    assert downscale_size(2400, 1200, "stretch", 1200) == ((1200, 1200), None)


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="downscale_mode"):
        downscale_size(100, 100, "zoom", 1200)
