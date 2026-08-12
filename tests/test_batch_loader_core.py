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
