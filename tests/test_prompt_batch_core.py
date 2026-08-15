# Tests for the Prompt Batch parse/select core — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import json
import pytest
from nodes.prompt_batch_core import parse_prompts, select_prompt, DEFAULT_PROMPTS_JSON


def test_parse_object_list():
    raw = json.dumps([{"positive": "fox", "negative": "blur"}, {"positive": "city"}])
    assert parse_prompts(raw) == [("fox", "blur"), ("city", "")]


def test_parse_string_shorthand_and_wrapper():
    assert parse_prompts(json.dumps(["lonely fox"])) == [("lonely fox", "")]
    assert parse_prompts(json.dumps({"prompts": [{"positive": "x"}]})) == [("x", "")]


def test_parse_errors():
    with pytest.raises(ValueError):
        parse_prompts("")
    with pytest.raises(ValueError):
        parse_prompts("{not json")
    with pytest.raises(ValueError):
        parse_prompts(json.dumps([{"positive": "  "}]))  # empty positive


def test_select_clamps():
    prompts = [("a", ""), ("b", "n")]
    assert select_prompt(prompts, 1) == ("b", "n", 1)
    assert select_prompt(prompts, 9) == ("b", "n", 1)   # overshoot clamps to last
    assert select_prompt(prompts, -3) == ("a", "", 0)   # negative clamps to 0
    assert select_prompt(prompts, None) == ("a", "", 0)  # non-int tolerated


def test_default_json_is_valid():
    assert len(parse_prompts(DEFAULT_PROMPTS_JSON)) >= 1
