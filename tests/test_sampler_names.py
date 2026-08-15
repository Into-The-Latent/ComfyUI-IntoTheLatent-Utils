# Tests for the ComfyUI->A1111 sampler-name mapping — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.civitai_metadata.sampler_names import to_a1111_sampler


def test_known_pairs():
    assert to_a1111_sampler("euler", "normal") == "Euler"
    assert to_a1111_sampler("euler_ancestral", "normal") == "Euler a"
    assert to_a1111_sampler("dpmpp_2m", "karras") == "DPM++ 2M Karras"
    assert to_a1111_sampler("dpmpp_2m", "exponential") == "DPM++ 2M Exponential"


def test_unknown_sampler_falls_back_readably():
    assert to_a1111_sampler("brand_new_sampler", "normal") == "Brand New Sampler"


def test_unknown_scheduler_appended():
    assert to_a1111_sampler("euler", "weird_sched") == "Euler weird_sched"


def test_empty_inputs():
    assert to_a1111_sampler("", "") == "Unknown"
