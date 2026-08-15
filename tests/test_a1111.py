# Tests for the A1111 parameters formatter — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.civitai_metadata.a1111 import format_parameters


def test_full_golden_string():
    out = format_parameters(
        positive="a red fox, 8k", negative="blurry",
        steps=30, cfg=6.5, seed=12345, sampler="DPM++ 2M Karras",
        width=1024, height=1024,
        model_name="myCkpt", model_hash="a1b2c3d4e5",
        loras=[{"name": "styleLora", "strength": 0.8, "hash": "1122aabbcc"}],
        denoise=1.0, clip_skip=None,
    )
    assert out == (
        "a red fox, 8k <lora:styleLora:0.8>\n"
        "Negative prompt: blurry\n"
        "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 6.5, Seed: 12345, "
        "Size: 1024x1024, Model hash: a1b2c3d4e5, Model: myCkpt, "
        'Lora hashes: "styleLora: 1122aabbcc", Version: ComfyUI'
    )


def test_integral_floats_trimmed_and_empty_negative():
    out = format_parameters(
        positive="cat", negative="", steps=20, cfg=7.0, seed=1, sampler="Euler",
        width=512, height=768,
    )
    lines = out.split("\n")
    assert lines[0] == "cat"
    assert lines[1] == "Negative prompt: "
    assert "CFG scale: 7," in lines[2]          # 7.0 -> 7
    assert "Size: 512x768" in lines[2]
    assert "Lora hashes" not in out             # no loras -> no key
    assert "Model" not in out                   # no model -> no key


def test_denoise_and_clip_skip_conditional():
    out = format_parameters(
        positive="x", negative="", steps=20, cfg=7, seed=1, sampler="Euler",
        width=64, height=64, denoise=0.55, clip_skip=2,
    )
    assert "Denoising strength: 0.55" in out
    assert "Clip skip: 2" in out


def test_lora_tag_without_hash_still_appended():
    out = format_parameters(
        positive="x", negative="", steps=20, cfg=7, seed=1, sampler="Euler",
        width=64, height=64, loras=[{"name": "noHashLora", "strength": 1.0}],
    )
    assert "<lora:noHashLora:1>" in out.split("\n")[0]
    assert "Lora hashes" not in out             # no hash -> not in settings line
