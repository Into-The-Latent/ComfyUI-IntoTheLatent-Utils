# Tests for tracer prompt resolution — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import json
from nodes.civitai_metadata.tracer import trace

CKPT = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "myCkpt.safetensors"}}
SAVE = {"class_type": "ITLSaveCivitaiMetadata", "inputs": {"images": ["6", 0]}}
VAE = {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}}


def _ksampler(pos_text, neg_text):
    return {
        "1": CKPT,
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_text, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_text, "clip": ["1", 1]}},
        "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0],
              "negative": ["3", 0], "seed": 1, "steps": 20, "cfg": 7,
              "sampler_name": "euler", "scheduler": "normal"}},
        "6": VAE, "7": SAVE,
    }


def test_static_clip_text():
    r = trace(_ksampler("a fox", "blurry"), "7")
    assert r.positive == "a fox" and r.negative == "blurry"
    assert "positive" not in r.unresolved and "negative" not in r.unresolved


def test_prompt_from_batch_node_at_index():
    p = _ksampler(["10", 0], ["10", 1])  # both CLIP text inputs come from the batch node
    p["10"] = {"class_type": "ITLPromptBatch", "inputs": {
        "prompts_json": json.dumps([{"positive": "fox", "negative": "blur"},
                                    {"positive": "city", "negative": ""}]),
        "index": 1}}
    r = trace(p, "7")
    assert r.positive == "city" and r.negative == ""


def test_unresolvable_source_flagged():
    p = _ksampler(["10", 0], "blurry")
    p["10"] = {"class_type": "SomeLLMNode", "inputs": {"seed": 5}}  # no string to read
    r = trace(p, "7")
    assert r.positive == "" and "positive" in r.unresolved
    assert r.negative == "blurry" and "negative" not in r.unresolved


def test_conditioning_passthrough_walked():
    p = _ksampler("base pos", "neg")
    # Insert a ConditioningConcat between CLIP and sampler positive.
    p["8"] = {"class_type": "ConditioningConcat", "inputs": {"conditioning_to": ["2", 0], "conditioning_from": ["2", 0]}}
    p["4"]["inputs"]["positive"] = ["8", 0]
    assert trace(p, "7").positive == "base pos"


def test_zeroed_negative_is_empty_not_the_positive():
    # Turbo / CFG-1 workflows: the negative is the positive encode run through ConditioningZeroOut.
    p = _ksampler("a cat", "unused")
    p["8"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    p["4"]["inputs"]["negative"] = ["8", 0]
    r = trace(p, "7")
    assert r.positive == "a cat" and r.negative == ""
    assert "negative" not in r.unresolved


def _wan_i2v(neg_link):
    # WanImageToVideo carries both conditionings: outputs positive(0), negative(1), latent(2).
    return {
        "1": CKPT,
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["1", 1]}},
        "5": {"class_type": "WanImageToVideo", "inputs": {"positive": ["2", 0], "negative": neg_link,
              "vae": ["1", 2], "width": 832, "height": 480, "length": 81, "batch_size": 1}},
        "4": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["1", 0], "positive": ["5", 0],
              "negative": ["5", 1], "latent_image": ["5", 2], "noise_seed": 1, "steps": 20, "cfg": 3.5,
              "sampler_name": "euler", "scheduler": "simple"}},
        "6": VAE, "7": SAVE,
    }


def test_wan_i2v_follows_the_matching_input():
    r = trace(_wan_i2v(["3", 0]), "7")
    assert r.positive == "a cat" and r.negative == "blurry"


def test_wan_i2v_zeroed_negative():
    p = _wan_i2v(["8", 0])
    p["8"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    r = trace(p, "7")
    assert r.positive == "a cat" and r.negative == ""
    assert "negative" not in r.unresolved


def test_sd3_template_combine_keeps_real_negative():
    # ComfyUI's SD3 template: Combine(SetTimestepRange(ZeroOut(enc)), SetTimestepRange(enc)).
    p = _ksampler("a cat", "blurry")
    p["8"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["3", 0]}}
    p["9"] = {"class_type": "ConditioningSetTimestepRange", "inputs": {"conditioning": ["8", 0], "start": 0.1, "end": 1.0}}
    p["10"] = {"class_type": "ConditioningSetTimestepRange", "inputs": {"conditioning": ["3", 0], "start": 0.0, "end": 0.1}}
    p["11"] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["9", 0], "conditioning_2": ["10", 0]}}
    p["4"]["inputs"]["negative"] = ["11", 0]
    r = trace(p, "7")
    assert r.positive == "a cat" and r.negative == "blurry"
    assert "negative" not in r.unresolved


def test_zero_out_beside_an_unknown_encoder_stays_unresolved():
    # One branch is zeroed, the other ends at an encoder we can't read: the negative is unknown, not empty.
    p = _ksampler("a cat", "unused")
    p["8"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    p["9"] = {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": "blurry", "text_l": "blurry", "clip": ["1", 1]}}
    p["10"] = {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["8", 0], "conditioning_2": ["9", 0]}}
    p["4"]["inputs"]["negative"] = ["10", 0]
    r = trace(p, "7")
    assert r.negative == "" and "negative" in r.unresolved


def test_pair_conditioning_follows_suffixed_inputs():
    # Hook nodes carry both conditionings under suffixed names (positive_NEW / negative_NEW, *_A / *_B).
    p = _ksampler("a cat", "blurry")
    p["8"] = {"class_type": "PairConditioningSetProperties", "inputs": {
        "positive_NEW": ["2", 0], "negative_NEW": ["3", 0], "strength": 1.0, "set_cond_area": "default"}}
    p["9"] = {"class_type": "PairConditioningCombine", "inputs": {
        "positive_A": ["8", 0], "negative_A": ["8", 1], "positive_B": ["2", 0], "negative_B": ["3", 0]}}
    p["4"]["inputs"]["positive"] = ["9", 0]
    p["4"]["inputs"]["negative"] = ["9", 1]
    r = trace(p, "7")
    assert r.positive == "a cat" and r.negative == "blurry"
