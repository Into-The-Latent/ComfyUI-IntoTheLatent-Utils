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
