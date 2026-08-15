# Tests for the tracer sampler detection — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.civitai_metadata.tracer import trace


def _prompt(sampler_extra=None, images_link=("6", 0)):
    sampler = {
        "class_type": "KSampler",
        "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                   "latent_image": ["5", 0], "seed": 12345, "steps": 30, "cfg": 6.5,
                   "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0},
    }
    if sampler_extra:
        sampler["inputs"].update(sampler_extra)
    return {
        "4": sampler,
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
        "7": {"class_type": "ITLSaveCivitaiMetadata",
              "inputs": {"images": list(images_link), "filename_prefix": "ITL", "save_workflow": False}},
    }


def test_finds_sampler_via_image_trace():
    r = trace(_prompt(), "7")
    assert r.steps == 30 and r.cfg == 6.5 and r.seed == 12345
    assert r.sampler_name == "dpmpp_2m" and r.scheduler == "karras" and r.denoise == 1.0
    assert "sampler" not in r.unresolved


def test_ksampler_advanced_noise_seed():
    p = _prompt()
    p["4"]["class_type"] = "KSamplerAdvanced"
    del p["4"]["inputs"]["seed"]
    p["4"]["inputs"]["noise_seed"] = 777
    assert trace(p, "7").seed == 777


def test_single_sampler_fallback_when_image_trace_dead_ends():
    p = _prompt(images_link=("99", 0))  # images point at a node that doesn't reach the sampler
    p["99"] = {"class_type": "SomePassThrough", "inputs": {"x": 1}}
    assert trace(p, "7").steps == 30  # only one sampler in graph -> used


def test_no_sampler_marks_unresolved():
    p = {"7": {"class_type": "ITLSaveCivitaiMetadata", "inputs": {"images": ["6", 0]}}}
    assert "sampler" in trace(p, "7").unresolved
