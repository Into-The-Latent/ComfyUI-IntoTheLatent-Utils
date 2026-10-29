# Tests for the tracer's Flux guidance (FluxGuidance / CLIPTextEncodeFlux) — part of
# ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.civitai_metadata.tracer import trace


def _ksampler_flux(pos_link=("2", 0), neg_link=("3", 0)):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"}},
        "13": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl.safetensors", "type": "flux"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["13", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["13", 0]}},
        "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": list(pos_link), "negative": list(neg_link),
              "seed": 1, "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple"}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
        "7": {"class_type": "ITLSaveCivitaiMetadata", "inputs": {"images": ["6", 0]}},
    }


def test_flux_guidance_on_the_positive():
    p = _ksampler_flux(pos_link=("8", 0))
    p["8"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["2", 0], "guidance": 3.5}}
    r = trace(p, "7")
    assert r.guidance == 3.5 and r.cfg == 1.0 and r.positive == "a cat"


def test_flux_guidance_behind_reference_latent():
    # Flux Kontext: FluxGuidance(ReferenceLatent(enc, latent)).
    p = _ksampler_flux(pos_link=("8", 0))
    p["9"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["2", 0], "latent": ["10", 0]}}
    p["10"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["1", 2]}}
    p["11"] = {"class_type": "LoadImage", "inputs": {"image": "ref.png"}}
    p["8"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["9", 0], "guidance": 2.5}}
    assert trace(p, "7").guidance == 2.5


def test_clip_text_encode_flux_carries_its_own_guidance():
    p = _ksampler_flux(pos_link=("8", 0))
    p["8"] = {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": ["13", 0], "clip_l": "a cat", "t5xxl": "a cat", "guidance": 4.0}}
    assert trace(p, "7").guidance == 4.0


def test_flux_disable_guidance_means_none():
    p = _ksampler_flux(pos_link=("8", 0))
    p["9"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["2", 0], "guidance": 3.5}}
    p["8"] = {"class_type": "FluxDisableGuidance", "inputs": {"conditioning": ["9", 0]}}
    assert trace(p, "7").guidance is None


def test_guidance_read_from_the_positive_only():
    p = _ksampler_flux(neg_link=("8", 0))
    p["8"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["3", 0], "guidance": 3.5}}
    assert trace(p, "7").guidance is None


def test_basic_guider_flux_template():
    p = _ksampler_flux()
    p["8"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["2", 0], "guidance": 3.5}}
    p["5"] = {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["8", 0]}}
    p["4"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"guider": ["5", 0]}}
    r = trace(p, "7")
    assert r.guidance == 3.5 and r.cfg is None and r.positive == "a cat"


def test_no_flux_nodes_no_guidance():
    assert trace(_ksampler_flux(), "7").guidance is None
