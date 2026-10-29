# Tests for the tracer on the modular samplers (SamplerCustomAdvanced / SamplerCustom) — part of
# ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.civitai_metadata.tracer import trace


def _flux(guider):
    # Flux / Krea template: everything but the latent sits on a node wired into SamplerCustomAdvanced.
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-krea-dev.safetensors", "weight_dtype": "default"}},
        "13": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl.safetensors", "type": "flux"}},
        "14": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "style.safetensors", "strength_model": 0.8}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["13", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["13", 0]}},
        "4": guider,
        "5": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "6": {"class_type": "BasicScheduler", "inputs": {"model": ["14", 0], "scheduler": "simple", "steps": 8, "denoise": 1.0}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "10": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "8": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0], "guider": ["4", 0], "sampler": ["5", 0],
              "sigmas": ["6", 0], "latent_image": ["10", 0]}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "12": {"class_type": "ITLSaveCivitaiMetadata", "inputs": {"images": ["11", 0]}},
    }


def _cfg_guider(negative=("3", 0)):
    return {"class_type": "CFGGuider", "inputs": {"model": ["14", 0], "positive": ["2", 0], "negative": list(negative), "cfg": 3.5}}


def test_cfg_guider_reads_every_field():
    r = trace(_flux(_cfg_guider()), "12")
    assert r.positive == "a cat" and r.negative == "blurry"
    assert r.cfg == 3.5 and r.seed == 42
    assert r.sampler_name == "euler" and r.scheduler == "simple" and r.steps == 8 and r.denoise == 1.0
    assert r.model_name == "flux1-krea-dev" and r.model_folder == "diffusion_models"
    assert r.loras == [{"name": "style", "strength": 0.8, "file": "style.safetensors"}]
    assert r.unresolved == []


def test_cfg_guider_zeroed_negative():
    p = _flux(_cfg_guider(negative=("9", 0)))
    p["9"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
    r = trace(p, "12")
    assert r.positive == "a cat" and r.negative == "" and "negative" not in r.unresolved


def test_basic_guider_has_no_negative_and_no_cfg():
    # BasicGuider runs without CFG: there is no negative and no CFG value, so neither is "unresolved".
    r = trace(_flux({"class_type": "BasicGuider", "inputs": {"model": ["14", 0], "conditioning": ["2", 0]}}), "12")
    assert r.positive == "a cat" and r.negative == "" and r.cfg is None
    assert r.model_name == "flux1-krea-dev" and len(r.loras) == 1
    assert r.unresolved == []


def test_dual_cfg_guider():
    r = trace(_flux({"class_type": "DualCFGGuider", "inputs": {"model": ["14", 0], "cond1": ["2", 0], "cond2": ["2", 0],
                     "negative": ["3", 0], "cfg_conds": 5.0, "cfg_cond2_negative": 2.0, "style": "regular"}}), "12")
    assert r.positive == "a cat" and r.negative == "blurry" and r.cfg == 5.0
    assert r.unresolved == []


def test_other_guider_read_by_input_names():
    # PerpNegGuider and similar custom guiders use the CFGGuider input names.
    r = trace(_flux({"class_type": "PerpNegGuider", "inputs": {"model": ["14", 0], "positive": ["2", 0], "negative": ["3", 0],
                     "empty_conditioning": ["3", 0], "cfg": 4.0, "neg_scale": 1.0}}), "12")
    assert r.positive == "a cat" and r.negative == "blurry" and r.cfg == 4.0
    assert r.model_name == "flux1-krea-dev"


def test_scheduler_denoise_is_read():
    p = _flux(_cfg_guider())
    p["6"]["inputs"]["denoise"] = 0.6
    assert trace(p, "12").denoise == 0.6


def test_sigmas_through_a_split_node():
    # SplitSigmas sits between the scheduler and the sampler in two-pass workflows.
    p = _flux(_cfg_guider())
    p["15"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": ["6", 0], "step": 4}}
    p["8"]["inputs"]["sigmas"] = ["15", 1]
    r = trace(p, "12")
    assert r.steps == 8 and r.scheduler == "simple"


def test_sampler_custom_follows_sampler_and_sigmas():
    # SamplerCustom has the prompts, cfg and seed itself, but sampler_name / steps on wired nodes.
    p = _flux(_cfg_guider())
    p["8"] = {"class_type": "SamplerCustom", "inputs": {"model": ["14", 0], "add_noise": True, "noise_seed": 7, "cfg": 2.0,
              "positive": ["2", 0], "negative": ["3", 0], "sampler": ["5", 0], "sigmas": ["6", 0], "latent_image": ["10", 0]}}
    r = trace(p, "12")
    assert r.positive == "a cat" and r.negative == "blurry" and r.cfg == 2.0 and r.seed == 7
    assert r.sampler_name == "euler" and r.steps == 8 and r.scheduler == "simple"
    assert r.model_name == "flux1-krea-dev"
    assert r.unresolved == []


def test_unwired_guider_stays_unresolved():
    p = _flux(_cfg_guider())
    del p["8"]["inputs"]["guider"]
    r = trace(p, "12")
    assert {"positive", "negative", "cfg"} <= set(r.unresolved)


def test_settings_walk_never_reads_past_a_sampler():
    # sigmas from an unknown node fed by an earlier KSampler: that sampler's steps are not this one's.
    p = _flux(_cfg_guider())
    p["20"] = {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "latent_image": ["10", 0], "seed": 1, "steps": 30, "cfg": 7, "sampler_name": "euler", "scheduler": "normal"}}
    p["21"] = {"class_type": "SomeSigmasFromLatent", "inputs": {"latent": ["20", 0]}}
    p["8"]["inputs"]["sigmas"] = ["21", 0]
    r = trace(p, "12")
    assert r.steps is None and "steps" in r.unresolved and r.scheduler is None


def test_split_sigmas_denoise_is_not_the_img2img_strength():
    # SplitSigmasDenoise's `denoise` is where the schedule is split; steps / scheduler / denoise all come
    # from the scheduler node itself.
    p = _flux(_cfg_guider())
    p["15"] = {"class_type": "SplitSigmasDenoise", "inputs": {"sigmas": ["6", 0], "denoise": 0.3}}
    p["8"]["inputs"]["sigmas"] = ["15", 1]
    r = trace(p, "12")
    assert r.denoise == 1.0 and r.steps == 8 and r.scheduler == "simple"
