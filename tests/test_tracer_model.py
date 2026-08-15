# Tests for tracer model/LoRA chain — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.civitai_metadata.tracer import trace


def _base():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sub/myCkpt.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "pos", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "neg", "clip": ["1", 1]}},
        "4": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0],
              "negative": ["3", 0], "seed": 1, "steps": 20, "cfg": 7,
              "sampler_name": "euler", "scheduler": "normal"}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
        "7": {"class_type": "ITLSaveCivitaiMetadata", "inputs": {"images": ["6", 0]}},
    }


def test_no_lora_checkpoint_name():
    r = trace(_base(), "7")
    assert r.model_name == "myCkpt" and r.model_file == "sub/myCkpt.safetensors"
    assert r.loras == []


def test_lora_chain_in_load_order():
    p = _base()
    p["8"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1],
              "lora_name": "A.safetensors", "strength_model": 0.8, "strength_clip": 0.8}}
    p["9"] = {"class_type": "LoraLoader", "inputs": {"model": ["8", 0], "clip": ["8", 1],
              "lora_name": "B.safetensors", "strength_model": 0.5, "strength_clip": 0.5}}
    p["4"]["inputs"]["model"] = ["9", 0]  # sampler <- B <- A <- checkpoint
    r = trace(p, "7")
    assert r.model_name == "myCkpt"
    assert [(lo["name"], lo["strength"], lo["file"]) for lo in r.loras] == [
        ("A", 0.8, "A.safetensors"), ("B", 0.5, "B.safetensors")]


def test_model_only_lora_and_passthrough():
    p = _base()
    p["8"] = {"class_type": "ModelSamplingDiscrete", "inputs": {"model": ["10", 0], "sampling": "eps"}}
    p["10"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0],
               "lora_name": "C.safetensors", "strength_model": 1.0}}
    p["4"]["inputs"]["model"] = ["8", 0]  # sampler <- passthrough <- C <- checkpoint
    r = trace(p, "7")
    assert [lo["name"] for lo in r.loras] == ["C"]
    assert r.model_name == "myCkpt"


def test_checkpoint_folder_is_checkpoints():
    r = trace(_base(), "7")
    assert r.model_folder == "checkpoints"


def test_unet_loader_diffusion_model():
    # Flux/Krea-style: model comes from a Load Diffusion Model (UNETLoader), not a checkpoint.
    p = _base()
    del p["1"]["inputs"]["ckpt_name"]
    p["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "flux/krea-turbo.safetensors",
              "weight_dtype": "default"}}
    r = trace(p, "7")
    assert r.model_name == "krea-turbo"
    assert r.model_file == "flux/krea-turbo.safetensors"
    assert r.model_folder == "diffusion_models"


def test_power_lora_loader_rgthree_multiple():
    # rgthree Power Lora Loader packs several loras into one node as lora_N dicts.
    p = _base()
    p["8"] = {"class_type": "Power Lora Loader (rgthree)", "inputs": {
        "model": ["1", 0], "clip": ["1", 1],
        "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
        "lora_1": {"on": True, "lora": "styleA.safetensors", "strength": 0.8, "strengthTwo": None},
        "lora_2": {"on": True, "lora": "sub/styleB.safetensors", "strength": 0.5, "strengthTwo": None},
        "lora_3": {"on": False, "lora": "disabledLora.safetensors", "strength": 1.0, "strengthTwo": None},
        "lora_4": {"on": True, "lora": "None", "strength": 1.0, "strengthTwo": None},
        "➕ Add Lora": ""}}
    p["4"]["inputs"]["model"] = ["8", 0]  # sampler <- PowerLora <- checkpoint
    r = trace(p, "7")
    assert r.model_name == "myCkpt"
    # enabled real loras only (disabled + empty "None" slot skipped), in listed order.
    assert [(lo["name"], lo["strength"], lo["file"]) for lo in r.loras] == [
        ("styleA", 0.8, "styleA.safetensors"), ("styleB", 0.5, "sub/styleB.safetensors")]


def test_power_lora_loader_mixed_with_standard_load_order():
    # PowerLora(A,B) closest to sampler, a standard LoraLoader(C) closest to the checkpoint.
    p = _base()
    p["8"] = {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1],
              "lora_name": "C.safetensors", "strength_model": 1.0, "strength_clip": 1.0}}
    p["9"] = {"class_type": "Power Lora Loader (rgthree)", "inputs": {
        "model": ["8", 0], "clip": ["8", 1],
        "lora_1": {"on": True, "lora": "A.safetensors", "strength": 0.8},
        "lora_2": {"on": True, "lora": "B.safetensors", "strength": 0.6}}}
    p["4"]["inputs"]["model"] = ["9", 0]  # sampler <- PowerLora(A,B) <- LoraLoader(C) <- ckpt
    r = trace(p, "7")
    # load order = nearest checkpoint first: C, then the power loader's A, B.
    assert [lo["name"] for lo in r.loras] == ["C", "A", "B"]


def test_lora_loader_stack_rgthree():
    # rgthree Lora Loader Stack: flat lora_0N / strength_0N pairs.
    p = _base()
    p["8"] = {"class_type": "Lora Loader Stack (rgthree)", "inputs": {
        "model": ["1", 0], "clip": ["1", 1],
        "lora_01": "A.safetensors", "strength_01": 0.8,
        "lora_02": "sub/B.safetensors", "strength_02": 0.5,
        "lora_03": "None", "strength_03": 1.0,          # empty slot skipped
        "lora_04": "zero.safetensors", "strength_04": 0}}  # zero strength skipped (rgthree won't load it)
    p["4"]["inputs"]["model"] = ["8", 0]
    r = trace(p, "7")
    assert [(lo["name"], lo["strength"], lo["file"]) for lo in r.loras] == [
        ("A", 0.8, "A.safetensors"), ("B", 0.5, "sub/B.safetensors")]


def test_unet_loader_with_model_only_lora():
    # sampler <- LoraLoaderModelOnly <- UNETLoader (common Flux LoRA wiring)
    p = _base()
    p["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "krea.safetensors", "weight_dtype": "default"}}
    p["8"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0],
              "lora_name": "styleFlux.safetensors", "strength_model": 0.9}}
    p["4"]["inputs"]["model"] = ["8", 0]
    r = trace(p, "7")
    assert r.model_name == "krea" and r.model_folder == "diffusion_models"
    assert [lo["name"] for lo in r.loras] == ["styleFlux"]
