# Static graph tracer over the ComfyUI API prompt — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Walks the API prompt (dict of {node_id: {class_type, inputs}}) backward from a save node's `images`
# input to recover the real generation parameters used this run. Mirrored in
# web/js/save_civitai_metadata.js — keep the two in sync.
#
# This file is built up across three concerns: (1) sampler detection + scalar fields, (2) prompt
# resolution (incl. our ITLPromptBatch), (3) the model/LoRA chain.
import os
from dataclasses import dataclass, field

from ..prompt_batch_core import parse_prompts, select_prompt

SAMPLER_CLASSES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
CLIP_ENCODE_CLASSES = {"CLIPTextEncode"}
BATCH_CLASS = "ITLPromptBatch"
# Model-source loaders → (widget holding the filename, folder_paths folder to hash it from).
# Covers checkpoints AND standalone diffusion models (UNETLoader "Load Diffusion Model" — Flux/SD3/
# Krea/etc.), plus common GGUF unet loaders. Unknown loaders fall through as pass-throughs.
MODEL_SOURCES = {
    "CheckpointLoaderSimple": ("ckpt_name", "checkpoints"),
    "CheckpointLoader": ("ckpt_name", "checkpoints"),
    "unCLIPCheckpointLoader": ("ckpt_name", "checkpoints"),
    "UNETLoader": ("unet_name", "diffusion_models"),
    "UnetLoaderGGUF": ("unet_name", "diffusion_models"),
    "UnetLoaderGGUFAdvanced": ("unet_name", "diffusion_models"),
}
LORA_CLASSES = {"LoraLoader", "LoraLoaderModelOnly"}
# rgthree Power Lora Loader packs many loras into one node: inputs `lora_1`, `lora_2`, … each a dict
# {on, lora, strength, strengthTwo}. Mirrored in web/js/save_civitai_metadata.js.
POWER_LORA_CLASSES = {"Power Lora Loader (rgthree)"}
# rgthree Lora Loader Stack is flatter: paired inputs `lora_01`/`strength_01` … up to four slots.
LORA_STACK_CLASSES = {"Lora Loader Stack (rgthree)"}


def _lora_stack_entries(ins):
    """Enabled loras from an rgthree Lora Loader Stack, ordered by slot number.

    Skips "None"/blank slots and (matching rgthree) zero-strength slots — those aren't loaded, so
    they never touch the image and shouldn't appear in the metadata.
    """
    out = []
    nums = [str(k)[5:] for k in ins if str(k).startswith("lora_") and str(k)[5:].isdigit()]
    for num in sorted(nums, key=int):
        name = ins.get("lora_" + num)
        if not isinstance(name, str) or not name or name == "None":
            continue
        strength = ins.get("strength_" + num)
        if not _is_link(strength) and strength == 0:
            continue
        out.append({"name": _stem(name),
                    "strength": None if _is_link(strength) else strength,
                    "file": name})
    return out


def _power_lora_entries(ins):
    """Enabled loras from an rgthree Power Lora Loader's inputs, in listed order.

    Matches rgthree's own rule: keys prefixed ``lora_`` whose value is a dict; skip disabled (``on``
    is False) and empty ("None"/blank) slots. The prefix + dict shape naturally excludes the header
    widget / "Add Lora" entries.
    """
    out = []
    for k, v in ins.items():
        if not (isinstance(v, dict) and str(k).upper().startswith("LORA_")):
            continue
        name = v.get("lora")
        if not isinstance(name, str) or not name or name == "None" or v.get("on") is False:
            continue
        strength = v.get("strength")
        out.append({"name": _stem(name),
                    "strength": None if _is_link(strength) else strength,
                    "file": name})
    return out


@dataclass
class TraceResult:
    positive: str = ""
    negative: str = ""
    steps: object = None
    cfg: object = None
    seed: object = None
    sampler_name: str = None
    scheduler: str = None
    denoise: object = None
    clip_skip: object = None
    model_name: str = None
    model_file: str = None
    model_folder: str = None
    loras: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)


def _node(prompt, node_id):
    if node_id is None:
        return None
    return prompt.get(str(node_id))


def _input(prompt, node_id, name):
    node = _node(prompt, node_id)
    if not node:
        return None
    return node.get("inputs", {}).get(name)


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def _find_sampler(prompt, start_id):
    """BFS backward over input links from start_id to the nearest sampler node."""
    seen = set()
    queue = [str(start_id)]
    while queue:
        nid = queue.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        node = _node(prompt, nid)
        if not node:
            continue
        if node.get("class_type") in SAMPLER_CLASSES:
            return nid, node
        for v in node.get("inputs", {}).values():
            if _is_link(v):
                queue.append(str(v[0]))
    return None, None


def _single_sampler(prompt):
    found = [(nid, n) for nid, n in prompt.items() if n.get("class_type") in SAMPLER_CLASSES]
    return found[0] if len(found) == 1 else (None, None)


def _read_sampler_fields(sampler, r):
    ins = sampler.get("inputs", {})

    def lit(name):
        v = ins.get(name)
        return None if _is_link(v) else v

    r.steps = lit("steps")
    r.cfg = lit("cfg")
    r.seed = lit("seed") if lit("seed") is not None else lit("noise_seed")
    r.sampler_name = lit("sampler_name")
    r.scheduler = lit("scheduler")
    r.denoise = lit("denoise")
    for f in ("steps", "cfg", "sampler_name"):
        if getattr(r, f) is None:
            r.unresolved.append(f)


def _find_clip_encode(prompt, start_id):
    """BFS backward to the nearest CLIPTextEncode (walks through conditioning combiners)."""
    seen = set()
    queue = [str(start_id)]
    while queue:
        nid = queue.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        node = _node(prompt, nid)
        if not node:
            continue
        if node.get("class_type") in CLIP_ENCODE_CLASSES:
            return nid, node
        for v in node.get("inputs", {}).values():
            if _is_link(v):
                queue.append(str(v[0]))
    return None, None


def _resolve_batch(origin, slot):
    raw = origin.get("inputs", {}).get("prompts_json")
    idx = origin.get("inputs", {}).get("index", 0)
    if not isinstance(raw, str):
        return "", False
    try:
        prompts = parse_prompts(raw)
        pos, neg, _ = select_prompt(prompts, idx)
    except Exception:
        return "", False
    return (pos if slot == 0 else neg), True


def _resolve_text(prompt, clip_node):
    text = clip_node.get("inputs", {}).get("text")
    if isinstance(text, str):
        return text, True
    if not _is_link(text):
        return "", False
    oid, slot = str(text[0]), text[1]
    origin = _node(prompt, oid)
    if not origin:
        return "", False
    if origin.get("class_type") == BATCH_CLASS:
        return _resolve_batch(origin, slot)
    for v in origin.get("inputs", {}).values():  # plain primitive/string node
        if isinstance(v, str):
            return v, True
    return "", False


def _resolve_conditioning_text(prompt, sampler, which):
    link = sampler.get("inputs", {}).get(which)
    if not _is_link(link):
        return "", False
    _cid, clip = _find_clip_encode(prompt, str(link[0]))
    if clip is None:
        return "", False
    return _resolve_text(prompt, clip)


def _resolve_prompts(prompt, sampler, r):
    r.positive, ok_p = _resolve_conditioning_text(prompt, sampler, "positive")
    r.negative, ok_n = _resolve_conditioning_text(prompt, sampler, "negative")
    if not ok_p:
        r.unresolved.append("positive")
    if not ok_n:
        r.unresolved.append("negative")


def _stem(name):
    return os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0]


def _trace_model_chain(prompt, sampler, r):
    model = sampler.get("inputs", {}).get("model")
    nid = str(model[0]) if _is_link(model) else None
    seen = set()
    while nid and nid not in seen:
        seen.add(nid)
        node = _node(prompt, nid)
        if not node:
            break
        cls = node.get("class_type")
        ins = node.get("inputs", {})
        if cls in LORA_CLASSES:
            name = ins.get("lora_name")
            strength = ins.get("strength_model", ins.get("strength"))
            if isinstance(name, str):
                r.loras.append({"name": _stem(name),
                                "strength": None if _is_link(strength) else strength,
                                "file": name})
            nxt = ins.get("model")
            nid = str(nxt[0]) if _is_link(nxt) else None
            continue
        if cls in POWER_LORA_CLASSES or cls in LORA_STACK_CLASSES:
            entries = _power_lora_entries(ins) if cls in POWER_LORA_CLASSES else _lora_stack_entries(ins)
            # Reversed within the node so the final r.loras.reverse() restores listed order.
            for lo in reversed(entries):
                r.loras.append(lo)
            nxt = ins.get("model")
            nid = str(nxt[0]) if _is_link(nxt) else None
            continue
        if cls in MODEL_SOURCES:
            widget, folder = MODEL_SOURCES[cls]
            name = ins.get(widget)
            if isinstance(name, str):
                r.model_name = _stem(name)
                r.model_file = name
                r.model_folder = folder
            break
        nxt = ins.get("model")  # pass-through node (e.g. ModelSamplingDiscrete)
        nid = str(nxt[0]) if _is_link(nxt) else None
    r.loras.reverse()  # collected sampler->checkpoint; emit in load order


def trace(prompt, node_id):
    r = TraceResult()
    prompt = prompt or {}

    images = _input(prompt, node_id, "images")
    start = str(images[0]) if _is_link(images) else None

    sid, sampler = (None, None)
    if start is not None:
        sid, sampler = _find_sampler(prompt, start)
    if sampler is None:
        sid, sampler = _single_sampler(prompt)
    if sampler is None:
        r.unresolved.append("sampler")
        return r

    _read_sampler_fields(sampler, r)
    _resolve_prompts(prompt, sampler, r)
    _trace_model_chain(prompt, sampler, r)
    return r
