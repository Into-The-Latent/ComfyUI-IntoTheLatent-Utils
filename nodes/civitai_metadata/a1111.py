# A1111 `parameters` chunk formatter — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Produces the flat text Civitai parses: positive (+ appended <lora:...> tags), a "Negative prompt:"
# line, then a comma-separated settings line. Unknown/None fields are omitted, never guessed.


def _num(x):
    """Render a number A1111-style: ints stay ints, integral floats drop the .0 (7.0 -> 7)."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def format_parameters(*, positive, negative, steps, cfg, seed, sampler, width, height,
                      model_name=None, model_hash=None, loras=None, denoise=None,
                      clip_skip=None, version="ComfyUI"):
    loras = loras or []
    tags = " ".join(
        f"<lora:{lo['name']}:{_num(lo['strength'])}>"
        for lo in loras if lo.get("strength") is not None
    )
    pos = (positive or "").strip()
    pos_line = (pos + " " + tags).strip() if tags else pos
    lines = [pos_line, f"Negative prompt: {negative or ''}"]

    parts = []

    def add(key, val):
        if val is not None and val != "":
            parts.append(f"{key}: {val}")

    add("Steps", _num(steps) if steps is not None else None)
    add("Sampler", sampler or None)
    add("CFG scale", _num(cfg) if cfg is not None else None)
    add("Seed", _num(seed) if seed is not None else None)
    add("Size", f"{width}x{height}")
    add("Model hash", model_hash)
    add("Model", model_name)
    if denoise is not None and float(denoise) < 1.0:
        add("Denoising strength", _num(denoise))
    if clip_skip is not None:
        add("Clip skip", _num(clip_skip))
    hashed = [lo for lo in loras if lo.get("hash")]
    if hashed:
        inner = ", ".join(f"{lo['name']}: {lo['hash']}" for lo in hashed)
        add("Lora hashes", f'"{inner}"')
    add("Version", version)

    lines.append(", ".join(parts))
    return "\n".join(lines)
