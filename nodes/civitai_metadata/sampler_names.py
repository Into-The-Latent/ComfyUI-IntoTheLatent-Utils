# ComfyUI -> A1111 sampler-label mapping — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Civitai/A1111 read a sampler *label* (e.g. "DPM++ 2M Karras"). ComfyUI stores a (sampler_name,
# scheduler) pair. Map the common pairs; fall back to a readable rendering so nothing is lost.

_SAMPLER_LABELS = {
    "euler": "Euler",
    "euler_cfg_pp": "Euler",
    "euler_ancestral": "Euler a",
    "euler_ancestral_cfg_pp": "Euler a",
    "heun": "Heun",
    "heunpp2": "Heun",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "lms": "LMS",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_sde_gpu": "DPM++ SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_2m_sde_gpu": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "dpmpp_3m_sde_gpu": "DPM++ 3M SDE",
    "ddim": "DDIM",
    "uni_pc": "UniPC",
    "uni_pc_bh2": "UniPC",
    "lcm": "LCM",
}

# Schedulers A1111 treats as the plain sampler (no suffix).
_PLAIN_SCHEDULERS = {"normal", "simple", "sgm_uniform", "ddim_uniform", "beta", "", None}


def to_a1111_sampler(sampler_name, scheduler):
    base = _SAMPLER_LABELS.get(sampler_name)
    if base is None:
        base = sampler_name.replace("_", " ").title() if sampler_name else "Unknown"
    if scheduler == "karras":
        return f"{base} Karras"
    if scheduler == "exponential":
        return f"{base} Exponential"
    if scheduler in _PLAIN_SCHEDULERS:
        return base
    return f"{base} {scheduler}"  # unknown scheduler: append raw, lose nothing
