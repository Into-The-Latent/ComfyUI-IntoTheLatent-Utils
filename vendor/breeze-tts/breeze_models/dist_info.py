"""Process rank without touching environment variables.

Why: `breeze_models` and `breeze_infer` are copied into the ComfyUI pack
ComfyUI-IntoTheLatent-Utils (its vendor/breeze-tts), and the Comfy registry flags a published pack
version on any environment variable access or network request in its files, which hides the
version from ComfyUI Manager. So nothing in these two packages reads or writes environment
variables or opens URLs. Guards: tests/test_registry_scanner_clean.py in this repo
(Into-The-Latent/breeze-tts; tests are not part of the vendored copy) and
tests/test_requirements.py in the pack.

Behaviour change against upstream Breeze TTS: upstream reads RANK / WORLD_SIZE / LOCAL_RANK, which
torchrun sets before any process group exists. Here the values come from torch.distributed, and
nothing in these packages calls `init_process_group`, so unless the caller initialises the process
group first, every process sees rank 0 / world size 1 and `resolve_device()` picks cuda:0. Multi
process callers must either initialise torch.distributed before loading or pass an explicit
device. ComfyUI is single process, where the result is the same as upstream's.
"""
import torch


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
