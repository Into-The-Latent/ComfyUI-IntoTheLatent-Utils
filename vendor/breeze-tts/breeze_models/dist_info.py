"""Process rank without touching environment variables.

The Comfy registry flags any environment variable access in a published pack, and this package is
vendored into one (tests/test_registry_scanner_clean.py), so the rank comes from torch.distributed
rather than from RANK. Before `init_process_group`, and always inside ComfyUI, that is rank 0.
"""
import torch


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0
