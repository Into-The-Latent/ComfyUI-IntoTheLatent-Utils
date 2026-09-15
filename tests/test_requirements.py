"""The pack's install must never make pip replace the host's torch.

Regression guard for a fresh ComfyUI install breaking after `pip install -r requirements.txt`: a
torch/torchaudio version floor anywhere in the dependency set makes pip upgrade torch, and on Windows
PyPI only carries CPU-only torch wheels, so ComfyUI then fails with "Torch not compiled with CUDA
enabled". These checks cover the files this repo owns; the pinned breeze-tts fork tag (comfyui-v1.4+)
declares `torch` without a floor and no torchaudio for the same reason; older tags are the bug.
"""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH_FAMILY = ("torch", "torchaudio", "torchvision")


def _requirements_lines():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def _pyproject_deps():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def _package_name(spec: str) -> str:
    return re.split(r"[\s@<>=!~;\[]", spec, maxsplit=1)[0].lower()


def test_no_torch_family_in_requirements():
    names = {_package_name(s) for s in _requirements_lines()}
    assert not names & set(TORCH_FAMILY), names


def test_no_torch_family_in_pyproject():
    names = {_package_name(s) for s in _pyproject_deps()}
    assert not names & set(TORCH_FAMILY), names


def test_fork_pin_is_a_tag_and_matches_in_both_files():
    pattern = re.compile(r"breeze-tts @ git\+https://github\.com/Into-The-Latent/breeze-tts@(comfyui-v[\d.]+)$")
    req = [m.group(1) for s in _requirements_lines() if (m := pattern.match(s))]
    proj = [m.group(1) for s in _pyproject_deps() if (m := pattern.match(s))]
    assert len(req) == 1 and len(proj) == 1, (req, proj)
    assert req == proj
    tag_version = tuple(int(part) for part in req[0].removeprefix("comfyui-v").split("."))
    assert tag_version >= (1, 5), tag_version  # comfyui-v1.4 and older declared torch>=2.9
