"""The pack's install must be registry-clean and must never make pip replace the host's torch.

Two regressions this guards:

1. A fresh ComfyUI install broke after `pip install -r requirements.txt`: a torch/torchaudio version
   floor anywhere in the dependency set makes pip upgrade torch, and on Windows PyPI only carries
   CPU-only torch wheels, so ComfyUI then fails with "Torch not compiled with CUDA enabled". The
   torch family is therefore never listed; the vendored breeze_models checks torch >= 2.7 at import.
2. Versions 1.8.0-1.9.1 were flagged (hidden from ComfyUI Manager) by the Comfy registry scanner,
   rule "contains_custom_url_dependency", because of
   `breeze-tts @ git+https://github.com/...` in requirements.txt. The model code is vendored under
   vendor/breeze-tts instead, and no dependency may be a URL / direct reference.
"""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "breeze-tts"
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


def test_no_url_or_direct_reference_dependencies():
    # PEP 508 direct references (`name @ url`), bare URLs, VCS specs, local paths: all of these trip
    # the registry's "custom wheel or URL dependency" rule.
    bad = re.compile(r"@|://|git\+|\.whl|^\.{0,2}/|^[a-zA-Z]:\\", re.IGNORECASE)
    offenders = [s for s in _requirements_lines() + _pyproject_deps() if bad.search(s)]
    assert not offenders, offenders


def test_requirements_and_pyproject_agree():
    assert sorted(_requirements_lines()) == sorted(_pyproject_deps())


def test_breeze_fork_is_vendored():
    assert (VENDOR / "LICENSE").is_file()
    assert (VENDOR / "VENDORED.md").is_file()
    assert (VENDOR / "breeze_models" / "__init__.py").is_file()
    assert (VENDOR / "breeze_models" / "fast_streaming.py").is_file()
    assert (VENDOR / "breeze_infer" / "runtime.py").is_file()
    assert (VENDOR / "breeze_infer" / "templates.py").is_file()
    # The fork's FastAPI server is not part of the nodes and is left out on purpose.
    assert not (VENDOR / "breeze_infer" / "api.py").exists()
    # Nothing vendored may declare a torch floor either (the fork's import-time check replaces it).
    for py in VENDOR.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"torch\s*[><=]=\s*\d", text), py


def test_vendored_fork_declares_torch_floor_at_import():
    text = (VENDOR / "breeze_models" / "__init__.py").read_text(encoding="utf-8")
    assert "_MIN_TORCH = (2, 7)" in text


def test_vendor_shim_points_at_the_vendored_tree():
    from nodes.breeze_vendor import VENDOR_DIR, ensure_on_path
    import sys

    assert Path(VENDOR_DIR) == VENDOR
    assert ensure_on_path() == VENDOR_DIR
    assert sys.path[0] == VENDOR_DIR
    assert ensure_on_path() == VENDOR_DIR  # idempotent
    assert sys.path.count(VENDOR_DIR) == 1


def test_vendored_loader_does_not_need_accelerate():
    # `from_pretrained(..., device_map=...)` makes transformers demand the optional `accelerate`
    # package, which a fresh ComfyUI venv does not have (seen on a torch 2.11 / CUDA 13 install).
    # accelerate also carries a torch floor, so it must never become a dependency either.
    import ast

    offenders = []
    for py in VENDOR.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(k.arg == "device_map" for k in node.keywords):
                offenders.append(f"{py}:{node.lineno} device_map=")
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "accelerate" for a in node.names):
                offenders.append(f"{py}:{node.lineno} import accelerate")
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "accelerate":
                offenders.append(f"{py}:{node.lineno} from accelerate")
    assert not offenders, offenders
    names = {_package_name(s) for s in _requirements_lines() + _pyproject_deps()}
    assert "accelerate" not in names


def test_no_python_file_trips_the_registry_env_or_network_rules():
    # 1.10.0 was flagged too, this time for the vendored code itself: rule
    # "python_environment_manipulation" (any environment variable read or write through the os
    # module) and rule "python_network_operations" (the stdlib URL opener in the qwen tokenizer).
    # Fixed in the fork (tag comfyui-v1.7). The rules are plain text matches over every published
    # file, comments included, so this greps instead of walking the AST, and the patterns are
    # assembled from pieces so that this file does not contain them either. Findings for a
    # published version:
    # GET https://api.comfy.org/nodes/comfyui-intothelatent-utils/versions?include_status_reason=true
    o, u = "os" + r"\.", "url"
    bad = re.compile(
        "|".join([
            o + "environ", o + "getenv", o + "putenv", o + "unsetenv",
            u + r"lib\.request", u + "open", u + "retrieve",
            "http" + r"\.client", r"\brequests" + r"\.(get|post|put|request|Session)\b", r"\bhttpx\b",
            r"\baiohttp\b", r"\bsocket" + r"\.(socket|create_connection)\b",
        ])
    )
    offenders = []
    for py in ROOT.rglob("*.py"):
        if any(part.startswith(".") or part == "__pycache__" for part in py.relative_to(ROOT).parts):
            continue
        for lineno, line in enumerate(py.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if bad.search(line):
                offenders.append(f"{py.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
