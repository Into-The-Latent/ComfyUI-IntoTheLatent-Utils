# Runs the frontend logic checks in tests/js with Node — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
# The pure helpers in web/js (no ComfyUI imports) are checked outside the browser; skipped without node.
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = sorted(f for f in os.listdir(os.path.join(HERE, "js")) if f.endswith(".check.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("check", CHECKS)
def test_js_check(check):
    r = subprocess.run(["node", os.path.join(HERE, "js", check)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
