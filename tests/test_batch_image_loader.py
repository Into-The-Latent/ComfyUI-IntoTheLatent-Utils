# Integration test for the Batch Image Loader nodes — part of ComfyUI-AI2Go-Utils. GPL-3.0.
import json

import pytest

pytest.importorskip("comfy_api")  # only runs inside a ComfyUI environment
import torch  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    import folder_paths
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path))
    return tmp_path


def test_advanced_images_masks_filenames(input_dir):
    from nodes.batch_image_loader import _run_image
    Image.new("RGBA", (32, 16), (255, 0, 0, 128)).save(input_dir / "a.png")
    Image.new("RGB", (8, 8), (0, 255, 0)).save(input_dir / "b.png")
    out = _run_image(json.dumps([{"name": "a.png"}, {"name": "b.png"}]), "off", 1200, advanced=True)

    assert len(out) == 25 and out[0] == 2
    img1, mask1, name1 = out[1], out[2], out[3]
    assert img1.shape == (1, 16, 32, 3)           # [1,H,W,C]
    assert mask1.shape == (1, 16, 32)             # inverted alpha, image-sized
    assert torch.allclose(mask1, torch.full_like(mask1, 1.0 - 128 / 255), atol=1e-3)
    assert name1 == "a.png"
    img2, mask2 = out[4], out[5]
    assert img2.shape == (1, 8, 8, 3)
    assert mask2.shape == (1, 64, 64) and mask2.max() == 0   # no alpha -> stock 64x64 zeros
    assert out[7] is None                          # padding starts after file 2's group


def test_simple_layout(input_dir):
    from nodes.batch_image_loader import _run_image
    Image.new("RGB", (8, 8), (0, 0, 255)).save(input_dir / "c.png")
    out = _run_image(json.dumps([{"name": "c.png"}]), "off", 1200, advanced=False)
    assert len(out) == 9 and out[0] == 1
    assert out[1].shape == (1, 8, 8, 3) and out[2] is None


def test_downscale_fit_applies(input_dir):
    from nodes.batch_image_loader import _run_image
    Image.new("RGB", (2400, 1200), (9, 9, 9)).save(input_dir / "big.png")
    out = _run_image(json.dumps([{"name": "big.png"}]), "fit", 1200, advanced=False)
    assert out[1].shape == (1, 600, 1200, 3)


def test_missing_file_raises(input_dir):
    from nodes.batch_image_loader import _run_image
    with pytest.raises(ValueError, match="gone.png"):
        _run_image(json.dumps([{"name": "gone.png"}]), "off", 1200, advanced=False)
