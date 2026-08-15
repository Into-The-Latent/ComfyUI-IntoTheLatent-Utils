# Tests for AutoV2 file hashing + cache — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
import hashlib
import nodes.civitai_metadata.hashing as hashing
from nodes.civitai_metadata.hashing import sha256_autov2, HashCache


def test_autov2_matches_sha256_prefix(tmp_path):
    f = tmp_path / "model.bin"
    f.write_bytes(b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()[:10]
    assert sha256_autov2(str(f)) == expected
    assert expected == "2cf24dba5f"


def test_missing_file_returns_none(tmp_path):
    cache = HashCache(str(tmp_path / "cache.json"))
    assert cache.get(str(tmp_path / "nope.bin")) is None


def test_cache_hit_does_not_recompute(tmp_path, monkeypatch):
    f = tmp_path / "model.bin"
    f.write_bytes(b"hello")
    cache = HashCache(str(tmp_path / "cache.json"))
    assert cache.get(str(f)) == "2cf24dba5f"

    # Second call must hit the cache: make recompute explode if it is called.
    def boom(_):
        raise AssertionError("recomputed despite cache hit")
    monkeypatch.setattr(hashing, "sha256_autov2", boom)
    cache2 = HashCache(str(tmp_path / "cache.json"))  # reloads persisted cache
    assert cache2.get(str(f)) == "2cf24dba5f"
