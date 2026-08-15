# AutoV2 file hashing + cache — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# Civitai matches a checkpoint/LoRA by AutoV2 = first 10 hex of the file's full SHA-256. Hashing a
# multi-GB checkpoint is slow, so results are cached to JSON keyed by abspath+size+mtime.
import hashlib
import json
import os


def sha256_autov2(path):
    """First 10 hex chars of the file's full SHA-256 (the value Civitai indexes)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


class HashCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (OSError, ValueError):
            self._data = {}

    def get(self, path):
        """AutoV2 hash for ``path``; compute+persist on miss. None if missing/unreadable."""
        if not path:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        key = os.path.abspath(path)
        entry = self._data.get(key)
        if entry and entry.get("size") == st.st_size and entry.get("mtime") == st.st_mtime:
            return entry.get("hash")
        try:
            # Module-qualified so tests can monkeypatch sha256_autov2 to assert cache hits.
            digest = sha256_autov2(path)
        except OSError:
            return None
        self._data[key] = {"size": st.st_size, "mtime": st.st_mtime, "hash": digest}
        self._save()
        return digest

    def _save(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except OSError:
            pass
