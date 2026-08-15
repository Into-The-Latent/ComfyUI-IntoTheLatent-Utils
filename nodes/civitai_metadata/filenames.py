# Filename token expansion — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
#
# ComfyUI core's folder_paths.get_save_image_path expands %year%/%month%/%day%/... and subfolders,
# but NOT the %date:FORMAT% token the stock Save Image tooltip advertises (that lives in the web
# frontend and doesn't reach a custom node's widget). This expands %date:FORMAT% (and bare %date%)
# so filename prefixes like "%date:yyyy-MM-dd%/Krea-Turbo" work. Pure/comfy-free — the caller passes
# `now`; other tokens are left for get_save_image_path.
import re

# .NET/luxon-style format tokens -> strftime. Longest-first so yyyy wins over yy in a single pass,
# and case distinguishes MM (month) from mm (minute), HH/hh (hour).
_FMT_TOKENS = [("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"),
               ("HH", "%H"), ("hh", "%I"), ("mm", "%M"), ("ss", "%S")]
_FMT_MAP = dict(_FMT_TOKENS)
_FMT_RE = re.compile("|".join(re.escape(tok) for tok, _ in _FMT_TOKENS))
_DATE_RE = re.compile(r"%date:([^%]*)%")


def _to_strftime(fmt):
    return _FMT_RE.sub(lambda m: _FMT_MAP[m.group(0)], fmt)


def expand_date_tokens(prefix, now):
    """Expand ``%date:FORMAT%`` (and bare ``%date%`` -> yyyy-MM-dd) in ``prefix`` using ``now``.

    ``now`` is a ``datetime``. Non-date tokens (``%year%``, ``%width%``, subfolders) are left
    untouched for ``folder_paths.get_save_image_path``.
    """
    if not prefix or "%date" not in prefix:
        return prefix
    prefix = _DATE_RE.sub(lambda m: now.strftime(_to_strftime(m.group(1))), prefix)
    return prefix.replace("%date%", now.strftime("%Y-%m-%d"))
