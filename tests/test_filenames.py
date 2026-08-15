# Tests for filename %date:FORMAT% token expansion — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from datetime import datetime

from nodes.civitai_metadata.filenames import expand_date_tokens

NOW = datetime(2026, 7, 12, 9, 8, 7)  # 2026-07-12 09:08:07


def test_date_with_format_and_subfolder():
    assert expand_date_tokens("%date:yyyy-MM-dd%/Krea-Turbo_9-16", NOW) == "2026-07-12/Krea-Turbo_9-16"


def test_date_year_only():
    assert expand_date_tokens("%date:yyyy%/x", NOW) == "2026/x"


def test_date_full_datetime():
    assert expand_date_tokens("%date:yyyyMMdd_HHmmss%", NOW) == "20260712_090807"


def test_bare_date_defaults_to_iso():
    assert expand_date_tokens("%date%/x", NOW) == "2026-07-12/x"


def test_month_vs_minute_case_sensitive():
    # MM = month (07), mm = minute (08)
    assert expand_date_tokens("%date:MM-mm%", NOW) == "07-08"


def test_no_token_left_untouched():
    assert expand_date_tokens("plain/name", NOW) == "plain/name"
    # %year% is get_save_image_path's job — expand_date_tokens must not touch it.
    assert expand_date_tokens("%year%/name", NOW) == "%year%/name"


def test_empty_and_none():
    assert expand_date_tokens("", NOW) == ""
    assert expand_date_tokens(None, NOW) is None
