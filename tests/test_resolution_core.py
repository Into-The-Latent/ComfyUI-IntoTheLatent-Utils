# Tests for the comfy-free resolution core — part of ComfyUI-IntoTheLatent-Utils. GPL-3.0.
from nodes.resolution_core import (
    resolve_dims, effective_ar, parse_ar, profile_clamps, aspect_options,
    clamp_snap_multiple,
)

SQ = "1:1 (Square)"
WS = "16:9 (Widescreen)"


def test_default_profile_never_clamps():
    assert resolve_dims("default", "raw", SQ, "landscape", 8, 1.0, 5000, 5000) == (5000, 5000)


def test_snap_multiple_rounds_to_8():
    # 1021/8 = 127.625 -> 128 -> 1024 ; 1000 already a multiple of 8
    assert resolve_dims("default", "raw", SQ, "landscape", 8, 1.0, 1021, 1000) == (1024, 1000)


def test_snap_multiple_is_configurable():
    # 1030/64 = 16.09 -> 16 -> 1024
    assert resolve_dims("default", "raw", SQ, "landscape", 64, 1.0, 1030, 1030) == (1024, 1024)


def test_ideogram_ignores_snap_multiple():
    # snap_multiple=64 is ignored; Ideogram stays mult 16: 1050/16 = 65.625 -> 66 -> 1056
    assert resolve_dims("Ideogram 4", "raw", SQ, "landscape", 64, 1.0, 1050, 1050) == (1056, 1056)


def test_default_uses_snap_multiple_not_model_mult():
    # same input as above but default profile + mult 64 -> 1024, proving the two profiles differ
    assert resolve_dims("default", "raw", SQ, "landscape", 64, 1.0, 1050, 1050) == (1024, 1024)


def test_ideogram_clamps_16_9_keeping_aspect():
    assert resolve_dims("Ideogram 4", "auto", WS, "landscape", 8, 1.0, 4000, 0) == (2048, 1152)


def test_orientation_transpose_portrait_is_taller():
    w, h = resolve_dims("default", "auto", WS, "portrait", 8, 1.0, 1080, 0)
    assert h > w and (w, h) == (1080, 1920)


def test_orientation_landscape_is_wider():
    w, h = resolve_dims("default", "auto", WS, "landscape", 8, 1.0, 1920, 0)
    assert w > h and (w, h) == (1920, 1080)


def test_effective_ar_inverts_for_portrait():
    assert round(effective_ar(WS, "landscape"), 4) == round(16 / 9, 4)
    assert round(effective_ar(WS, "portrait"), 4) == round(9 / 16, 4)


def test_backward_compat_bare_ratio_matches_label():
    bare = resolve_dims("default", "auto", "16:9", "landscape", 8, 1.0, 1920, 0)
    label = resolve_dims("default", "auto", WS, "landscape", 8, 1.0, 1920, 0)
    assert bare == label == (1920, 1080)
    assert parse_ar("16:9") == parse_ar(WS)


def test_new_21_9_ratio():
    # 21:9 ; 2016 is a multiple of 8 and 2016*9/21 = 864 (also /8) -> no rounding ambiguity
    assert resolve_dims("default", "auto", "21:9 (Cinemascope)", "landscape", 8, 1.0, 2016, 0) == (2016, 864)


def test_profile_clamps_flag():
    assert profile_clamps("Ideogram 4") is True
    assert profile_clamps("default") is False


def test_aspect_options_shape():
    opts = aspect_options()
    assert opts[0] == "1:1 (Square)"
    assert "16:9 (Widescreen)" in opts
    assert "21:9 (Cinemascope)" in opts
    assert len(opts) == 9
    # square + landscape only — no portrait ratios listed
    assert not any(lbl.startswith(("9:16", "2:3", "3:4", "4:5", "1:2", "9:21", "1:3")) for lbl in opts)


def test_megapixel_mode_hits_target_and_ratio():
    # megapixel: default profile, 16:9, 2.0 MP -> ~2 MP at ~16:9, snapped to /8
    w, h = resolve_dims("default", "megapixel", WS, "landscape", 8, 2.0, 0, 0)
    assert (w, h) == (1888, 1064)


def test_backward_compat_portrait_bare_ratio():
    # old workflow bare "9:16" with orientation defaulting to landscape still yields portrait dims
    w, h = resolve_dims("default", "auto", "9:16", "landscape", 8, 1.0, 1080, 0)
    assert h > w and (w, h) == (1080, 1920)


def test_flipped_9_21_portrait():
    # 21:9 flipped to portrait (9:21) -> tall; 864 and 2016 are both multiples of 8
    w, h = resolve_dims("default", "auto", "21:9 (Cinemascope)", "portrait", 8, 1.0, 864, 0)
    assert h > w and (w, h) == (864, 2016)


# ── snap_multiple robustness (regression: an emptied number widget serialized "" -> the node's
#    INT validation rejected the whole prompt, blocking even the Ideogram profile that ignores it). ──

def test_clamp_snap_multiple_heals_bad_values_to_default():
    # Anything non-numeric / < 1 heals to the default 8 so a stray value can never break a profile.
    assert clamp_snap_multiple("") == 8       # the empty string an emptied number widget serializes
    assert clamp_snap_multiple(0) == 8
    assert clamp_snap_multiple(None) == 8
    assert clamp_snap_multiple(-5) == 8
    assert clamp_snap_multiple("abc") == 8


def test_clamp_snap_multiple_keeps_and_bounds_valid_values():
    assert clamp_snap_multiple(16) == 16
    assert clamp_snap_multiple("64") == 64    # numeric strings accepted (widget can hand back a string)
    assert clamp_snap_multiple(8.9) == 8      # truncated to int
    assert clamp_snap_multiple(3000) == 1024  # clamped to the widget max


def test_default_profile_empty_snap_multiple_heals_to_8():
    # "" (the exact reported failure value) must not break the resolver: default snaps at mult 8.
    # 1005/8 = 125.625 -> 126 -> 1008 (no .5 boundary, so Python round == JS Math.round).
    assert resolve_dims("default", "raw", SQ, "landscape", "", 1.0, 1005, 1005) == (1008, 1008)


def test_ideogram_ignores_empty_snap_multiple():
    # The core invariant: Ideogram always snaps to 16 regardless of a garbage snap_multiple.
    assert resolve_dims("Ideogram 4", "raw", SQ, "landscape", "", 1.0, 1050, 1050) == (1056, 1056)


def test_ideogram_output_invariant_across_bad_snap_multiple():
    # Property: for ANY snap_multiple, Ideogram yields identical dims, all multiples of 16 in [256,2048].
    ref = resolve_dims("Ideogram 4", "megapixel", WS, "landscape", 8, 4.0, 0, 0)
    for snap in ("", 0, -5, None, 1, 64, 3000):
        w, h = resolve_dims("Ideogram 4", "megapixel", WS, "landscape", snap, 4.0, 0, 0)
        assert (w, h) == ref
        assert w % 16 == 0 and h % 16 == 0
        assert 256 <= w <= 2048 and 256 <= h <= 2048
