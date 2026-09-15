"""Tests for the protocol dispatch flag on estimate_architecture."""

from __future__ import annotations

import math

import numpy as np

from muscle_arc.geometry.metrics import estimate_architecture


def _synthetic_masks(h: int = 160, w: int = 220) -> tuple[np.ndarray, np.ndarray]:
    """Two horizontal apo bands (superficial ~y=50, deep ~y=100) with a few
    fascicle fragments tilted between them, similar to prior test fixtures
    used elsewhere in this module's own test-style helper functions."""
    apo = np.zeros((h, w), dtype=np.uint8)
    apo[48:53, :] = 1  # superficial band
    apo[98:103, :] = 1  # deep band

    fasc = np.zeros((h, w), dtype=np.uint8)
    # A few short diagonal fascicle strokes between the bands, drawn thick
    # enough (radius 2) that cv2.fitLine gets a well-conditioned point set.
    import cv2

    for x0 in (40, 100, 160):
        p1 = (x0, 55)
        p2 = (x0 + 12, 95)
        cv2.line(fasc, p1, p2, color=1, thickness=3)
    return apo, fasc


def test_legacy_protocol_is_default_and_matches_direct_call() -> None:
    apo, fasc = _synthetic_masks()
    mm_per_pixel = 0.1

    result_default = estimate_architecture(apo, fasc, mm_per_pixel=mm_per_pixel)
    result_explicit_legacy = estimate_architecture(
        apo, fasc, mm_per_pixel=mm_per_pixel, protocol="legacy"
    )

    assert result_default.pa_deg == result_explicit_legacy.pa_deg
    assert result_default.fl_mm == result_explicit_legacy.fl_mm
    assert result_default.mt_mm == result_explicit_legacy.mt_mm


def test_host_v1_protocol_returns_finite_params_on_valid_input() -> None:
    apo, fasc = _synthetic_masks()
    mm_per_pixel = 0.1

    result = estimate_architecture(apo, fasc, mm_per_pixel=mm_per_pixel, protocol="host_v1")

    assert math.isfinite(result.pa_deg)
    assert math.isfinite(result.fl_mm)
    assert math.isfinite(result.mt_mm)


def test_host_v1_protocol_falls_back_to_legacy_when_surfaces_unavailable() -> None:
    # Degenerate apo mask (too little area) -> extract_apo_surfaces returns
    # None -> _estimate_host_v1 returns None -> should fall back to legacy
    # values / defaults exactly like protocol="legacy" would.
    h, w = 160, 220
    apo = np.zeros((h, w), dtype=np.uint8)
    fasc = np.zeros((h, w), dtype=np.uint8)
    mm_per_pixel = 0.1

    result_host = estimate_architecture(apo, fasc, mm_per_pixel=mm_per_pixel, protocol="host_v1")
    result_legacy = estimate_architecture(apo, fasc, mm_per_pixel=mm_per_pixel, protocol="legacy")

    assert result_host.pa_deg == result_legacy.pa_deg
    assert result_host.fl_mm == result_legacy.fl_mm
    assert result_host.mt_mm == result_legacy.mt_mm
