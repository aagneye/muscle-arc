"""Tests for the host-confirmed 1cm tick-spacing prior in depth_scale.py."""

from __future__ import annotations

import numpy as np

from muscle_arc.geometry.depth_scale import (
    NO_SCALE_RESOLUTION,
    TICK_SPACING_CM,
    estimate_depth_scale,
)


def test_tick_spacing_constant_is_one_cm() -> None:
    # Host forum answer (2026-04-07/08): "spaced 1 cm apart" — a fact, not a
    # tunable. Locks the constant's value so an accidental edit is caught.
    assert TICK_SPACING_CM == 1.0


def test_no_scale_resolution_matches_hosts_stated_512_rescale() -> None:
    assert NO_SCALE_RESOLUTION == (512, 512)


def test_blank_512x512_image_flags_no_scale_recoverable() -> None:
    # A blank (all-zero) 512x512 image has no chrome, no ruler, no OCR text
    # — every scale source in estimate_depth_scale should fail, landing on
    # the terminal "none" result, which should set no_scale_recoverable
    # given the resolution exactly matches the host's stated rescale target.
    img = np.zeros((512, 512), dtype=np.uint8)

    result = estimate_depth_scale(img)

    assert result.mm_per_pixel is None
    assert result.no_scale_recoverable is True


def test_non_matching_resolution_does_not_set_no_scale_recoverable() -> None:
    img = np.zeros((300, 400), dtype=np.uint8)

    result = estimate_depth_scale(img)

    assert result.no_scale_recoverable is False


def test_default_no_scale_recoverable_is_false_for_backward_compat() -> None:
    from muscle_arc.geometry.depth_scale import DepthScaleResult

    result = DepthScaleResult(
        mm_per_pixel=0.08,
        px_per_cm=125.0,
        depth_cm=5.0,
        sector=None,
        source="ocr+sector",
        confidence=0.9,
    )

    assert result.no_scale_recoverable is False
