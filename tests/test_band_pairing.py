"""Aponeurosis pair selection on multi-band frames (geometry/metrics.py APO_PAIRING)."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from muscle_arc.geometry import metrics as geom
from muscle_arc.geometry.metrics import _apo_bands, muscle_thickness_px


@pytest.fixture
def pairing():
    """Run a test under a given pairing mode and always restore the module default."""
    old = geom.APO_PAIRING

    def _set(mode: str) -> None:
        geom.APO_PAIRING = mode

    yield _set
    geom.APO_PAIRING = old


def _band(mask: np.ndarray, y_left: float, y_right: float, thick: int, x0: int = 0, x1: int | None = None) -> None:
    w = mask.shape[1]
    x1 = w - 1 if x1 is None else x1
    for x in range(x0, x1 + 1):
        y = int(round(y_left + (y_right - y_left) * x / (w - 1)))
        mask[y : y + thick, x] = 1


def _fascicles(mask: np.ndarray, y_lo: int, y_hi: int, n: int, length: int = 80, deg: float = 18.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    w = mask.shape[1]
    for _ in range(n):
        cx, cy = rng.uniform(60, w - 60), rng.uniform(y_lo + 15, y_hi - 15)
        dx, dy = math.cos(math.radians(deg)) * length / 2, -math.sin(math.radians(deg)) * length / 2
        cv2.line(mask, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 1, 3)


def _stacked(h: int = 400, w: int = 600):
    """Thick upper muscle (gap 0.5 h) over a thin lower one: top, middle, bottom bands."""
    apo = np.zeros((h, w), np.uint8)
    _band(apo, 40, 40, 8)  # superficial, inner edge y = 47
    _band(apo, 260, 240, 8)  # middle: deep apo of the upper muscle
    _band(apo, 360, 350, 8, x0=100)  # deep apo of the lower muscle
    fasc = np.zeros((h, w), np.uint8)
    _fascicles(fasc, 47, 245, n=25)
    _fascicles(fasc, 262, 348, n=2, length=40, seed=1)  # sparse detections in the lower muscle
    return apo, fasc


def test_upper_dense_measures_the_thick_upper_muscle(pairing) -> None:
    apo, fasc = _stacked()
    pairing("legacy")
    mt_legacy = muscle_thickness_px(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    pairing("upper_dense")
    mt_new = muscle_thickness_px(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    assert mt_legacy < 150  # documents the bug: per-column surfaces drop gaps > 0.42 h
    assert 185 < mt_new < 215  # top inner edge 47 -> middle inner edge ~250


def test_upper_dense_never_skips_a_long_middle_band(pairing) -> None:
    apo, fasc = _stacked()
    _fascicles(fasc, 262, 348, n=6, length=40, seed=2)  # lower muscle denser, but still second
    pairing("upper_dense")
    sup, deep = _apo_bands(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    assert np.median(np.where(sup)[0]) < 50
    assert 235 < np.median(np.where(deep)[0]) < 270  # the middle band, not the bottom one


def test_fragment_cannot_become_the_deep_band(pairing) -> None:
    h, w = 400, 600
    apo = np.zeros((h, w), np.uint8)
    _band(apo, 60, 60, 8)
    _band(apo, 280, 260, 8)
    apo[340:346, 500:530] = 1  # 30-px fragment under the deep apo
    fasc = np.zeros((h, w), np.uint8)
    _fascicles(fasc, 68, 262, n=25)
    fasc[320:335, 480:540] = 1  # fascicle-like blob near the fragment
    pairing("upper_dense")
    sup, deep = _apo_bands(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    assert deep[340:346, 500:530].sum() == 0
    assert 255 < np.median(np.where(deep)[0]) < 290


def test_sparse_subcutaneous_pair_is_skipped(pairing) -> None:
    h, w = 400, 600
    apo = np.zeros((h, w), np.uint8)
    _band(apo, 20, 20, 5)  # skin line
    _band(apo, 104, 104, 7)  # superficial apo, 8 mm below the skin at 0.1 mm/px
    _band(apo, 320, 300, 8)  # deep apo
    fasc = np.zeros((h, w), np.uint8)
    _fascicles(fasc, 111, 300, n=25)
    fasc[60:64, 200:240] = 1  # one false positive in the subcutaneous fat
    pairing("upper_dense")
    sup, deep = _apo_bands(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    assert 100 < np.median(np.where(sup)[0]) < 112
    assert 295 < np.median(np.where(deep)[0]) < 330


def test_plain_two_band_frame_is_unchanged(pairing) -> None:
    h, w = 400, 600
    apo = np.zeros((h, w), np.uint8)
    _band(apo, 60, 50, 8)
    _band(apo, 250, 230, 8)
    fasc = np.zeros((h, w), np.uint8)
    _fascicles(fasc, 60, 235, n=25)
    pairing("legacy")
    mt_legacy = muscle_thickness_px(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    pairing("upper_dense")
    mt_new = muscle_thickness_px(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    assert mt_new == pytest.approx(mt_legacy, abs=0.5)


def test_without_fascicles_falls_back_to_legacy(pairing) -> None:
    apo, _ = _stacked()
    pairing("upper_dense")
    assert geom._select_band_pair(apo, fasc_mask=None, mm_per_pixel=0.1) is None


def test_long_deep_band_beats_a_short_shallow_piece(pairing) -> None:
    """IMG_00194: a 24 %-wide piece above the deep line on the left, the long deep apo on the right."""
    h, w = 420, 800
    apo = np.zeros((h, w), np.uint8)
    _band(apo, 60, 80, 8)  # superficial, full width
    _band(apo, 300, 300, 6, x0=0, x1=190)  # short piece, shallower than the deep apo
    _band(apo, 330, 370, 8, x0=230)  # long deep apo
    fasc = np.zeros((h, w), np.uint8)
    _fascicles(fasc, 90, 300, n=40)
    pairing("upper_dense")
    sup, deep = _apo_bands(apo, fasc_mask=fasc, mm_per_pixel=0.1)
    cols = np.flatnonzero(deep.any(axis=0))
    assert cols.min() >= 220 and cols.max() > 700  # the long band, not the left piece
    assert np.median(np.where(sup)[0]) < 100
