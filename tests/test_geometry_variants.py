"""Rater-convention geometry: least-extrapolation fascicle choice and apo pairing."""

from __future__ import annotations

import math

import cv2
import numpy as np

from muscle_arc.geometry.metrics import (
    fascicle_length_least_extrap_px,
    muscle_thickness_px,
)


def _apo_two_bands(h: int, w: int, top: tuple[int, int], deep: tuple[int, int]) -> np.ndarray:
    apo = np.zeros((h, w), np.uint8)
    apo[top[0] : top[1], :] = 1
    apo[deep[0] : deep[1], :] = 1
    return apo


def _segment(mask: np.ndarray, cx: float, cy: float, length: float, deg: float, thick: int) -> None:
    dx = math.cos(math.radians(deg)) * length / 2
    dy = -math.sin(math.radians(deg)) * length / 2  # rising to the right
    cv2.line(mask, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 1, thick)


def test_least_extrap_prefers_long_central_fragment() -> None:
    h, w = 160, 400
    apo = _apo_two_bands(h, w, (20, 25), (125, 130))  # inner edges y=24 and y=125
    fasc = np.zeros((h, w), np.uint8)
    _segment(fasc, cx=200, cy=75, length=90, deg=20, thick=3)  # long, central
    _segment(fasc, cx=40, cy=60, length=20, deg=35, thick=3)  # short, at the edge
    fl = fascicle_length_least_extrap_px(fasc, apo, top_k=1)
    expected = 101 / math.sin(math.radians(20))  # gap between inner edges along a 20 deg line
    assert abs(fl - expected) / expected < 0.04


def test_least_extrap_returns_nan_without_fragments() -> None:
    apo = _apo_two_bands(160, 400, (20, 25), (125, 130))
    assert math.isnan(fascicle_length_least_extrap_px(np.zeros_like(apo), apo, top_k=3))


def test_three_band_mask_picks_superficial_pair() -> None:
    h, w = 300, 400
    apo = np.zeros((h, w), np.uint8)
    apo[10:14, :] = 1  # skin line
    apo[80:85, :] = 1  # superficial apo, inner edge y=84
    apo[180:185, :] = 1  # deep apo, inner edge y=180
    fasc = np.zeros((h, w), np.uint8)
    for cx in (120, 200, 280):
        _segment(fasc, cx=cx, cy=132, length=70, deg=20, thick=3)
    mt = muscle_thickness_px(apo, fasc_mask=fasc)
    assert abs(mt - 96) <= 3
