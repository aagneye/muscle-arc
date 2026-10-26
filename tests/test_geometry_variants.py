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


def test_geo_fl_parallel_apos_equals_trig() -> None:
    from muscle_arc.geometry.metrics import fascicle_length_geo_px

    h, w = 300, 600
    apo = _apo_two_bands(h, w, (40, 46), (245, 251))  # inner edges y = 45 and y = 245
    fasc = np.zeros((h, w), np.uint8)
    for cx, cy in ((200, 120), (300, 150), (400, 180), (250, 200)):
        _segment(fasc, cx=cx, cy=cy, length=90, deg=20, thick=3)
    fl = fascicle_length_geo_px(fasc, apo, pa_deg=20.0)
    expected = 200 / math.sin(math.radians(20))
    assert abs(fl - expected) / expected < 0.03


def test_geo_fl_follows_converging_apos_not_trig() -> None:
    from muscle_arc.geometry.metrics import fascicle_length_geo_px

    h, w = 300, 600
    apo = np.zeros((h, w), np.uint8)
    apo[40:46, :] = 1  # superficial, horizontal: inner edge y = 45
    for x in range(w):  # deep apo rising 6 deg to the right: inner edge y = 245 - tan(6) (x - 300)
        y = int(round(245 - math.tan(math.radians(6)) * (x - 300)))
        apo[y : y + 6, x] = 1
    fasc = np.zeros((h, w), np.uint8)
    for cx, cy in ((240, 130), (300, 150), (360, 170)):  # lean toward the superficial apo on the right
        _segment(fasc, cx=cx, cy=cy, length=70, deg=26, thick=3)  # 20 deg to the deep apo
    fl = fascicle_length_geo_px(fasc, apo, pa_deg=20.0)
    # analytic: line through the belly centre (300, 145) at 26 deg from horizontal
    t = math.radians(26)
    up = (145 - 45) / math.sin(t)  # to the superficial apo
    # to the deep apo: solve 145 + s sin t = 245 - tan6 (300 - s cos t - 300) for s
    s = (245 - 145) / (math.sin(t) - math.tan(math.radians(6)) * math.cos(t))
    expected = up + s
    trig = 200 / math.sin(math.radians(20))
    assert abs(fl - expected) / expected < 0.05
    assert abs(fl - trig) / trig > 0.08  # parallel-apo trig is wrong here


def test_pa_tangent_at_insertion_uses_the_curved_deep_apo_where_fascicles_meet_it() -> None:
    from muscle_arc.geometry import metrics as geom
    from muscle_arc.geometry.metrics import pennation_angle_deg

    h, w = 320, 700
    apo = np.zeros((h, w), np.uint8)
    apo[30:36, :] = 1  # flat superficial apo
    # deep apo: flat on the left, bending upward on the right (y = 260 - k (x - 350)^2 for x > 350)
    for x in range(w):
        y = 260 - (0.0008 * (x - 350) ** 2 if x > 350 else 0.0)
        apo[int(round(y)) : int(round(y)) + 6, x] = 1
    fasc = np.zeros((h, w), np.uint8)
    for cx, cy in ((220, 120), (260, 150), (300, 100)):  # fragments on the left, leaning down-right
        _segment(fasc, cx=cx, cy=cy, length=80, deg=-12, thick=3)  # they insert on the curved right part
    old = geom.PA_TANGENT_AT
    try:
        geom.PA_TANGENT_AT = "center"
        pa_center = pennation_angle_deg(fasc, apo)
        geom.PA_TANGENT_AT = "insertion"
        pa_ins = pennation_angle_deg(fasc, apo)
    finally:
        geom.PA_TANGENT_AT = old
    assert np.isfinite(pa_center) and np.isfinite(pa_ins)
    assert pa_ins > pa_center + 2.0  # the rising deep apo at the insertion steepens the angle
