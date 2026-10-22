"""Manual host-protocol measurements: raw clicks -> PA / FL / MT in deg and mm."""

from __future__ import annotations

import math

import pytest

from muscle_arc.geometry.manual import (
    acute_angle_deg,
    polyline_length_px,
    summarize_record,
)


def test_polyline_length_sums_segments() -> None:
    pts = [[0, 0], [30, 40], [30, 100], [90, 100]]  # 50 + 60 + 60
    assert polyline_length_px(pts) == pytest.approx(170.0)


def test_acute_angle_between_segments() -> None:
    fasc = [[0, 100], [100 * math.cos(math.radians(20)), 100 - 100 * math.sin(math.radians(20))]]
    apo = [[0, 100], [200, 100]]
    assert acute_angle_deg(fasc, apo) == pytest.approx(20.0, abs=1e-6)
    # direction of drawing must not matter, and the result is always acute
    assert acute_angle_deg(fasc[::-1], apo) == pytest.approx(20.0, abs=1e-6)
    assert acute_angle_deg([[0, 0], [-10, 10]], [[0, 0], [10, 0]]) == pytest.approx(45.0)


def test_summarize_record_means_in_mm() -> None:
    rec = {
        "mt": [[[10, 20], [10, 220]], [[50, 20], [50, 230]], [[90, 20], [90, 210]]],  # 200, 210, 190 px
        "fl": [[[0, 0], [300, 400]], [[0, 0], [0, 500]]],  # 500, 500 px
        "pa": [
            {"fasc": [[0, 0], [10, -10]], "apo": [[0, 0], [10, 0]]},  # 45
            {"fasc": [[0, 0], [10, 0]], "apo": [[0, 0], [10, 0]]},  # 0
        ],
        "skip": False,
    }
    out = summarize_record(rec, mm_per_pixel=0.1)
    assert out["mt_mm"] == pytest.approx(20.0)
    assert out["fl_mm"] == pytest.approx(50.0)
    assert out["pa_deg"] == pytest.approx(22.5)
    assert (out["n_mt"], out["n_fl"], out["n_pa"]) == (3, 2, 2)


def test_summarize_record_missing_parts_are_nan() -> None:
    out = summarize_record({"mt": [], "fl": [], "pa": [], "skip": False}, mm_per_pixel=0.1)
    assert all(math.isnan(out[k]) for k in ("pa_deg", "fl_mm", "mt_mm"))


def test_skipped_record_returns_none() -> None:
    assert summarize_record({"mt": [], "fl": [], "pa": [], "skip": True}, mm_per_pixel=0.1) is None


def test_incomplete_shapes_are_ignored() -> None:
    rec = {
        "mt": [[[10, 20]]],  # a single click is not a line
        "fl": [[[0, 0]]],
        "pa": [{"fasc": [[0, 0], [10, -10]], "apo": [[0, 0]]}],
        "skip": False,
    }
    out = summarize_record(rec, mm_per_pixel=0.1)
    assert (out["n_mt"], out["n_fl"], out["n_pa"]) == (0, 0, 0)
