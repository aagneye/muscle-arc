"""Unit tests for muscle_arc.geometry.protocol (host_v1 measurement protocol)."""

from __future__ import annotations

import math

import numpy as np

from muscle_arc.geometry.protocol import mt_host_protocol
from muscle_arc.geometry.surfaces import ApoSurfaces


def _make_surfaces(
    y_super_fn,
    y_deep_fn,
    x0: float = 0.0,
    x1: float = 200.0,
) -> ApoSurfaces:
    xs = np.linspace(x0, x1, 50)
    y_s = np.array([y_super_fn(x) for x in xs])
    y_d = np.array([y_deep_fn(x) for x in xs])
    poly_super = np.poly1d(np.polyfit(xs, y_s, deg=1))
    poly_deep = np.poly1d(np.polyfit(xs, y_d, deg=1))
    full_x = np.arange(int(x1) + 1, dtype=np.float64)
    return ApoSurfaces(
        y_super=poly_super(full_x),
        y_deep=poly_deep(full_x),
        poly_super=poly_super,
        poly_deep=poly_deep,
        x0=x0,
        x1=x1,
    )


def test_mt_host_protocol_horizontal_parallel_apo_matches_gap_exactly() -> None:
    # Both aponeuroses flat/horizontal, gap = 20 px everywhere.
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 70.0)

    mt_px = mt_host_protocol(surfaces, mm_per_pixel=None)

    assert math.isclose(mt_px, 20.0, rel_tol=1e-6)


def test_mt_host_protocol_applies_mm_per_pixel_scale() -> None:
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 70.0)

    mt_mm = mt_host_protocol(surfaces, mm_per_pixel=0.1)

    assert math.isclose(mt_mm, 2.0, rel_tol=1e-6)


def test_mt_host_protocol_slanted_apo_is_larger_than_true_perpendicular_gap() -> None:
    """Host protocol measures the VERTICAL gap, not the perpendicular gap.

    For two parallel lines separated by a true perpendicular distance d and
    tilted at angle theta from horizontal, the vertical gap is d / cos(theta),
    which is strictly larger than d for theta != 0. This documents the
    expected bias direction vs a perpendicular-distance estimator.
    """
    theta = math.radians(15.0)
    slope = math.tan(theta)
    perpendicular_gap = 20.0
    # Vertical offset between two parallel lines of slope `slope` separated
    # by perpendicular distance `perpendicular_gap`.
    vertical_gap = perpendicular_gap / math.cos(theta)

    surfaces = _make_surfaces(
        lambda x: 50.0 + slope * x,
        lambda x: 50.0 + slope * x + vertical_gap,
    )

    mt_px = mt_host_protocol(surfaces, mm_per_pixel=None)

    assert mt_px > perpendicular_gap
    assert math.isclose(mt_px, vertical_gap, rel_tol=1e-3)


def test_mt_host_protocol_missing_surfaces_returns_nan() -> None:
    surfaces = ApoSurfaces(
        y_super=np.full(10, np.nan),
        y_deep=np.full(10, np.nan),
        poly_super=None,
        poly_deep=None,
        x0=0.0,
        x1=10.0,
    )

    result = mt_host_protocol(surfaces, mm_per_pixel=None)

    assert math.isnan(result)


def test_mt_host_protocol_degenerate_x_range_returns_nan() -> None:
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 70.0, x0=5.0, x1=5.0)

    result = mt_host_protocol(surfaces, mm_per_pixel=None)

    assert math.isnan(result)
