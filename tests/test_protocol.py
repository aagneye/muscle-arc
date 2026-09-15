"""Unit tests for muscle_arc.geometry.protocol (host_v1 measurement protocol)."""

from __future__ import annotations

import math

import numpy as np

from muscle_arc.geometry.protocol import fl_host_protocol, mt_host_protocol, pa_host_protocol
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


def test_pa_host_protocol_vertical_fascicle_against_horizontal_apo() -> None:
    # Deep apo flat at y=100. Fascicle fragment straight down (dy=1, dx=0)
    # from a point above the apo -> angle between fragment and apo tangent
    # (horizontal) should be ~90 degrees... but pa_range default excludes
    # that; use a tilted fascicle at a known angle instead.
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 100.0, x0=0.0, x1=200.0)

    theta = math.radians(20.0)
    # direction tilted 20 deg from vertical (i.e. 70 deg from horizontal apo)
    direction = np.array([math.sin(theta), math.cos(theta)])
    point = np.array([100.0, 60.0])
    fragments = [(point, direction, 30.0)]

    pa = pa_host_protocol(fragments, surfaces, pa_range=(0.0, 90.0))

    # Angle between fragment direction and horizontal tangent = 90 - theta
    expected = 90.0 - math.degrees(theta)
    assert math.isclose(pa, expected, abs_tol=1.0)


def test_pa_host_protocol_no_fragments_returns_nan() -> None:
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 100.0)
    result = pa_host_protocol([], surfaces)
    assert math.isnan(result)


def test_pa_host_protocol_averages_up_to_three_longest_fragments() -> None:
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 100.0, x0=0.0, x1=200.0)
    theta = math.radians(20.0)
    direction = np.array([math.sin(theta), math.cos(theta)])

    # 5 fragments at the same angle/insertion behaviour but different lengths;
    # only the 3 longest should be used (result should still equal the
    # common angle regardless, but this exercises the ranking/limit path).
    fragments = [
        (np.array([50.0, 60.0]), direction, 10.0),
        (np.array([80.0, 60.0]), direction, 40.0),
        (np.array([110.0, 60.0]), direction, 35.0),
        (np.array([140.0, 60.0]), direction, 30.0),
        (np.array([170.0, 60.0]), direction, 5.0),
    ]

    pa = pa_host_protocol(fragments, surfaces, pa_range=(0.0, 90.0))
    expected = 90.0 - math.degrees(theta)
    assert math.isclose(pa, expected, abs_tol=1.0)


def test_fl_host_protocol_prefers_least_extrapolated_fragments() -> None:
    # Vertical apo gap of 50px; fascicle fragments are vertical (dx=0,dy=1)
    # so FL == 50px for every fragment regardless of visible length, but
    # extrapolation_fraction differs by visible length_px.
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 100.0, x0=0.0, x1=200.0)
    direction = np.array([0.0, 1.0])

    fragments = [
        (np.array([50.0, 60.0]), direction, 5.0),   # heavily extrapolated
        (np.array([80.0, 60.0]), direction, 45.0),  # barely extrapolated
        (np.array([110.0, 60.0]), direction, 40.0),
        (np.array([140.0, 60.0]), direction, 42.0),
        (np.array([170.0, 60.0]), direction, 8.0),  # heavily extrapolated
    ]

    fl_px = fl_host_protocol(fragments, surfaces, mm_per_pixel=None)

    # All candidate FLs are exactly 50px given vertical fragments/apo gap.
    assert math.isclose(fl_px, 50.0, rel_tol=1e-6)


def test_fl_host_protocol_applies_mm_per_pixel_scale() -> None:
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 100.0, x0=0.0, x1=200.0)
    direction = np.array([0.0, 1.0])
    fragments = [
        (np.array([80.0, 60.0]), direction, 45.0),
        (np.array([110.0, 60.0]), direction, 40.0),
        (np.array([140.0, 60.0]), direction, 42.0),
    ]

    fl_mm = fl_host_protocol(fragments, surfaces, mm_per_pixel=0.2)

    assert math.isclose(fl_mm, 10.0, rel_tol=1e-6)


def test_fl_host_protocol_no_fragments_returns_nan() -> None:
    surfaces = _make_surfaces(lambda x: 50.0, lambda x: 100.0)
    result = fl_host_protocol([], surfaces)
    assert math.isnan(result)

