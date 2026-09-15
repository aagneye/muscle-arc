"""Host ground-truth measurement protocol (host_v1).

Reimplements PA / FL / MT to match the exact manual annotation protocol the
UMUD Challenge host team used to build ground truth, as described by Paul
Ritsche on the Kaggle forum ("Test data leakage | manual labeling", comment
2026-04-18):

- **MT**: 3 straight lines from upper to lower aponeurosis — left, middle,
  right — each drawn **perpendicular to the global image frame** (i.e.
  vertical, not perpendicular to either aponeurosis), then averaged. Host's
  stated rationale: perpendicular-to-apo introduces ~0.1mm bias vs vertical
  when only one aponeurosis is slanted, and DL_Track itself uses the
  vertical/global convention.
- **FL**: 3 fascicles, each a continuous line extended to its intersection
  with an aponeurosis, chosen to **minimise the amount of extrapolation**
  needed (i.e. prefer fragments that are already close to spanning the
  fascicle, not short stubs needing a long extension), then averaged.
- **PA**: 3 angles measured at **fascicle-fragment insertion points**
  (*not* the same fragments used for FL, and *not* computed as the angle at
  a line/aponeurosis intersection), then averaged.

Ground truth itself is the mean of 6 values (3 per rater x 2 raters), so all
protocol functions here return a **mean of 3**, matching the host's
per-rater aggregation — this is a deliberate difference from the existing
`geometry/metrics.py` functions, which mostly use a (more outlier-robust)
median over many more samples. This module exists to A/B against that
legacy convention (see `--geometry-protocol` flag in `eval_osf_pipeline.py`
and `scripts/calibrate_predict.py`), not to replace it outright — only
adopt whichever wins on OSF Gate2 (GT scale) per-parameter MAE.
"""

from __future__ import annotations

import numpy as np

from muscle_arc.geometry.surfaces import ApoSurfaces, intersect_ray_surface

# Host protocol samples 3 evenly-spaced positions across the image width.
_HOST_SAMPLE_FRACS = (0.25, 0.5, 0.75)

# Host protocol averages 3 fascicle fragments / 3 angles per image.
_N_HOST_SAMPLES = 3


def mt_host_protocol(
    surfaces: ApoSurfaces,
    mm_per_pixel: float | None = None,
) -> float:
    """Mean of 3 vertical (global-image-perpendicular) apo-to-apo gaps.

    Samples the fitted superficial/deep polynomial surfaces at 25%/50%/75%
    of the valid x-range and takes the plain vertical difference
    (y_deep - y_super) at each — this is the *global-frame* gap, not a
    gap projected onto either aponeurosis' local normal. Returns NaN if
    fewer than 2 of the 3 sample columns are valid (unlike the legacy
    48-sample-median path in `metrics.muscle_thickness_px`, this protocol
    intentionally does not fall back to band-scanning, to stay faithful to
    "3 lines, left/middle/right" rather than a dense scan).
    """
    if surfaces.poly_super is None or surfaces.poly_deep is None:
        return float("nan")

    x0, x1 = surfaces.x0, surfaces.x1
    if not (np.isfinite(x0) and np.isfinite(x1)) or x1 <= x0:
        return float("nan")

    gaps: list[float] = []
    for frac in _HOST_SAMPLE_FRACS:
        x = x0 + frac * (x1 - x0)
        y_super = float(surfaces.poly_super(x))
        y_deep = float(surfaces.poly_deep(x))
        gap = y_deep - y_super
        if gap > 0:
            gaps.append(gap)

    if len(gaps) < 2:
        return float("nan")

    mt_px = float(np.mean(gaps))
    if mm_per_pixel is not None and np.isfinite(mm_per_pixel):
        return mt_px * float(mm_per_pixel)
    return mt_px


def _fragment_angle_at_insertion(
    point: np.ndarray,
    direction: np.ndarray,
    surfaces: ApoSurfaces,
) -> float | None:
    """Angle (deg) between a fascicle fragment and the deep-apo tangent at
    the fragment's own insertion point (host protocol: measured at the
    fragment's insertion, not at a line/aponeurosis intersection point)."""
    insertion = intersect_ray_surface(point, direction, surfaces, which="deep")
    if insertion is None:
        return None
    x_ins = float(np.clip(insertion[0], surfaces.x0, surfaces.x1))
    # Local tangent direction at x_ins via derivative of the deep polynomial.
    if surfaces.poly_deep is None:
        return None
    dpoly = np.polyder(surfaces.poly_deep)
    slope = float(dpoly(x_ins))
    tangent = np.array([1.0, slope], dtype=np.float64)
    tnorm = float(np.linalg.norm(tangent))
    if tnorm < 1e-8:
        return None
    tangent = tangent / tnorm

    dnorm = float(np.linalg.norm(direction))
    if dnorm < 1e-8:
        return None
    d = direction / dnorm

    cos_ang = float(np.clip(abs(np.dot(d, tangent)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_ang)))


def pa_host_protocol(
    fragments: list[tuple[np.ndarray, np.ndarray, float]],
    surfaces: ApoSurfaces,
    pa_range: tuple[float, float] = (3.0, 50.0),
) -> float:
    """Mean of up to 3 pennation angles at fascicle-fragment insertion points.

    Host protocol: 3 angles measured where fascicle fragments meet the deep
    aponeurosis (their natural insertion), not derived from the same
    fragments used for FL and not computed as a line/aponeurosis
    intersection angle. Fragments are ranked by length (longest first, as a
    proxy for "most visually confident") and the first 3 whose insertion
    angle falls within `pa_range` are averaged.
    """
    if not fragments or surfaces.poly_deep is None:
        return float("nan")

    ranked = sorted(fragments, key=lambda f: -f[2])
    angles: list[float] = []
    for point, direction, _length in ranked:
        if len(angles) >= _N_HOST_SAMPLES:
            break
        ang = _fragment_angle_at_insertion(point, direction, surfaces)
        if ang is not None and pa_range[0] <= ang <= pa_range[1]:
            angles.append(ang)

    if not angles:
        return float("nan")
    return float(np.mean(angles))


def _extrapolation_fraction(
    point: np.ndarray,
    direction: np.ndarray,
    length_px: float,
    surfaces: ApoSurfaces,
) -> tuple[float, float] | None:
    """Return (extrapolation_fraction, fl_px) for one fragment extended to
    both aponeuroses. extrapolation_fraction = extra length needed beyond
    the visible fragment / total extended length (lower = fragment already
    spans most of the true fascicle, matching host's "minimise
    extrapolation" selection rule)."""
    p_super = intersect_ray_surface(point, direction, surfaces, which="super")
    p_deep = intersect_ray_surface(point, direction, surfaces, which="deep")
    if p_super is None or p_deep is None:
        return None
    fl_px = float(np.linalg.norm(p_deep - p_super))
    if fl_px <= 0:
        return None
    extra = max(0.0, fl_px - float(length_px))
    frac = extra / fl_px
    return frac, fl_px


def fl_host_protocol(
    fragments: list[tuple[np.ndarray, np.ndarray, float]],
    surfaces: ApoSurfaces,
    mm_per_pixel: float | None = None,
) -> float:
    """Mean FL of the 3 fascicle fragments needing the LEAST extrapolation.

    Host protocol: fascicles are drawn as continuous lines extended to
    their apo intersections, and raters explicitly "selected the fascicles
    in a way that minimised extrapolation" — i.e. prefer fragments that
    already visually span most of the gap, not short stubs stretched a
    long way. Ranks all fragments by extrapolation_fraction ascending and
    averages the FL of the best 3.
    """
    if not fragments or surfaces.poly_super is None or surfaces.poly_deep is None:
        return float("nan")

    candidates: list[tuple[float, float]] = []  # (extrapolation_fraction, fl_px)
    for point, direction, length_px in fragments:
        result = _extrapolation_fraction(point, direction, length_px, surfaces)
        if result is not None:
            candidates.append(result)

    if not candidates:
        return float("nan")

    candidates.sort(key=lambda c: c[0])
    best = candidates[:_N_HOST_SAMPLES]
    fl_px = float(np.mean([c[1] for c in best]))

    if mm_per_pixel is not None and np.isfinite(mm_per_pixel):
        return fl_px * float(mm_per_pixel)
    return fl_px

