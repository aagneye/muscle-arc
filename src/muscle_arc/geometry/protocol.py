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

from muscle_arc.geometry.surfaces import ApoSurfaces

# Host protocol samples 3 evenly-spaced positions across the image width.
_HOST_SAMPLE_FRACS = (0.25, 0.5, 0.75)


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
