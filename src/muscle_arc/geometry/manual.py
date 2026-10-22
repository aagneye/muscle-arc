"""Turn manual host-protocol clicks into PA / FL / MT.

The host's protocol (Kaggle data page, OSF benchmark README, DL_Track_US manual
mode): three straight muscle-thickness lines, three fascicles drawn as up to
three connected segments extended to the aponeuroses, and three pennation
angles drawn as one segment along the fascicle and one along the deep
aponeurosis. Each parameter is the mean of its drawings.

Record format (written by tools/labeler): coordinates are original-image pixels
::

    {"mt": [[[x, y], [x, y]], ...],
     "fl": [[[x, y], [x, y], ...], ...],
     "pa": [{"fasc": [[x, y], [x, y]], "apo": [[x, y], [x, y]]}, ...],
     "skip": false}
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = Sequence[float]


def polyline_length_px(points: Sequence[Point]) -> float:
    return float(sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:])))


def acute_angle_deg(seg_a: Sequence[Point], seg_b: Sequence[Point]) -> float:
    """Acute angle (0-90 deg) between the lines through two segments."""
    ax, ay = seg_a[1][0] - seg_a[0][0], seg_a[1][1] - seg_a[0][1]
    bx, by = seg_b[1][0] - seg_b[0][0], seg_b[1][1] - seg_b[0][1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return float("nan")
    cos = abs(ax * bx + ay * by) / (na * nb)
    return math.degrees(math.acos(min(1.0, cos)))


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize_record(rec: dict, mm_per_pixel: float) -> dict | None:
    """Mean PA (deg), FL (mm), MT (mm) and counts; None if the image was skipped."""
    if rec.get("skip"):
        return None
    mt = [polyline_length_px(line) * mm_per_pixel for line in rec.get("mt", []) if len(line) >= 2]
    fl = [polyline_length_px(f) * mm_per_pixel for f in rec.get("fl", []) if len(f) >= 2]
    pa = [
        acute_angle_deg(a["fasc"], a["apo"])
        for a in rec.get("pa", [])
        if len(a.get("fasc", [])) >= 2 and len(a.get("apo", [])) >= 2
    ]
    pa = [v for v in pa if not math.isnan(v)]
    return {
        "pa_deg": _mean(pa),
        "fl_mm": _mean(fl),
        "mt_mm": _mean(mt),
        "n_mt": len(mt),
        "n_fl": len(fl),
        "n_pa": len(pa),
    }
