from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArchitectureParams:
    pa_deg: float
    fl_mm: float
    mt_mm: float


def _fit_line(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (point, unit_direction) for a binary mask, or None."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([float(vx), float(vy)], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    direction /= norm
    point = np.array([float(x0), float(y0)], dtype=np.float64)
    return point, direction


def _angle_between_dirs(d1: np.ndarray, d2: np.ndarray) -> float:
    cos = float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _apo_bands(apo_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split apo mask into superficial (top) and deep (bottom) via connected components."""
    binary = (apo_mask > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # stats: [label, x, y, w, h, area] — skip background 0
    comps = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        comps.append((lab, area, float(centroids[lab][1])))
    if len(comps) < 2:
        # Fallback: median-y split
        ys = np.where(binary > 0)[0]
        if len(ys) < 20:
            return None
        mid = float(np.median(ys))
        super_m = ((binary > 0) & (np.arange(binary.shape[0])[:, None] < mid)).astype(np.uint8)
        deep_m = ((binary > 0) & (np.arange(binary.shape[0])[:, None] >= mid)).astype(np.uint8)
        if super_m.sum() < 10 or deep_m.sum() < 10:
            return None
        return super_m, deep_m

    comps.sort(key=lambda t: t[1], reverse=True)
    top2 = comps[:2]
    top2.sort(key=lambda t: t[2])  # smaller mean-y = superficial
    super_m = (labels == top2[0][0]).astype(np.uint8)
    deep_m = (labels == top2[1][0]).astype(np.uint8)
    return super_m, deep_m


def muscle_thickness_px(apo_mask: np.ndarray) -> float:
    bands = _apo_bands(apo_mask)
    if bands is None:
        ys = np.where(apo_mask > 0)[0]
        if len(ys) < 20:
            return float("nan")
        return float(np.percentile(ys, 95) - np.percentile(ys, 5))

    super_m, deep_m = bands
    s_line = _fit_line(super_m)
    d_line = _fit_line(deep_m)
    if s_line is not None and d_line is not None:
        # Perpendicular distance between roughly-parallel apo lines:
        # use deep point to superficial line distance (and vice versa), median.
        sp, sd = s_line
        dp, dd = d_line
        # Normal to superficial direction
        nrm = np.array([-sd[1], sd[0]])
        dist1 = abs(float(np.dot(dp - sp, nrm)))
        nrm2 = np.array([-dd[1], dd[0]])
        dist2 = abs(float(np.dot(sp - dp, nrm2)))
        return float(0.5 * (dist1 + dist2))

    # Column-wise fallback
    h, w = apo_mask.shape
    dists: list[float] = []
    step = max(1, w // 64)
    for x in range(0, w, step):
        sy = np.where(super_m[:, x] > 0)[0]
        dy = np.where(deep_m[:, x] > 0)[0]
        if len(sy) == 0 or len(dy) == 0:
            continue
        dists.append(float(np.mean(dy) - np.mean(sy)))
    if not dists:
        return float("nan")
    return float(np.median(np.abs(dists)))


def pennation_angle_deg(fasc_mask: np.ndarray, apo_mask: np.ndarray | None = None) -> float:
    fasc_line = _fit_line(fasc_mask)
    if fasc_line is None:
        return float("nan")
    _, fasc_dir = fasc_line

    if apo_mask is not None:
        bands = _apo_bands(apo_mask)
        if bands is not None:
            deep_line = _fit_line(bands[1])
            if deep_line is not None:
                return _angle_between_dirs(fasc_dir, deep_line[1])

    # Angle vs horizontal (deep apo often near-horizontal)
    horizontal = np.array([1.0, 0.0])
    return _angle_between_dirs(fasc_dir, horizontal)


def fascicle_length_px(fasc_mask: np.ndarray, apo_mask: np.ndarray | None = None) -> float:
    fasc_line = _fit_line(fasc_mask)
    if fasc_line is None:
        return float("nan")
    f0, fd = fasc_line

    if apo_mask is not None:
        bands = _apo_bands(apo_mask)
        if bands is not None:
            s_line = _fit_line(bands[0])
            d_line = _fit_line(bands[1])
            if s_line is not None and d_line is not None:
                p1 = _line_intersection(f0, fd, s_line[0], s_line[1])
                p2 = _line_intersection(f0, fd, d_line[0], d_line[1])
                if p1 is not None and p2 is not None:
                    return float(np.linalg.norm(p1 - p2))

    # PCA extent fallback on fascicle pixels
    ys, xs = np.where(fasc_mask > 0)
    if len(xs) < 20:
        return float("nan")
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts_c = pts - pts.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(pts_c, full_matrices=False)
    proj = pts_c @ vt[0]
    return float(proj.max() - proj.min())


def _line_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
    """Intersection of lines p1+t*d1 and p2+s*d2."""
    a = np.column_stack([d1, -d2])
    det = float(np.linalg.det(a))
    if abs(det) < 1e-8:
        return None
    t = float(np.linalg.solve(a, p2 - p1)[0])
    return p1 + t * d1


def estimate_architecture(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray,
    mm_per_pixel: float,
    defaults: tuple[float, float, float] = (15.0, 70.0, 20.0),
) -> ArchitectureParams:
    pa = pennation_angle_deg(fasc_mask, apo_mask)
    fl_px = fascicle_length_px(fasc_mask, apo_mask)
    mt_px = muscle_thickness_px(apo_mask)

    pa_deg = defaults[0] if not np.isfinite(pa) else pa
    fl_mm = defaults[1] if not np.isfinite(fl_px) else fl_px * mm_per_pixel
    mt_mm = defaults[2] if not np.isfinite(mt_px) else mt_px * mm_per_pixel
    return ArchitectureParams(pa_deg=float(pa_deg), fl_mm=float(fl_mm), mt_mm=float(mt_mm))


def clip_params(
    params: ArchitectureParams,
    pa_range: tuple[float, float],
    fl_range: tuple[float, float],
    mt_range: tuple[float, float],
) -> ArchitectureParams:
    return ArchitectureParams(
        pa_deg=float(np.clip(params.pa_deg, *pa_range)),
        fl_mm=float(np.clip(params.fl_mm, *fl_range)),
        mt_mm=float(np.clip(params.mt_mm, *mt_range)),
    )
