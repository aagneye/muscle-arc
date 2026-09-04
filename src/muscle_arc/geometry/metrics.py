from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArchitectureParams:
    pa_deg: float
    fl_mm: float
    mt_mm: float


def _angle_between_dirs(d1: np.ndarray, d2: np.ndarray) -> float:
    cos = float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _fit_line(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (point, unit_direction) for a binary mask, or None."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 15:
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


def _fit_line_weighted(prob: np.ndarray, min_mass: float = 5.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Probability-weighted PCA line fit (works on soft fascicle maps)."""
    if prob.ndim != 2:
        return None
    ys, xs = np.where(prob > 1e-4)
    if len(xs) < 15:
        return None
    w = prob[ys, xs].astype(np.float64)
    if float(w.sum()) < min_mass:
        return None
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    w = w / w.sum()
    mean = (pts * w[:, None]).sum(axis=0)
    centered = pts - mean
    cov = (centered * w[:, None]).T @ centered
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, int(np.argmax(vals))]
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    direction = direction / norm
    return mean, direction


def _apo_bands(apo_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split apo mask into superficial (top) and deep (bottom) via connected components."""
    binary = (apo_mask > 0).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        comps.append((lab, area, float(centroids[lab][1])))
    if len(comps) < 2:
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
    top2.sort(key=lambda t: t[2])
    super_m = (labels == top2[0][0]).astype(np.uint8)
    deep_m = (labels == top2[1][0]).astype(np.uint8)
    return super_m, deep_m


def _roi_between_apo(apo_mask: np.ndarray) -> np.ndarray | None:
    bands = _apo_bands(apo_mask)
    if bands is None:
        return None
    super_m, deep_m = bands
    h, w = apo_mask.shape
    ys_s = np.where(super_m > 0)[0]
    ys_d = np.where(deep_m > 0)[0]
    if len(ys_s) == 0 or len(ys_d) == 0:
        return None
    y0 = int(np.percentile(ys_s, 50))
    y1 = int(np.percentile(ys_d, 50))
    if y1 <= y0 + 5:
        return None
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y0:y1, :] = 1
    return roi


def _hough_fasc_direction(
    gray: np.ndarray, apo_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Classical Hough fallback for fascicle orientation in the apo ROI."""
    if gray.ndim != 2:
        return None
    h, w = gray.shape
    roi = _roi_between_apo(apo_mask) if apo_mask is not None else np.ones_like(gray, dtype=np.uint8)
    if roi is None:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    blur = cv2.GaussianBlur(enh, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = edges * roi

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=40, minLineLength=max(30, w // 10), maxLineGap=20
    )
    if lines is None:
        return None

    # Keep lines with acute angle to horizontal in [5, 40] degrees (typical PA range)
    scored: list[tuple[float, np.ndarray, np.ndarray]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 10:
            continue
        direction = np.array([dx, dy], dtype=np.float64) / length
        ang = _angle_between_dirs(direction, np.array([1.0, 0.0]))
        if 5.0 <= ang <= 40.0:
            scored.append((length, np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)]), direction))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    # Average top-k directions (same quadrant)
    top = scored[: min(8, len(scored))]
    dirs = np.stack([t[2] for t in top], axis=0)
    # Flip to consistent hemisphere
    ref = dirs[0]
    for i in range(len(dirs)):
        if np.dot(dirs[i], ref) < 0:
            dirs[i] = -dirs[i]
    direction = dirs.mean(axis=0)
    direction /= np.linalg.norm(direction) + 1e-8
    point = np.stack([t[1] for t in top], axis=0).mean(axis=0)
    return point, direction


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
        sp, sd = s_line
        dp, dd = d_line
        nrm = np.array([-sd[1], sd[0]])
        dist1 = abs(float(np.dot(dp - sp, nrm)))
        nrm2 = np.array([-dd[1], dd[0]])
        dist2 = abs(float(np.dot(sp - dp, nrm2)))
        return float(0.5 * (dist1 + dist2))

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


def _fasc_line(
    fasc_mask: np.ndarray,
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
    apo_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if fasc_prob is not None:
        line = _fit_line_weighted(fasc_prob)
        if line is not None:
            return line
    line = _fit_line(fasc_mask)
    if line is not None and int(fasc_mask.sum()) >= 40:
        return line
    if gray is not None:
        return _hough_fasc_direction(gray, apo_mask)
    return line


def pennation_angle_deg(
    fasc_mask: np.ndarray,
    apo_mask: np.ndarray | None = None,
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> float:
    fasc_line = _fasc_line(fasc_mask, fasc_prob=fasc_prob, gray=gray, apo_mask=apo_mask)
    if fasc_line is None:
        return float("nan")
    _, fasc_dir = fasc_line
    horizontal = np.array([1.0, 0.0])
    fasc_vs_h = _angle_between_dirs(fasc_dir, horizontal)

    if apo_mask is not None:
        bands = _apo_bands(apo_mask)
        if bands is not None:
            deep_line = _fit_line(bands[1])
            if deep_line is not None:
                deep_vs_h = _angle_between_dirs(deep_line[1], horizontal)
                if deep_vs_h <= 20.0:
                    return _angle_between_dirs(fasc_dir, deep_line[1])
    return fasc_vs_h


def fascicle_length_px(
    fasc_mask: np.ndarray,
    apo_mask: np.ndarray | None = None,
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
    pa_deg: float | None = None,
    mt_px: float | None = None,
) -> float:
    fasc_line = _fasc_line(fasc_mask, fasc_prob=fasc_prob, gray=gray, apo_mask=apo_mask)
    h, w = fasc_mask.shape
    diag = float(np.hypot(h, w))

    if fasc_line is not None and apo_mask is not None:
        f0, fd = fasc_line
        bands = _apo_bands(apo_mask)
        if bands is not None:
            s_line = _fit_line(bands[0])
            d_line = _fit_line(bands[1])
            if s_line is not None and d_line is not None:
                p1 = _line_intersection(f0, fd, s_line[0], s_line[1])
                p2 = _line_intersection(f0, fd, d_line[0], d_line[1])
                if p1 is not None and p2 is not None:
                    length = float(np.linalg.norm(p1 - p2))
                    in_pad = (
                        -0.25 * w <= p1[0] <= 1.25 * w
                        and -0.25 * h <= p1[1] <= 1.25 * h
                        and -0.25 * w <= p2[0] <= 1.25 * w
                        and -0.25 * h <= p2[1] <= 1.25 * h
                    )
                    if in_pad and 5.0 < length < 1.5 * diag:
                        return length

    # Trig fallback using reliable MT + PA: FL = MT / sin(PA)
    if (
        pa_deg is not None
        and mt_px is not None
        and np.isfinite(pa_deg)
        and np.isfinite(mt_px)
        and 2.0 < abs(pa_deg) < 80.0
        and mt_px > 5.0
    ):
        return float(mt_px / max(np.sin(np.radians(abs(pa_deg))), 1e-3))

    # Last resort: PCA extent of fascicle pixels
    ys, xs = np.where(fasc_mask > 0)
    if len(xs) < 15:
        return float("nan")
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts_c = pts - pts.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(pts_c, full_matrices=False)
    proj = pts_c @ vt[0]
    return float(proj.max() - proj.min())


def _line_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
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
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> ArchitectureParams:
    mt_px = muscle_thickness_px(apo_mask)
    pa = pennation_angle_deg(fasc_mask, apo_mask, fasc_prob=fasc_prob, gray=gray)
    fl_px = fascicle_length_px(
        fasc_mask,
        apo_mask,
        fasc_prob=fasc_prob,
        gray=gray,
        pa_deg=pa if np.isfinite(pa) else None,
        mt_px=mt_px if np.isfinite(mt_px) else None,
    )

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
