from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass


@dataclass
class ArchitectureParams:
    pa_deg: float
    fl_mm: float
    mt_mm: float


def _fit_line_angle_deg(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = abs(float(np.degrees(np.arctan2(float(vy), float(vx)))))
    if angle > 90:
        angle = 180 - angle
    return angle


def _apo_bands(apo_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Split aponeurosis mask into superficial (top) and deep (bottom) bands."""
    ys, xs = np.where(apo_mask > 0)
    if len(ys) < 20:
        return None
    mid = float(np.median(ys))
    super_m = np.zeros_like(apo_mask)
    deep_m = np.zeros_like(apo_mask)
    super_m[(apo_mask > 0) & (np.arange(apo_mask.shape[0])[:, None] < mid)] = 1
    deep_m[(apo_mask > 0) & (np.arange(apo_mask.shape[0])[:, None] >= mid)] = 1
    if super_m.sum() < 10 or deep_m.sum() < 10:
        return None
    return super_m, deep_m


def muscle_thickness_px(apo_mask: np.ndarray) -> float:
    bands = _apo_bands(apo_mask)
    if bands is None:
        ys = np.where(apo_mask > 0)[0]
        if len(ys) < 20:
            return float("nan")
        return float(np.percentile(ys, 95) - np.percentile(ys, 5))

    super_m, deep_m = bands
    # Column-wise distance between mean y of each band
    h, w = apo_mask.shape
    dists: list[float] = []
    for x in range(0, w, max(1, w // 64)):
        sy = np.where(super_m[:, x] > 0)[0]
        dy = np.where(deep_m[:, x] > 0)[0]
        if len(sy) == 0 or len(dy) == 0:
            continue
        dists.append(float(np.mean(dy) - np.mean(sy)))
    if not dists:
        return float("nan")
    return float(np.median(np.abs(dists)))


def pennation_angle_deg(fasc_mask: np.ndarray, apo_mask: np.ndarray | None = None) -> float:
    fasc_ang = _fit_line_angle_deg(fasc_mask)
    if fasc_ang is None:
        return float("nan")

    if apo_mask is not None:
        bands = _apo_bands(apo_mask)
        if bands is not None:
            deep_ang = _fit_line_angle_deg(bands[1])
            if deep_ang is not None:
                diff = abs(fasc_ang - deep_ang)
                return float(min(diff, 180 - diff))
    return float(fasc_ang)


def fascicle_length_px(fasc_mask: np.ndarray) -> float:
    ys, xs = np.where(fasc_mask > 0)
    if len(xs) < 20:
        return float("nan")
    # Major-axis extent via PCA
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts -= pts.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(pts, full_matrices=False)
    direction = vt[0]
    proj = pts @ direction
    return float(proj.max() - proj.min())


def estimate_architecture(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray,
    mm_per_pixel: float,
    defaults: tuple[float, float, float] = (20.0, 100.0, 25.0),
) -> ArchitectureParams:
    pa = pennation_angle_deg(fasc_mask, apo_mask)
    fl_px = fascicle_length_px(fasc_mask)
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
