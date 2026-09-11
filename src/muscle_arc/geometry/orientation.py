"""Dense fascicle orientation helpers (Radon ±angles, streamline FL)."""

from __future__ import annotations

import cv2
import numpy as np


def vesselness_enhance(gray: np.ndarray) -> np.ndarray:
    """Cheap multi-scale line enhance (Laplacian-of-Gaussian stack)."""
    g = gray.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    acc = np.zeros_like(g)
    for s in (1.0, 2.0, 3.0):
        blur = cv2.GaussianBlur(g, (0, 0), s)
        lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        acc = np.maximum(acc, -lap)
    acc = acc - acc.min()
    if acc.max() > 1e-6:
        acc = acc / acc.max()
    return (acc * 255.0).astype(np.uint8)


def radon_orientation_field(
    gray: np.ndarray,
    apo_mask: np.ndarray | None = None,
    angle_lo: float = -40.0,
    angle_hi: float = 40.0,
    n_angles: int = 81,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (point, direction, length) from ±angle Radon-like projection search.

    Searches both pennation signs (fixes half-range bug).
    """
    if gray is None or gray.ndim != 2:
        return None
    h, w = gray.shape
    y0, y1 = int(0.2 * h), int(0.8 * h)
    if apo_mask is not None and int(apo_mask.sum()) > 20:
        ys = np.where(apo_mask > 0)[0]
        if len(ys):
            y0 = int(np.clip(np.percentile(ys, 15), 0, h - 2))
            y1 = int(np.clip(np.percentile(ys, 85), y0 + 10, h))
    roi = gray[y0:y1, int(0.1 * w) : int(0.9 * w)]
    if roi.size < 100:
        return None
    enh = vesselness_enhance(roi).astype(np.float32)
    best_ang = None
    best_score = -1.0
    for ang in np.linspace(angle_lo, angle_hi, n_angles):
        if abs(ang) < 3.0:
            continue
        M = cv2.getRotationMatrix2D((enh.shape[1] / 2, enh.shape[0] / 2), float(ang), 1.0)
        rot = cv2.warpAffine(enh, M, (enh.shape[1], enh.shape[0]), flags=cv2.INTER_LINEAR)
        proj = rot.mean(axis=1)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_ang = float(ang)
    if best_ang is None:
        return None
    rad = np.radians(best_ang)
    direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
    if direction[1] < 0:
        direction = -direction
    point = np.array([0.5 * w, 0.5 * (y0 + y1)], dtype=np.float64)
    return point, direction, float(0.4 * w)


def orientation_from_mask_window(
    fasc_mask: np.ndarray,
    window: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel double-angle targets on labelled fascicle pixels.

    Returns (cos2θ, sin2θ) maps (zeros elsewhere).
    """
    h, w = fasc_mask.shape
    c2 = np.zeros((h, w), dtype=np.float32)
    s2 = np.zeros((h, w), dtype=np.float32)
    ys, xs = np.where(fasc_mask > 0)
    if len(xs) < 20:
        return c2, s2
    half = window // 2
    # Subsample for speed
    step = max(1, len(xs) // 2000)
    for y, x in zip(ys[::step], xs[::step]):
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        patch = fasc_mask[y0:y1, x0:x1]
        py, px = np.where(patch > 0)
        if len(px) < 8:
            continue
        pts = np.column_stack([px.astype(np.float32), py.astype(np.float32)])
        pts -= pts.mean(axis=0)
        cov = pts.T @ pts
        eigvals, eigvecs = np.linalg.eigh(cov)
        vx, vy = float(eigvecs[0, -1]), float(eigvecs[1, -1])
        theta = float(np.arctan2(vy, vx))
        c2[y, x] = np.cos(2 * theta)
        s2[y, x] = np.sin(2 * theta)
    return c2, s2


def streamline_fl_px(
    ori_cos2: np.ndarray,
    ori_sin2: np.ndarray,
    y_super: np.ndarray,
    y_deep: np.ndarray,
    n_seeds: int = 40,
    step: float = 2.0,
    max_steps: int = 400,
) -> float:
    """Integrate streamlines from deep→superficial through orientation field."""
    h, w = ori_cos2.shape
    valid = np.isfinite(y_super) & np.isfinite(y_deep) & (y_deep > y_super + 2)
    xs = np.flatnonzero(valid)
    if len(xs) < 4:
        return float("nan")
    seeds_x = np.linspace(xs.min(), xs.max(), n_seeds)
    lengths: list[float] = []
    weights: list[float] = []
    for sx in seeds_x:
        x = float(sx)
        xi = int(np.clip(round(x), 0, w - 1))
        if not valid[xi]:
            continue
        y = float(y_deep[xi]) - 1.0
        length = 0.0
        conf = 0.0
        n_conf = 0
        for _ in range(max_steps):
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            c2, s2 = float(ori_cos2[yi, xi]), float(ori_sin2[yi, xi])
            mag = (c2 * c2 + s2 * s2) ** 0.5
            if mag < 0.05:
                # fallback: aim toward superficial at same x
                y_target = float(y_super[xi]) if np.isfinite(y_super[xi]) else y - step
                dy = y_target - y
                dx = 0.0
                nrm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                dx, dy = dx / nrm * step, dy / nrm * step
            else:
                # recover θ from 2θ (choose branch pointing up / toward superficial)
                theta = 0.5 * float(np.arctan2(s2, c2))
                dx, dy = np.cos(theta) * step, np.sin(theta) * step
                # Prefer moving toward decreasing y (superficial is above)
                if dy > 0:
                    dx, dy = -dx, -dy
                conf += mag
                n_conf += 1
            x2, y2 = x + dx, y + dy
            length += float(np.hypot(dx, dy))
            x, y = x2, y2
            if not (0 <= x < w and 0 <= y < h):
                break
            xi2 = int(np.clip(round(x), 0, w - 1))
            if np.isfinite(y_super[xi2]) and y <= float(y_super[xi2]) + 1.0:
                break
        if 8.0 < length < 1.5 * float(np.hypot(h, w)):
            lengths.append(length)
            weights.append(conf / max(n_conf, 1))
    if not lengths:
        return float("nan")
    order = np.argsort(lengths)
    vals = np.asarray(lengths, dtype=np.float64)[order]
    wts = np.asarray(weights, dtype=np.float64)[order]
    cum = np.cumsum(wts)
    mid = 0.5 * cum[-1]
    idx = int(np.searchsorted(cum, mid))
    return float(vals[min(idx, len(vals) - 1)])
