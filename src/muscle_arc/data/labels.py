"""Derive SOF supervision: apo surfaces + fascicle orientation from GT masks."""

from __future__ import annotations

import numpy as np

from muscle_arc.geometry.orientation import orientation_from_mask_window
from muscle_arc.geometry.surfaces import extract_apo_surfaces


def apo_three_class(apo_mask: np.ndarray, fasc_mask: np.ndarray | None = None) -> np.ndarray:
    """Return HxW int64 labels: 0=bg, 1=superficial, 2=deep."""
    h, w = apo_mask.shape
    out = np.zeros((h, w), dtype=np.int64)
    surfaces = extract_apo_surfaces(apo_mask, fasc_mask=fasc_mask)
    if surfaces is None:
        # Fallback: upper/lower half of apo pixels
        ys, xs = np.where(apo_mask > 0)
        if len(ys) == 0:
            return out
        mid = float(np.median(ys))
        out[(apo_mask > 0) & (np.arange(h)[:, None] < mid)] = 1
        out[(apo_mask > 0) & (np.arange(h)[:, None] >= mid)] = 2
        return out
    for x in range(w):
        ys, yd = surfaces.y_super[x], surfaces.y_deep[x]
        if not (np.isfinite(ys) and np.isfinite(yd)):
            continue
        y1 = int(np.clip(round(ys), 0, h - 1))
        y2 = int(np.clip(round(yd), 0, h - 1))
        for dy in range(-2, 3):
            if 0 <= y1 + dy < h:
                out[y1 + dy, x] = 1
            if 0 <= y2 + dy < h:
                out[y2 + dy, x] = 2
    return out


def surface_targets(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_super, y_deep, valid) each length W; y in [0,1] normalized by H."""
    h, w = apo_mask.shape
    y_s = np.full(w, np.nan, dtype=np.float32)
    y_d = np.full(w, np.nan, dtype=np.float32)
    surfaces = extract_apo_surfaces(apo_mask, fasc_mask=fasc_mask)
    if surfaces is not None:
        y_s = surfaces.y_super.astype(np.float32)
        y_d = surfaces.y_deep.astype(np.float32)
    valid = (np.isfinite(y_s) & np.isfinite(y_d) & (y_d > y_s + 2)).astype(np.float32)
    y_s_n = np.where(valid > 0, y_s / max(h - 1, 1), 0.0).astype(np.float32)
    y_d_n = np.where(valid > 0, y_d / max(h - 1, 1), 0.0).astype(np.float32)
    return y_s_n, y_d_n, valid


def orientation_targets(fasc_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cos2θ, sin2θ, presence) float32 HxW."""
    c2, s2 = orientation_from_mask_window(fasc_mask)
    presence = (fasc_mask > 0).astype(np.float32)
    return c2.astype(np.float32), s2.astype(np.float32), presence
