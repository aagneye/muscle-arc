"""Per-column aponeurosis surfaces (superficial / deep inner edges).

Replaces brittle connected-component apo pairing when two clear bands exist.
Falls back to None so callers can use the legacy ``_apo_bands`` path.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ApoSurfaces:
    """Polynomial surfaces y = poly(x) for superficial (bottom face) and deep (top face)."""

    y_super: np.ndarray  # length W, nan where invalid
    y_deep: np.ndarray
    poly_super: np.poly1d | None
    poly_deep: np.poly1d | None
    x0: float
    x1: float

    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.y_super) & np.isfinite(self.y_deep) & (self.y_deep > self.y_super + 2)


def _column_inner_edges(apo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each column, superficial inner (bottom of upper run) and deep inner (top of lower)."""
    h, w = apo.shape
    y_s = np.full(w, np.nan, dtype=np.float64)
    y_d = np.full(w, np.nan, dtype=np.float64)
    binary = (apo > 0).astype(np.uint8)
    for x in range(w):
        col = binary[:, x]
        ys = np.flatnonzero(col)
        if len(ys) < 2:
            continue
        # Split into contiguous runs
        gaps = np.where(np.diff(ys) > 2)[0]
        runs: list[np.ndarray] = []
        start = 0
        for g in gaps:
            runs.append(ys[start : g + 1])
            start = g + 1
        runs.append(ys[start:])
        runs = [r for r in runs if len(r) >= 1]
        if len(runs) >= 2:
            upper, lower = runs[0], runs[-1]
            # Prefer runs with meaningful vertical separation
            if float(lower[0] - upper[-1]) < 0.04 * h:
                # try adjacent pair with max separation among first/last few
                best = None
                best_sep = -1.0
                for i in range(len(runs) - 1):
                    sep = float(runs[i + 1][0] - runs[i][-1])
                    if sep > best_sep:
                        best_sep = sep
                        best = (runs[i], runs[i + 1])
                if best is None or best_sep < 0.04 * h:
                    continue
                upper, lower = best
            y_s[x] = float(upper[-1])  # inner bottom of superficial
            y_d[x] = float(lower[0])  # inner top of deep
        elif len(runs) == 1 and len(runs[0]) >= 4:
            # Single thick band — use 20/80 percentiles as faces (weak)
            r = runs[0]
            y_s[x] = float(np.percentile(r, 20))
            y_d[x] = float(np.percentile(r, 80))
            if y_d[x] - y_s[x] < 0.05 * h:
                y_s[x] = y_d[x] = np.nan
    return y_s, y_d


def _fit_poly(xs: np.ndarray, ys: np.ndarray, degree: int = 2) -> np.poly1d | None:
    if len(xs) < max(4, degree + 2):
        return None
    try:
        coeff = np.polyfit(xs, ys, deg=min(degree, len(xs) - 1))
        return np.poly1d(coeff)
    except Exception:  # noqa: BLE001
        return None


def extract_apo_surfaces(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray | None = None,
    degree: int = 2,
    min_valid_frac: float = 0.08,
) -> ApoSurfaces | None:
    """Extract labelled superficial/deep inner surfaces from an apo mask."""
    if apo_mask is None or int(apo_mask.sum()) < 40:
        return None
    h, w = apo_mask.shape
    y_s, y_d = _column_inner_edges(apo_mask)
    valid = np.isfinite(y_s) & np.isfinite(y_d) & (y_d > y_s + 2)
    # Optional: prefer columns that contain fascicle mass between surfaces
    if fasc_mask is not None and int(fasc_mask.sum()) >= 50:
        fasc = fasc_mask > 0
        score = np.zeros(w, dtype=np.float64)
        for x in np.flatnonzero(valid):
            ys0, ys1 = int(y_s[x]), int(y_d[x])
            if ys1 <= ys0:
                continue
            score[x] = float(fasc[ys0:ys1, x].mean()) if ys1 > ys0 else 0.0
        # Drop columns with zero fasc if enough remain
        keep = valid & ((score > 0.01) | (score.max() < 1e-6))
        if float(keep.mean()) >= min_valid_frac:
            valid = keep
    if float(valid.mean()) < min_valid_frac:
        return None
    xs = np.flatnonzero(valid).astype(np.float64)
    poly_s = _fit_poly(xs, y_s[valid], degree=degree)
    poly_d = _fit_poly(xs, y_d[valid], degree=degree)
    # Smooth fill via poly where valid is sparse
    y_s_out = y_s.copy()
    y_d_out = y_d.copy()
    if poly_s is not None and poly_d is not None:
        for x in range(w):
            ys_p = float(poly_s(x))
            yd_p = float(poly_d(x))
            if not (0 <= ys_p < h and 0 <= yd_p < h and yd_p > ys_p + 2):
                continue
            if not np.isfinite(y_s_out[x]):
                y_s_out[x] = ys_p
            if not np.isfinite(y_d_out[x]):
                y_d_out[x] = yd_p
    x0, x1 = float(xs.min()), float(xs.max())
    return ApoSurfaces(
        y_super=y_s_out,
        y_deep=y_d_out,
        poly_super=poly_s,
        poly_deep=poly_d,
        x0=x0,
        x1=x1,
    )


def surfaces_to_bands(surfaces: ApoSurfaces, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize ±2px bands around surfaces for legacy callers."""
    h, w = shape
    super_m = np.zeros((h, w), dtype=np.uint8)
    deep_m = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        ys, yd = surfaces.y_super[x], surfaces.y_deep[x]
        if not (np.isfinite(ys) and np.isfinite(yd)):
            continue
        for dy in range(-2, 3):
            y1 = int(round(ys)) + dy
            y2 = int(round(yd)) + dy
            if 0 <= y1 < h:
                super_m[y1, x] = 1
            if 0 <= y2 < h:
                deep_m[y2, x] = 1
    return super_m, deep_m


def thickness_from_surfaces(surfaces: ApoSurfaces) -> float:
    """Median perpendicular-ish gap (vertical proxy) between surfaces."""
    valid = surfaces.valid_mask()
    if not np.any(valid):
        return float("nan")
    return float(np.median(surfaces.y_deep[valid] - surfaces.y_super[valid]))


def line_from_surface(
    surfaces: ApoSurfaces, which: str = "deep"
) -> tuple[np.ndarray, np.ndarray] | None:
    """Deg-1 fallback line (point, direction) from a poly surface."""
    poly = surfaces.poly_deep if which == "deep" else surfaces.poly_super
    if poly is None:
        return None
    x_mid = 0.5 * (surfaces.x0 + surfaces.x1)
    y_mid = float(poly(x_mid))
    # dy/dx from derivative
    dpoly = np.polyder(poly)
    slope = float(dpoly(x_mid))
    direction = np.array([1.0, slope], dtype=np.float64)
    n = float(np.linalg.norm(direction))
    if n < 1e-8:
        return None
    direction = direction / n
    point = np.array([x_mid, y_mid], dtype=np.float64)
    return point, direction


def intersect_ray_surface(
    point: np.ndarray,
    direction: np.ndarray,
    surfaces: ApoSurfaces,
    which: str = "super",
    x_pad: float = 50.0,
) -> np.ndarray | None:
    """Intersect ray with polynomial surface y=poly(x) via 1D search on x."""
    poly = surfaces.poly_super if which in ("super", "superficial") else surfaces.poly_deep
    if poly is None:
        return None
    dx = float(direction[0])
    dy = float(direction[1])
    if abs(dx) < 1e-8:
        # nearly vertical — sample at point x
        x = float(point[0])
        y = float(poly(x))
        return np.array([x, y], dtype=np.float64)
    # Parametrize x = px + t*dx; require y_line(x) = poly(x)
    # px + t*dx = x  → t = (x-px)/dx
    # py + t*dy = poly(x)
    xs = np.linspace(surfaces.x0 - x_pad, surfaces.x1 + x_pad, 400)
    best = None
    best_abs = 1e18
    px, py = float(point[0]), float(point[1])
    for x in xs:
        t = (x - px) / dx
        y_line = py + t * dy
        y_surf = float(poly(x))
        err = abs(y_line - y_surf)
        if err < best_abs:
            best_abs = err
            best = np.array([x, y_surf], dtype=np.float64)
    if best is None or best_abs > 3.0:
        return None
    return best
