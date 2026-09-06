"""Architecture geometry: PA / FL / MT from apo + fascicle masks.

DL_Track-style recipe: fit apo and fascicles as straight lines, extrapolate
beyond the visible mask, measure PA against the deep aponeurosis, FL as the
apo-to-apo intersection distance, MT as the perpendicular gap between apos.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArchitectureParams:
    pa_deg: float
    fl_mm: float
    mt_mm: float


@dataclass
class GeometryStats:
    """Path counters for diagnose prints (reset per run if desired)."""

    fasc_components: int = 0
    hough_fallback: int = 0
    radon_fallback: int = 0
    pa_from_apo: int = 0
    pa_nan: int = 0
    fl_intersection: int = 0
    fl_trig: int = 0
    fl_nan: int = 0
    mt_ok: int = 0
    mt_nan: int = 0

    def summary(self) -> str:
        return (
            f"geom_stats fasc_comp={self.fasc_components} hough={self.hough_fallback} "
            f"radon={self.radon_fallback} pa_apo={self.pa_from_apo} pa_nan={self.pa_nan} "
            f"fl_x={self.fl_intersection} fl_trig={self.fl_trig} fl_nan={self.fl_nan} "
            f"mt_ok={self.mt_ok} mt_nan={self.mt_nan}"
        )


# Module-level counters updated during measurement (read by calibrate_predict).
STATS = GeometryStats()


def reset_stats() -> None:
    """Clear counters in place so importers keep a live reference."""
    STATS.fasc_components = 0
    STATS.hough_fallback = 0
    STATS.radon_fallback = 0
    STATS.pa_from_apo = 0
    STATS.pa_nan = 0
    STATS.fl_intersection = 0
    STATS.fl_trig = 0
    STATS.fl_nan = 0
    STATS.mt_ok = 0
    STATS.mt_nan = 0


def _angle_between_dirs(d1: np.ndarray, d2: np.ndarray) -> float:
    cos = float(np.clip(abs(np.dot(d1, d2)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _normalize(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return None
    return v / n


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    vals = np.asarray(values, dtype=np.float64)[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    cum = np.cumsum(w)
    mid = 0.5 * cum[-1]
    idx = int(np.searchsorted(cum, mid))
    return float(vals[min(idx, len(vals) - 1)])


def _fit_line_l2(mask: np.ndarray, min_pts: int = 15) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_pts:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    direction = _normalize(np.array([float(vx), float(vy)], dtype=np.float64))
    if direction is None:
        return None
    point = np.array([float(x0), float(y0)], dtype=np.float64)
    return point, direction


def _fit_apo_poly_edge(
    band: np.ndarray, edge: str = "inner_bottom", degree: int = 2
) -> tuple[np.poly1d, float, float] | None:
    """Fit y=poly(x) to an apo inner edge for local tangent evaluation."""
    h, w = band.shape
    xs: list[float] = []
    ys: list[float] = []
    step = max(1, w // 256)
    for x in range(0, w, step):
        col = np.where(band[:, x] > 0)[0]
        if len(col) == 0:
            continue
        if edge == "inner_bottom":
            y = float(col.max())
        elif edge == "inner_top":
            y = float(col.min())
        else:
            y = float(np.median(col))
        xs.append(float(x))
        ys.append(y)
    if len(xs) < 25:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    deg = min(degree, max(1, len(xs) // 20))
    try:
        coeffs = np.polyfit(x_arr, y_arr, deg)
    except np.linalg.LinAlgError:
        return None
    return np.poly1d(coeffs), float(x_arr.min()), float(x_arr.max())


def _fit_line_deg1_edge(
    band: np.ndarray, edge: str = "inner_bottom"
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit y = a*x + b to an apo band edge as a degree-1 line."""
    poly = _fit_apo_poly_edge(band, edge=edge, degree=1)
    if poly is None:
        return None
    p, x0, x1 = poly
    a = float(p.c[0]) if len(p.c) >= 1 else 0.0
    # For degree 1, p(x)=a*x+b
    if len(p.c) == 2:
        a = float(p.c[0])
        b = float(p.c[1])
    else:
        b = float(p(0.5 * (x0 + x1)))
        a = float(np.polyder(p)(0.5 * (x0 + x1)))
    direction = _normalize(np.array([1.0, a], dtype=np.float64))
    if direction is None:
        return None
    xm = 0.5 * (x0 + x1)
    point = np.array([xm, float(a * xm + b)], dtype=np.float64)
    return point, direction


def _local_apo_tangent(
    band: np.ndarray, x: float, edge: str = "inner_top"
) -> np.ndarray | None:
    """Unit tangent of deep/superficial apo at x using local quadratic fit."""
    fitted = _fit_apo_poly_edge(band, edge=edge, degree=2)
    if fitted is None:
        line = _fit_line_deg1_edge(band, edge=edge)
        return None if line is None else line[1]
    poly, x0, x1 = fitted
    xx = float(np.clip(x, x0, x1))
    dy = float(np.polyder(poly)(xx))
    return _normalize(np.array([1.0, dy], dtype=np.float64))


def _apo_bands(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray | None = None,
    mm_per_pixel: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split apo into superficial (top) and deep (bottom) with sanity checks.

    Do NOT reward vertical separation — that picks skin-to-bone on 3-band
    console screenshots. Score by fascicle mass contained between bands,
    physiological MT when scale is known, and prefer the narrower belly pair.
    """
    binary = (apo_mask > 0).astype(np.uint8)
    h, _w = binary.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # (lab, area, cy) sorted top→bottom for adjacent/near-adjacent pairing
    comps: list[tuple[int, int, float]] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 30:
            continue
        comps.append((lab, area, float(centroids[lab][1])))
    comps.sort(key=lambda t: t[2])

    fasc_ys: np.ndarray | None = None
    fasc_med: float | None = None
    if fasc_mask is not None and int(fasc_mask.sum()) >= 50:
        fasc_ys = np.where(fasc_mask > 0)[0]
        if len(fasc_ys):
            fasc_med = float(np.median(fasc_ys))

    if len(comps) >= 2:
        best: tuple[np.ndarray, np.ndarray] | None = None
        best_score = -1e9
        # Adjacent (gap=1) and near-adjacent (gap=2) pairs only — never widest
        for gap in (1, 2):
            for i in range(len(comps) - gap):
                top, bot = comps[i], comps[i + gap]
                sep = abs(bot[2] - top[2])
                if sep < 0.05 * h:
                    continue
                # Soft positional priors (not decisive)
                top_ok = top[2] < 0.50 * h
                bot_ok = bot[2] > 0.25 * h
                if not (top_ok and bot_ok):
                    continue
                if bot[2] < 0.40 * h:
                    continue  # both still in upper frame → skin+superficial

                # Fascicle containment: fraction of fasc mass strictly between bands
                contain = 0.0
                if fasc_ys is not None and len(fasc_ys):
                    y_lo, y_hi = top[2], bot[2]
                    inside = float(np.mean((fasc_ys > y_lo) & (fasc_ys < y_hi)))
                    outside = 1.0 - inside
                    contain = inside - outside  # in [-1, 1]; belly pairs → near +1
                    if fasc_med is not None and not (y_lo < fasc_med < y_hi):
                        contain -= 0.5

                # Hard physiological MT gate when scale known
                mt_term = 0.0
                if mm_per_pixel is not None and mm_per_pixel > 0:
                    mt_mm = sep * float(mm_per_pixel)
                    if mt_mm < 8.0 or mt_mm > 35.0:
                        continue  # reject, do not merely penalize
                    # Prefer mid-range thickness
                    mt_term = 1.0 - abs(mt_mm - 18.0) / 18.0

                # Prefer narrower pairs (anti skin-to-bone): smaller sep is better
                narrow = 1.0 - sep / max(h, 1)
                area_term = (top[1] + bot[1]) / max(float(binary.sum()), 1.0)
                # Primary: containment; secondary: narrow + area + mt
                score = (
                    2.0 * contain
                    + 0.8 * narrow
                    + 0.3 * area_term
                    + 0.5 * mt_term
                    + (0.15 if gap == 1 else 0.0)  # slight adjacent preference
                )
                if score > best_score:
                    best_score = score
                    best = (
                        (labels == top[0]).astype(np.uint8),
                        (labels == bot[0]).astype(np.uint8),
                    )
        if best is not None:
            super_m, deep_m = best
            if _bands_sane(super_m, deep_m, h):
                return super_m, deep_m

    # Fallback: split by median y of all apo pixels (optionally near fasc median)
    ys = np.where(binary > 0)[0]
    if len(ys) < 30:
        return None
    mid = float(np.median(ys))
    if fasc_med is not None:
        mid = float(np.clip(fasc_med, float(np.percentile(ys, 20)), float(np.percentile(ys, 80))))
    super_m = ((binary > 0) & (np.arange(h)[:, None] < mid)).astype(np.uint8)
    deep_m = ((binary > 0) & (np.arange(h)[:, None] >= mid)).astype(np.uint8)
    if not _bands_sane(super_m, deep_m, h):
        return None
    return super_m, deep_m


def _bands_sane(super_m: np.ndarray, deep_m: np.ndarray, h: int) -> bool:
    if int(super_m.sum()) < 20 or int(deep_m.sum()) < 20:
        return False
    ys_s = np.where(super_m > 0)[0]
    ys_d = np.where(deep_m > 0)[0]
    if len(ys_s) == 0 or len(ys_d) == 0:
        return False
    # Superficial centroid must be above deep
    if float(np.mean(ys_s)) >= float(np.mean(ys_d)):
        return False
    # Minimum vertical separation
    if float(np.median(ys_d) - np.median(ys_s)) < 0.04 * h:
        return False
    # Deep band should not live entirely in the top quarter (3rd-band trap)
    if float(np.mean(ys_d)) < 0.22 * h:
        return False
    return True


def _fit_apo_lines(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray | None = None,
    mm_per_pixel: float | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """Return ((p_s, d_s), (p_d, d_d)) for superficial and deep apo lines."""
    bands = _apo_bands(apo_mask, fasc_mask=fasc_mask, mm_per_pixel=mm_per_pixel)
    if bands is None:
        return None
    super_m, deep_m = bands
    # Inner edges facing the muscle belly
    line_s = _fit_line_deg1_edge(super_m, edge="inner_bottom")
    line_d = _fit_line_deg1_edge(deep_m, edge="inner_top")
    if line_s is None:
        line_s = _fit_line_l2(super_m)
    if line_d is None:
        line_d = _fit_line_l2(deep_m)
    if line_s is None or line_d is None:
        return None
    return line_s, line_d


def muscle_thickness_px(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray | None = None,
    mm_per_pixel: float | None = None,
) -> float:
    lines = _fit_apo_lines(apo_mask, fasc_mask=fasc_mask, mm_per_pixel=mm_per_pixel)
    h, w = apo_mask.shape
    if lines is not None:
        (ps, ds), (pd, dd) = lines
        # Sample mid-image x and measure perpendicular gap between lines
        gaps: list[float] = []
        for x in np.linspace(0.15 * w, 0.85 * w, 48):
            # Point on each line at this x: p + t*d with p_x + t*d_x = x
            ys: list[float] = []
            for p, d in ((ps, ds), (pd, dd)):
                if abs(d[0]) < 1e-6:
                    ys.append(float(p[1]))
                else:
                    t = (x - p[0]) / d[0]
                    ys.append(float(p[1] + t * d[1]))
            gap = abs(ys[1] - ys[0])
            # Approximate perpendicular using average tangent
            t_avg = _normalize(ds + dd)
            if t_avg is None:
                continue
            nrm = np.array([-t_avg[1], t_avg[0]])
            # Vertical gap projected onto normal
            perp = abs(float(np.dot(np.array([0.0, ys[1] - ys[0]]), nrm)))
            if perp > 1.0:
                gaps.append(perp if perp > 0 else gap)
            elif gap > 1.0:
                gaps.append(gap)
        if gaps:
            STATS.mt_ok += 1
            return float(np.median(gaps))

    bands = _apo_bands(apo_mask, fasc_mask=fasc_mask, mm_per_pixel=mm_per_pixel)
    if bands is None:
        STATS.mt_nan += 1
        return float("nan")
    super_m, deep_m = bands
    dists: list[float] = []
    step = max(1, w // 64)
    for x in range(0, w, step):
        sy = np.where(super_m[:, x] > 0)[0]
        dy = np.where(deep_m[:, x] > 0)[0]
        if len(sy) == 0 or len(dy) == 0:
            continue
        dists.append(float(np.mean(dy) - np.mean(sy)))
    if not dists:
        STATS.mt_nan += 1
        return float("nan")
    STATS.mt_ok += 1
    return float(np.median(np.abs(dists)))


def _segment_length(point: np.ndarray, direction: np.ndarray, mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    proj = (pts - point) @ direction
    return float(proj.max() - proj.min())


def _fascicle_components(
    fasc_mask: np.ndarray,
    min_area: int = 30,
    min_length: float = 12.0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Fit one line per connected fascicle fragment (no horizontal angle pre-filter)."""
    binary = (fasc_mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        line = _fit_line_l2(comp, min_pts=min(12, max(6, min_area // 3)))
        if line is None:
            continue
        point, direction = line
        length = _segment_length(point, direction, comp)
        if length < min_length:
            continue
        # Orient downward for stable averaging
        if direction[1] < 0:
            direction = -direction
        out.append((point, direction, length))
    STATS.fasc_components += len(out)
    return out


def _dedupe_fragments(
    comps: list[tuple[np.ndarray, np.ndarray, float]],
    angle_tol: float = 4.0,
    dist_tol: float = 25.0,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Drop near-duplicate overlapping fascicle fragments (DL_Track-style)."""
    if len(comps) <= 1:
        return comps
    kept: list[tuple[np.ndarray, np.ndarray, float]] = []
    for point, direction, length in sorted(comps, key=lambda c: -c[2]):
        duplicate = False
        for kp, kd, _ in kept:
            if _angle_between_dirs(direction, kd) > angle_tol:
                continue
            # Distance between parallel lines ~ cross product magnitude
            delta = point - kp
            dist = abs(float(delta[0] * kd[1] - delta[1] * kd[0]))
            if dist < dist_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append((point, direction, length))
    return kept


def _hough_fasc_components(
    gray: np.ndarray, apo_mask: np.ndarray | None
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Classical Hough fallback; robust to OpenCV HoughLinesP array shapes."""
    if gray is None or gray.ndim != 2:
        return []
    h, w = gray.shape
    bands = _apo_bands(apo_mask) if apo_mask is not None else None
    roi = np.ones((h, w), dtype=np.uint8)
    if bands is not None:
        super_m, deep_m = bands
        ys_s = np.where(super_m > 0)[0]
        ys_d = np.where(deep_m > 0)[0]
        if len(ys_s) and len(ys_d):
            y0 = int(np.percentile(ys_s, 50))
            y1 = int(np.percentile(ys_d, 50))
            if y1 > y0 + 5:
                roi = np.zeros((h, w), dtype=np.uint8)
                roi[y0:y1, :] = 1

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(gray)
    blur = cv2.GaussianBlur(enh, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = (edges > 0).astype(np.uint8) * 255
    edges = edges * roi
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=40, minLineLength=max(30, w // 10), maxLineGap=20
    )
    if lines is None:
        return []
    arr = np.asarray(lines)
    # OpenCV may return (N,1,4) or (N,4)
    if arr.ndim == 3:
        segs = arr.reshape(-1, 4)
    elif arr.ndim == 2 and arr.shape[1] >= 4:
        segs = arr[:, :4]
    else:
        return []

    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for seg in segs:
        x1, y1, x2, y2 = [float(v) for v in seg[:4]]
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 15:
            continue
        direction = _normalize(np.array([dx, dy], dtype=np.float64))
        if direction is None:
            continue
        if direction[1] < 0:
            direction = -direction
        point = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float64)
        out.append((point, direction, length))
    if out:
        STATS.hough_fallback += 1
    return out


def _radon_orientation(
    gray: np.ndarray, apo_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Local Radon-like orientation via rotating projection energy in apo ROI."""
    if gray is None or gray.ndim != 2:
        return None
    h, w = gray.shape
    bands = _apo_bands(apo_mask) if apo_mask is not None else None
    y0, y1 = int(0.2 * h), int(0.8 * h)
    if bands is not None:
        ys_s = np.where(bands[0] > 0)[0]
        ys_d = np.where(bands[1] > 0)[0]
        if len(ys_s) and len(ys_d):
            y0 = int(np.percentile(ys_s, 60))
            y1 = int(np.percentile(ys_d, 40))
            if y1 <= y0 + 10:
                y0, y1 = int(0.2 * h), int(0.8 * h)
    roi = gray[y0:y1, int(0.1 * w) : int(0.9 * w)].astype(np.float32)
    if roi.size < 100:
        return None
    # Enhance linear texture
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enh = clahe.apply(np.clip(roi, 0, 255).astype(np.uint8)).astype(np.float32)
    best_ang = None
    best_score = -1.0
    # Search angles 5..40 deg from horizontal (image coords, +y down)
    for ang in np.linspace(5.0, 40.0, 36):
        M = cv2.getRotationMatrix2D((enh.shape[1] / 2, enh.shape[0] / 2), float(ang), 1.0)
        rot = cv2.warpAffine(enh, M, (enh.shape[1], enh.shape[0]), flags=cv2.INTER_LINEAR)
        # Projection variance along rows — fascicles near-horizontal after rotate
        proj = rot.mean(axis=1)
        score = float(np.var(proj))
        if score > best_score:
            best_score = score
            best_ang = float(ang)
    if best_ang is None:
        return None
    rad = np.radians(best_ang)
    direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
    point = np.array([0.5 * w, 0.5 * (y0 + y1)], dtype=np.float64)
    STATS.radon_fallback += 1
    return point, direction, float(0.4 * w)


def _fascicle_lines(
    fasc_mask: np.ndarray,
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
    apo_mask: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    mask = fasc_mask
    if fasc_prob is not None and int(fasc_mask.sum()) < 40:
        thr = float(np.percentile(fasc_prob, 85))
        thr = max(0.15, min(thr, 0.5))
        mask = (fasc_prob > thr).astype(np.uint8)
    comps = _dedupe_fragments(_fascicle_components(mask))
    if len(comps) >= 2:
        return comps
    if len(comps) == 1:
        # Still try Hough/Radon to enrich
        extra: list[tuple[np.ndarray, np.ndarray, float]] = []
        if gray is not None:
            extra.extend(_hough_fasc_components(gray, apo_mask))
            if not extra:
                rad = _radon_orientation(gray, apo_mask)
                if rad is not None:
                    extra.append(rad)
        return _dedupe_fragments(comps + extra)
    # No components — classical fallbacks
    if gray is not None:
        hough = _hough_fasc_components(gray, apo_mask)
        if hough:
            return _dedupe_fragments(hough)
        rad = _radon_orientation(gray, apo_mask)
        if rad is not None:
            return [rad]
    return []


def _line_line_intersection(
    p1: np.ndarray,
    d1: np.ndarray,
    p2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray | None:
    """Intersect two infinite lines p + t d. Extrapolates freely (DL_Track-style)."""
    # Solve p1 + t d1 = p2 + s d2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], dtype=np.float64)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-8:
        return None  # parallel
    rhs = p2 - p1
    try:
        ts = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    t = float(ts[0])
    return p1 + t * d1


def pennation_angle_deg(
    fasc_mask: np.ndarray,
    apo_mask: np.ndarray | None = None,
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
    pa_range: tuple[float, float] = (5.0, 45.0),
    mm_per_pixel: float | None = None,
) -> float:
    """PA = acute angle between fascicle and local deep-apo tangent; Radon ensemble."""
    if apo_mask is None or int(apo_mask.sum()) < 20:
        STATS.pa_nan += 1
        return float("nan")

    bands = _apo_bands(apo_mask, fasc_mask=fasc_mask, mm_per_pixel=mm_per_pixel)
    apo_lines = _fit_apo_lines(apo_mask, fasc_mask=fasc_mask, mm_per_pixel=mm_per_pixel)
    if bands is None and apo_lines is None:
        STATS.pa_nan += 1
        return float("nan")
    dd_global = apo_lines[1][1] if apo_lines is not None else np.array([1.0, 0.0])

    comps = _fascicle_lines(fasc_mask, fasc_prob=fasc_prob, gray=gray, apo_mask=apo_mask)
    angles: list[float] = []
    weights: list[float] = []
    for point, direction, length in comps:
        if bands is not None:
            tangent = _local_apo_tangent(bands[1], float(point[0]), edge="inner_top")
            dd = tangent if tangent is not None else dd_global
        else:
            dd = dd_global
        ang = _angle_between_dirs(direction, dd)
        if pa_range[0] <= ang <= pa_range[1]:
            angles.append(ang)
            weights.append(length)

    if not angles and comps:
        for point, direction, length in comps:
            if bands is not None:
                tangent = _local_apo_tangent(bands[1], float(point[0]), edge="inner_top")
                dd = tangent if tangent is not None else dd_global
            else:
                dd = dd_global
            ang = _angle_between_dirs(direction, dd)
            if 3.0 <= ang <= 50.0:
                angles.append(ang)
                weights.append(length)

    radon_ang: float | None = None
    if gray is not None:
        rad = _radon_orientation(gray, apo_mask)
        if rad is not None:
            _rp, rd, _rl = rad
            ra = _angle_between_dirs(rd, dd_global)
            if pa_range[0] <= ra <= pa_range[1]:
                radon_ang = ra

    if angles:
        mask_pa = _weighted_median(angles, weights)
        STATS.pa_from_apo += 1
        if radon_ang is not None:
            return float(0.65 * mask_pa + 0.35 * radon_ang)
        return mask_pa

    if radon_ang is not None:
        STATS.pa_from_apo += 1
        return float(radon_ang)

    STATS.pa_nan += 1
    return float("nan")


def _edge_fit_fl_px(
    fasc_mask: np.ndarray,
    apo_mask: np.ndarray,
    pa_deg: float | None = None,
    mt_px: float | None = None,
) -> float | None:
    """DL_Track-style FL: fit fascicle edges inside the inter-aponeurosis band."""
    bands = _apo_bands(apo_mask, fasc_mask=fasc_mask)
    if bands is None:
        return None
    super_m, deep_m = bands
    h, w = apo_mask.shape
    ys_s = np.where(super_m > 0)[0]
    ys_d = np.where(deep_m > 0)[0]
    if len(ys_s) == 0 or len(ys_d) == 0:
        return None
    y0 = int(np.percentile(ys_s, 60))
    y1 = int(np.percentile(ys_d, 40))
    if y1 - y0 < 8:
        return None
    band = fasc_mask.copy()
    band[:y0, :] = 0
    band[y1:, :] = 0
    if int(band.sum()) < 40:
        return None
    # Contour edge points → fitLine
    ys, xs = np.where(band > 0)
    if len(xs) < 30:
        return None
    # Sample edge via morphological gradient
    edge = cv2.morphologyEx(band, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    ey, ex = np.where(edge > 0)
    if len(ex) < 20:
        ey, ex = ys, xs
    pts = np.column_stack([ex.astype(np.float32), ey.astype(np.float32)])
    vx, vy, x0, y0p = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    direction = _normalize(np.array([float(vx), float(vy)], dtype=np.float64))
    if direction is None:
        return None
    point = np.array([float(x0), float(y0p)], dtype=np.float64)
    apo_lines = _fit_apo_lines(apo_mask, fasc_mask=fasc_mask)
    if apo_lines is None:
        return None
    (ps, ds), (pd, dd) = apo_lines
    p1 = _line_line_intersection(point, direction, ps, ds)
    p2 = _line_line_intersection(point, direction, pd, dd)
    if p1 is None or p2 is None:
        return None
    fl = float(np.linalg.norm(p1 - p2))
    diag = float(np.hypot(h, w))
    if not (8.0 < fl < 1.5 * diag):
        return None
    if (
        pa_deg is not None
        and np.isfinite(pa_deg)
        and mt_px is not None
        and np.isfinite(mt_px)
        and 5.0 <= abs(pa_deg) <= 45.0
        and mt_px > 5.0
    ):
        trig = float(mt_px / max(np.sin(np.radians(abs(pa_deg))), 1e-3))
        if trig > 1 and (fl > 2.5 * trig or fl < 0.35 * trig):
            return None
    return fl


def fascicle_length_px(
    fasc_mask: np.ndarray,
    apo_mask: np.ndarray | None = None,
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
    pa_deg: float | None = None,
    mt_px: float | None = None,
    mm_per_pixel: float | None = None,
) -> float:
    h, w = fasc_mask.shape
    diag = float(np.hypot(h, w))
    # Equivalent to PA roughly in 10.5–30 deg for a given MT
    fl_lo = fl_hi = None
    if mt_px is not None and np.isfinite(mt_px) and mt_px > 5.0:
        fl_lo = 2.0 * float(mt_px)
        fl_hi = 5.5 * float(mt_px)

    def _in_clamp(fl: float) -> bool:
        if fl_lo is None or fl_hi is None:
            return True
        return fl_lo <= fl <= fl_hi

    inter: float | None = None
    edge_fl: float | None = None

    if apo_mask is not None and int(apo_mask.sum()) >= 20:
        apo_lines = _fit_apo_lines(apo_mask, fasc_mask=fasc_mask, mm_per_pixel=mm_per_pixel)
        comps = _fascicle_lines(fasc_mask, fasc_prob=fasc_prob, gray=gray, apo_mask=apo_mask)
        if apo_lines is not None and comps:
            (ps, ds), (pd, dd) = apo_lines
            lengths: list[float] = []
            weights: list[float] = []
            for point, direction, length in comps:
                p1 = _line_line_intersection(point, direction, ps, ds)
                p2 = _line_line_intersection(point, direction, pd, dd)
                if p1 is None or p2 is None:
                    continue
                fl = float(np.linalg.norm(p1 - p2))
                pad_x, pad_y = 0.75 * w, 0.75 * h
                in_pad = (
                    -pad_x <= p1[0] <= w + pad_x
                    and -pad_y <= p1[1] <= h + pad_y
                    and -pad_x <= p2[0] <= w + pad_x
                    and -pad_y <= p2[1] <= h + pad_y
                )
                if not in_pad:
                    continue
                if not (8.0 < fl < 1.5 * diag):
                    continue
                if mt_px is not None and np.isfinite(mt_px) and fl < 0.7 * mt_px:
                    continue
                if not _in_clamp(fl):
                    continue
                if (
                    pa_deg is not None
                    and np.isfinite(pa_deg)
                    and mt_px is not None
                    and np.isfinite(mt_px)
                    and 10.0 <= abs(pa_deg) <= 35.0
                ):
                    trig = float(mt_px / max(np.sin(np.radians(abs(pa_deg))), 1e-3))
                    if trig > 1 and (fl > 2.5 * trig or fl < 0.35 * trig):
                        continue
                lengths.append(fl)
                weights.append(length)
            if lengths:
                inter = float(_weighted_median(lengths, weights))
                if not _in_clamp(inter):
                    inter = None

        edge_fl = _edge_fit_fl_px(fasc_mask, apo_mask, pa_deg=pa_deg, mt_px=mt_px)
        if edge_fl is not None and not _in_clamp(float(edge_fl)):
            edge_fl = None

    # Trig only when PA is numerically stable (avoid 1/sin(5°) explosions)
    trig: float | None = None
    if (
        pa_deg is not None
        and mt_px is not None
        and np.isfinite(pa_deg)
        and np.isfinite(mt_px)
        and 10.0 <= abs(pa_deg) <= 35.0
        and mt_px > 5.0
    ):
        trig = float(mt_px / max(np.sin(np.radians(abs(pa_deg))), 1e-3))
        if not _in_clamp(trig):
            trig = None

    vals: list[float] = []
    if inter is not None and np.isfinite(inter):
        vals.append(inter)
        STATS.fl_intersection += 1
    if edge_fl is not None and np.isfinite(edge_fl):
        vals.append(float(edge_fl))
        STATS.fl_intersection += 1
    if trig is not None:
        vals.append(trig)
        STATS.fl_trig += 1
    if vals:
        # Median resists one exploded estimator dragging the mean
        return float(np.median(vals))

    # Last resort: clamp a raw intersection/edge if only out-of-range candidates existed
    if mt_px is not None and np.isfinite(mt_px) and mt_px > 5.0 and fl_lo is not None and fl_hi is not None:
        if inter is not None and np.isfinite(inter):
            STATS.fl_intersection += 1
            return float(np.clip(inter, fl_lo, fl_hi))
        if (
            pa_deg is not None
            and np.isfinite(pa_deg)
            and 10.0 <= abs(pa_deg) <= 35.0
        ):
            t = float(mt_px / max(np.sin(np.radians(abs(pa_deg))), 1e-3))
            STATS.fl_trig += 1
            return float(np.clip(t, fl_lo, fl_hi))

    STATS.fl_nan += 1
    return float("nan")


def estimate_architecture(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray,
    mm_per_pixel: float,
    defaults: tuple[float, float, float] = (15.0, 70.0, 20.0),
    fasc_prob: np.ndarray | None = None,
    gray: np.ndarray | None = None,
) -> ArchitectureParams:
    mt_px = muscle_thickness_px(apo_mask, fasc_mask=fasc_mask, mm_per_pixel=mm_per_pixel)
    pa = pennation_angle_deg(
        fasc_mask, apo_mask, fasc_prob=fasc_prob, gray=gray, mm_per_pixel=mm_per_pixel
    )
    fl_px = fascicle_length_px(
        fasc_mask,
        apo_mask,
        fasc_prob=fasc_prob,
        gray=gray,
        pa_deg=pa if np.isfinite(pa) else None,
        mt_px=mt_px if np.isfinite(mt_px) else None,
        mm_per_pixel=mm_per_pixel,
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
