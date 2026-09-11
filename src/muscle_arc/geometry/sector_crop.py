"""Detect B-mode ultrasound sector and crop console chrome."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class SectorCrop:
    crop: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h
    kind: str  # "console" | "cropped"
    chrome_ratio: float
    applied: bool

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("crop", None)
        d["bbox"] = list(self.bbox)
        return d


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    best = cur = None
    best_len = 0
    for i, f in enumerate(flags):
        if f:
            cur = i if cur is None else cur
            length = i - cur + 1
            if length > best_len:
                best_len, best = length, (cur, i + 1)
        else:
            cur = None
    return best


def _extend(profile: np.ndarray, run: tuple[int, int] | None, frac: float) -> tuple[int, int] | None:
    if run is None:
        return None
    lo, hi = run
    cut = frac * float(profile.max())
    while lo > 0 and profile[lo - 1] > cut:
        lo -= 1
    while hi < len(profile) and profile[hi] > cut:
        hi += 1
    return lo, hi


def find_sector(gray: np.ndarray, floor: int = 12) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of the B-mode sector via row/column occupancy.

    Siemens 800×1200 consoles often light the full width (UI + colour bar), so a
    plain occupancy run yields a false full-frame box like (0, 39, 1200, 761).
    Prefer an *interior* high-occupancy band when edge chrome is present.
    """
    nz = gray > floor
    if not nz.any():
        return None
    h0, w0 = gray.shape[:2]
    col = nz.mean(axis=0)
    row = nz.mean(axis=1)
    cmax = float(col.max())
    rmax = float(row.max())
    cs = _extend(col, _longest_run(col > 0.5 * cmax), 0.15)
    rs = _extend(row, _longest_run(row > 0.5 * rmax), 0.15)
    if cs is None or rs is None:
        return None
    x, w = cs[0], cs[1] - cs[0]
    y, h = rs[0], rs[1] - rs[0]

    # Full-width / near-full-width: try a stricter interior column band.
    # Typical console: dark/UI margins + bright sector in the middle third+.
    fill_w = w / max(w0, 1)
    if fill_w > 0.88 and w0 >= 600:
        # Ignore outer 8% columns (colour bar / menu) and re-detect.
        margin = max(8, int(0.08 * w0))
        col_i = col.copy()
        col_i[:margin] = 0.0
        col_i[-margin:] = 0.0
        if float(col_i.max()) > 0.15:
            cs2 = _extend(col_i, _longest_run(col_i > 0.55 * float(col_i.max())), 0.12)
            if cs2 is not None:
                x2, w2 = cs2[0], cs2[1] - cs2[0]
                # Accept only if we actually found an interior band
                if w2 < 0.88 * w0 and w2 > 0.35 * w0:
                    x, w = x2, w2
        # Secondary: brightest contiguous mid-band by column energy
        if w / max(w0, 1) > 0.88:
            energy = gray.mean(axis=0).astype(np.float64)
            energy[:margin] = 0.0
            energy[-margin:] = 0.0
            thr = 0.55 * float(energy.max()) if energy.max() > 0 else 0.0
            cs3 = _extend(energy, _longest_run(energy > thr), 0.10)
            if cs3 is not None:
                x3, w3 = cs3[0], cs3[1] - cs3[0]
                if 0.35 * w0 < w3 < 0.88 * w0:
                    x, w = x3, w3

    if w < 40 or h < 40:
        return None
    return int(x), int(y), int(w), int(h)


def classify_kind(gray: np.ndarray, sector: tuple[int, int, int, int] | None) -> tuple[str, float]:
    """Console screenshots have substantial non-black chrome outside the sector."""
    h, w = gray.shape[:2]
    if sector is None:
        return "cropped", 0.0
    x, y, sw, sh = sector
    area = float(h * w)
    sector_area = float(sw * sh)
    fill = sector_area / max(area, 1.0)
    # Outside sector: fraction of lit pixels (UI text / rulers)
    mask = np.ones((h, w), dtype=bool)
    mask[y : y + sh, x : x + sw] = False
    outside = gray[mask]
    chrome = float((outside > 20).mean()) if outside.size else 0.0
    # Console if sector doesn't fill the frame OR outside chrome is dense
    if fill < 0.72 or chrome > 0.04:
        return "console", chrome
    return "cropped", chrome


def sector_crop(
    gray: np.ndarray,
    *,
    force: bool = False,
    pad: int = 2,
) -> SectorCrop:
    """
    Crop to B-mode sector for console screenshots; leave cropped frames unchanged.

    Geometry/segmentation then run in crop space. Scale OCR still uses full frame.
    """
    full = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    sector = find_sector(full)
    kind, chrome = classify_kind(full, sector)
    if sector is None:
        h, w = full.shape[:2]
        return SectorCrop(full, (0, 0, w, h), kind, chrome, applied=False)

    x, y, sw, sh = sector
    apply = force or kind == "console"
    # Also crop if sector is a clear interior box (<90% of frame)
    h, w = full.shape[:2]
    if (sw * sh) / max(h * w, 1) < 0.90:
        apply = True
        kind = "console" if kind == "cropped" and chrome > 0.02 else kind

    if not apply:
        return SectorCrop(full, (0, 0, w, h), kind, chrome, applied=False)

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + sw + pad)
    y1 = min(h, y + sh + pad)
    crop = full[y0:y1, x0:x1].copy()
    return SectorCrop(crop, (x0, y0, x1 - x0, y1 - y0), kind, chrome, applied=True)


def mask_chrome(
    gray: np.ndarray,
    *,
    pad: int = 2,
    min_fill: float = 0.35,
) -> tuple[np.ndarray, SectorCrop]:
    """Zero pixels outside the B-mode sector **without cropping**.

    Keeps full-frame coordinates (scale + geometry stay aligned) while removing
    console chrome that confuses DL_Track / apo pairing. No-op on cropped frames.
    """
    full = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = full.shape[:2]
    sector = find_sector(full)
    kind, chrome = classify_kind(full, sector)
    info = SectorCrop(full, (0, 0, w, h), kind, chrome, applied=False)
    # Require *real* outside chrome (UI text/rulers). Black-border crops
    # (OSF) often have fill<0.72 and get mis-labeled console — do not mask them.
    if sector is None or chrome < 0.035:
        return full, info
    x, y, sw, sh = sector
    fill = (sw * sh) / max(h * w, 1.0)
    # Refuse tiny/wrong boxes (over-mask). Near-full frames after Siemens
    # interior detection still get a soft edge fade rather than a no-op.
    if fill < min_fill:
        return full, info
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + sw + pad)
    y1 = min(h, y + sh + pad)
    # If the box still spans almost the whole frame, only zero thin margins
    # when chrome is dense outside a slightly inset rectangle.
    if fill > 0.92:
        inset = max(4, int(0.03 * min(h, w)))
        x0, y0 = inset, inset
        x1, y1 = w - inset, h - inset
        if chrome < 0.05:
            return full, info
    masked = np.zeros_like(full)
    masked[y0:y1, x0:x1] = full[y0:y1, x0:x1]
    info = SectorCrop(masked, (x0, y0, x1 - x0, y1 - y0), "console", chrome, applied=True)
    return masked, info
