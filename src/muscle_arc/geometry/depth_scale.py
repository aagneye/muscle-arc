"""Recover mm-per-pixel from depth OCR, tick combs, groups, and OSF shape lookup."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from muscle_arc.geometry.scale import is_placeholder_dpi_scale
from muscle_arc.geometry.sector_crop import classify_kind, find_sector

_CM = re.compile(r"(\d{1,2})\s*[.,]?\s*(\d)?\s*c\s*m", re.I)
# Typical ultrasound mm/px band (OSF expert ≈ 0.04–0.11)
_MM_LO, _MM_HI = 0.025, 0.12
_MM_PRIOR = 0.07


@dataclass
class DepthScaleResult:
    mm_per_pixel: float | None
    px_per_cm: float | None
    depth_cm: float | None
    sector: tuple[int, int, int, int] | None
    source: str
    confidence: float
    text: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        if self.sector is not None:
            d["sector"] = list(self.sector)
        return d


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _ocr_region(region: np.ndarray) -> str:
    try:
        import pytesseract
    except Exception:  # noqa: BLE001
        return ""
    if region.size == 0:
        return ""
    r = cv2.resize(region, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    r = 255 - r
    r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    try:
        return pytesseract.image_to_string(
            r,
            config="--psm 11 -c tessedit_char_whitelist=0123456789.,cmTiefeDepth ",
        )
    except Exception:  # noqa: BLE001
        return ""


def read_depth_cm(gray: np.ndarray, sector: tuple[int, int, int, int] | None) -> tuple[float | None, str]:
    g = gray.copy()
    if sector is not None:
        x, y, w, h = sector
        g[y : y + h, x : x + w] = 0
    text = _ocr_region(g)
    cands: list[float] = []
    for whole, frac in _CM.findall(text):
        val = float(whole) + (float(frac) / 10.0 if frac else 0.0)
        if 1.0 <= val <= 15.0:
            cands.append(val)
        elif 15.0 < val <= 150.0 and 1.0 <= val / 10.0 <= 15.0:
            cands.append(val / 10.0)
    if not cands:
        return None, text
    # Prefer depths that land mm/px in the ultrasound band when sector height is known.
    if sector is not None and sector[3] > 0:
        sh = float(sector[3])
        scored: list[tuple[float, float]] = []
        for d in cands:
            mm = 10.0 * d / sh
            if _MM_LO <= mm <= _MM_HI:
                scored.append((abs(mm - _MM_PRIOR), d))
        if scored:
            scored.sort(key=lambda t: t[0])
            return scored[0][1], text
    return max(cands), text


def _comb_strength(pos: np.ndarray, pitch: float) -> float:
    if pitch <= 0:
        return 0.0
    ph = 2 * np.pi * (np.asarray(pos) / pitch)
    return float(abs(np.exp(1j * ph).mean()))


def _fit_comb(pos: np.ndarray, min_pitch: float = 6.0, strength: float = 0.90) -> tuple[float, float]:
    pos = np.sort(np.asarray(pos, dtype=np.float64))
    if len(pos) < 4:
        return 0.0, 0.0
    span = float(pos[-1] - pos[0])
    if span < min_pitch:
        return 0.0, 0.0
    grid = np.linspace(min_pitch, span, 2000)
    best_p = 0.0
    for p in grid[::-1]:
        if _comb_strength(pos, p) >= strength:
            best_p = float(p)
            break
    if best_p == 0.0:
        return 0.0, 0.0
    fine = np.linspace(best_p * 0.96, best_p * 1.04, 200)
    best_p = float(max(fine, key=lambda p: _comb_strength(pos, p)))
    return best_p, _comb_strength(pos, best_p)


def _tick_components(region: np.ndarray, axis: str) -> list[tuple[float, float, float]]:
    if region.size == 0:
        return []
    hi = float(region.max())
    if hi < 20:
        return []
    thresh = max(25.0, 0.45 * hi)
    binary = (region > thresh).astype(np.uint8)
    n, _lbl, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 3 or w * h == 0 or area / (w * h) < 0.35:
            continue
        if axis == "y":
            thick, length = h, w
            along, across = float(cent[i][1]), float(cent[i][0])
        else:
            thick, length = w, h
            along, across = float(cent[i][0]), float(cent[i][1])
        if not (1 <= thick <= 8 and 2 <= length <= 45):
            continue
        out.append((along, across, float(length)))
    return out


def tick_pitch_px(
    gray: np.ndarray,
    axis: str = "y",
    edge_frac: float = 0.22,
    *,
    faint: bool = False,
) -> tuple[float | None, float]:
    """Return (pitch_px, confidence) for a side/top ruler.

    When faint=True (cropped frames), use autocorrelation + duty-cycle prior:
    true rulers are sparse combs (duty ~0.01–0.05), not dense UI stripes.
    """
    h, w = gray.shape[:2]
    extent = w if axis == "y" else h
    reach = max(20 if faint else 30, int(extent * (0.12 if faint else edge_frac)))
    best_pitch, best_conf = None, 0.0

    if faint:
        for lo, hi in ((0, reach), (extent - reach, extent)):
            region = gray[:, lo:hi] if axis == "y" else gray[lo:hi, :]
            if axis == "y":
                profile = region.max(axis=1).astype(np.float64)
            else:
                profile = region.max(axis=0).astype(np.float64)
            if profile.max() < 15:
                continue
            p = profile - profile.mean()
            if float(np.std(p)) < 1e-6:
                continue
            corr = np.correlate(p, p, mode="full")
            corr = corr[len(corr) // 2 :]
            if len(corr) < 20:
                continue
            corr = corr / max(corr[0], 1e-8)
            lo_p, hi_p = 8, min(120, len(corr) - 1)
            thr = profile.mean() + 0.6 * profile.std()
            duty = float((profile > thr).mean())
            if duty < 0.005 or duty > 0.08:
                continue
            peaks: list[tuple[float, int]] = []
            for lag in range(lo_p, hi_p):
                if corr[lag] < 0.25:
                    continue
                if lag > lo_p and corr[lag] <= corr[lag - 1]:
                    continue
                if lag + 1 < hi_p and corr[lag] < corr[lag + 1]:
                    continue
                peaks.append((float(corr[lag]), lag))
            if not peaks:
                continue
            peaks.sort(reverse=True)
            strength, pitch = peaks[0]
            harm = (
                float(corr[min(int(round(2 * pitch)), len(corr) - 1)])
                if pitch * 2 < len(corr)
                else 0.0
            )
            conf = float(
                strength
                * (0.5 + 0.5 * harm)
                * np.clip((0.05 - abs(duty - 0.02)) / 0.05, 0.3, 1.0)
            )
            if conf > best_conf:
                best_conf = conf
                best_pitch = float(pitch)
        return best_pitch, best_conf

    for lo, hi in ((0, reach), (extent - reach, extent)):
        region = gray[:, lo:hi] if axis == "y" else gray[lo:hi, :]
        comps = _tick_components(region, axis)
        if len(comps) < 4:
            continue
        along = np.array([c[0] for c in comps])
        across = np.array([c[1] for c in comps]) + lo
        length = np.array([c[2] for c in comps])
        for centre in np.unique(np.round(across)):
            sel = np.abs(across - centre) <= 6.0
            if int(sel.sum()) < 4:
                continue
            lens = length[sel]
            if lens.mean() <= 0 or lens.std() / lens.mean() > 0.35:
                continue
            pitch, strength = _fit_comb(along[sel])
            if pitch <= 2:
                continue
            conf = float(strength * np.clip(sel.sum() / 6.0, 0.0, 1.0))
            if conf > best_conf:
                best_conf = conf
                best_pitch = pitch
    return best_pitch, best_conf


def estimate_depth_scale(img: np.ndarray) -> DepthScaleResult:
    """Primary: OCR depth + sector height. Secondary: tick comb (console / cropped)."""
    gray = _to_gray(img)
    sector = find_sector(gray)
    kind, _chrome = classify_kind(gray, sector)
    depth, text = read_depth_cm(gray, sector)

    if sector is not None and depth is not None and depth > 0:
        _x, _y, _w, sh = sector
        px_per_cm = float(sh) / float(depth)
        mm = 10.0 / px_per_cm
        conf = 0.95 if kind == "console" else 0.90
        if not is_placeholder_dpi_scale(mm) and _MM_LO <= mm <= _MM_HI:
            return DepthScaleResult(
                mm_per_pixel=mm,
                px_per_cm=px_per_cm,
                depth_cm=depth,
                sector=sector,
                source="ocr+sector",
                confidence=conf,
                text=text.strip(),
            )

    faint = kind == "cropped"
    # Cropped frames: try vertical then horizontal faint combs (border ticks)
    axes = ("y", "x") if faint else ("y",)
    best_result: DepthScaleResult | None = None
    for axis in axes:
        pitch, conf = tick_pitch_px(gray, axis=axis, faint=faint)
        min_conf = 0.40 if faint else 0.60
        if pitch is None or conf < min_conf:
            continue
        cands: list[tuple[float, float, float, str]] = []
        tick_order = (10.0, 5.0) if faint else (5.0, 10.0)
        for tick_mm in tick_order:
            mm = tick_mm / pitch
            if _MM_LO <= mm <= _MM_HI:
                score = abs(mm - _MM_PRIOR) / max(conf, 1e-3)
                cands.append((score, mm, tick_mm, f"ticks_{tick_mm:g}mm"))
        mm25 = 2.5 / pitch
        if (not faint) and conf >= 0.85 and 0.04 <= mm25 <= 0.09:
            score = abs(mm25 - _MM_PRIOR) / max(conf, 1e-3) + 0.02
            cands.append((score, mm25, 2.5, "ticks_2.5mm"))
        if not cands:
            continue
        cands.sort(key=lambda t: t[0])
        _score, mm, _tick_mm, src = cands[0]
        out_conf = float(conf) * (0.85 if faint else 1.0)
        if out_conf < 0.40:
            continue
        cand = DepthScaleResult(
            mm_per_pixel=float(mm),
            px_per_cm=10.0 / float(mm),
            depth_cm=None,
            sector=sector,
            source=src + ("_faint" if faint else "") + (f"_{axis}" if faint and axis != "y" else ""),
            confidence=out_conf,
            text=text.strip(),
        )
        if best_result is None or cand.confidence > best_result.confidence:
            best_result = cand
    if best_result is not None:
        return best_result

    return DepthScaleResult(
        mm_per_pixel=None,
        px_per_cm=None,
        depth_cm=depth,
        sector=sector,
        source="none",
        confidence=0.0,
        text=text.strip(),
    )


def load_depth_scale_table(path: Path | str) -> dict[str, float]:
    """Build image_id/stem -> mm_per_pixel from audit CSV."""
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, float] = {}
    if "mm_per_pixel" not in df.columns:
        return out
    for _, r in df.iterrows():
        mm = r.get("mm_per_pixel")
        if pd.isna(mm):
            continue
        mm = float(mm)
        if is_placeholder_dpi_scale(mm):
            continue
        conf = float(r.get("confidence", 1.0)) if "confidence" in df.columns else 1.0
        if conf < 0.4:
            continue
        src = str(r.get("source", "")) if "source" in df.columns else ""
        if "ticks_2.5mm" in src and (mm > 0.09 or conf < 0.85):
            continue
        if not (_MM_LO <= mm <= _MM_HI):
            continue
        iid = str(r.get("image_id", r.get("stem", "")))
        if not iid:
            continue
        out[iid] = mm
        out[Path(iid).stem] = mm
    return out


def share_scales_in_groups(
    image_ids: list[str],
    scales: dict[str, float],
    groups: list[list[int]],
) -> dict[str, float]:
    """Propagate median confident scale within each 5-frame (or shape) group."""
    out = dict(scales)
    for idxs in groups:
        vals = []
        for i in idxs:
            iid = image_ids[i]
            if iid in scales:
                vals.append(scales[iid])
            elif Path(iid).stem in scales:
                vals.append(scales[Path(iid).stem])
        if not vals:
            continue
        med = float(np.median(vals))
        for i in idxs:
            iid = image_ids[i]
            out[iid] = med
            out[Path(iid).stem] = med
    return out


def osf_shape_scale_lookup(
    h: int,
    w: int,
    osf_gt: Path | str | None = None,
) -> float | None:
    """Map (h,w) to median OSF mm_per_pixel when shape matches expert set."""
    path = Path(osf_gt or "experiments/osf_expert_gt.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "mm_per_pixel" not in df.columns:
        return None
    scales = []
    for _, r in df.iterrows():
        p = Path(str(r.get("path", "")))
        if not p.exists():
            continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if im.shape[0] == h and im.shape[1] == w and pd.notna(r.get("mm_per_pixel")):
            scales.append(float(r["mm_per_pixel"]))
    if scales:
        mm = float(np.median(scales))
        if not is_placeholder_dpi_scale(mm):
            return mm
    if "scale_pixel_per_cm" in df.columns and df["scale_pixel_per_cm"].notna().any():
        aspects = []
        for _, r in df.iterrows():
            p = Path(str(r.get("path", "")))
            if not p.exists() or pd.isna(r.get("scale_pixel_per_cm")):
                continue
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            aspects.append(
                (
                    abs(im.shape[1] / max(im.shape[0], 1) - w / max(h, 1)),
                    float(r["scale_pixel_per_cm"]),
                )
            )
        if aspects:
            aspects.sort()
            ppc = aspects[0][1]
            mm = 10.0 / ppc
            if not is_placeholder_dpi_scale(mm):
                return mm
    return None
