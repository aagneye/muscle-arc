"""Pixel-to-mm scale resolution helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def is_placeholder_dpi_scale(mm_per_pixel: float, xres: float | None = None) -> bool:
    """Reject classic 72 DPI placeholders that look like mm/px ≈ 0.353."""
    if not np.isfinite(mm_per_pixel):
        return True
    # 25.4/72 ≈ 0.35278
    if abs(mm_per_pixel - 25.4 / 72.0) < 1e-3:
        return True
    if xres is not None and abs(float(xres) - 72.0) < 0.5:
        return True
    # Ultrasound mm/px is typically ~0.02–0.20; >0.25 almost always bogus metadata
    if mm_per_pixel > 0.25 or mm_per_pixel < 0.005:
        return True
    return False


def load_scale_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "mm_per_pixel" not in df.columns:
        return df
    ok = []
    for _, r in df.iterrows():
        mm = r.get("mm_per_pixel")
        xres = r.get("xres")
        if pd.isna(mm) or is_placeholder_dpi_scale(float(mm), None if pd.isna(xres) else float(xres)):
            ok.append(False)
        else:
            ok.append(True)
    df = df.copy()
    df["scale_usable"] = ok
    return df


def intensity_signature(gray: np.ndarray, bins: int = 16) -> np.ndarray:
    """Compact intensity histogram for device/gain clustering."""
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten().astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def cluster_key(h: int, w: int, gray: np.ndarray | None = None) -> str:
    """Richer than raw HxW: aspect bucket + height bucket + brightness bucket."""
    aspect = w / max(h, 1)
    if aspect < 1.1:
        ab = "sq"
    elif aspect < 1.4:
        ab = "wide"
    elif aspect < 1.7:
        ab = "wider"
    else:
        ab = "pano"
    if h < 550:
        hb = "hS"
    elif h < 750:
        hb = "hM"
    else:
        hb = "hL"
    if gray is None:
        return f"{ab}_{hb}"
    mean = float(np.mean(gray))
    if mean < 40:
        ib = "dark"
    elif mean < 80:
        ib = "mid"
    else:
        ib = "bright"
    return f"{ab}_{hb}_{ib}"


def assign_scales(
    raw: pd.DataFrame,
    sample_scales: list[dict],
    default_scale: float,
    meta_by_id: dict[str, float] | None = None,
    ocr_by_id: dict[str, float] | None = None,
    shape_by_hw: dict[tuple[int, int], float] | None = None,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Assign per-row mt/fl mm_per_pixel.

    Priority:
      1) usable TIFF/ImageJ metadata
      2) depth OCR / tick scale (ocr_by_id)
      3) same cluster_key as a sample-GT calibrated row
      4) nearest cluster by shared aspect/height prefix
      5) OSF/shape lookup (shape_by_hw)
      6) median of all sample scales / default
    """
    meta_by_id = meta_by_id or {}
    ocr_by_id = ocr_by_id or {}
    shape_by_hw = shape_by_hw or {}
    # sample_scales entries: {image_id, cluster, mt_scale, fl_scale}
    by_cluster_mt: dict[str, list[float]] = {}
    by_cluster_fl: dict[str, list[float]] = {}
    for s in sample_scales:
        by_cluster_mt.setdefault(s["cluster"], []).append(s["mt_scale"])
        by_cluster_fl.setdefault(s["cluster"], []).append(s["fl_scale"])
    cluster_mt = {k: float(np.median(v)) for k, v in by_cluster_mt.items()}
    cluster_fl = {k: float(np.median(v)) for k, v in by_cluster_fl.items()}
    primary_mt = float(np.median([s["mt_scale"] for s in sample_scales])) if sample_scales else default_scale
    primary_fl = float(np.median([s["fl_scale"] for s in sample_scales])) if sample_scales else primary_mt

    mt_out = []
    fl_out = []
    sources = []
    for row in raw.itertuples():
        iid = str(row.image_id)
        cluster = str(getattr(row, "cluster", "unk"))
        stem = Path(iid).stem
        # 1) metadata
        if iid in meta_by_id or stem in meta_by_id:
            mm = meta_by_id.get(iid, meta_by_id.get(stem))
            mt_out.append(float(mm))
            fl_out.append(float(mm))
            sources.append("meta")
            continue
        # 2) depth OCR / ticks (beats cluster — per-image truth)
        if iid in ocr_by_id or stem in ocr_by_id:
            mm = ocr_by_id.get(iid, ocr_by_id.get(stem))
            mt_out.append(float(mm))
            fl_out.append(float(mm))
            sources.append("ocr")
            continue
        # 3) exact cluster
        if cluster in cluster_mt:
            mt_out.append(cluster_mt[cluster])
            fl_out.append(cluster_fl.get(cluster, cluster_mt[cluster]))
            sources.append("cluster")
            continue
        # 4) prefix match aspect_height
        parts = cluster.rsplit("_", 1)
        prefix = parts[0] if len(parts) == 2 else cluster
        matches_mt = [v for k, v in cluster_mt.items() if k.startswith(prefix)]
        matches_fl = [v for k, v in cluster_fl.items() if k.startswith(prefix)]
        if matches_mt:
            mt_out.append(float(np.median(matches_mt)))
            fl_out.append(float(np.median(matches_fl)) if matches_fl else float(np.median(matches_mt)))
            sources.append("cluster_prefix")
            continue
        # 5) OSF / shape lookup
        h = int(getattr(row, "h", 0) or 0)
        w = int(getattr(row, "w", 0) or 0)
        if (h, w) in shape_by_hw:
            mm = float(shape_by_hw[(h, w)])
            mt_out.append(mm)
            fl_out.append(mm)
            sources.append("shape")
            continue
        mt_out.append(primary_mt)
        fl_out.append(primary_fl)
        sources.append("primary")

    info = {
        "n_meta": sources.count("meta"),
        "n_ocr": sources.count("ocr"),
        "n_cluster": sources.count("cluster"),
        "n_prefix": sources.count("cluster_prefix"),
        "n_shape": sources.count("shape"),
        "n_primary": sources.count("primary"),
        "primary_mt": primary_mt,
        "primary_fl": primary_fl,
        "clusters": sorted(cluster_mt.keys()),
        "sources": sources,
    }
    return pd.Series(mt_out), pd.Series(fl_out), info
