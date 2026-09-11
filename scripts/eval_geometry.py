#!/usr/bin/env python3
"""Scale-free local validator for geometry (GT masks).

Reports PA distribution and FL/MT finiteness. Geometry-only mode feeds GT masks
to isolate measurement bugs from segmentation quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from muscle_arc.data.dataset import pair_images_masks, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry.metrics import (
    fascicle_length_px,
    muscle_thickness_px,
    pennation_angle_deg,
    reset_stats,
    STATS,
)


def _read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if shape is not None and m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _geom(apo: np.ndarray, fasc: np.ndarray, gray: np.ndarray | None = None) -> tuple[float, float, float]:
    try:
        mt = muscle_thickness_px(apo)
        pa = pennation_angle_deg(fasc, apo, gray=gray)
        fl = fascicle_length_px(
            fasc,
            apo,
            gray=gray,
            pa_deg=pa if np.isfinite(pa) else None,
            mt_px=mt if np.isfinite(mt) else None,
        )
        return (
            float(pa) if np.isfinite(pa) else float("nan"),
            float(fl) if np.isfinite(fl) else float("nan"),
            float(mt) if np.isfinite(mt) else float("nan"),
        )
    except Exception as exc:  # noqa: BLE001 — keep validator running
        print(f"WARN geom failed: {type(exc).__name__}: {exc}")
        return float("nan"), float("nan"), float("nan")


def _summarize(name: str, errs: list[float], relative: bool = False) -> None:
    arr = np.asarray([e for e in errs if np.isfinite(e)], dtype=np.float64)
    if len(arr) == 0:
        print(f"{name}: no finite errors")
        return
    unit = "%" if relative else "deg"
    print(
        f"{name}: n={len(arr)} mean={arr.mean():.3f}{unit} "
        f"median={np.median(arr):.3f}{unit} p90={np.percentile(arr, 90):.3f}{unit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_train_present()
    reset_stats()

    fasc_pairs = pair_images_masks(paths.fasc_imgs, paths.fasc_masks)
    apo_pairs = pair_images_masks(paths.apo_imgs, paths.apo_masks)
    apo_by_stem = {p[0].stem: p for p in apo_pairs}

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(fasc_pairs))
    rng.shuffle(idxs)
    idxs = idxs[: args.max_samples]

    pa_vals: list[float] = []
    fl_vals: list[float] = []
    mt_vals: list[float] = []

    print(f"Geometry-only on GT fasc masks (n={len(idxs)})")
    for i in idxs:
        img_p, fasc_p = fasc_pairs[int(i)]
        gray = read_gray(img_p)
        fasc = _read_mask(fasc_p, gray.shape[:2])
        apo = np.zeros_like(fasc)
        if img_p.stem in apo_by_stem:
            apo = _read_mask(apo_by_stem[img_p.stem][1], gray.shape[:2])
        pa, fl, mt = _geom(apo, fasc, gray=gray)
        pa_vals.append(pa)
        fl_vals.append(fl)
        mt_vals.append(mt)

    pa_arr = np.asarray(pa_vals, dtype=np.float64)
    finite_pa = pa_arr[np.isfinite(pa_arr)]
    if len(finite_pa) == 0:
        print("GT PA: no finite values")
    else:
        print(
            f"GT PA: n={len(finite_pa)} mean={finite_pa.mean():.2f} "
            f"median={np.median(finite_pa):.2f} "
            f"[{np.percentile(finite_pa, 10):.1f}, {np.percentile(finite_pa, 90):.1f}]"
        )
    fl_arr = np.asarray(fl_vals, dtype=np.float64)
    mt_arr = np.asarray(mt_vals, dtype=np.float64)
    print(
        f"GT FL_px finite={np.isfinite(fl_arr).sum()}/{len(fl_arr)} "
        f"median={np.nanmedian(fl_arr):.1f} "
        f"p90={np.nanpercentile(fl_arr, 90) if np.isfinite(fl_arr).any() else float('nan'):.1f}"
    )
    print(
        f"GT MT_px finite={np.isfinite(mt_arr).sum()}/{len(mt_arr)} "
        f"median={np.nanmedian(mt_arr):.1f}"
    )

    ratios = []
    for pa, fl, mt in zip(pa_vals, fl_vals, mt_vals):
        if not (np.isfinite(pa) and np.isfinite(fl) and np.isfinite(mt)):
            continue
        if abs(pa) < 2:
            continue
        expected = mt / max(np.sin(np.radians(abs(pa))), 1e-3)
        ratios.append(abs(fl - expected) / max(expected, 1e-3) * 100.0)
    _summarize("FL vs MT/sin(PA) relative", ratios, relative=True)

    common = sorted(set(p[0].stem for p in fasc_pairs) & set(apo_by_stem.keys()))
    print(f"\nMatched apo+fasc stems: {len(common)}")
    matched_pa_median = float("nan")
    if common:
        sample = list(common)
        rng.shuffle(sample)
        sample = sample[: min(len(sample), args.max_samples)]
        pa2, fl2, mt2 = [], [], []
        for stem in sample:
            fasc_img = next(p[0] for p in fasc_pairs if p[0].stem == stem)
            fasc_m = next(p[1] for p in fasc_pairs if p[0].stem == stem)
            apo_m = apo_by_stem[stem][1]
            gray = read_gray(fasc_img)
            fasc = _read_mask(fasc_m, gray.shape[:2])
            apo = _read_mask(apo_m, gray.shape[:2])
            pa, fl, mt = _geom(apo, fasc, gray=gray)
            pa2.append(pa)
            fl2.append(fl)
            mt2.append(mt)
        pa2a = np.asarray(pa2, dtype=np.float64)
        matched_pa_median = float(np.nanmedian(pa2a))
        print(
            f"Matched GT PA median={matched_pa_median:.2f} "
            f"mean={np.nanmean(pa2a):.2f} finite={np.isfinite(pa2a).mean()*100:.0f}%"
        )
        print(
            f"Matched GT FL_px median={np.nanmedian(fl2):.1f} "
            f"MT_px median={np.nanmedian(mt2):.1f} "
            f"FL finite={np.isfinite(fl2).mean()*100:.0f}%"
        )

    print(STATS.summary())

    # Prefer matched-apo median when available; else fasc-only PA
    check_med = matched_pa_median if np.isfinite(matched_pa_median) else (
        float(np.median(finite_pa)) if len(finite_pa) else float("nan")
    )
    ok = np.isfinite(check_med) and 8.0 <= check_med <= 25.0
    print("\nPASS" if ok else "\nCHECK: PA median outside 8-25 deg — geometry may still be wrong")
    print(f"gate_median_pa={check_med:.2f}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
