#!/usr/bin/env python3
"""Render what the geometry sees on cached test predictions, for visual failure audits.

Per image: chrome-masked frame, apo (green) and fascicle (red) probabilities, the fitted
superficial / deep aponeurosis lines, every fascicle line the FL / PA code uses with its
extrapolation between the two aponeuroses, and the competing estimates (candidate value,
contour FL, trig FL, least-extrapolation FL, PA with and without Radon, both MT
conventions). Images are picked per device layout where the FL estimators disagree
most, plus one typical image, or given explicitly with --ids.

Outputs go to experiments/ (git-ignored: test images are competition data).

  python scripts/overlay_audit.py --tag k10 --per-layout 3 --out-dir experiments/audit_k10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry import metrics as geom
from muscle_arc.geometry.metrics import _apo_bands, _fascicle_lines, _fit_apo_lines, _line_line_intersection
from muscle_arc.geometry.sector_crop import mask_chrome


def _binarize(apo_p: np.ndarray, fasc_p: np.ndarray, apo_thr: float, fasc_thr: float):
    """Same binarization as scripts/geometry_sweep.py."""
    apo = (apo_p > apo_thr).astype(np.uint8)
    fasc = (fasc_p > fasc_thr).astype(np.uint8)
    if int(fasc.sum()) < 5000:
        fasc = cv2.morphologyEx(fasc, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if int(apo.sum()) > 5000:
        apo = cv2.medianBlur(apo, 5)
    return apo, fasc


def _full_line(p: np.ndarray, d: np.ndarray, w: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if abs(d[0]) < 1e-6:
        return None
    t0, t1 = (0 - p[0]) / d[0], (w - 1 - p[0]) / d[0]
    a, b = p + t0 * d, p + t1 * d
    return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))


def render(image_id: str, img_path: Path, npz: Path, info: dict, apo_thr: float, fasc_thr: float,
           max_w: int = 1400) -> np.ndarray:
    d = np.load(npz)
    apo_p, fasc_p = d["apo"].astype(np.float32), d["fasc"].astype(np.float32)
    gray, _ = mask_chrome(read_gray(img_path))
    if gray.shape != apo_p.shape:
        gray = cv2.resize(gray, (apo_p.shape[1], apo_p.shape[0]))
    h, w = gray.shape
    apo, fasc = _binarize(apo_p, fasc_p, apo_thr, fasc_thr)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    vis[..., 1] += 110 * apo_p
    vis[..., 2] += 140 * fasc_p
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    th = max(1, w // 600)

    bands = _apo_bands(apo, fasc_mask=fasc, mm_per_pixel=float(d["mm_per_pixel"]))
    if bands is not None:  # component pair used for the PA tangent: blue = superficial, orange = deep
        for band, col in zip(bands, ((255, 120, 0), (0, 140, 255))):
            cnts, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, col, th + 1)

    lines = _fit_apo_lines(apo, fasc_mask=fasc, mm_per_pixel=float(d["mm_per_pixel"]))
    if lines is not None:
        for (p, dd), col in zip(lines, ((255, 255, 0), (0, 255, 255))):
            seg = _full_line(p, dd, w)
            if seg:
                cv2.line(vis, *seg, col, th)
        (ps, ds), (pd_, dd_) = lines
        for point, direction, length in _fascicle_lines(fasc, fasc_prob=fasc_p, gray=gray, apo_mask=apo):
            a = _line_line_intersection(point, direction, ps, ds)
            b = _line_line_intersection(point, direction, pd_, dd_)
            if a is None or b is None:
                continue
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 0, 255), max(1, th // 2))
            half = 0.5 * length * direction
            cv2.line(vis, tuple(int(v) for v in point - half), tuple(int(v) for v in point + half), (255, 255, 255), th + 1)

    lines_txt = [
        f"{image_id}  {info.get('layout', '')}  mm/px {float(d['mm_per_pixel']):.4f}",
        f"cand   PA {info.get('pa_deg', np.nan):5.1f}  FL {info.get('fl_mm', np.nan):5.1f}  MT {info.get('mt_mm', np.nan):5.1f}",
        f"FL legacy {info.get('fl_legacy', np.nan):5.1f}  contour {info.get('fl_contour', np.nan):5.1f}  "
        f"trig {info.get('fl_trig', np.nan):5.1f}  le5 {info.get('fl_le5', np.nan):5.1f}",
        f"PA legacy {info.get('pa_legacy', np.nan):5.1f}  noradon {info.get('pa_noradon', np.nan):5.1f}   "
        f"MT legacy {info.get('mt_legacy', np.nan):5.1f}  host {info.get('mt_host', np.nan):5.1f}",
    ]
    scale = max_w / w if w > max_w else 1.0
    if scale != 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pad = np.zeros((26 * len(lines_txt) + 8, vis.shape[1], 3), np.uint8)
    for i, t in enumerate(lines_txt):
        cv2.putText(pad, t, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([pad, vis])


def pick(measures: pd.DataFrame, per_layout: int) -> pd.DataFrame:
    """Per layout: the images with the largest |contour - trig| FL disagreement, plus the median one."""
    m = measures.copy()
    m["disagree"] = (m["fl_contour"] - m["fl_trig"]).abs()
    out = []
    for _, g in m.dropna(subset=["disagree"]).groupby("layout"):
        g = g.sort_values("disagree", ascending=False)
        out.append(g.head(per_layout))
        out.append(g.iloc[[len(g) // 2]])
    return pd.concat(out).drop_duplicates("image_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/dual_v3.yaml"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cand", type=Path, default=None, help="candidate CSV whose values are printed (default cand_<tag>.csv)")
    ap.add_argument("--table", type=Path, default=Path("experiments/depth_scale_table_exact.csv"))
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--per-layout", type=int, default=3)
    ap.add_argument("--apo-thr", type=float, default=0.35)
    ap.add_argument("--fasc-thr", type=float, default=0.10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--apo-pairing", choices=("legacy", "upper_dense"), default="legacy")
    ap.add_argument("--measures-tag", default=None, help="sweep output tag to print (default: --tag)")
    args = ap.parse_args()
    geom.APO_PAIRING = args.apo_pairing

    meas = pd.read_csv(f"experiments/geometry_sweep_{args.measures_tag or args.tag}_test_measures.csv")
    layout = pd.read_csv(args.table).set_index("image_id")["source"]
    meas["layout"] = meas["image_id"].map(layout)
    meas["fl_trig"] = meas["mt_legacy"] / np.sin(np.radians(meas["pa_legacy"] + 2.0))
    cand = pd.read_csv(args.cand or Path(f"submissions/cand_{args.tag}.csv"))
    meas = meas.merge(cand, on="image_id", how="left")
    chosen = meas[meas["image_id"].isin(args.ids)] if args.ids else pick(meas, args.per_layout)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    paths = {p.name: p for p in list_images(DataPaths.from_config(cfg["data"]).test_images)}
    cache = Path("experiments/prob_cache") / args.tag / "test"
    for r in chosen.itertuples():
        img = render(r.image_id, paths[r.image_id], cache / f"{r.image_id}.npz", r._asdict(), args.apo_thr, args.fasc_thr)
        name = f"{r.layout}__{Path(r.image_id).stem}.jpg"
        cv2.imwrite(str(args.out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cols = ["image_id", "layout", "pa_deg", "fl_mm", "mt_mm", "fl_contour", "fl_trig", "fl_le5", "pa_noradon"]
    with pd.option_context("display.width", 220):
        print(chosen[cols + (["disagree"] if "disagree" in chosen else [])].round(1).to_string(index=False))
    print(f"wrote {len(chosen)} overlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
