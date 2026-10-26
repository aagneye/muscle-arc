#!/usr/bin/env python3
"""Sweep geometry conventions on cached probabilities (OSF expert GT + test agreement).

Per image, every candidate measurement is computed once from the probability
cache (scripts/cache_probs.py). Because the UMUD score is a sum of independent
per-parameter terms, each parameter's variants are scored separately:

  OSF : MAE against the six-rater expert benchmark (GT scale) -> the decision metric
  test: MAE against the public-anchor CSV (agreement proxy only, never a target)

Writes experiments/geometry_sweep_<tag>_{measures,scores}.csv and, with
--write-candidate, a submission CSV built from the chosen variant per parameter.
"""

from __future__ import annotations

import argparse
import importlib.util
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry import metrics as geom
from muscle_arc.geometry.calibration import apply_calibration, fit_linear_calibration, loo_gain
from muscle_arc.geometry.metrics import (
    fascicle_length_least_extrap_px,
    fascicle_length_px,
    muscle_thickness_px,
    pennation_angle_deg,
)
from muscle_arc.geometry.protocol import mt_host_protocol
from muscle_arc.geometry.sector_crop import mask_chrome
from muscle_arc.geometry.surfaces import extract_apo_surfaces

PA_OFFSETS = (0.0, 1.0, 2.0, 3.0)
# fl_legacy already holds 0.5 trig (raw PA) + 0.5 contour inside fascicle_length_px;
# fl_contour is the same route with the internal trig weight set to 0.
TRIG_W = (0.0, 0.25, 0.5, 0.75)
FL_BASES = ("fl_legacy", "fl_contour", "fl_le3", "fl_le5")
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
CLIP = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def _binarize(apo_p: np.ndarray, fasc_p: np.ndarray, apo_thr: float, fasc_thr: float):
    apo = (apo_p > apo_thr).astype(np.uint8)
    fasc = (fasc_p > fasc_thr).astype(np.uint8)
    if int(fasc.sum()) < 5000:
        fasc = cv2.morphologyEx(fasc, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if int(apo.sum()) > 5000:
        apo = cv2.medianBlur(apo, 5)
    return apo, fasc


def measure_one(job: tuple[str, str, str, float, float]) -> dict:
    image_id, npz_path, gray_path, apo_thr, fasc_thr = job
    d = np.load(npz_path)
    apo_p, fasc_p = d["apo"].astype(np.float32), d["fasc"].astype(np.float32)
    mm = float(d["mm_per_pixel"])
    gray = read_gray(Path(gray_path))
    apo, fasc = _binarize(apo_p, fasc_p, apo_thr, fasc_thr)
    if gray.shape != apo.shape:
        gray = cv2.resize(gray, (apo.shape[1], apo.shape[0]))
    nan = float("nan")
    mt_leg = muscle_thickness_px(apo, fasc_mask=fasc, mm_per_pixel=mm)
    surf = extract_apo_surfaces(apo, fasc_mask=fasc, degree=2)
    mt_host = mt_host_protocol(surf) if surf is not None else nan
    pa_leg = pennation_angle_deg(fasc, apo, fasc_prob=fasc_p, gray=gray, mm_per_pixel=mm)
    pa_nor = pennation_angle_deg(fasc, apo, fasc_prob=fasc_p, gray=None, mm_per_pixel=mm)
    fl_kw = dict(fasc_prob=fasc_p, gray=gray, pa_deg=pa_leg if np.isfinite(pa_leg) else None,
                 mt_px=mt_leg if np.isfinite(mt_leg) else None, mm_per_pixel=mm)
    fl_leg = fascicle_length_px(fasc, apo, **fl_kw)
    blend = geom.FL_BLEND_TRIG
    try:
        geom.FL_BLEND_TRIG = 0.0  # contour route only (intersection + edge fits)
        fl_con = fascicle_length_px(fasc, apo, **fl_kw)
    finally:
        geom.FL_BLEND_TRIG = blend
    fl_le3 = fascicle_length_least_extrap_px(fasc, apo, fasc_prob=fasc_p, top_k=3, mm_per_pixel=mm)
    fl_le5 = fascicle_length_least_extrap_px(fasc, apo, fasc_prob=fasc_p, top_k=5, mm_per_pixel=mm)
    px = lambda v: float(v) * mm if np.isfinite(v) else nan  # noqa: E731
    return {
        "image_id": image_id, "mm_per_pixel": mm,
        "mt_legacy": px(mt_leg), "mt_host": px(mt_host),
        "pa_legacy": float(pa_leg), "pa_noradon": float(pa_nor),
        "fl_legacy": px(fl_leg), "fl_contour": px(fl_con), "fl_le3": px(fl_le3), "fl_le5": px(fl_le5),
    }


def _fill(s: pd.Series, groups: list[list[int]] | None) -> pd.Series:
    s = s.copy()
    if groups:
        for g in groups:
            vals = s.iloc[g].dropna()
            if len(vals):
                s.iloc[g] = s.iloc[g].fillna(float(vals.median()))
    return s.fillna(float(s.median()))


def build_variants(m: pd.DataFrame, groups: list[list[int]] | None) -> dict[str, dict[str, pd.Series]]:
    """Return {param: {variant_name: series}} with NaNs filled and host clip applied."""
    out: dict[str, dict[str, pd.Series]] = {"pa_deg": {}, "fl_mm": {}, "mt_mm": {}}
    for mt_name in ("mt_legacy", "mt_host"):
        out["mt_mm"][mt_name] = _fill(m[mt_name], groups).clip(*CLIP["mt_mm"])
    for pa_name in ("pa_legacy", "pa_noradon"):
        base = _fill(m[pa_name], groups)
        for off in PA_OFFSETS:
            out["pa_deg"][f"{pa_name}+{off:g}"] = (base + off).clip(*CLIP["pa_deg"])
    mt_ref = out["mt_mm"]["mt_legacy"]
    pa_ref = out["pa_deg"]["pa_legacy+2"]
    trig = (mt_ref / np.sin(np.radians(pa_ref))).clip(*CLIP["fl_mm"])
    out["fl_mm"]["fl_trig"] = trig  # pure MT / sin(PA + 2)
    out["fl_mm"]["fl_trig_raw"] = (mt_ref / np.sin(np.radians(out["pa_deg"]["pa_legacy+0"]))).clip(*CLIP["fl_mm"])
    for fl_name in FL_BASES:
        if fl_name not in m.columns:  # measures written before fl_contour existed
            continue
        base = _fill(m[fl_name], groups)
        for w in TRIG_W:
            out["fl_mm"][f"{fl_name}|trig{w:g}"] = ((1 - w) * base + w * trig).clip(*CLIP["fl_mm"])
    return out


def smooth_groups(s: pd.Series, groups: list[list[int]], w: float = 0.5) -> pd.Series:
    """Pull each clip frame toward its clip median (w = 1 is the plain clip median)."""
    s = s.copy()
    for g in groups:
        med = float(s.iloc[g].median())
        s.iloc[g] = (1 - w) * s.iloc[g] + w * med
    return s


def score_table(variants, ref: pd.DataFrame, groups=None) -> pd.DataFrame:
    rows = []
    for param, vs in variants.items():
        for name, series in vs.items():
            pred = smooth_groups(series, groups) if groups else series
            err = (pred.to_numpy() - ref[param].to_numpy())
            mae = float(np.nanmean(np.abs(err)))
            rows.append({"param": param, "variant": name, "mae": mae,
                         "bias": float(np.nanmean(err)), "norm": mae / TOL[param]})
    return pd.DataFrame(rows).sort_values(["param", "norm"])


def _groups_for(paths: list[Path]) -> list[list[int]]:
    spec = importlib.util.spec_from_file_location("cal", "scripts/calibrate_predict.py")
    cal = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cal)
    return [g for g in cal.sequence_groups(paths) if len(g) > 1]


def _set_pairing(mode: str) -> None:
    """Pool initializer: the pairing switch must be set inside every worker."""
    geom.APO_PAIRING = mode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/dual_v3.yaml"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("experiments/prob_cache"))
    ap.add_argument("--osf-gt", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    ap.add_argument("--anchor", type=Path, default=Path("experiments/vera_raw.csv"))
    ap.add_argument("--apo-thr", type=float, default=0.35)
    ap.add_argument("--fasc-thr", type=float, default=0.10)
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--pick", nargs=3, metavar=("PA", "FL", "MT"), default=None,
                    help="variant names to write as a candidate (default: best on OSF)")
    ap.add_argument("--write-candidate", type=Path, default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="apply OSF-fitted linear calibration per parameter when LOO helps")
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--smooth-w", type=float, default=0.5,
                    help="clip smoothing weight toward the clip median (1 = clip median)")
    ap.add_argument("--apo-pairing", choices=("legacy", "upper_dense"), default="legacy",
                    help="aponeurosis pair selection (geometry/metrics.py APO_PAIRING)")
    ap.add_argument("--out-tag", default=None, help="name for the output files (default: --tag)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cache = args.cache_root / args.tag
    osf = pd.read_csv(args.osf_gt)
    osf = osf[[Path(str(p)).exists() for p in osf["path"]]].reset_index(drop=True)
    test_paths = list_images(DataPaths.from_config(cfg["data"]).test_images)

    jobs = [(str(r.image_id), str(cache / "osf" / f"{r.image_id}.npz"), str(r.path),
             args.apo_thr, args.fasc_thr) for r in osf.itertuples()]
    tjobs = [(p.name, str(cache / "test" / f"{p.name}.npz"), str(p), args.apo_thr, args.fasc_thr)
             for p in test_paths]
    with Pool(args.procs, initializer=_set_pairing, initargs=(args.apo_pairing,)) as pool:
        mo = pd.DataFrame(pool.map(measure_one, jobs))
        mt = pd.DataFrame(pool.map(_measure_test, tjobs))
    stem = f"experiments/geometry_sweep_{args.out_tag or args.tag}"
    mo.to_csv(f"{stem}_osf_measures.csv", index=False)
    mt.to_csv(f"{stem}_test_measures.csv", index=False)

    groups = _groups_for(test_paths)
    vo = build_variants(mo, None)
    vt = build_variants(mt, groups)
    so = score_table(vo, osf)
    anchor = pd.read_csv(args.anchor).set_index("image_id").reindex(mt["image_id"]).reset_index()
    st = score_table(vt, anchor, groups)
    scores = so.merge(st, on=["param", "variant"], suffixes=("_osf", "_test"))
    scores.to_csv(f"{stem}_scores.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(scores.round(3).to_string(index=False))

    best = {p: scores[scores.param == p].sort_values("norm_osf").iloc[0]["variant"] for p in TOL}
    pick = dict(zip(("pa_deg", "fl_mm", "mt_mm"), args.pick)) if args.pick else best
    osf_umud = sum(scores[(scores.param == p) & (scores.variant == v)]["norm_osf"].iloc[0]
                   for p, v in pick.items()) / 3
    test_agree = sum(scores[(scores.param == p) & (scores.variant == v)]["norm_test"].iloc[0]
                     for p, v in pick.items()) / 3
    print(f"PICK {pick}  OSF UMUD={osf_umud:.3f}  test-agreement={test_agree:.3f}")

    calib: dict[str, tuple[float, float]] = {}
    if args.calibrate:
        for p, v in pick.items():
            pred, gt = vo[p][v].to_numpy(), osf[p].to_numpy()
            gain_loo = loo_gain(pred, gt, shrink=args.shrink)
            g, o = fit_linear_calibration(pred, gt, shrink=args.shrink)
            use = gain_loo < 0
            print(f"CALIB {p}: gain={g:.3f} offset={o:+.3f} loo_delta_mae={gain_loo:+.3f} -> {'APPLY' if use else 'skip'}")
            if use:
                calib[p] = (g, o)

    if args.write_candidate:
        sub = pd.DataFrame({"image_id": mt["image_id"]})
        for p, v in pick.items():
            series = vt[p][v]
            if p in calib:
                series = pd.Series(apply_calibration(series.to_numpy(), *calib[p])).clip(*CLIP[p])
            sub[p] = smooth_groups(series, groups, w=args.smooth_w).round(4)
        sub[["image_id", "pa_deg", "fl_mm", "mt_mm"]].to_csv(args.write_candidate, index=False)
        print(f"wrote {args.write_candidate} ({len(sub)} rows)")


def _measure_test(job: tuple[str, str, str, float, float]) -> dict:
    """Test images: the cache was computed on the chrome-masked frame; measure on the same."""
    image_id, npz_path, img_path, apo_thr, fasc_thr = job
    d = np.load(npz_path)
    gray, _ = mask_chrome(read_gray(Path(img_path)))
    tmp = Path(npz_path).with_suffix(".gray.png")
    if not tmp.exists():
        cv2.imwrite(str(tmp), gray)
    return measure_one((image_id, npz_path, str(tmp), apo_thr, fasc_thr)) | {"h": int(d["shape"][0])}


if __name__ == "__main__":
    main()
