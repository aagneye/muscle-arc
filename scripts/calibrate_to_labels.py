#!/usr/bin/env python3
"""Calibrate a candidate to our manual host-protocol measurements (declared external data).

Per parameter, a gain/offset map per device layout with partial pooling toward the
global map (geometry/calibration.py); applied only when its leave-one-out MAE over the
labelled images beats the uncalibrated MAE. The fitted maps are written as JSON so the
final pipeline re-applies them deterministically on any new data (nothing from the
leaderboard is used). The challenge evidence for this design is in
docs/research/2026-10-26-rater-calibration-and-fusion.md (section 2).

  python scripts/calibrate_to_labels.py --pred submissions/cand_k15.csv \
      --out submissions/cand_k15_cal.csv --params experiments/calibration_k15.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from muscle_arc.geometry.calibration import (
    GLOBAL,
    apply_pooled,
    fit_pooled_calibration,
    loo_pooled_mae,
)
from muscle_arc.geometry.umud_metric import umud_score

PARAMS = ("pa_deg", "fl_mm", "mt_mm")
CLIP = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def _eval_local():
    spec = importlib.util.spec_from_file_location("eval_local", Path(__file__).resolve().parent / "eval_local.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def calibrate(pred: pd.DataFrame, gt: pd.DataFrame, groups: pd.Series, *, shrink: float, pool: float,
              force: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """(calibrated predictions, fitted maps per parameter, LOO report).

    ``groups`` maps every image_id of ``pred`` to its device layout. A parameter's map
    is kept only if the leave-one-out MAE improves (or ``force``).
    """
    m = gt.merge(pred, on="image_id", suffixes=("_gt", ""))
    if len(m) < 5:
        raise ValueError(f"only {len(m)} labelled images with predictions; need at least 5")
    g_lab = groups.reindex(m["image_id"]).fillna("unknown").to_numpy()
    out = pred.copy()
    maps: dict[str, dict] = {}
    rows = []
    mae_before, mae_after = {}, {}
    for p in PARAMS:
        x, y = m[p].to_numpy(), m[f"{p}_gt"].to_numpy()
        before, after = loo_pooled_mae(x, y, g_lab, shrink=shrink, pool=pool)
        use = force or after < before
        mae_before[p], mae_after[p] = before, after if use else before
        params = fit_pooled_calibration(x, y, g_lab, shrink=shrink, pool=pool)
        rows.append({"param": p, "n": len(m), "mae": before, "mae_loo_cal": after, "applied": use,
                     "global_gain": params[GLOBAL][0], "global_offset": params[GLOBAL][1]})
        if use:
            maps[p] = {k: [float(v[0]), float(v[1])] for k, v in params.items()}
            g_all = groups.reindex(out["image_id"]).fillna("unknown").to_numpy()
            out[p] = np.clip(apply_pooled(out[p].to_numpy(), g_all, params), *CLIP[p]).round(4)
    report = pd.DataFrame(rows)
    report.attrs["umud_before"] = umud_score(mae_before["pa_deg"], mae_before["fl_mm"], mae_before["mt_mm"])
    report.attrs["umud_after_loo"] = umud_score(mae_after["pa_deg"], mae_after["fl_mm"], mae_after["mt_mm"])
    return out, maps, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=Path("data/labeling/labels.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/labeling/manifest.json"))
    ap.add_argument("--table", type=Path, default=Path("experiments/depth_scale_table_exact.csv"))
    ap.add_argument("--group-col", default="source", help="device-layout column of --table")
    ap.add_argument("--shrink", type=float, default=0.5, help="blend of the OLS map with the identity")
    ap.add_argument("--pool", type=float, default=10.0, help="labels a layout needs to weigh as much as the global fit")
    ap.add_argument("--force", action="store_true", help="apply even when leave-one-out does not improve")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--params", type=Path, default=None, help="write the fitted maps as JSON")
    args = ap.parse_args()

    ev = _eval_local()
    gt = ev.load_manual_labels(args.labels, args.manifest)
    pred = pd.read_csv(args.pred)
    table = pd.read_csv(args.table)
    groups = table.set_index("image_id")[args.group_col].astype(str)
    out, maps, report = calibrate(pred, gt, groups, shrink=args.shrink, pool=args.pool, force=args.force)
    with pd.option_context("display.width", 200):
        print(report.round(4).to_string(index=False))
    print(f"labelled UMUD before {report.attrs['umud_before']:.4f} -> leave-one-out after {report.attrs['umud_after_loo']:.4f}")
    out[["image_id", *PARAMS]].to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    if args.params:
        args.params.write_text(json.dumps({"source": str(args.pred), "shrink": args.shrink, "pool": args.pool,
                                           "maps": maps}, indent=2))
        print(f"wrote {args.params}")


if __name__ == "__main__":
    main()
