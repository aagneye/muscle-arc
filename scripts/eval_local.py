#!/usr/bin/env python3
"""Local replica of the public score: official UMUD metric against manual test measurements.

The metric is published (host notebook paulritsche/umud-score, implemented in
geometry/umud_metric.py); the hidden answer key is replaced by our own
host-protocol measurements (data/labeling/labels.json, declared external data).
Prints per-candidate UMUD, per-parameter MAE and bias, and a bootstrap 95 %
interval, so candidates are ranked before any Kaggle submission is spent.

  python scripts/eval_local.py --pred submissions/cand_k10.csv submissions/cand_k5.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from muscle_arc.geometry.manual import summarize_record
from muscle_arc.geometry.umud_metric import umud_from_errors, umud_score

PARAMS = ("pa_deg", "fl_mm", "mt_mm")


def load_manual_labels(labels_json: Path, manifest_json: Path) -> pd.DataFrame:
    """One row per measured image (skipped and blank records excluded)."""
    labels = json.loads(Path(labels_json).read_text())
    mm = {m["image_id"]: m["mm_per_pixel"] for m in json.loads(Path(manifest_json).read_text())["images"]}
    rows = []
    for image_id, rec in labels.items():
        out = summarize_record(rec, mm[image_id])
        if out is None or (out["n_mt"] + out["n_fl"] + out["n_pa"]) == 0:
            continue
        rows.append({"image_id": image_id, **out, "seconds": rec.get("seconds")})
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def score(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Official UMUD on the images present in ``gt`` (NaN labels drop that parameter only)."""
    m = gt.merge(pred, on="image_id", how="left", suffixes=("_gt", ""))
    if m[[f"{p}" for p in PARAMS]].isna().any().any():
        missing = m.loc[m[list(PARAMS)].isna().any(axis=1), "image_id"].tolist()[:5]
        raise ValueError(f"prediction missing for labelled images, e.g. {missing}")
    errs = {p: (m[p] - m[f"{p}_gt"]).abs().to_numpy() for p in PARAMS}
    ok = {p: np.isfinite(errs[p]) for p in PARAMS}
    out = umud_from_errors(errs["pa_deg"][ok["pa_deg"]], errs["fl_mm"][ok["fl_mm"]], errs["mt_mm"][ok["mt_mm"]])
    out["n"] = len(m)
    for p in PARAMS:
        out[f"{p}_bias"] = float(np.nanmean(m[p] - m[f"{p}_gt"]))
    return out


def bootstrap_ci(pred: pd.DataFrame, gt: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    m = gt.merge(pred, on="image_id", suffixes=("_gt", ""))
    e = {p: (m[p] - m[f"{p}_gt"]).abs().to_numpy() for p in PARAMS}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(m), len(m))
        vals.append(umud_score(*(np.nanmean(e[p][idx]) for p in PARAMS)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("data/labeling/labels.json"))
    ap.add_argument("--manifest", type=Path, default=Path("data/labeling/manifest.json"))
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    gt = load_manual_labels(args.labels, args.manifest)
    print(f"labelled images: {len(gt)}  (median {gt['seconds'].median():.0f} s per image)")
    rows = []
    for p in args.pred:
        s = score(pd.read_csv(p), gt)
        lo, hi = bootstrap_ci(pd.read_csv(p), gt)
        rows.append({"candidate": p.name, "umud": s["umud"], "ci95": f"{lo:.3f}-{hi:.3f}",
                     "pa_mae": s["pa_mae"], "fl_mae": s["fl_mae"], "mt_mae": s["mt_mae"],
                     "pa_bias": s["pa_deg_bias"], "fl_bias": s["fl_mm_bias"], "mt_bias": s["mt_mm_bias"]})
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(rows).sort_values("umud").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
