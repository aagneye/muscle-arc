#!/usr/bin/env python3
"""Local UMUD-style MAE gate on sample_submission (+ optional external CSV).

Reports mean absolute error for PA / FL / MT so we only Kaggle-submit when local
metrics improve vs a stored baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from muscle_arc.geometry.umud_metric import umud_score
from muscle_arc.infer.predict import load_sample_submission


def mae(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(a[m] - b[m])))


def match_pred(pred: pd.DataFrame, rid: str) -> pd.Series | None:
    stem = Path(rid).stem
    m = pred[pred["image_id"] == rid]
    if m.empty:
        m = pred[pred["stem"] == stem]
    if m.empty:
        return None
    return m.iloc[0]


def collect_rows(pred: pd.DataFrame, gt: pd.DataFrame, id_col: str) -> list[dict]:
    rows = []
    for _, s in gt.iterrows():
        rid = str(s[id_col])
        m = match_pred(pred, rid)
        if m is None:
            continue
        pa_gt, fl_gt, mt_gt = float(s["pa_deg"]), float(s["fl_mm"]), float(s["mt_mm"])
        # Skip degenerate stub rows (all-zero / non-positive lengths)
        if not (np.isfinite(fl_gt) and np.isfinite(mt_gt) and fl_gt > 0 and mt_gt > 0):
            continue
        rows.append(
            {
                "id": rid,
                "pa_gt": pa_gt,
                "fl_gt": fl_gt,
                "mt_gt": mt_gt,
                "pa_pr": float(m["pa_deg"]),
                "fl_pr": float(m["fl_mm"]),
                "mt_pr": float(m["mt_mm"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--pred", type=Path, required=True, help="Prediction CSV")
    parser.add_argument("--baseline", type=Path, default=Path("experiments/local_mae_baseline.json"))
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--external-gt", type=Path, default=None, help="Optional external GT CSV")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    pred = pd.read_csv(args.pred)
    pred["stem"] = pred["image_id"].map(lambda x: Path(str(x)).stem)

    rows: list[dict] = []
    sample_path = Path(cfg["data"]["root"]) / cfg["data"]["sample_submission"]
    if sample_path.exists():
        sample = load_sample_submission(sample_path, sep=cfg["data"].get("csv_sep", ";"))
        id_col = [c for c in sample.columns if "id" in c.lower()][0]
        rows.extend(collect_rows(pred, sample, id_col))

    if args.external_gt and args.external_gt.exists():
        ext = pd.read_csv(args.external_gt)
        id_col = "image_id" if "image_id" in ext.columns else ("id" if "id" in ext.columns else "stem")
        before = len(rows)
        rows.extend(collect_rows(pred, ext, id_col))
        print(f"external_gt added {len(rows) - before} matched rows from {args.external_gt}")

    if not rows:
        print("NO_GT_ROWS")
        raise SystemExit(2)

    df = pd.DataFrame(rows)
    metrics = {
        "n": int(len(df)),
        "pa_mae": mae(df["pa_pr"].to_numpy(), df["pa_gt"].to_numpy()),
        "fl_mae": mae(df["fl_pr"].to_numpy(), df["fl_gt"].to_numpy()),
        "mt_mae": mae(df["mt_pr"].to_numpy(), df["mt_gt"].to_numpy()),
    }
    # Official UMUD score: S = (1/3)(MAE_PA/6 + MAE_FL/12 + MAE_MT/3)
    metrics["umud"] = umud_score(metrics["pa_mae"], metrics["fl_mae"], metrics["mt_mae"])
    metrics["combo"] = metrics["umud"]
    print(json.dumps(metrics, indent=2))
    for _, r in df.iterrows():
        print(
            f"  {r['id']}: PA {r['pa_pr']:.2f}/{r['pa_gt']:.2f} "
            f"FL {r['fl_pr']:.2f}/{r['fl_gt']:.2f} MT {r['mt_pr']:.2f}/{r['mt_gt']:.2f}"
        )

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    if args.save_baseline or not args.baseline.exists():
        args.baseline.write_text(json.dumps(metrics, indent=2))
        print(f"saved baseline -> {args.baseline}")
        raise SystemExit(0)

    base = json.loads(args.baseline.read_text())
    improved = metrics["combo"] < float(base.get("combo", 1e9)) - 1e-6
    print(f"baseline_combo={base.get('combo')} current_combo={metrics['combo']} improved={improved}")
    raise SystemExit(0 if improved else 1)


if __name__ == "__main__":
    main()
