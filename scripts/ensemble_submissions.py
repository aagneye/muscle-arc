#!/usr/bin/env python3
"""Ensemble / pick-best columns across submissions using OSF + sample MAE."""

from __future__ import annotations

import argparse
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


def score_vs_gt(pred: pd.DataFrame, gt: pd.DataFrame, id_col: str) -> dict:
    pred = pred.copy()
    pred["stem"] = pred["image_id"].map(lambda x: Path(str(x)).stem)
    pa_e, fl_e, mt_e = [], [], []
    for _, s in gt.iterrows():
        rid = str(s[id_col])
        stem = Path(rid).stem
        m = pred[pred["image_id"] == rid]
        if m.empty:
            m = pred[pred["stem"] == stem]
        if m.empty:
            continue
        pa_e.append(abs(float(m.iloc[0]["pa_deg"]) - float(s["pa_deg"])))
        fl_e.append(abs(float(m.iloc[0]["fl_mm"]) - float(s["fl_mm"])))
        mt_e.append(abs(float(m.iloc[0]["mt_mm"]) - float(s["mt_mm"])))
    if not pa_e:
        return {"n": 0, "combo": 9e9}
    return {
        "n": len(pa_e),
        "pa_mae": float(np.mean(pa_e)),
        "fl_mae": float(np.mean(fl_e)),
        "mt_mae": float(np.mean(mt_e)),
        "combo": umud_score(float(np.mean(pa_e)), float(np.mean(fl_e)), float(np.mean(mt_e))),
        "umud": umud_score(float(np.mean(pa_e)), float(np.mean(fl_e)), float(np.mean(mt_e))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--cands",
        nargs="+",
        default=[
            "submissions/submission_v8.csv",
            "submissions/submission_v10.csv",
            "submissions/submission_v11.csv",
        ],
    )
    parser.add_argument("--external-gt", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    parser.add_argument("--out", type=Path, default=Path("submissions/submission_ensemble.csv"))
    parser.add_argument("--mode", choices=["mean", "best_combo"], default="best_combo")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    frames = []
    for c in args.cands:
        p = Path(c)
        if not p.exists():
            print(f"skip missing {p}")
            continue
        frames.append((p.name, pd.read_csv(p)))
    if not frames:
        raise SystemExit("no candidate submissions")

    sample_path = Path(cfg["data"]["root"]) / cfg["data"]["sample_submission"]
    gts = []
    if sample_path.exists():
        sample = load_sample_submission(sample_path, sep=cfg["data"].get("csv_sep", ";"))
        id_col = [c for c in sample.columns if "id" in c.lower()][0]
        gts.append(("sample", sample, id_col))
    if args.external_gt.exists():
        ext = pd.read_csv(args.external_gt)
        id_col = "image_id" if "image_id" in ext.columns else "stem"
        gts.append(("osf", ext, id_col))

    scored = []
    for name, df in frames:
        combos = []
        for gname, gt, id_col in gts:
            s = score_vs_gt(df, gt, id_col)
            print(f"{name} vs {gname}: {s}")
            if s["n"] > 0:
                combos.append(s["combo"])
        scored.append((float(np.mean(combos)) if combos else 9e9, name, df))

    scored.sort(key=lambda t: t[0])
    print("ranking:", [(n, c) for c, n, _ in scored])

    if args.mode == "best_combo":
        best = scored[0][2]
        print(f"BEST {scored[0][1]} combo={scored[0][0]:.4f}")
    else:
        # column-wise mean of available
        base = frames[0][1].sort_values("image_id").reset_index(drop=True)
        for col in ("pa_deg", "fl_mm", "mt_mm"):
            stack = []
            for _, df in frames:
                d = df.sort_values("image_id").reset_index(drop=True)
                stack.append(d[col].to_numpy(dtype=float))
            base[col] = np.mean(np.stack(stack, axis=0), axis=0)
        best = base
        print("MEAN ensemble")

    # Sanity: reject if FL std collapsed vs v8
    v8 = Path("submissions/submission_v8.csv")
    if v8.exists():
        d8 = pd.read_csv(v8)
        if float(best["fl_mm"].std()) < 0.45 * float(d8["fl_mm"].std()):
            print(
                f"WARN FL std collapsed ({best['fl_mm'].std():.2f} vs v8 {d8['fl_mm'].std():.2f}); "
                "preferring best non-collapsed candidate"
            )
            for _c, name, df in scored:
                if float(df["fl_mm"].std()) >= 0.45 * float(d8["fl_mm"].std()):
                    best = df
                    print(f"using {name}")
                    break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = best[["image_id", "pa_deg", "fl_mm", "mt_mm"]].sort_values("image_id")
    best.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(best.describe().round(3).to_string())


if __name__ == "__main__":
    main()
