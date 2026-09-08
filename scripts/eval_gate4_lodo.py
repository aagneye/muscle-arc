#!/usr/bin/env python3
"""Leave-one-device-out (or leave-one-row-group) stability check on OSF.

Blocks submit if a candidate blend/gain would be needed per fold — we only
report UMUD variance across held-out groups (no fitting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from muscle_arc.geometry.umud_metric import umud_from_errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate2-csv",
        type=Path,
        default=Path("experiments/gate2_osf_umud.csv"),
        help="Per-image Gate2 predictions with pa_err/fl_err/mt columns or errs",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/gate4_lodo.json"))
    parser.add_argument("--max-umud-std", type=float, default=0.15)
    args = parser.parse_args()

    if not args.gate2_csv.exists():
        print(f"missing {args.gate2_csv}")
        raise SystemExit(1)
    df = pd.read_csv(args.gate2_csv)
    # Group by filename prefix / device hint if present; else chunks of 5
    if "device" in df.columns:
        keys = df["device"].astype(str)
    else:
        keys = df["image_id"].astype(str).map(lambda s: s.split("_")[0] if "_" in s else s[:6])
    fold_scores = []
    for k, g in df.groupby(keys):
        if len(g) < 2:
            continue
        pa = g["pa_err"].to_numpy(dtype=float) if "pa_err" in g else np.abs(g["pa_deg"] - g["pa_gt"])
        fl = (
            np.abs(g["fl_mm"] - g["fl_gt"]).to_numpy(dtype=float)
            if "fl_gt" in g
            else g.get("fl_err", pd.Series(dtype=float)).to_numpy(dtype=float)
        )
        mt = (
            np.abs(g["mt_mm"] - g["mt_gt"]).to_numpy(dtype=float)
            if "mt_gt" in g
            else g.get("mt_err", pd.Series(dtype=float)).to_numpy(dtype=float)
        )
        m = umud_from_errors(pa, fl, mt)
        fold_scores.append({"fold": str(k), "n": m["n"], "umud": m["umud"]})
    umuds = [f["umud"] for f in fold_scores if np.isfinite(f["umud"])]
    info = {
        "folds": fold_scores,
        "umud_mean": float(np.mean(umuds)) if umuds else float("nan"),
        "umud_std": float(np.std(umuds)) if umuds else float("nan"),
        "gate4_lodo_ok": bool(umuds and float(np.std(umuds)) <= args.max_umud_std),
    }
    print(json.dumps(info, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2))
    print("GATE4_LODO_OK" if info["gate4_lodo_ok"] else "GATE4_LODO_FAIL")
    raise SystemExit(0 if info["gate4_lodo_ok"] else 1)


if __name__ == "__main__":
    main()
