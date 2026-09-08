#!/usr/bin/env python3
"""Average probability-map ensembles are preferred; this averages PA/FL/MT safely.

For true SOF ensembles, average apo/fasc probs before geometry in calibrate_predict.
This helper mean-averages submission CSVs as a last-resort ensemble (not preferred).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cands", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["mean", "median"],
        default="mean",
        help="Average final measurements (prefer prob-level ensemble upstream)",
    )
    args = parser.parse_args()
    dfs = [pd.read_csv(p).sort_values("image_id").reset_index(drop=True) for p in args.cands]
    base = dfs[0][["image_id"]].copy()
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        stack = pd.concat([d[col] for d in dfs], axis=1)
        base[col] = stack.mean(axis=1) if args.mode == "mean" else stack.median(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out, index=False)
    print(f"Wrote {args.out} from {len(dfs)} cands mode={args.mode}")


if __name__ == "__main__":
    main()
