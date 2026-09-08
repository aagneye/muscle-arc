#!/usr/bin/env python3
"""Write a constant-value submission for leaderboard probing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from muscle_arc.data.dataset import list_images
from muscle_arc.data.paths import DataPaths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--pa", type=float, default=15.0)
    parser.add_argument("--fl", type=float, default=75.0)
    parser.add_argument("--mt", type=float, default=19.0)
    parser.add_argument("--out", type=Path, default=Path("submissions/probe_constant.csv"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_test_present()
    ids = [p.name for p in list_images(paths.test_images)]
    df = pd.DataFrame(
        {
            "image_id": ids,
            "pa_deg": args.pa,
            "fl_mm": args.fl,
            "mt_mm": args.mt,
        }
    ).sort_values("image_id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows) pa={args.pa} fl={args.fl} mt={args.mt}")


if __name__ == "__main__":
    main()
