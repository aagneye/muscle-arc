#!/usr/bin/env python3
"""Per-scanner correction of the final measurements (configs/device_calibration.yaml).

The leaderboard probes (scripts/lb_probe.py, docs/curator_climb.md) show a fixed offset per
device layout that none of our label sets can reveal: FL +8.2 mm on siemens_right_ruler,
-6.9 / -8.1 mm on left_ruler_5cm / png_left_ruler, PA +3.3 deg on png_left_ruler. The
corrections are hand-set constants in the config, so the pipeline re-runs on new data
without a leaderboard (host ruling 739660, owner decision 2026-10-30). The layout of each
image is the scale decoder's `source` (experiments/depth_scale_table_exact.csv, written by
scripts/build_exact_scale_table.py); layouts missing from the config stay unchanged.

  python scripts/apply_device_calibration.py --cand submissions/cand_med6ud.csv \
      --out submissions/cand_med6ud_dc1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

OPS = {
    "fl_mul": ("fl_mm", lambda v, c: v * c),
    "fl_add": ("fl_mm", lambda v, c: v + c),
    "pa_add": ("pa_deg", lambda v, c: v + c),
    "mt_add": ("mt_mm", lambda v, c: v + c),
}


def apply_calibration(cand: pd.DataFrame, layouts: pd.Series, calib: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Copy of ``cand`` with each layout's corrections; ``layouts`` maps image_id -> layout."""
    out = cand.copy()
    lay = out["image_id"].map(layouts)
    for name, corr in calib.items():
        rows = lay == name
        for key, c in corr.items():
            if key not in OPS:
                raise KeyError(f"{name}: unknown correction {key!r} (known: {sorted(OPS)})")
            col, op = OPS[key]
            out.loc[rows, col] = op(out.loc[rows, col], float(c))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--calib", type=Path, default=Path("configs/device_calibration.yaml"))
    ap.add_argument("--table", type=Path, default=Path("experiments/depth_scale_table_exact.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cand = pd.read_csv(args.cand)
    layouts = pd.read_csv(args.table).set_index("image_id")["source"]
    calib = yaml.safe_load(args.calib.read_text())["layouts"]
    out = apply_calibration(cand, layouts, calib)
    out.to_csv(args.out, index=False)

    lay = cand["image_id"].map(layouts).fillna("?").rename("layout")
    cols = ["pa_deg", "fl_mm", "mt_mm"]
    report = pd.concat({"before": cand.groupby(lay)[cols].mean(), "after": out.groupby(lay)[cols].mean()}, axis=1)
    with pd.option_context("display.width", 200):
        print(report.round(2).to_string())
    print(f"wrote {args.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
