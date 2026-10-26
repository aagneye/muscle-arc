#!/usr/bin/env python3
"""Measurement-level fusion of several model runs (seeds, resolutions, encoders).

Averaging probability maps and measuring once is what the k5/k10/k15 ensembles do; the
challenge evidence (docs/research/2026-10-26-rater-calibration-and-fusion.md) says the
gains for a derived scalar come from fusing the *measurements*: each run is measured on
its own cache (scripts/geometry_sweep.py writes experiments/geometry_sweep_<tag>_test_measures.csv),
the chosen variant per parameter is taken per image, and the runs are combined with a
median (or mean) per image before the clip smoothing. The spread between runs is written
as a label-free uncertainty per image.

  python scripts/fuse_measurements.py --tags k5 s2 s3 --pick pa_legacy+2 "fl_legacy|trig0" mt_legacy \
      --out submissions/cand_med3.csv --spread experiments/fuse_spread_med3.csv
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PARAMS = ("pa_deg", "fl_mm", "mt_mm")
_SWEEP = Path(__file__).resolve().parent / "geometry_sweep.py"


def _sweep():
    spec = importlib.util.spec_from_file_location("geometry_sweep", _SWEEP)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def picked_series(measures: list[pd.DataFrame], pick: dict[str, str],
                  groups: list[list[int]] | None, build_variants) -> dict[str, pd.DataFrame]:
    """{param: DataFrame(rows = images in the order of the first run, one column per run)}."""
    ids = measures[0]["image_id"].tolist()
    out = {p: pd.DataFrame(index=ids) for p in PARAMS}
    for k, m in enumerate(measures):
        m = m.set_index("image_id").reindex(ids).reset_index()
        if m[["mt_legacy", "pa_legacy", "fl_legacy"]].isna().all().any():
            raise ValueError(f"run {k} has no measurements for the images of run 0")
        v = build_variants(m, groups)
        for p in PARAMS:
            out[p][f"run{k}"] = v[p][pick[p]].to_numpy()
    return out


def fuse(per_run: dict[str, pd.DataFrame], stat: str = "median") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(fused values, spread) per image; spread = mean absolute deviation from the fused value."""
    fused = pd.DataFrame(index=per_run["pa_deg"].index)
    spread = pd.DataFrame(index=per_run["pa_deg"].index)
    for p in PARAMS:
        runs = per_run[p]
        fused[p] = runs.median(axis=1) if stat == "median" else runs.mean(axis=1)
        spread[p] = (runs.sub(fused[p], axis=0)).abs().mean(axis=1)
    return fused, spread


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/dual_v3.yaml"))
    ap.add_argument("--tags", nargs="+", required=True, help="sweep tags, each with a _test_measures.csv")
    ap.add_argument("--pick", nargs=3, metavar=("PA", "FL", "MT"), required=True)
    ap.add_argument("--stat", choices=("median", "mean"), default="median")
    ap.add_argument("--smooth-w", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spread", type=Path, default=None, help="write per-image between-run spread")
    args = ap.parse_args()

    sw = _sweep()
    cfg = yaml.safe_load(args.config.read_text())
    test_paths = sw.list_images(sw.DataPaths.from_config(cfg["data"]).test_images)
    groups = sw._groups_for(test_paths)
    measures = [pd.read_csv(f"experiments/geometry_sweep_{t}_test_measures.csv") for t in args.tags]
    order = [p.name for p in test_paths]
    measures = [m.set_index("image_id").reindex(order).reset_index() for m in measures]
    pick = dict(zip(PARAMS, args.pick))
    per_run = picked_series(measures, pick, groups, sw.build_variants)
    fused, spread = fuse(per_run, args.stat)

    sub = pd.DataFrame({"image_id": fused.index})
    for p in PARAMS:
        sub[p] = sw.smooth_groups(fused[p].reset_index(drop=True), groups, w=args.smooth_w).round(4).to_numpy()
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows, {args.stat} of {len(args.tags)} runs)")
    for p in PARAMS:
        q = spread[p].quantile([0.5, 0.9, 0.99]).to_numpy()
        print(f"  {p}: between-run spread median {q[0]:.3f}  p90 {q[1]:.3f}  p99 {q[2]:.3f}")
    if args.spread:
        spread.rename_axis("image_id").reset_index().to_csv(args.spread, index=False)


if __name__ == "__main__":
    main()
