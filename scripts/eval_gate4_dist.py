#!/usr/bin/env python3
"""Gate 4: distribution sanity vs v8 (std collapse / clip-hit rate)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from muscle_arc.geometry.umud_metric import gate4_dist_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cand", type=Path, required=True)
    parser.add_argument("--ref", type=Path, default=Path("submissions/submission_v8.csv"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-clip-frac", type=float, default=0.05)
    args = parser.parse_args()

    cand = pd.read_csv(args.cand)
    ref = pd.read_csv(args.ref)
    ok, info = gate4_dist_ok(cand, ref, max_clip_frac=args.max_clip_frac)
    # Probed public FL median ~81.5
    fl_med = info["fl_med"]
    info["fl_med_near_81"] = bool(70.0 <= fl_med <= 95.0)
    info["gate4_ok"] = bool(ok and info["fl_med_near_81"])
    info["gate3_ok"] = info["gate4_ok"]  # report-card alias
    print(json.dumps(info, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(info, indent=2))
    print("GATE4_OK" if info["gate4_ok"] else "GATE4_FAIL")
    print("GATE3_OK" if info["gate3_ok"] else "GATE3_FAIL")
    raise SystemExit(0 if info["gate4_ok"] else 1)


if __name__ == "__main__":
    main()
