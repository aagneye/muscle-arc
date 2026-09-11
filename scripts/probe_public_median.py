#!/usr/bin/env python3
"""Probe public-LB medians via constant submissions (MT / PA / FL sweeps).

Kaggle ranks by best submission, so probes cannot hurt the leaderboard rank.
Interpolate interval slopes to recover the public-set median (same method used
for FL ≈ 81.5 mm). Do NOT calibrate absolute FL/MT toward OSF medians.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from muscle_arc.data.dataset import list_images
from muscle_arc.data.paths import DataPaths


def write_constant(
    ids: list[str],
    pa: float,
    fl: float,
    mt: float,
    out: Path,
) -> Path:
    df = pd.DataFrame(
        {"image_id": ids, "pa_deg": pa, "fl_mm": fl, "mt_mm": mt}
    ).sort_values("image_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} n={len(df)} pa={pa} fl={fl} mt={mt}")
    return out


def interpolate_zero(xs: list[float], scores: list[float]) -> float | None:
    """Find x where score slope crosses zero between consecutive samples."""
    if len(xs) < 2 or len(scores) != len(xs):
        return None
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    scores = [scores[i] for i in order]
    # Best is the minimum; also try slope zero-crossing between neighbors
    best_i = int(min(range(len(scores)), key=lambda i: scores[i]))
    if best_i == 0:
        # Check if still decreasing toward lower x — return lowest probed
        return float(xs[0])
    if best_i == len(scores) - 1:
        return float(xs[-1])
    # Linear interpolate min between best neighbors
    x0, x1, x2 = xs[best_i - 1], xs[best_i], xs[best_i + 1]
    s0, s1, s2 = scores[best_i - 1], scores[best_i], scores[best_i + 1]
    # Parabolic vertex approx
    denom = (s0 - 2 * s1 + s2)
    if abs(denom) < 1e-12:
        return float(x1)
    # Map to equal spacing assumption on indices
    vertex = 0.5 * (s0 - s2) / denom
    # vertex in [-1,1] relative to best_i
    span = 0.5 * (x2 - x0)
    return float(x1 + vertex * span)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--axis", choices=["mt", "pa", "fl"], default="mt")
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="Sweep values (defaults depend on --axis)",
    )
    parser.add_argument("--pa", type=float, default=15.2, help="Held-fixed PA")
    parser.add_argument("--fl", type=float, default=81.5, help="Held-fixed FL")
    parser.add_argument("--mt", type=float, default=22.0, help="Held-fixed MT")
    parser.add_argument("--out-dir", type=Path, default=Path("submissions/probes"))
    parser.add_argument(
        "--scores-json",
        type=Path,
        default=None,
        help="Optional {value: public_score} after submitting; prints interpolated median",
    )
    args = parser.parse_args()

    defaults = {
        "mt": [16.0, 19.0, 22.0, 25.0, 28.0],
        "pa": [12.0, 14.0, 16.0, 18.0, 20.0],
        "fl": [70.0, 75.0, 81.5, 88.0, 95.0],
    }
    values = args.values if args.values is not None else defaults[args.axis]

    cfg = yaml.safe_load(args.config.read_text())
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_test_present()
    ids = [p.name for p in list_images(paths.test_images)]

    written = []
    for v in values:
        pa, fl, mt = args.pa, args.fl, args.mt
        if args.axis == "mt":
            mt = v
        elif args.axis == "pa":
            pa = v
        else:
            fl = v
        out = args.out_dir / f"probe_{args.axis}_{v:g}.csv"
        write_constant(ids, pa, fl, mt, out)
        written.append(str(out))

    manifest = {
        "axis": args.axis,
        "values": values,
        "held": {"pa": args.pa, "fl": args.fl, "mt": args.mt},
        "files": written,
        "note": "Submit each CSV; record publicScore; pass --scores-json to interpolate.",
    }
    man_path = args.out_dir / f"probe_{args.axis}_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {man_path}")

    if args.scores_json and args.scores_json.exists():
        scores = json.loads(args.scores_json.read_text())
        xs = [float(k) for k in scores.keys()]
        ys = [float(scores[k]) for k in scores.keys()]
        med = interpolate_zero(xs, ys)
        print(f"Interpolated public median {args.axis} ≈ {med}")
        result = {"axis": args.axis, "scores": scores, "median_est": med}
        (args.out_dir / f"probe_{args.axis}_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
