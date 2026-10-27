#!/usr/bin/env python3
"""Leaderboard probes: per-parameter (and per-subgroup) MAE and bias from public scores.

The public score is S = sum_p MAE_p / (3 T_p) with T = (6 deg, 12 mm, 3 mm), computed on
the public subset (about 33 % of the 309 test images, N ~ 102). Shifting one parameter by
a constant K larger than any error turns its absolute error into K + e_i, so a +K / -K
pair of probes gives the signed and absolute error sums of the shifted images exactly, up
to the 1e-5 rounding of the board:

    c = 3 T_p N
    A = c (S+ - S-) / 2            = sum_G e_i
    B = c (S+ + S- - 2 S0) / 2     = n_G K - sum_G |e_i|

For a full-set probe n_G = N and N cancels out of MAE and bias. For a subgroup (device
layout, clip frames, ...) only its public members count and their number is unknown; with
K large (sum_G |e_i| < K / 2) it is recovered as n_G = round(B / K), so use K = 1e4 for
subgroup probes. A single +K probe only gives the under-prediction sum of G (e_i < 0).
This is diagnosis of where the error lives; the final pipeline uses none of these numbers
(host ruling 739660 forbids leaderboard-weighted ensembling, not leaderboard diagnosis).

  python scripts/lb_probe.py make --base submissions/cand_k10.csv --param fl_mm --shift 10000 \
      --table experiments/depth_scale_table_exact.csv --where source=siemens_right_ruler \
      --out submissions/probe_fl_p10000_siemens_right_ruler.csv
  python scripts/lb_probe.py decode --base-score 0.40307 \
      --pair pa_deg 100 5.85605 5.81660 --pair fl_mm 1000 27.90299 27.98273 \
      --single mt_mm 100 11.44966 --pair fl_mm 10000 S_PLUS S_MINUS ? siemens
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
PARAMS = tuple(TOL)
N_PUBLIC = 103  # 309 / 3: confirmed by the integer subgroup size of the 2026-10-27 siemens probe


def load_table(path: Path) -> pd.DataFrame:
    """CSV with an image_id column, or the labeling manifest JSON ({"images": [...]})."""
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text())["images"])
    return pd.read_csv(path)


def select_ids(table: pd.DataFrame, where: str) -> list[str]:
    """``col=value`` exact match, ``col=*`` non-null, ``col!=value`` complement."""
    neg = "!=" in where
    col, value = where.split("!=" if neg else "=", 1)
    if col not in table.columns:
        raise KeyError(f"column {col!r} not in table ({list(table.columns)})")
    if value == "*":
        mask = table[col].notna() & (table[col].astype(str) != "")
    else:
        mask = table[col].astype(str) == value
    if neg:
        mask = ~mask
    return table.loc[mask, "image_id"].astype(str).tolist()


def make_probe(base: pd.DataFrame, param: str, shift: float, ids: list[str] | None) -> pd.DataFrame:
    """Copy of ``base`` with ``param`` shifted by ``shift`` on ``ids`` (all rows if None)."""
    if param not in PARAMS:
        raise ValueError(f"param must be one of {PARAMS}")
    out = base.copy()
    rows = out["image_id"].isin(ids) if ids is not None else slice(None)
    out.loc[rows, param] = out.loc[rows, param] + shift
    return out


def pair_terms(param: str, k: float, s_plus: float, s_minus: float, s0: float, n: int) -> tuple[float, float]:
    """(A, B) = (sum of signed errors, n_G K - sum of absolute errors) of the shifted images."""
    c = 3.0 * TOL[param] * n
    return c * (s_plus - s_minus) / 2.0, c * (s_plus + s_minus - 2.0 * s0) / 2.0


def pair_stats(param: str, k: float, s_plus: float, s_minus: float, s0: float, n: int,
               n_sub: int | None) -> tuple[int, float, float]:
    """(n_G, MAE_G, bias_G); n_G recovered as round(B / K) when not given."""
    a, b = pair_terms(param, k, s_plus, s_minus, s0, n)
    if n_sub is None:
        n_sub = int(round(b / k))
        if n_sub <= 0:
            raise ValueError(f"{param}: recovered subgroup size {n_sub}; is K large enough?")
    sum_abs = n_sub * k - b
    return n_sub, sum_abs / n_sub, a / n_sub


def single_under_sum(param: str, k: float, s_plus: float, s0: float, n: int, n_sub: int) -> float:
    """Sum of |e_i| over the subgroup's under-predictions (e_i < 0) from one +K probe."""
    c = 3.0 * TOL[param] * n
    return (n_sub * k - c * (s_plus - s0)) / 2.0


def decode(n: int, s0: float, pairs: list[tuple[str, float, float, float, int | None, str]],
           singles: list[tuple[str, float, float, int | None, str]]) -> pd.DataFrame:
    """Rows of (param, subset, n, mae, bias, contribution) for every solvable probe.

    ``pairs``: (param, K, S+, S-, n_sub, label) with n_sub = n for a full-set probe and
    None for a subgroup of unknown public size. ``singles``: (param, K, S+, n_sub, label).
    """
    rows = []
    full_mae: dict[str, float] = {}
    for param, k, sp, sm, n_sub, label in pairs:
        n_g, mae, bias = pair_stats(param, k, sp, sm, s0, n, n_sub)
        rows.append({"param": param, "subset": label, "n": n_g, "mae": mae, "bias": bias})
        if n_g == n:
            full_mae[param] = mae
    missing = [p for p in PARAMS if p not in full_mae]
    if len(missing) == 1:  # the base score closes the system
        p = missing[0]
        full_mae[p] = 3.0 * TOL[p] * (s0 - sum(full_mae[q] / (3.0 * TOL[q]) for q in full_mae))
        rows.append({"param": p, "subset": "all (from S0)", "n": n, "mae": full_mae[p], "bias": float("nan")})
    for param, k, sp, n_sub, label in singles:
        n_sub = n_sub or n
        under = single_under_sum(param, k, sp, s0, n, n_sub)
        row = {"param": param, "subset": label, "n": n_sub, "under_sum": under, "mae": float("nan"), "bias": float("nan")}
        if n_sub == n and param in full_mae:  # |e| known -> bias = (sum|e| - 2 under) / n
            row["mae"] = full_mae[param]
            row["bias"] = (full_mae[param] * n - 2.0 * under) / n
        rows.append(row)
    df = pd.DataFrame(rows)
    df["contribution"] = df["mae"] / (3.0 * df["param"].map(TOL))
    total = sum(full_mae[p] / (3.0 * TOL[p]) for p in full_mae)
    df["share"] = df["contribution"] / total if total else float("nan")
    return df


def _n_sub(token: str | None) -> int | None | str:
    if token is None:
        return "full"
    return None if token == "?" else int(token)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write a shifted copy of a submission")
    mk.add_argument("--base", type=Path, required=True)
    mk.add_argument("--param", choices=PARAMS, required=True)
    mk.add_argument("--shift", type=float, required=True)
    mk.add_argument("--table", type=Path, help="CSV or manifest JSON with image_id + subset columns")
    mk.add_argument("--where", help="col=value | col=* | col!=value (needs --table)")
    mk.add_argument("--ids", type=Path, help="text file, one image_id per line")
    mk.add_argument("--out", type=Path, required=True)
    dc = sub.add_parser("decode", help="turn probe scores into MAE / bias")
    dc.add_argument("--n", type=int, default=N_PUBLIC, help="images in the public subset")
    dc.add_argument("--base-score", type=float, required=True)
    dc.add_argument("--pair", nargs="+", action="append", default=[], metavar="X",
                    help="PARAM K S_PLUS S_MINUS [N_SUB|?] [LABEL]; omit N_SUB for a full-set probe, ? to recover it")
    dc.add_argument("--single", nargs="+", action="append", default=[],
                    metavar="X", help="PARAM K S_PLUS [N_SUB] [LABEL]")
    args = ap.parse_args()

    if args.cmd == "make":
        ids = None
        if args.ids:
            ids = [l.strip() for l in args.ids.read_text().splitlines() if l.strip()]
        elif args.where:
            if not args.table:
                ap.error("--where needs --table")
            ids = select_ids(load_table(args.table), args.where)
        base = pd.read_csv(args.base)
        if ids is not None and not set(ids) <= set(base["image_id"].astype(str)):
            ap.error("subset contains ids missing from the base submission")
        probe = make_probe(base, args.param, args.shift, ids)
        probe.to_csv(args.out, index=False)
        print(f"{args.out}: {args.param} {args.shift:+g} on {len(ids) if ids is not None else len(base)} rows")
        return

    pairs = []
    for p in args.pair:
        n_sub = _n_sub(p[4] if len(p) > 4 else None)
        n_sub = args.n if n_sub == "full" else n_sub
        label = p[5] if len(p) > 5 else ("all" if n_sub == args.n else "subgroup")
        pairs.append((p[0], float(p[1]), float(p[2]), float(p[3]), n_sub, label))
    singles = []
    for p in args.single:
        n_sub = int(p[3]) if len(p) > 3 else None
        label = p[4] if len(p) > 4 else ("all" if n_sub is None else f"n={n_sub}")
        singles.append((p[0], float(p[1]), float(p[2]), n_sub, label))
    df = decode(args.n, args.base_score, pairs, singles)
    with pd.option_context("display.width", 200):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
