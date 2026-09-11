#!/usr/bin/env python3
"""Aggregate Gate0/1/2/3 into one report card (SOF climb)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate0", type=Path, default=Path("experiments/gate0_reference.json"))
    parser.add_argument("--gate1", type=Path, default=Path("experiments/gate1_geometry.json"))
    parser.add_argument("--gate2", type=Path, default=Path("experiments/gate2_osf_umud.json"))
    parser.add_argument("--gate2b", type=Path, default=Path("experiments/gate2b_osf_pred_scale.json"))
    parser.add_argument("--gate3", type=Path, default=Path("experiments/gate3_dist.json"))
    parser.add_argument("--out", type=Path, default=Path("experiments/grow_report_card.json"))
    parser.add_argument("--require-gate0", action="store_true")
    args = parser.parse_args()

    g0, g1, g2, g2b, g3 = (
        _load(args.gate0),
        _load(args.gate1),
        _load(args.gate2),
        _load(args.gate2b),
        _load(args.gate3),
    )
    gate0_ok = bool(g0.get("gate0_ok"))
    gate1_ok = bool(g1.get("gate1_ok") or g1.get("geom_ok"))
    gate2_ok = bool(g2.get("gate2_ok"))
    gate2b_ok = True
    if not g2b.get("missing"):
        # pred-scale gap vs gt-scale
        umud_gt = float(g2.get("umud", 9))
        umud_pred = float(g2b.get("umud", 9))
        gate2b_ok = bool((umud_pred - umud_gt) <= 0.10) if umud_gt < 9 else bool(g2b.get("gate2_ok"))
    gate3_ok = bool(g3.get("gate3_ok") or g3.get("gate4_ok"))

    grown = gate1_ok and gate2_ok and gate3_ok and gate2b_ok
    if args.require_gate0:
        grown = grown and gate0_ok

    card = {
        "gate0_ok": gate0_ok,
        "gate1_ok": gate1_ok,
        "gate2_ok": gate2_ok,
        "gate2b_ok": gate2b_ok,
        "gate3_ok": gate3_ok,
        "grown": grown,
        "submit": "SUBMIT" if grown else "HOLD",
        "gate0": {k: g0.get(k) for k in ("n", "umud", "gate0_ok", "masks_from", "scale_mode")},
        "gate1": {
            k: g1.get(k)
            for k in ("n", "split", "pa_mae_deg", "fl_rel_mae", "mt_rel_mae", "geom_ok", "route")
        },
        "gate2": {
            k: g2.get(k)
            for k in (
                "n",
                "umud",
                "n_mt_bad",
                "n_mt_ok",
                "mt_ratio_median",
                "fl_ratio_median",
                "gate2_ok",
            )
        },
        "gate2b": {k: g2b.get(k) for k in ("n", "umud", "gate2_ok")},
        "gate3": {
            k: g3.get(k)
            for k in ("fl_med", "fl_med_near_81", "gate3_warn_fl_med", "clip_frac", "gate3_ok")
        },
    }
    print(json.dumps(card, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(card, indent=2))
    print(card["submit"])
    raise SystemExit(0 if grown else 1)


if __name__ == "__main__":
    main()
