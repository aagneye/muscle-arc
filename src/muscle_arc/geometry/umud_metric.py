"""Official UMUD score helpers (host notebook tolerances)."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Official divisors from host UMUD score notebook
TOL_PA = 6.0
TOL_FL = 12.0
TOL_MT = 3.0


def umud_score(pa_mae: float, fl_mae: float, mt_mae: float) -> float:
    """S = (1/3) * (MAE_PA/6 + MAE_FL/12 + MAE_MT/3)."""
    return float(
        (1.0 / 3.0)
        * (float(pa_mae) / TOL_PA + float(fl_mae) / TOL_FL + float(mt_mae) / TOL_MT)
    )


def umud_from_errors(pa_err: np.ndarray, fl_err: np.ndarray, mt_err: np.ndarray) -> dict:
    pa_mae = float(np.mean(pa_err)) if len(pa_err) else float("nan")
    fl_mae = float(np.mean(fl_err)) if len(fl_err) else float("nan")
    mt_mae = float(np.mean(mt_err)) if len(mt_err) else float("nan")
    return {
        "n": int(len(pa_err)),
        "pa_mae": pa_mae,
        "fl_mae": fl_mae,
        "mt_mae": mt_mae,
        "umud": umud_score(pa_mae, fl_mae, mt_mae) if len(pa_err) else float("nan"),
        # alias for older callers
        "combo": umud_score(pa_mae, fl_mae, mt_mae) if len(pa_err) else float("nan"),
    }


def gate4_dist_ok(
    cand: pd.DataFrame,
    ref: pd.DataFrame,
    *,
    fl_std_min_frac: float = 0.45,
    mt_std_min_frac: float = 0.45,
    max_clip_frac: float = 0.05,
    fl_hi: float = 140.0,
    mt_hi: float = 45.0,
    fl_lo: float = 25.0,
    mt_lo: float = 6.0,
) -> tuple[bool, dict]:
    """Reject collapsed std (v9) or excessive clip-hits (v10b)."""
    info: dict = {}
    fl_std_c = float(cand["fl_mm"].std())
    fl_std_r = float(ref["fl_mm"].std())
    mt_std_c = float(cand["mt_mm"].std())
    mt_std_r = float(ref["mt_mm"].std())
    n = max(len(cand), 1)
    fl_clip = int(((cand["fl_mm"] <= fl_lo + 1e-6) | (cand["fl_mm"] >= fl_hi - 1e-6)).sum())
    mt_clip = int(((cand["mt_mm"] <= mt_lo + 1e-6) | (cand["mt_mm"] >= mt_hi - 1e-6)).sum())
    info.update(
        {
            "fl_std_cand": fl_std_c,
            "fl_std_ref": fl_std_r,
            "mt_std_cand": mt_std_c,
            "mt_std_ref": mt_std_r,
            "fl_clip_frac": fl_clip / n,
            "mt_clip_frac": mt_clip / n,
            "fl_med": float(cand["fl_mm"].median()),
            "mt_med": float(cand["mt_mm"].median()),
        }
    )
    ok = True
    reasons = []
    if fl_std_r > 0 and fl_std_c < fl_std_min_frac * fl_std_r:
        ok = False
        reasons.append("fl_std_collapse")
    if mt_std_r > 0 and mt_std_c < mt_std_min_frac * mt_std_r:
        ok = False
        reasons.append("mt_std_collapse")
    if fl_clip / n > max_clip_frac:
        ok = False
        reasons.append("fl_clip_high")
    if mt_clip / n > max_clip_frac:
        ok = False
        reasons.append("mt_clip_high")
    info["ok"] = ok
    info["reasons"] = reasons
    return ok, info
