"""Per-parameter linear calibration toward the expert benchmark, gated by leave-one-out.

The host recommends the six-rater OSF benchmark (35 images) for calibrating an
automated pipeline to the raters. With only 35 images a fitted gain/offset can
overfit, so (1) the fit is shrunk toward identity and (2) it is applied only if
its leave-one-out MAE beats the uncalibrated MAE (``loo_gain < 0``).
"""

from __future__ import annotations

import numpy as np


def fit_linear_calibration(pred: np.ndarray, gt: np.ndarray, shrink: float = 0.5) -> tuple[float, float]:
    """OLS fit gt ~ a * pred + b, blended with the identity: the returned map is
    pred + (1 - shrink) * ((a * pred + b) - pred)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(gt)
    if ok.sum() < 3:
        return 1.0, 0.0
    a, b = np.polyfit(pred[ok], gt[ok], 1)
    return 1.0 + (1.0 - shrink) * (float(a) - 1.0), (1.0 - shrink) * float(b)


def apply_calibration(pred: np.ndarray, gain: float, offset: float) -> np.ndarray:
    return gain * np.asarray(pred, dtype=np.float64) + offset


def loo_gain(pred: np.ndarray, gt: np.ndarray, shrink: float = 0.5) -> float:
    """Leave-one-out calibrated MAE minus uncalibrated MAE (negative = calibration helps)."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    ok = np.flatnonzero(np.isfinite(pred) & np.isfinite(gt))
    if len(ok) < 5:
        return 0.0
    errs = []
    for i in ok:
        rest = ok[ok != i]
        g, o = fit_linear_calibration(pred[rest], gt[rest], shrink)
        errs.append(abs(g * pred[i] + o - gt[i]))
    return float(np.mean(errs) - np.mean(np.abs(pred[ok] - gt[ok])))


GLOBAL = "__global__"


def fit_pooled_calibration(
    pred: np.ndarray,
    gt: np.ndarray,
    groups: np.ndarray,
    *,
    shrink: float = 0.5,
    pool: float = 10.0,
) -> dict[str, tuple[float, float]]:
    """Per-group (device layout) gain/offset with partial pooling toward the global fit.

    A group with n labelled images gets weight n / (n + pool) on its own shrunk OLS
    fit and the rest on the global one, so small groups inherit the global map and
    large groups keep their own. The result maps every group name plus ``GLOBAL``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    g0, o0 = fit_linear_calibration(pred, gt, shrink)
    out = {GLOBAL: (g0, o0)}
    for name in np.unique(groups):
        sel = (groups == name) & np.isfinite(pred) & np.isfinite(gt)
        n = int(sel.sum())
        if n < 3:
            out[name] = (g0, o0)
            continue
        gg, og = fit_linear_calibration(pred[sel], gt[sel], shrink)
        w = n / (n + pool)
        out[name] = (w * gg + (1.0 - w) * g0, w * og + (1.0 - w) * o0)
    return out


def apply_pooled(pred: np.ndarray, groups: np.ndarray, params: dict[str, tuple[float, float]]) -> np.ndarray:
    """Apply the group's map to each element; unknown groups use the global map."""
    pred = np.asarray(pred, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    out = np.empty_like(pred)
    for i, (x, g) in enumerate(zip(pred, groups)):
        gain, offset = params.get(g, params[GLOBAL])
        out[i] = gain * x + offset
    return out


def loo_pooled_mae(
    pred: np.ndarray,
    gt: np.ndarray,
    groups: np.ndarray,
    *,
    shrink: float = 0.5,
    pool: float = 10.0,
) -> tuple[float, float]:
    """(uncalibrated MAE, leave-one-out pooled-calibrated MAE) over the labelled images."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    ok = np.flatnonzero(np.isfinite(pred) & np.isfinite(gt))
    if len(ok) < 5:
        return float("nan"), float("nan")
    errs = []
    for i in ok:
        rest = ok[ok != i]
        params = fit_pooled_calibration(pred[rest], gt[rest], groups[rest], shrink=shrink, pool=pool)
        errs.append(abs(apply_pooled(pred[[i]], groups[[i]], params)[0] - gt[i]))
    return float(np.mean(np.abs(pred[ok] - gt[ok]))), float(np.mean(errs))
