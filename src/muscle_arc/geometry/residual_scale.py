"""Phase A' residual / scale regressors when TIFF metadata is unusable."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

FEATURE_COLS = [
    "h",
    "w",
    "aspect",
    "area",
    "log_area",
    "mean_i",
    "std_i",
    "pa_deg",
    "fl_px",
    "mt_px",
    "fl_over_mt",
    "trig_fl_px",
    "sin_pa",
]


def row_features(
    h: float,
    w: float,
    mean_i: float,
    std_i: float,
    pa_deg: float,
    fl_px: float,
    mt_px: float,
) -> dict[str, float]:
    """Compact geometry + shape features for scale / mm residual models."""
    aspect = float(w) / max(float(h), 1.0)
    area = float(h) * float(w)
    pa = float(pa_deg) if np.isfinite(pa_deg) else 15.0
    fl = float(fl_px) if np.isfinite(fl_px) else np.nan
    mt = float(mt_px) if np.isfinite(mt_px) else np.nan
    sin_pa = float(np.sin(np.radians(abs(pa))))
    trig = float(mt / max(sin_pa, 1e-3)) if np.isfinite(mt) else np.nan
    fl_over_mt = float(fl / mt) if np.isfinite(fl) and np.isfinite(mt) and mt > 1 else np.nan
    return {
        "h": float(h),
        "w": float(w),
        "aspect": aspect,
        "area": area,
        "log_area": float(np.log(max(area, 1.0))),
        "mean_i": float(mean_i),
        "std_i": float(std_i),
        "pa_deg": pa,
        "fl_px": fl,
        "mt_px": mt,
        "fl_over_mt": fl_over_mt,
        "trig_fl_px": trig,
        "sin_pa": sin_pa,
    }


def features_matrix(df: pd.DataFrame) -> np.ndarray:
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise KeyError(f"missing feature column {c}")
    return df[FEATURE_COLS].to_numpy(dtype=np.float64)


def train_mm_regressor(
    feats: pd.DataFrame,
    fl_mm: np.ndarray,
    mt_mm: np.ndarray,
    *,
    max_iter: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.06,
    min_samples_leaf: int = 3,
) -> MultiOutputRegressor:
    """Direct FL_mm / MT_mm regressor (Huber-like via absolute error)."""
    X = features_matrix(feats)
    y = np.column_stack([fl_mm, mt_mm]).astype(np.float64)
    base = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=0.1,
        random_state=42,
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y)
    return model


def train_scale_regressor(
    feats: pd.DataFrame,
    mm_per_pixel: np.ndarray,
    *,
    max_iter: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.06,
    min_samples_leaf: int = 3,
) -> HistGradientBoostingRegressor:
    """Predict mm/px from shape+intensity (uses OSF Scale_pixel_per_cm)."""
    X = features_matrix(feats)
    y = np.asarray(mm_per_pixel, dtype=np.float64)
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X, y)
    return model


def predict_mm(model: MultiOutputRegressor, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(features_matrix(feats))
    return pred[:, 0], pred[:, 1]


def predict_scale(model: HistGradientBoostingRegressor, feats: pd.DataFrame) -> np.ndarray:
    return model.predict(features_matrix(feats))


def save_models(
    out_dir: Path | str,
    *,
    mm_model: MultiOutputRegressor | None = None,
    scale_model: HistGradientBoostingRegressor | None = None,
    meta: dict | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mm_model is not None:
        joblib.dump(mm_model, out_dir / "mm_regressor.joblib")
    if scale_model is not None:
        joblib.dump(scale_model, out_dir / "scale_regressor.joblib")
    joblib.dump({"feature_cols": FEATURE_COLS, **(meta or {})}, out_dir / "meta.joblib")
    return out_dir


def load_models(model_dir: Path | str) -> dict:
    model_dir = Path(model_dir)
    out: dict = {"dir": model_dir, "mm": None, "scale": None, "meta": {}}
    if not model_dir.exists():
        return out
    mm_path = model_dir / "mm_regressor.joblib"
    sc_path = model_dir / "scale_regressor.joblib"
    meta_path = model_dir / "meta.joblib"
    if mm_path.exists():
        out["mm"] = joblib.load(mm_path)
    if sc_path.exists():
        out["scale"] = joblib.load(sc_path)
    if meta_path.exists():
        out["meta"] = joblib.load(meta_path)
    return out


def apply_residual_mm(
    raw: pd.DataFrame,
    models: dict,
    *,
    prefer: str = "blend",
    blend_alpha: float = 0.65,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Replace/blend FL_mm MT_mm using residual models.

    prefer:
      - "mm": use direct mm regressor only
      - "scale": fl_px*scale, mt_px*scale only
      - "blend": alpha * mm_model + (1-alpha) * (px * scale_model)
    """
    info = {"n": len(raw), "mode": prefer, "used_mm": False, "used_scale": False}
    fl_base = raw["fl_mm"].to_numpy(dtype=np.float64) if "fl_mm" in raw.columns else np.full(len(raw), np.nan)
    mt_base = raw["mt_mm"].to_numpy(dtype=np.float64) if "mt_mm" in raw.columns else np.full(len(raw), np.nan)

    feats = raw.copy()
    for c in FEATURE_COLS:
        if c not in feats.columns:
            # derive common ones if missing
            if c == "aspect" and {"h", "w"}.issubset(feats.columns):
                feats[c] = feats["w"] / feats["h"].clip(lower=1)
            elif c == "area" and {"h", "w"}.issubset(feats.columns):
                feats[c] = feats["h"] * feats["w"]
            elif c == "log_area" and "area" in feats.columns:
                feats[c] = np.log(feats["area"].clip(lower=1))
            elif c == "fl_over_mt" and {"fl_px", "mt_px"}.issubset(feats.columns):
                feats[c] = feats["fl_px"] / feats["mt_px"].replace(0, np.nan)
            elif c == "sin_pa" and "pa_deg" in feats.columns:
                feats[c] = np.sin(np.radians(feats["pa_deg"].abs()))
            elif c == "trig_fl_px" and {"mt_px", "sin_pa"}.issubset(feats.columns):
                feats[c] = feats["mt_px"] / feats["sin_pa"].clip(lower=1e-3)
            else:
                feats[c] = np.nan

    fl_out = fl_base.copy()
    mt_out = mt_base.copy()

    fl_mm_pred = mt_mm_pred = None
    scale_pred = None
    if models.get("mm") is not None:
        fl_mm_pred, mt_mm_pred = predict_mm(models["mm"], feats)
        info["used_mm"] = True
    if models.get("scale") is not None:
        scale_pred = predict_scale(models["scale"], feats)
        # Physiological ultrasound mm/px band
        scale_pred = np.clip(scale_pred, 0.02, 0.20)
        info["used_scale"] = True
        info["scale_median"] = float(np.nanmedian(scale_pred))

    fl_from_scale = mt_from_scale = None
    if scale_pred is not None:
        fl_px = raw["fl_px"].to_numpy(dtype=np.float64)
        mt_px = raw["mt_px"].to_numpy(dtype=np.float64)
        fl_from_scale = fl_px * scale_pred
        mt_from_scale = mt_px * scale_pred

    if prefer == "mm" and fl_mm_pred is not None:
        fl_out, mt_out = fl_mm_pred, mt_mm_pred
    elif prefer == "scale" and fl_from_scale is not None:
        fl_out, mt_out = fl_from_scale, mt_from_scale
    elif prefer == "blend" and fl_mm_pred is not None and fl_from_scale is not None:
        a = float(np.clip(blend_alpha, 0.0, 1.0))
        fl_out = a * fl_mm_pred + (1 - a) * fl_from_scale
        mt_out = a * mt_mm_pred + (1 - a) * mt_from_scale
    elif fl_mm_pred is not None:
        fl_out, mt_out = fl_mm_pred, mt_mm_pred
    elif fl_from_scale is not None:
        fl_out, mt_out = fl_from_scale, mt_from_scale

    return pd.Series(fl_out), pd.Series(mt_out), info
