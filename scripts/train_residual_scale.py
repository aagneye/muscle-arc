#!/usr/bin/env python3
"""Train Phase A' residual FL/MT (+ scale) models on OSF expert benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from muscle_arc.data.dataset import read_gray
from muscle_arc.geometry.metrics import (
    fascicle_length_px,
    muscle_thickness_px,
    pennation_angle_deg,
    reset_stats,
)
from muscle_arc.geometry.residual_scale import (
    FEATURE_COLS,
    apply_residual_mm,
    load_models,
    row_features,
    save_models,
    train_mm_regressor,
    train_scale_regressor,
)
from muscle_arc.geometry.scale import cluster_key
from muscle_arc.infer.predict import predict_prob
from muscle_arc.models.segmentation import build_segmentation_model


def load_model(ckpt: Path, model_cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_segmentation_model(model_cfg).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def geometry_row(
    path: Path,
    apo_model: torch.nn.Module,
    fasc_model: torch.nn.Module,
    img_size: int,
    device: torch.device,
    thr: float,
) -> dict:
    gray = read_gray(path)
    h, w = gray.shape[:2]
    apo_prob = predict_prob(apo_model, gray, img_size, device, True, True, False, False)
    fasc_prob = predict_prob(fasc_model, gray, img_size, device, True, True, False, False)
    apo = (apo_prob > thr).astype(np.uint8)
    fasc = (fasc_prob > thr).astype(np.uint8)
    if int(fasc.sum()) < 5000:
        fasc = cv2.morphologyEx(fasc, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if int(apo.sum()) > 5000:
        apo = cv2.medianBlur(apo, 5)
    mt_px = muscle_thickness_px(apo)
    pa = pennation_angle_deg(fasc, apo, fasc_prob=fasc_prob, gray=gray)
    fl_px = fascicle_length_px(
        fasc,
        apo,
        fasc_prob=fasc_prob,
        gray=gray,
        pa_deg=pa if np.isfinite(pa) else None,
        mt_px=mt_px if np.isfinite(mt_px) else None,
    )
    if (not np.isfinite(fl_px)) and np.isfinite(pa) and np.isfinite(mt_px) and 5 < abs(pa) < 45:
        fl_px = float(mt_px / max(np.sin(np.radians(abs(pa))), 1e-3))
    feats = row_features(
        h,
        w,
        float(np.mean(gray)),
        float(np.std(gray)),
        float(pa) if np.isfinite(pa) else np.nan,
        float(fl_px) if np.isfinite(fl_px) else np.nan,
        float(mt_px) if np.isfinite(mt_px) else np.nan,
    )
    feats.update(
        {
            "image_id": path.name,
            "stem": path.stem,
            "path": str(path),
            "cluster": cluster_key(h, w, gray),
        }
    )
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--apo-ckpt", type=Path, default=Path("experiments/checkpoints_phase4/apo_best.pt"))
    parser.add_argument("--fasc-ckpt", type=Path, default=Path("experiments/checkpoints_phase4/fasc_best.pt"))
    parser.add_argument("--gt", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/residual_models"))
    parser.add_argument("--thr", type=float, default=0.30)
    parser.add_argument("--features-out", type=Path, default=Path("experiments/osf_features.csv"))
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Skip segmentation/geometry if --features-out already exists",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = pd.read_csv(args.gt)
    if gt.empty:
        raise SystemExit("empty GT")

    if args.reuse_features and args.features_out.exists():
        feats = pd.read_csv(args.features_out)
        print(f"Reusing features {args.features_out} n={len(feats)}", flush=True)
    else:
        # Fall back to v1 checkpoints if phase4 missing
        apo_ckpt = args.apo_ckpt if args.apo_ckpt.exists() else Path("experiments/checkpoints/apo_best.pt")
        fasc_ckpt = args.fasc_ckpt if args.fasc_ckpt.exists() else Path("experiments/checkpoints/fasc_best.pt")
        print(f"Using ckpts apo={apo_ckpt} fasc={fasc_ckpt} device={device}", flush=True)

        apo_model = load_model(apo_ckpt, cfg["model"], device)
        fasc_model = load_model(fasc_ckpt, cfg["model"], device)
        img_size = int(cfg["img_size"])
        reset_stats()

        rows = []
        for _, g in gt.iterrows():
            p = Path(str(g["path"])) if str(g.get("path", "")) else None
            if p is None or not p.exists():
                alt = Path("data/external/umud_osf/unpacked/benchmark_dataset_architecture_v0.1.0") / f"{g['stem']}.tif"
                p = alt if alt.exists() else None
            if p is None or not p.exists():
                print(f"skip missing {g.get('stem')}", flush=True)
                continue
            print(f"geometry {p.name} ...", flush=True)
            feat = geometry_row(p, apo_model, fasc_model, img_size, device, args.thr)
            feat["pa_gt"] = float(g["pa_deg"])
            feat["fl_gt"] = float(g["fl_mm"])
            feat["mt_gt"] = float(g["mt_mm"])
            feat["mm_per_pixel_gt"] = float(g["mm_per_pixel"]) if pd.notna(g.get("mm_per_pixel")) else np.nan
            rows.append(feat)

        feats = pd.DataFrame(rows)
        args.features_out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(args.features_out, index=False)
        print(f"Wrote features {args.features_out} n={len(feats)}", flush=True)

    ok = feats.dropna(subset=["fl_gt", "mt_gt", "fl_px", "mt_px"]).copy()
    if len(ok) < 8:
        raise SystemExit(f"too few rows for residual training: {len(ok)}")

    # Holdout MAE estimate (honest small-N gate; avoid LOO OpenMP thrash)
    from sklearn.model_selection import KFold

    fl_pred_cv, mt_pred_cv, fl_gt_cv, mt_gt_cv = [], [], [], []
    kf = KFold(n_splits=min(5, len(ok)), shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(ok):
        tr, te = ok.iloc[train_idx], ok.iloc[test_idx]
        mm_m = train_mm_regressor(
            tr, tr["fl_gt"].to_numpy(), tr["mt_gt"].to_numpy(), max_iter=80, max_depth=2
        )
        sc_m = None
        if tr["mm_per_pixel_gt"].notna().sum() >= 5:
            sc_m = train_scale_regressor(
                tr, tr["mm_per_pixel_gt"].to_numpy(), max_iter=80, max_depth=2
            )
        models = {"mm": mm_m, "scale": sc_m}
        te2 = te.copy()
        te2["fl_mm"] = te2["fl_px"] * 0.06
        te2["mt_mm"] = te2["mt_px"] * 0.06
        fl_s, mt_s, _ = apply_residual_mm(te2, models, prefer="blend")
        fl_pred_cv.extend(fl_s.to_numpy().tolist())
        mt_pred_cv.extend(mt_s.to_numpy().tolist())
        fl_gt_cv.extend(te["fl_gt"].to_numpy().tolist())
        mt_gt_cv.extend(te["mt_gt"].to_numpy().tolist())

    fl_mae = float(mean_absolute_error(fl_gt_cv, fl_pred_cv))
    mt_mae = float(mean_absolute_error(mt_gt_cv, mt_pred_cv))
    # Baseline: constant scale 0.06
    fl_base = float(mean_absolute_error(ok["fl_gt"], ok["fl_px"] * 0.06))
    mt_base = float(mean_absolute_error(ok["mt_gt"], ok["mt_px"] * 0.06))
    # Also baseline using median OSF mm/px
    med_scale = float(ok["mm_per_pixel_gt"].median())
    fl_base_osf = float(mean_absolute_error(ok["fl_gt"], ok["fl_px"] * med_scale))
    mt_base_osf = float(mean_absolute_error(ok["mt_gt"], ok["mt_px"] * med_scale))
    metrics = {
        "n": int(len(ok)),
        "cv_fl_mae": fl_mae,
        "cv_mt_mae": mt_mae,
        "base06_fl_mae": fl_base,
        "base06_mt_mae": mt_base,
        "base_osfscale_fl_mae": fl_base_osf,
        "base_osfscale_mt_mae": mt_base_osf,
        "osf_median_mm_per_pixel": med_scale,
        "improved_vs_06": (fl_mae + mt_mae) < (fl_base + mt_base),
    }
    print(json.dumps(metrics, indent=2), flush=True)

    mm_model = train_mm_regressor(ok, ok["fl_gt"].to_numpy(), ok["mt_gt"].to_numpy())
    scale_model = None
    if ok["mm_per_pixel_gt"].notna().sum() >= 5:
        scale_model = train_scale_regressor(ok, ok["mm_per_pixel_gt"].to_numpy())
    save_models(args.out_dir, mm_model=mm_model, scale_model=scale_model, meta=metrics)
    print(f"saved models -> {args.out_dir}", flush=True)

    # Sanity: reload + in-sample
    models = load_models(args.out_dir)
    tmp = ok.copy()
    tmp["fl_mm"] = tmp["fl_px"] * 0.06
    tmp["mt_mm"] = tmp["mt_px"] * 0.06
    fl_s, mt_s, info = apply_residual_mm(tmp, models, prefer="blend")
    print("apply_info", info)
    print(
        "insample FL MAE",
        float(mean_absolute_error(ok["fl_gt"], fl_s)),
        "MT MAE",
        float(mean_absolute_error(ok["mt_gt"], mt_s)),
    )


if __name__ == "__main__":
    main()
