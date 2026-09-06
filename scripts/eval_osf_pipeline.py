#!/usr/bin/env python3
"""Gate 2: full pipeline on OSF expert images → official UMUD score.

Runs sector_crop → depth scale → predict → geometry → mm, vs osf_expert_gt.csv.
Does NOT use osf_shape_scale_lookup as the validator path.

Also reports per-image mt_ratio / fl_ratio so we can count images that escape
the skin-to-bone failure bucket (target: <5/32 outside 0.9–1.1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from muscle_arc.data.dataset import read_gray
from muscle_arc.geometry.depth_scale import estimate_depth_scale
from muscle_arc.geometry.metrics import (
    fascicle_length_px,
    muscle_thickness_px,
    pennation_angle_deg,
    reset_stats,
)
from muscle_arc.geometry.sector_crop import sector_crop
from muscle_arc.geometry.umud_metric import umud_from_errors
from muscle_arc.infer.predict import predict_prob
from muscle_arc.models.segmentation import build_segmentation_model


def _load_model(ckpt: Path, model_cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_segmentation_model(model_cfg).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--osf-gt", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    parser.add_argument("--apo-ckpt", type=Path, default=Path("experiments/checkpoints_phase4/apo_best.pt"))
    parser.add_argument("--fasc-ckpt", type=Path, default=Path("experiments/checkpoints_phase4/fasc_best.pt"))
    parser.add_argument("--apo-thr", type=float, default=0.35)
    parser.add_argument("--fasc-thr", type=float, default=0.10)
    parser.add_argument("--sector-crop", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--out", type=Path, default=Path("experiments/gate2_osf_umud.json"))
    parser.add_argument("--max-images", type=int, default=35)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    gt = pd.read_csv(args.osf_gt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apo_model = _load_model(args.apo_ckpt, cfg["model"], device)
    fasc_model = _load_model(args.fasc_ckpt, cfg["model"], device)
    img_size = int(cfg["img_size"])
    do_sector = args.sector_crop == "true"
    reset_stats()

    pa_err, fl_err, mt_err = [], [], []
    rows = []
    n_scale = 0
    for _, s in gt.head(args.max_images).iterrows():
        path = Path(str(s.get("path", "")))
        if not path.exists():
            print(f"skip missing {path}")
            continue
        full = read_gray(path)
        crop_info = sector_crop(full) if do_sector else None
        gray = crop_info.crop if (crop_info is not None and crop_info.applied) else full

        est = estimate_depth_scale(full)
        gt_mm = float(s["mm_per_pixel"]) if pd.notna(s.get("mm_per_pixel")) else None
        if gt_mm is not None and gt_mm > 0:
            mm = gt_mm
            scale_src = "osf_gt"
        elif est.mm_per_pixel is not None:
            mm = float(est.mm_per_pixel)
            scale_src = est.source
            n_scale += 1
        else:
            mm = 0.06
            scale_src = "default"

        apo_prob = predict_prob(apo_model, gray, img_size, device, True, True, False, False)
        fasc_prob = predict_prob(fasc_model, gray, img_size, device, True, True, False, False)
        apo = (apo_prob > args.apo_thr).astype(np.uint8)
        fasc = (fasc_prob > args.fasc_thr).astype(np.uint8)

        mt_px = muscle_thickness_px(apo, fasc_mask=fasc, mm_per_pixel=mm)
        pa = pennation_angle_deg(
            fasc, apo, fasc_prob=fasc_prob, gray=gray, mm_per_pixel=mm
        )
        fl_px = fascicle_length_px(
            fasc,
            apo,
            fasc_prob=fasc_prob,
            gray=gray,
            pa_deg=pa if np.isfinite(pa) else None,
            mt_px=mt_px if np.isfinite(mt_px) else None,
            mm_per_pixel=mm,
        )
        fl_mm = float(fl_px * mm) if np.isfinite(fl_px) else float("nan")
        mt_mm = float(mt_px * mm) if np.isfinite(mt_px) else float("nan")
        pa_deg = float(pa) if np.isfinite(pa) else float("nan")

        pa_gt, fl_gt, mt_gt = float(s["pa_deg"]), float(s["fl_mm"]), float(s["mt_mm"])
        if np.isfinite(pa_deg):
            pa_err.append(abs(pa_deg - pa_gt))
        if np.isfinite(fl_mm):
            fl_err.append(abs(fl_mm - fl_gt))
        if np.isfinite(mt_mm):
            mt_err.append(abs(mt_mm - mt_gt))

        mt_ratio = float(mt_mm / mt_gt) if np.isfinite(mt_mm) and mt_gt > 0 else float("nan")
        fl_ratio = float(fl_mm / fl_gt) if np.isfinite(fl_mm) and fl_gt > 0 else float("nan")
        pa_e = abs(pa_deg - pa_gt) if np.isfinite(pa_deg) else float("nan")

        rows.append(
            {
                "image_id": s.get("image_id", path.name),
                "pa_deg": pa_deg,
                "fl_mm": fl_mm,
                "mt_mm": mt_mm,
                "pa_gt": pa_gt,
                "fl_gt": fl_gt,
                "mt_gt": mt_gt,
                "pa_err": pa_e,
                "mt_ratio": mt_ratio,
                "fl_ratio": fl_ratio,
                "mm_per_pixel": mm,
                "scale_src": scale_src,
                "est_mm": est.mm_per_pixel,
                "est_src": est.source,
            }
        )

    df = pd.DataFrame(rows)
    metrics = umud_from_errors(
        np.asarray(pa_err, dtype=float),
        np.asarray(fl_err, dtype=float),
        np.asarray(mt_err, dtype=float),
    )
    metrics["n_est_scale"] = n_scale
    metrics["n_rows"] = len(rows)

    mt_ok = df["mt_ratio"].notna() & (df["mt_ratio"] >= 0.9) & (df["mt_ratio"] <= 1.1)
    fl_ok = df["fl_ratio"].notna() & (df["fl_ratio"] >= 0.9) & (df["fl_ratio"] <= 1.1)
    n_scored = int(df["mt_ratio"].notna().sum())
    n_mt_bad = int((~mt_ok & df["mt_ratio"].notna()).sum())
    n_fl_bad = int((~fl_ok & df["fl_ratio"].notna()).sum())
    metrics["n_mt_ok"] = int(mt_ok.sum())
    metrics["n_mt_bad"] = n_mt_bad
    metrics["n_fl_ok"] = int(fl_ok.sum())
    metrics["n_fl_bad"] = n_fl_bad
    metrics["mt_ratio_median"] = float(df["mt_ratio"].median()) if n_scored else float("nan")
    metrics["fl_ratio_median"] = float(df["fl_ratio"].median()) if n_scored else float("nan")

    # Pass: UMUD < 0.50 and fewer than 5 images with mt_ratio outside 0.9–1.1
    metrics["gate2_ok"] = bool(
        metrics["n"] >= 10 and metrics["umud"] < 0.50 and n_mt_bad < 5
    )

    print(json.dumps(metrics, indent=2))
    print("\nper-image ratios (mt_ratio | fl_ratio | pa_err):")
    for _, r in df.iterrows():
        flag = "OK" if 0.9 <= r["mt_ratio"] <= 1.1 else "BAD"
        print(
            f"  {flag} {r['image_id']}: mt={r['mt_ratio']:.2f} fl={r['fl_ratio']:.2f} "
            f"pa_err={r['pa_err']:.2f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"Wrote {args.out}")
    print(f"mt_bad={n_mt_bad}/{n_scored} fl_bad={n_fl_bad}/{n_scored}")
    print("GATE2_OK" if metrics["gate2_ok"] else "GATE2_FAIL")
    raise SystemExit(0 if metrics["gate2_ok"] else 1)


if __name__ == "__main__":
    main()
