#!/usr/bin/env python3
"""Run inference and write Kaggle submission CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from muscle_arc.data.paths import DataPaths
from muscle_arc.infer.predict import (
    load_sample_submission,
    run_folder_inference,
    temporal_smooth,
)
from muscle_arc.models.segmentation import build_segmentation_model


def load_model(ckpt: Path, model_cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_segmentation_model(model_cfg).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--apo-ckpt", type=Path, default=Path("experiments/checkpoints/apo_best.pt"))
    parser.add_argument(
        "--fasc-ckpt", type=Path, default=Path("experiments/checkpoints/fasc_best.pt")
    )
    parser.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    device_name = cfg.get("device", "cuda")
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )

    paths = DataPaths.from_config(cfg["data"])
    paths.assert_test_present()

    apo_model = load_model(args.apo_ckpt, cfg["model"], device)
    fasc_model = load_model(args.fasc_ckpt, cfg["model"], device)

    infer_cfg = cfg["infer"]
    pred = run_folder_inference(
        apo_model=apo_model,
        fasc_model=fasc_model,
        image_dir=paths.test_images,
        img_size=int(cfg["img_size"]),
        device=device,
        mm_per_pixel=float(infer_cfg["mm_per_pixel"]),
        clip=infer_cfg["clip"],
        tta_hflip=bool(infer_cfg.get("tta_hflip", True)),
        mask_percentile=float(infer_cfg.get("mask_percentile", 75)),
    )

    if paths.sample_submission.exists():
        sample = load_sample_submission(
            paths.sample_submission, sep=cfg["data"].get("csv_sep", ";")
        )
        id_col = [c for c in sample.columns if "id" in c.lower()][0]
        # Align to sample order / ids
        pred = pred.set_index("image_id")
        ordered = []
        for raw_id in sample[id_col].astype(str):
            key = Path(raw_id).stem
            if key in pred.index:
                row = pred.loc[key]
            else:
                # try full string
                row = pred.loc[raw_id] if raw_id in pred.index else None
            if row is None:
                ordered.append(
                    {
                        "image_id": raw_id,
                        "pa_deg": 20.0,
                        "fl_mm": 100.0,
                        "mt_mm": 25.0,
                    }
                )
            else:
                ordered.append(
                    {
                        "image_id": raw_id,
                        "pa_deg": float(row["pa_deg"]),
                        "fl_mm": float(row["fl_mm"]),
                        "mt_mm": float(row["mt_mm"]),
                    }
                )
        out_df = pd.DataFrame(ordered)
    else:
        out_df = pred

    if infer_cfg.get("temporal_smooth", True):
        out_df = temporal_smooth(out_df, window=int(infer_cfg.get("temporal_window", 5)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
