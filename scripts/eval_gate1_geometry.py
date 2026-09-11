#!/usr/bin/env python3
"""Gate 1: scale-free geometry MAE — GT masks vs predicted masks (PA / FL_px / MT_px).

Default exam set is the sequence-safe **holdout** split (final exam never used
to pick checkpoints). Isolates drawing quality from mm/px.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from muscle_arc.data.dataset import pair_images_masks, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.data.splits import load_split_manifest
from muscle_arc.geometry.metrics import (
    STATS,
    fascicle_length_px,
    muscle_thickness_px,
    pennation_angle_deg,
    reset_stats,
)
from muscle_arc.infer.predict import predict_prob
from muscle_arc.models.segmentation import build_segmentation_model


def _read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if shape is not None and m.shape[:2] != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _geom(apo: np.ndarray, fasc: np.ndarray, gray: np.ndarray | None = None) -> tuple[float, float, float]:
    mt = muscle_thickness_px(apo, fasc_mask=fasc)
    pa = pennation_angle_deg(fasc, apo, gray=gray)
    fl = fascicle_length_px(
        fasc,
        apo,
        gray=gray,
        pa_deg=pa if np.isfinite(pa) else None,
        mt_px=mt if np.isfinite(mt) else None,
    )
    return (
        float(pa) if np.isfinite(pa) else float("nan"),
        float(fl) if np.isfinite(fl) else float("nan"),
        float(mt) if np.isfinite(mt) else float("nan"),
    )


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
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all holdout / sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apo-ckpt", type=Path, default=Path("experiments/checkpoints_grow/apo_best.pt"))
    parser.add_argument("--fasc-ckpt", type=Path, default=Path("experiments/checkpoints_grow/fasc_best.pt"))
    parser.add_argument("--apo-thr", type=float, default=0.35)
    parser.add_argument("--fasc-thr", type=float, default=0.10)
    parser.add_argument("--split-dir", type=Path, default=Path("experiments/splits"))
    parser.add_argument(
        "--split",
        choices=("holdout", "val", "train", "all"),
        default="holdout",
        help="Which split to exam (default: holdout final exam)",
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/gate1_geometry.json"))
    parser.add_argument("--baseline", type=Path, default=None, help="Optional prior Gate1 JSON to compare")
    parser.add_argument("--gt-only", action="store_true", help="Skip predicted masks (legacy)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_train_present()
    reset_stats()

    fasc_pairs = pair_images_masks(paths.fasc_imgs, paths.fasc_masks)
    apo_pairs = pair_images_masks(paths.apo_imgs, paths.apo_masks)
    apo_by_stem = {p[0].stem: p for p in apo_pairs}
    fasc_by_stem = {p[0].stem: p for p in fasc_pairs}
    common = sorted(set(fasc_by_stem) & set(apo_by_stem))

    sample: list[str]
    if args.split == "all":
        sample = list(common)
        split_note = "all_matched"
    else:
        apo_man = args.split_dir / "apo_splits.json"
        fasc_man = args.split_dir / "fasc_splits.json"
        if apo_man.exists() and fasc_man.exists():
            apo_s = set(load_split_manifest(apo_man)["splits"][args.split])
            fasc_s = set(load_split_manifest(fasc_man)["splits"][args.split])
            sample = sorted((apo_s & fasc_s) & set(common))
            split_note = f"{args.split}_intersect"
        else:
            rng = np.random.default_rng(args.seed)
            sample = list(common)
            rng.shuffle(sample)
            n = args.max_samples if args.max_samples > 0 else min(160, len(sample))
            sample = sample[:n]
            split_note = f"fallback_random_{args.split}"
            print(f"WARN missing split manifests under {args.split_dir}; using {split_note}")

    if args.max_samples > 0 and len(sample) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(sample)
        sample = sample[: args.max_samples]

    print(f"Gate1 split={split_note} stems: {len(sample)} / matched={len(common)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apo_model = fasc_model = None
    if not args.gt_only:
        if not args.apo_ckpt.exists() or not args.fasc_ckpt.exists():
            print("WARN missing ckpts; falling back to --gt-only")
            args.gt_only = True
        else:
            apo_model = _load_model(args.apo_ckpt, cfg["model"], device)
            fasc_model = _load_model(args.fasc_ckpt, cfg["model"], device)

    img_size = int(cfg["img_size"])
    pa_e, fl_e, mt_e = [], [], []
    gt_pa, gt_fl, gt_mt = [], [], []

    for stem in sample:
        fasc_img = fasc_by_stem[stem][0]
        fasc_m = fasc_by_stem[stem][1]
        apo_m = apo_by_stem[stem][1]
        gray = read_gray(fasc_img)
        apo_gt = _read_mask(apo_m, gray.shape[:2])
        fasc_gt = _read_mask(fasc_m, gray.shape[:2])
        pa_g, fl_g, mt_g = _geom(apo_gt, fasc_gt, gray=gray)
        gt_pa.append(pa_g)
        gt_fl.append(fl_g)
        gt_mt.append(mt_g)

        if args.gt_only or apo_model is None:
            continue

        apo_prob = predict_prob(apo_model, gray, img_size, device, True, True, False, False)
        fasc_prob = predict_prob(fasc_model, gray, img_size, device, True, True, False, False)
        apo_pr = (apo_prob > args.apo_thr).astype(np.uint8)
        fasc_pr = (fasc_prob > args.fasc_thr).astype(np.uint8)
        pa_p, fl_p, mt_p = _geom(apo_pr, fasc_pr, gray=gray)
        if np.isfinite(pa_g) and np.isfinite(pa_p):
            pa_e.append(abs(pa_p - pa_g))
        if np.isfinite(fl_g) and np.isfinite(fl_p) and fl_g > 1:
            fl_e.append(abs(fl_p - fl_g) / fl_g)
        if np.isfinite(mt_g) and np.isfinite(mt_p) and mt_g > 1:
            mt_e.append(abs(mt_p - mt_g) / mt_g)

    report = {
        "n": len(sample),
        "split": args.split,
        "split_note": split_note,
        "gt_pa_median": float(np.nanmedian(gt_pa)) if gt_pa else float("nan"),
        "gt_fl_px_median": float(np.nanmedian(gt_fl)) if gt_fl else float("nan"),
        "gt_mt_px_median": float(np.nanmedian(gt_mt)) if gt_mt else float("nan"),
        "mode": "gt_only" if args.gt_only else "gt_vs_pred",
        "apo_ckpt": str(args.apo_ckpt),
        "fasc_ckpt": str(args.fasc_ckpt),
    }
    if not args.gt_only:
        report["pa_mae_deg"] = float(np.mean(pa_e)) if pa_e else float("nan")
        report["fl_rel_mae"] = float(np.mean(fl_e)) if fl_e else float("nan")
        report["mt_rel_mae"] = float(np.mean(mt_e)) if mt_e else float("nan")
        report["n_pa"] = len(pa_e)
        report["n_fl"] = len(fl_e)
        report["n_mt"] = len(mt_e)
        geom_ok = (
            (report["pa_mae_deg"] < 5.0 if pa_e else False)
            and (report["fl_rel_mae"] < 0.35 if fl_e else False)
            and (report["mt_rel_mae"] < 0.35 if mt_e else False)
        )
        report["geom_ok"] = bool(geom_ok)
        report["gate1_ok"] = bool(geom_ok)
        report["route"] = "scale_first" if geom_ok else "geometry_first"

        if args.baseline is not None and args.baseline.exists():
            base = json.loads(args.baseline.read_text())
            report["baseline_path"] = str(args.baseline)
            improved = True
            for key in ("pa_mae_deg", "fl_rel_mae", "mt_rel_mae"):
                if key in base and key in report and np.isfinite(report[key]) and np.isfinite(base[key]):
                    report[f"delta_{key}"] = float(report[key] - base[key])
                    if report[key] > base[key] * 1.02:
                        improved = False
            report["vs_baseline_improved"] = bool(improved)
            report["gate1_ok"] = bool(geom_ok or improved)
    else:
        report["geom_ok"] = None
        report["gate1_ok"] = False
        report["route"] = "unknown_gt_only"

    print(json.dumps(report, indent=2))
    print(STATS.summary())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")
    print("GATE1_OK" if report.get("gate1_ok") else "GATE1_FAIL")
    if report.get("route") == "geometry_first":
        print("GATE1_ROUTE geometry_first")
    elif report.get("route") == "scale_first":
        print("GATE1_ROUTE scale_first")
    else:
        print("GATE1_ROUTE unknown")
    raise SystemExit(0 if report.get("gate1_ok") else 1)


if __name__ == "__main__":
    main()
