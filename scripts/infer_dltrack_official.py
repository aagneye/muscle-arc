#!/usr/bin/env python3
"""DL_Track letterbox + chrome-mask + official doCalculations geometry."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from muscle_arc.data.dataset import list_images, read_gray  # noqa: E402
from muscle_arc.data.paths import DataPaths  # noqa: E402
from muscle_arc.geometry.depth_scale import (  # noqa: E402
    build_osf_shape_scale_map,
    load_depth_scale_table,
    share_scales_in_groups,
)
from muscle_arc.geometry.sector_crop import mask_chrome  # noqa: E402
from muscle_arc.models.dl_track import load_dl_track  # noqa: E402
from muscle_arc.models.dl_track_official import official_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("submissions/submission_dltrack_win.csv"),
    )
    ap.add_argument("--dl-track-dir", type=Path, default=Path("data/external/dl_track"))
    ap.add_argument(
        "--depth-scale-table",
        type=Path,
        default=Path("experiments/depth_scale_table.csv"),
    )
    ap.add_argument("--osf-gt", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    ap.add_argument("--apo-thr", type=float, default=0.35)
    ap.add_argument("--fasc-thr", type=float, default=0.10)
    ap.add_argument(
        "--chrome-mask",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_test_present()

    spec = importlib.util.spec_from_file_location(
        "calibrate_predict", Path("scripts/calibrate_predict.py")
    )
    cal = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cal)

    seg = load_dl_track(args.dl_track_dir, img_size=512)

    ocr_by_id, ocr_conf = load_depth_scale_table(
        args.depth_scale_table, return_confidence=True
    )
    build_osf_shape_scale_map(args.osf_gt)
    test_paths = list_images(paths.test_images)
    groups = cal.sequence_groups(test_paths)
    image_ids = [p.name for p in test_paths]
    ocr_merged = share_scales_in_groups(
        image_ids, dict(ocr_by_id), groups, confidences=ocr_conf, min_conf=0.55
    )

    rows = []
    n_masked = 0
    defaults = cfg["infer"]["clip"]
    for p in test_paths:
        full = read_gray(p)
        if args.chrome_mask == "true":
            gray, info = mask_chrome(full)
            if info.applied:
                n_masked += 1
        else:
            gray = full
        apo, fasc, _ap, _fp = seg.predict_masks(
            gray, apo_thr=args.apo_thr, fasc_thr=args.fasc_thr
        )
        h, w = full.shape[:2]
        if apo.shape[:2] != (h, w):
            import cv2

            apo = cv2.resize(apo, (w, h), interpolation=cv2.INTER_NEAREST)
            fasc = cv2.resize(fasc, (w, h), interpolation=cv2.INTER_NEAREST)

        mm = ocr_merged.get(p.name, ocr_merged.get(p.stem))
        if mm is None or not (0.025 <= float(mm) <= 0.16):
            mm = float(cfg["infer"].get("mm_per_pixel", 0.06))
        try:
            pa, fl, mt = official_metrics(
                apo,
                fasc,
                full,
                mm_per_pixel=float(mm),
                apo_thr=args.apo_thr,
                fasc_thr=args.fasc_thr,
                model_apo=seg.apo,
                model_fasc=seg.fasc,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"fail {p.name}: {exc}")
            pa = fl = mt = float("nan")
        rows.append({"image_id": p.name, "pa_deg": pa, "fl_mm": fl, "mt_mm": mt})
        if len(rows) % 25 == 0:
            print(f"{len(rows)}/{len(test_paths)} masked_so_far={n_masked}", flush=True)

    out = pd.DataFrame(rows)
    for col, (lo, hi) in (
        ("pa_deg", defaults["pa_deg"]),
        ("fl_mm", defaults["fl_mm"]),
        ("mt_mm", defaults["mt_mm"]),
    ):
        med = (
            float(out[col].median())
            if out[col].notna().any()
            else float((lo + hi) / 2)
        )
        out[col] = out[col].fillna(med).clip(lo, hi)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out, "n", len(out), "chrome_masked", n_masked)
    print(out.describe())


if __name__ == "__main__":
    main()
