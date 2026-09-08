#!/usr/bin/env python3
"""Fit mm_per_pixel via metadata/clusters, then write full submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry.metrics import (
    STATS,
    fascicle_length_px,
    muscle_thickness_px,
    pennation_angle_deg,
    reset_stats,
)
from muscle_arc.geometry.scale import (
    assign_scales,
    cluster_key,
    load_scale_table,
)
from muscle_arc.geometry.depth_scale import (
    estimate_depth_scale,
    load_depth_scale_table,
    osf_shape_scale_lookup,
    share_scales_in_groups,
)
from muscle_arc.geometry.sector_crop import sector_crop
from muscle_arc.geometry.residual_scale import (
    apply_residual_mm,
    load_models as load_residual_models,
    row_features,
)
from muscle_arc.infer.predict import load_sample_submission, predict_mask, predict_prob
from muscle_arc.models.segmentation import build_segmentation_model


def load_model(ckpt: Path, model_cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_segmentation_model(model_cfg).to(device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def sequence_groups(paths: list[Path]) -> list[list[int]]:
    """Group acquisition runs: prefer contiguous numeric stem runs of length ~5.

    Host test set has ~27 runs of exactly 5 frames that share scale. Falls back to
    contiguous same-(H,W) buckets when stems are not numeric sequences.
    """
    import re

    stems = [p.stem for p in paths]
    shapes: list[tuple[int, int] | None] = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        shapes.append(None if im is None else (int(im.shape[0]), int(im.shape[1])))

    # Parse trailing digits → (prefix, frame_idx)
    parsed: list[tuple[str, int] | None] = []
    for s in stems:
        m = re.match(r"^(.*?)(\d+)$", s)
        if m:
            parsed.append((m.group(1), int(m.group(2))))
        else:
            parsed.append(None)

    used = [False] * len(paths)
    groups: list[list[int]] = []

    # Pass 1: contiguous indices with same prefix and consecutive frame numbers
    i = 0
    while i < len(paths):
        if used[i] or parsed[i] is None:
            i += 1
            continue
        prefix, start_n = parsed[i]
        run = [i]
        j = i + 1
        expect = start_n + 1
        while j < len(paths) and not used[j] and parsed[j] is not None:
            pref_j, n_j = parsed[j]
            if pref_j != prefix or n_j != expect:
                break
            # Prefer same shape within a run when readable
            if shapes[i] is not None and shapes[j] is not None and shapes[j] != shapes[i]:
                break
            run.append(j)
            expect += 1
            j += 1
        if len(run) >= 2:
            for k in run:
                used[k] = True
            groups.append(run)
            i = j
        else:
            i += 1

    # Pass 2: leftover contiguous same-shape runs (size capped near 5)
    i = 0
    while i < len(paths):
        if used[i]:
            i += 1
            continue
        run = [i]
        j = i + 1
        while j < len(paths) and not used[j] and shapes[j] == shapes[i] and len(run) < 5:
            run.append(j)
            j += 1
        for k in run:
            used[k] = True
        groups.append(run)
        i = j

    groups.sort(key=lambda g: g[0])
    return groups


def median_smooth_groups(df: pd.DataFrame, groups: list[list[int]], cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for idxs in groups:
        if len(idxs) < 2:
            continue
        for col in cols:
            med = float(np.median(out.iloc[idxs][col].to_numpy()))
            out.iloc[idxs, out.columns.get_loc(col)] = med
    return out


def ema_smooth_groups(
    df: pd.DataFrame, groups: list[list[int]], cols: list[str], alpha: float = 0.45
) -> pd.DataFrame:
    """Light EMA within each sequence group (forward then backward average)."""
    out = df.copy()
    for idxs in groups:
        if len(idxs) < 3:
            continue
        for col in cols:
            vals = out.iloc[idxs][col].to_numpy(dtype=np.float64).copy()
            fwd = vals.copy()
            for i in range(1, len(fwd)):
                fwd[i] = alpha * vals[i] + (1 - alpha) * fwd[i - 1]
            bwd = vals.copy()
            for i in range(len(bwd) - 2, -1, -1):
                bwd[i] = alpha * vals[i] + (1 - alpha) * bwd[i + 1]
            smooth = 0.5 * (fwd + bwd)
            out.iloc[idxs, out.columns.get_loc(col)] = smooth
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--apo-ckpt", type=Path, default=Path("experiments/checkpoints/apo_best.pt"))
    parser.add_argument("--fasc-ckpt", type=Path, default=Path("experiments/checkpoints/fasc_best.pt"))
    parser.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument("--thr", type=float, default=None, help="Legacy shared thr (overrides both if set alone)")
    parser.add_argument("--apo-thr", type=float, default=None, help="Apo mask threshold (default 0.35)")
    parser.add_argument("--fasc-thr", type=float, default=None, help="Fasc mask threshold (default 0.10)")
    parser.add_argument("--debug-csv", type=Path, default=None, help="Per-row debug CSV path")
    parser.add_argument("--temporal-smooth", type=str, default=None, choices=["true", "false"])
    parser.add_argument("--scale-table", type=Path, default=Path("experiments/scale_table.csv"))
    parser.add_argument("--apo-ckpt2", type=Path, default=None, help="Optional 2nd apo for ensemble")
    parser.add_argument("--fasc-ckpt2", type=Path, default=None, help="Optional 2nd fasc for ensemble")
    parser.add_argument(
        "--residual-model-dir",
        type=Path,
        default=Path("experiments/residual_models"),
        help="Phase A' residual mm/scale models (used when metadata absent)",
    )
    parser.add_argument(
        "--residual-prefer",
        type=str,
        default="off",
        choices=["blend", "mm", "scale", "off"],
        help="Phase A' residual on non-cluster rows (default off after v9 LB regression)",
    )
    parser.add_argument(
        "--depth-scale-table",
        type=Path,
        default=Path("experiments/depth_scale_table.csv"),
        help="Per-image OCR/tick scales from audit_depth_scale.py",
    )
    parser.add_argument(
        "--sector-crop",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Crop B-mode sector before segmentation (console screenshots)",
    )
    parser.add_argument(
        "--live-depth-scale",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Estimate depth OCR/ticks live when table missing an image",
    )
    parser.add_argument(
        "--scale-detector",
        type=Path,
        default=None,
        help="Optional trained ScaleStripDetector ckpt; fills gaps when OCR/ticks miss",
    )
    parser.add_argument(
        "--osf-gt",
        type=Path,
        default=Path("experiments/osf_expert_gt.csv"),
        help="OSF expert GT for shape→mm/px lookup on leftover rows",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = DataPaths.from_config(cfg["data"])
    paths.assert_test_present()
    reset_stats()

    apo_model = load_model(args.apo_ckpt, cfg["model"], device)
    fasc_model = load_model(args.fasc_ckpt, cfg["model"], device)
    apo_model2 = load_model(args.apo_ckpt2, cfg["model"], device) if args.apo_ckpt2 else None
    fasc_model2 = load_model(args.fasc_ckpt2, cfg["model"], device) if args.fasc_ckpt2 else None

    infer = cfg["infer"]
    img_size = int(cfg["img_size"])
    # Split thresholds (DL_Track defaults) unless --thr forces shared
    apo_thr = float(args.apo_thr if args.apo_thr is not None else infer.get("apo_thr", 0.35))
    fasc_thr = float(args.fasc_thr if args.fasc_thr is not None else infer.get("fasc_thr", 0.10))
    if args.thr is not None:
        apo_thr = fasc_thr = float(args.thr)
    thr = apo_thr  # legacy alias
    print(f"Thresholds apo={apo_thr:.3f} fasc={fasc_thr:.3f}")
    do_temporal = bool(infer.get("temporal_smooth", True))
    if args.temporal_smooth is not None:
        do_temporal = args.temporal_smooth == "true"
    use_ms_tta = bool(infer.get("multiscale_tta", True))
    frangi_fasc = bool(infer.get("frangi_enhance_fasc", False))

    # Usable metadata scales (placeholder 72dpi already filtered)
    scale_df = load_scale_table(args.scale_table)
    meta_by_id: dict[str, float] = {}
    if len(scale_df) and "scale_usable" in scale_df.columns:
        usable = scale_df[scale_df["scale_usable"]]
        for _, r in usable.iterrows():
            meta_by_id[str(r["image_id"])] = float(r["mm_per_pixel"])
            meta_by_id[str(r["stem"])] = float(r["mm_per_pixel"])
    print(f"Usable metadata scale keys: {len(meta_by_id)}")

    do_sector = args.sector_crop == "true"
    do_live_depth = args.live_depth_scale == "true"
    ocr_by_id, ocr_conf = load_depth_scale_table(
        args.depth_scale_table, return_confidence=True
    )
    print(f"Depth-scale table keys: {len(ocr_by_id)}")

    scale_det = None
    if args.scale_detector is not None and args.scale_detector.exists():
        from muscle_arc.models.scale_detector import load_scale_detector, predict_mm_per_pixel

        scale_det = load_scale_detector(args.scale_detector, device)
        print(f"Loaded scale detector {args.scale_detector}")

    test_paths = list_images(paths.test_images)
    groups = sequence_groups(test_paths)
    rows = []
    live_scales: dict[str, float] = {}
    live_conf: dict[str, float] = {}
    for p in test_paths:
        full = read_gray(p)
        crop_info = sector_crop(full) if do_sector else None
        gray = crop_info.crop if (crop_info is not None and crop_info.applied) else full
        h, w = gray.shape[:2]
        fh, fw = full.shape[:2]

        # Live depth scale on FULL frame (chrome/text outside sector)
        prov_mm: float | None = None
        if p.name in ocr_by_id:
            prov_mm = float(ocr_by_id[p.name])
        elif p.stem in ocr_by_id:
            prov_mm = float(ocr_by_id[p.stem])
        elif do_live_depth:
            est = estimate_depth_scale(full)
            if est.mm_per_pixel is not None and est.confidence >= 0.4:
                live_scales[p.name] = float(est.mm_per_pixel)
                live_scales[p.stem] = float(est.mm_per_pixel)
                live_conf[p.name] = float(est.confidence)
                live_conf[p.stem] = float(est.confidence)
                prov_mm = float(est.mm_per_pixel)
        if prov_mm is None and scale_det is not None:
            try:
                mm_nn = float(predict_mm_per_pixel(scale_det, full, device))
                if 0.025 <= mm_nn <= 0.12:
                    live_scales[p.name] = mm_nn
                    live_scales[p.stem] = mm_nn
                    live_conf[p.name] = 0.50  # CNN gap-fill: below group vote threshold
                    live_conf[p.stem] = 0.50
                    prov_mm = mm_nn
            except Exception as exc:  # noqa: BLE001
                print(f"scale_detector fail {p.name}: {exc}")

        apo_prob = predict_prob(
            apo_model,
            gray,
            img_size,
            device,
            tta_hflip=True,
            use_letterbox=True,
            multiscale=use_ms_tta,
            frangi_enhance=False,
        )
        fasc_prob = predict_prob(
            fasc_model,
            gray,
            img_size,
            device,
            tta_hflip=True,
            use_letterbox=True,
            multiscale=use_ms_tta,
            frangi_enhance=frangi_fasc,
        )
        if apo_model2 is not None:
            apo_prob = 0.5 * (
                apo_prob
                + predict_prob(
                    apo_model2, gray, img_size, device, True, True, use_ms_tta, False
                )
            )
        if fasc_model2 is not None:
            fasc_prob = 0.5 * (
                fasc_prob
                + predict_prob(
                    fasc_model2, gray, img_size, device, True, True, use_ms_tta, frangi_fasc
                )
            )
        apo = (apo_prob > apo_thr).astype(np.uint8)
        fasc = (fasc_prob > fasc_thr).astype(np.uint8)
        if int(fasc.sum()) < 5000:
            fasc = cv2.morphologyEx(fasc, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if int(apo.sum()) > 5000:
            apo = cv2.medianBlur(apo, 5)

        # Pass provisional OCR mm so apo-pair MT gate can reject skin-to-bone
        mt_px = muscle_thickness_px(apo, fasc_mask=fasc, mm_per_pixel=prov_mm)
        pa = pennation_angle_deg(
            fasc, apo, fasc_prob=fasc_prob, gray=gray, mm_per_pixel=prov_mm
        )
        fl_px = fascicle_length_px(
            fasc,
            apo,
            fasc_prob=fasc_prob,
            gray=gray,
            pa_deg=pa if np.isfinite(pa) else None,
            mt_px=mt_px if np.isfinite(mt_px) else None,
            mm_per_pixel=prov_mm,
        )
        # Trig fallback only when PA is stable (matches fascicle_length_px guard)
        if (
            (not np.isfinite(fl_px))
            and np.isfinite(pa)
            and np.isfinite(mt_px)
            and 10 < abs(pa) < 35
        ):
            fl_px = float(mt_px / max(np.sin(np.radians(abs(pa))), 1e-3))
            if np.isfinite(mt_px) and mt_px > 5:
                fl_px = float(np.clip(fl_px, 2.0 * mt_px, 5.5 * mt_px))

        n_fasc = int(cv2.connectedComponents((fasc > 0).astype(np.uint8), connectivity=8)[0]) - 1
        rows.append(
            {
                "image_id": p.name,
                "path": str(p),
                "h": h,
                "w": w,
                "full_h": fh,
                "full_w": fw,
                "shape": f"{h}x{w}",
                "cluster": cluster_key(h, w, gray),
                "kind": crop_info.kind if crop_info is not None else "unknown",
                "crop_applied": bool(crop_info.applied) if crop_info is not None else False,
                "n_fasc_comp": n_fasc,
                "apo_px": int(apo.sum()),
                "fasc_px": int(fasc.sum()),
                "pa_deg": float(pa) if np.isfinite(pa) else np.nan,
                "fl_px": float(fl_px) if np.isfinite(fl_px) else np.nan,
                "mt_px": float(mt_px) if np.isfinite(mt_px) else np.nan,
                **row_features(
                    h,
                    w,
                    float(np.mean(gray)),
                    float(np.std(gray)),
                    float(pa) if np.isfinite(pa) else np.nan,
                    float(fl_px) if np.isfinite(fl_px) else np.nan,
                    float(mt_px) if np.isfinite(mt_px) else np.nan,
                ),
            }
        )
    raw = pd.DataFrame(rows)

    # Merge live OCR into table scales and share within sequence groups.
    # High-conf frames vote; low-conf / missing inherit the group median.
    ocr_merged = dict(ocr_by_id)
    ocr_merged.update(live_scales)
    conf_merged = dict(ocr_conf)
    conf_merged.update(live_conf)
    image_ids = [p.name for p in test_paths]
    ocr_merged = share_scales_in_groups(
        image_ids, ocr_merged, groups, confidences=conf_merged, min_conf=0.55
    )
    print(
        f"OCR/tick scales after live+group: {len(ocr_merged)} keys "
        f"(live={len(live_scales)}, crop_applied={int(raw['crop_applied'].sum())})"
    )

    # OSF shape lookup for leftover primary candidates
    shape_by_hw: dict[tuple[int, int], float] = {}
    for (h, w), g in raw.groupby(["h", "w"]):
        mm = osf_shape_scale_lookup(int(h), int(w), args.osf_gt)
        if mm is not None:
            shape_by_hw[(int(h), int(w))] = float(mm)
    print(f"OSF shape lookup entries: {len(shape_by_hw)}")

    default_scale = float(infer.get("mm_per_pixel", 0.06))
    sample_scales: list[dict] = []
    if paths.sample_submission.exists():
        sample = load_sample_submission(paths.sample_submission, sep=cfg["data"].get("csv_sep", ";"))
        id_col = [c for c in sample.columns if "id" in c.lower()][0]
        for _, srow in sample.iterrows():
            rid = str(srow[id_col])
            stem = Path(rid).stem
            match = raw[raw["image_id"] == rid]
            if match.empty:
                match = raw[raw["image_id"].map(lambda x: Path(x).stem) == stem]
            if match.empty:
                continue
            m = match.iloc[0]
            print(
                f"GT {rid} ({m['cluster']}): pa={srow['pa_deg']} fl={srow['fl_mm']} mt={srow['mt_mm']} | "
                f"pred pa={m['pa_deg']:.2f} fl_px={m['fl_px']:.1f} mt_px={m['mt_px']:.1f}"
            )
            entry = {"image_id": rid, "cluster": str(m["cluster"])}
            if np.isfinite(m["mt_px"]) and m["mt_px"] > 1 and float(srow["mt_mm"]) > 0:
                entry["mt_scale"] = float(srow["mt_mm"]) / float(m["mt_px"])
            if np.isfinite(m["fl_px"]) and m["fl_px"] > 1 and float(srow["fl_mm"]) > 0:
                entry["fl_scale"] = float(srow["fl_mm"]) / float(m["fl_px"])
            if "mt_scale" in entry and "fl_scale" in entry:
                sample_scales.append(entry)
            elif "mt_scale" in entry:
                entry["fl_scale"] = entry["mt_scale"]
                sample_scales.append(entry)

    mt_scales, fl_scales, info = assign_scales(
        raw,
        sample_scales,
        default_scale,
        meta_by_id,
        ocr_by_id=ocr_merged,
        shape_by_hw=shape_by_hw,
    )
    print("Scale assignment:", {k: v for k, v in info.items() if k != "sources"})
    sources = info.get("sources") or ["primary"] * len(raw)

    out = raw.copy()
    out["mt_mm"] = [
        (row.mt_px * mt_scales.iloc[i]) if np.isfinite(row.mt_px) else np.nan
        for i, row in enumerate(out.itertuples())
    ]
    out["fl_mm"] = [
        (row.fl_px * fl_scales.iloc[i]) if np.isfinite(row.fl_px) else np.nan
        for i, row in enumerate(out.itertuples())
    ]
    out["scale_source"] = sources

    # Phase A': residual — never overwrite meta/ocr/cluster
    residual_models = load_residual_models(args.residual_model_dir)
    apply_on = {"primary", "cluster_prefix", "shape"}
    if args.residual_prefer != "off" and (
        residual_models.get("mm") is not None or residual_models.get("scale") is not None
    ):
        fl_res, mt_res, rinfo = apply_residual_mm(out, residual_models, prefer=args.residual_prefer)
        n_replaced = 0
        for i, src in enumerate(sources):
            if src in {"meta", "ocr", "cluster"}:
                continue
            if src not in apply_on and args.residual_prefer != "mm":
                continue
            if np.isfinite(fl_res.iloc[i]):
                out.iloc[i, out.columns.get_loc("fl_mm")] = float(fl_res.iloc[i])
            if np.isfinite(mt_res.iloc[i]):
                out.iloc[i, out.columns.get_loc("mt_mm")] = float(mt_res.iloc[i])
            out.iloc[i, out.columns.get_loc("scale_source")] = f"residual_{args.residual_prefer}"
            n_replaced += 1
        print(f"Residual models applied to {n_replaced}/{len(out)} non-ocr rows:", rinfo)
    else:
        print("Residual models skipped (off or missing)")

    # Sample-GT gain correction DELETED (n=2 public-fit; v9/v14c lesson).
    n_ocr = int(info.get("n_ocr", 0))
    print(f"Sample residual gains disabled (n_ocr={n_ocr})")

    out["pa_deg"] = out["pa_deg"].fillna(15.0)
    out["fl_mm"] = out["fl_mm"].fillna(70.0)
    out["mt_mm"] = out["mt_mm"].fillna(20.0)

    clip = infer["clip"]
    out["pa_deg"] = out["pa_deg"].clip(*clip["pa_deg"])
    out["fl_mm"] = out["fl_mm"].clip(*clip["fl_mm"])
    out["mt_mm"] = out["mt_mm"].clip(*clip["mt_mm"])

    out_df = out[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()

    if do_temporal:
        # With per-image OCR scales, median wipe on FL/MT destroys depth diversity.
        # Only smooth PA (and lightly EMA FL/MT) when OCR coverage is high.
        if n_ocr >= max(20, int(0.25 * len(out))):
            out_df = ema_smooth_groups(out_df, groups, ["pa_deg"], alpha=0.35)
            print(f"Temporal PA-only EMA over {len(groups)} groups (OCR scales preserved)")
        else:
            out_df = ema_smooth_groups(out_df, groups, ["pa_deg", "fl_mm", "mt_mm"])
            out_df = median_smooth_groups(out_df, groups, ["pa_deg", "fl_mm", "mt_mm"])
            print(f"Temporal EMA+median over {len(groups)} groups")
    else:
        print("Temporal smooth disabled")

    out_df = out_df.sort_values("image_id").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out_df)} rows)")
    print(out_df.describe().to_string())
    print(STATS.summary())
    for col, lo, hi in [
        ("pa_deg", *clip["pa_deg"]),
        ("fl_mm", *clip["fl_mm"]),
        ("mt_mm", *clip["mt_mm"]),
    ]:
        n_lo = int((out_df[col] <= lo + 1e-6).sum())
        n_hi = int((out_df[col] >= hi - 1e-6).sum())
        print(f"clip hits {col}: low={n_lo} high={n_hi}")

    debug_path = args.debug_csv or args.out.with_name(args.out.stem + "_debug.csv")
    dbg_cols = [
        c
        for c in (
            "image_id",
            "kind",
            "crop_applied",
            "cluster",
            "h",
            "w",
            "n_fasc_comp",
            "apo_px",
            "fasc_px",
            "fl_px",
            "mt_px",
            "scale_source",
            "pa_deg",
            "fl_mm",
            "mt_mm",
        )
        if c in out.columns or c in out_df.columns
    ]
    dbg = out.copy()
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        dbg[c] = out_df.set_index("image_id").reindex(dbg["image_id"])[c].to_numpy()
    dbg[[c for c in dbg_cols if c in dbg.columns]].to_csv(debug_path, index=False)
    print(f"Wrote debug {debug_path}")

    if paths.sample_submission.exists():
        sample = load_sample_submission(paths.sample_submission, sep=cfg["data"].get("csv_sep", ";"))
        id_col = [c for c in sample.columns if "id" in c.lower()][0]
        for _, srow in sample.iterrows():
            rid = str(srow[id_col])
            match = out_df[out_df["image_id"] == rid]
            if match.empty:
                match = out_df[out_df["image_id"].map(lambda x: Path(x).stem) == Path(rid).stem]
            if match.empty:
                continue
            r = match.iloc[0]
            print(
                f"GATE_B {rid}: PA_err={abs(float(r.pa_deg)-float(srow['pa_deg'])):.2f} "
                f"FL_err={abs(float(r.fl_mm)-float(srow['fl_mm'])):.2f} "
                f"MT_err={abs(float(r.mt_mm)-float(srow['mt_mm'])):.2f}"
            )


if __name__ == "__main__":
    main()
