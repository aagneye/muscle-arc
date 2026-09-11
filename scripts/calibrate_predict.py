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
from muscle_arc.geometry.scale_fusion import ScaleCandidate, confidence_weighted_vote
from muscle_arc.geometry.sector_crop import mask_chrome, sector_crop
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


def _tick_keypoint_estimate(
    model,
    gray: "np.ndarray",
    device: "torch.device",
) -> tuple[float | None, float]:
    """Run TickKeypointNet on the four border strips of `gray`, return the
    highest-confidence (mm_per_pixel, confidence) among them.

    Mirrors the border-strip convention already used by
    scale_detector.extract_border_strips, but feeds each 1D max-intensity
    profile through the heatmap keypoint model instead of a scalar-regression
    CNN. See src/muscle_arc/models/tick_keypoint.py for the detection +
    RulerNet-Median pitch recovery, and docs/research_scale_reader.md for the
    design rationale.
    """
    import cv2

    from muscle_arc.models.tick_keypoint import pitch_to_mm, predict_pitch_px

    h, w = gray.shape[:2]
    s = max(8, min(32, h // 4, w // 4))
    if s < 8:
        return None, 0.0
    strips = {
        "left": gray[:, :s],
        "right": gray[:, w - s :],
        "top": gray[:s, :].T,
        "bottom": gray[h - s :, :].T,
    }
    best_mm: float | None = None
    best_conf = 0.0
    for region in strips.values():
        if region.size == 0:
            continue
        profile = region.max(axis=1).astype(np.float32)
        profile_resized = cv2.resize(profile[None, :], (256, 1), interpolation=cv2.INTER_LINEAR)[0]
        profile_norm = profile_resized / 255.0
        pitch_px, conf = predict_pitch_px(model, profile_norm, device)
        if pitch_px is None or conf <= 0:
            continue
        # Undo the 256-resize to recover true pixel pitch on this strip.
        scale = 256 / max(region.shape[0], 1)
        pitch_px_native = pitch_px / scale
        mm, _src = pitch_to_mm(pitch_px_native)
        if not (0.025 <= mm <= 0.16):
            continue
        if conf > best_conf:
            best_conf, best_mm = conf, mm
    return best_mm, best_conf


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
        default="false",
        choices=["true", "false", "auto"],
        help="Crop B-mode sector: true=always, false=never, auto=console chrome only",
    )
    parser.add_argument(
        "--chrome-mask",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Zero console chrome in-place (no crop) before masks; keeps full-frame coords",
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
        "--tick-keypoint-ckpt",
        type=Path,
        default=None,
        help=(
            "Optional trained TickKeypointNet ckpt (scripts/train_tick_keypoint.py); "
            "off by default. When present, joins the scale_fusion jury alongside "
            "OCR/ticks/OSF-shape/scale_det instead of a single hardcoded fallback."
        ),
    )
    parser.add_argument(
        "--sof-ckpt",
        type=Path,
        default=None,
        help="Optional SOF multitask ckpt (uses apo_bin/fasc_bin instead of dual Unets)",
    )
    parser.add_argument(
        "--masks-from",
        type=str,
        default="ours",
        choices=["ours", "dl_track"],
        help="ours=SOF/dual U-Nets; dl_track=Gate0 reference VGG-UNets",
    )
    parser.add_argument(
        "--geom",
        type=str,
        default="auto",
        choices=["auto", "ours", "official"],
        help="auto=official when masks-from=dl_track else ours; official=doCalculations",
    )
    parser.add_argument(
        "--dl-track-dir",
        type=Path,
        default=Path("data/external/dl_track"),
        help="Directory with DL_Track apo/fasc .h5 weights",
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

    sof_model = None
    apo_model = fasc_model = apo_model2 = fasc_model2 = None
    dl_track = None
    if args.masks_from == "dl_track":
        from muscle_arc.models.dl_track import load_dl_track

        dl_track = load_dl_track(args.dl_track_dir, img_size=int(cfg["img_size"]))
        print(f"Loaded DL_Track from {args.dl_track_dir}")
    elif args.sof_ckpt is not None and args.sof_ckpt.exists():
        from muscle_arc.models.multitask import build_sof_model

        sof_model = build_sof_model(cfg).to(device)
        payload = torch.load(args.sof_ckpt, map_location=device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        sof_model.load_state_dict(state, strict=False)
        sof_model.eval()
        print(f"Loaded SOF multitask {args.sof_ckpt}")
    else:
        apo_model = load_model(args.apo_ckpt, cfg["model"], device)
        fasc_model = load_model(args.fasc_ckpt, cfg["model"], device)
        apo_model2 = load_model(args.apo_ckpt2, cfg["model"], device) if args.apo_ckpt2 else None
        fasc_model2 = load_model(args.fasc_ckpt2, cfg["model"], device) if args.fasc_ckpt2 else None

    geom_mode = args.geom
    if geom_mode == "auto":
        # Official doCalculations is available via --geom official; OSF Gate2b
        # ≤0.42 is currently met by our geometry (official ~0.78 on pred scale).
        geom_mode = "ours"
    print(f"Geometry mode: {geom_mode}")
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

    # sector-crop: false=off, true=force crop, auto=console chrome only (sector_crop default)
    do_live_depth = args.live_depth_scale == "true"
    ocr_by_id, ocr_conf = load_depth_scale_table(
        args.depth_scale_table, return_confidence=True
    )
    print(f"Depth-scale table keys: {len(ocr_by_id)}")
    # Warm OSF shape→scale cache once (avoid per-image TIFF re-reads).
    from muscle_arc.geometry.depth_scale import build_osf_shape_scale_map

    _shape_map = build_osf_shape_scale_map(args.osf_gt)
    print(f"OSF shape cache warmed: {len(_shape_map)} shapes")

    scale_det = None
    if args.scale_detector is not None and args.scale_detector.exists():
        from muscle_arc.models.scale_detector import load_scale_detector, predict_mm_per_pixel

        scale_det = load_scale_detector(args.scale_detector, device)
        print(f"Loaded scale detector {args.scale_detector}")

    tick_kp_model = None
    if args.tick_keypoint_ckpt is not None and args.tick_keypoint_ckpt.exists():
        from muscle_arc.models.tick_keypoint import load_tick_keypoint_net

        tick_kp_model = load_tick_keypoint_net(args.tick_keypoint_ckpt, device)
        print(f"Loaded TickKeypointNet {args.tick_keypoint_ckpt}")

    test_paths = list_images(paths.test_images)
    groups = sequence_groups(test_paths)
    rows = []
    live_scales: dict[str, float] = {}
    live_conf: dict[str, float] = {}
    for p in test_paths:
        full = read_gray(p)
        if args.sector_crop == "false":
            crop_info = None
            gray = full
        elif args.sector_crop == "true":
            crop_info = sector_crop(full, force=True)
            gray = crop_info.crop if crop_info.applied else full
        else:
            crop_info = sector_crop(full, force=False)
            gray = crop_info.crop if (crop_info is not None and crop_info.applied) else full
        # Optional: mask chrome without cropping (default on; preferred with DL_Track)
        if args.chrome_mask == "true" and args.sector_crop == "false":
            gray, crop_info = mask_chrome(full)
        elif args.chrome_mask == "true" and args.sector_crop == "auto":
            # auto crop off for win path: chrome-mask only
            gray, crop_info = mask_chrome(full)
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
            jury_candidates = []
            if est.mm_per_pixel is not None and est.confidence > 0:
                jury_candidates.append(
                    ScaleCandidate(est.source, float(est.mm_per_pixel), float(est.confidence))
                )
            if scale_det is not None:
                try:
                    mm_nn = float(predict_mm_per_pixel(scale_det, full, device))
                    if 0.025 <= mm_nn <= 0.12:
                        jury_candidates.append(ScaleCandidate("scale_det_cnn", mm_nn, 0.50))
                except Exception as exc:  # noqa: BLE001
                    print(f"scale_detector fail {p.name}: {exc}")
            if tick_kp_model is not None:
                try:
                    mm_tk, conf_tk = _tick_keypoint_estimate(tick_kp_model, full, device)
                    if mm_tk is not None:
                        jury_candidates.append(ScaleCandidate("tick_keypoint", mm_tk, conf_tk))
                except Exception as exc:  # noqa: BLE001
                    print(f"tick_keypoint fail {p.name}: {exc}")
            fused = confidence_weighted_vote(jury_candidates)
            if fused.mm_per_pixel is not None and fused.confidence >= 0.4:
                live_scales[p.name] = float(fused.mm_per_pixel)
                live_scales[p.stem] = float(fused.mm_per_pixel)
                live_conf[p.name] = float(fused.confidence)
                live_conf[p.stem] = float(fused.confidence)
                prov_mm = float(fused.mm_per_pixel)
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

        apo_prob = fasc_prob = None
        if dl_track is not None:
            _, _, apo_prob, fasc_prob = dl_track.predict_masks(
                gray, apo_thr=apo_thr, fasc_thr=fasc_thr
            )
        elif sof_model is not None:
            from muscle_arc.data.dataset import letterbox, unletterbox

            canvas, meta = letterbox(gray, img_size, is_mask=False)
            rgb = np.stack([canvas, canvas, canvas], axis=-1)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            with torch.no_grad():
                out = sof_model(tensor)
                apo_c = torch.sigmoid(out["apo_bin"])[0, 0].cpu().numpy()
                fasc_c = torch.sigmoid(out["fasc_bin"])[0, 0].cpu().numpy()
            apo_prob = unletterbox(apo_c.astype(np.float32), meta)
            fasc_prob = unletterbox(fasc_c.astype(np.float32), meta)
            if apo_prob.shape[:2] != (h, w):
                apo_prob = cv2.resize(apo_prob, (w, h), interpolation=cv2.INTER_LINEAR)
            if fasc_prob.shape[:2] != (h, w):
                fasc_prob = cv2.resize(fasc_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
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

        fl_mm_direct = mt_mm_direct = np.nan
        if geom_mode == "official":
            from muscle_arc.models.dl_track_official import official_metrics

            mm_geo = float(prov_mm) if prov_mm is not None else float(
                cfg["infer"].get("mm_per_pixel", 0.06)
            )
            # Geometry on full-frame coords; masks are already HxW of gray
            # (chrome-mask keeps HxW == full). Resize if sector-crop shrank.
            apo_g, fasc_g = apo, fasc
            if (h, w) != (fh, fw):
                apo_g = cv2.resize(apo, (fw, fh), interpolation=cv2.INTER_NEAREST)
                fasc_g = cv2.resize(fasc, (fw, fh), interpolation=cv2.INTER_NEAREST)
            try:
                pa, fl_mm_direct, mt_mm_direct = official_metrics(
                    apo_g,
                    fasc_g,
                    full,
                    mm_per_pixel=mm_geo,
                    apo_thr=apo_thr,
                    fasc_thr=fasc_thr,
                    model_apo=getattr(dl_track, "apo", None) if dl_track else None,
                    model_fasc=getattr(dl_track, "fasc", None) if dl_track else None,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"official geom fail {p.name}: {exc}")
                pa = fl_mm_direct = mt_mm_direct = float("nan")
            # Store px placeholders so scale path can still run for non-official rows
            fl_px = float(fl_mm_direct / mm_geo) if np.isfinite(fl_mm_direct) else np.nan
            mt_px = float(mt_mm_direct / mm_geo) if np.isfinite(mt_mm_direct) else np.nan
        else:
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
                "fl_mm_direct": float(fl_mm_direct) if np.isfinite(fl_mm_direct) else np.nan,
                "mt_mm_direct": float(mt_mm_direct) if np.isfinite(mt_mm_direct) else np.nan,
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
    # Official doCalculations already applied OCR/ticks via calib_dist — prefer it.
    if "fl_mm_direct" in out.columns:
        n_off = 0
        for i, row in enumerate(out.itertuples()):
            if np.isfinite(getattr(row, "fl_mm_direct", np.nan)):
                out.iloc[i, out.columns.get_loc("fl_mm")] = float(row.fl_mm_direct)
                n_off += 1
            if np.isfinite(getattr(row, "mt_mm_direct", np.nan)):
                out.iloc[i, out.columns.get_loc("mt_mm")] = float(row.mt_mm_direct)
            if np.isfinite(getattr(row, "fl_mm_direct", np.nan)) or np.isfinite(
                getattr(row, "mt_mm_direct", np.nan)
            ):
                out.iloc[i, out.columns.get_loc("scale_source")] = "official_calib"
        print(f"Official geometry mm rows: {n_off}/{len(out)}")

    # Phase A': residual — never overwrite meta/ocr/cluster
    residual_models: dict = {}
    apply_on = {"primary", "cluster_prefix", "shape"}
    if args.residual_prefer != "off":
        try:
            residual_models = load_residual_models(args.residual_model_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Residual models skipped (load failed): {exc}")
            residual_models = {}
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
