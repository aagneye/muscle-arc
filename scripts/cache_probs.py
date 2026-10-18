#!/usr/bin/env python3
"""Cache apo/fascicle probability maps (float16 npz per image) for fast geometry sweeps.

Several checkpoints per branch are averaged in probability space (fold ensemble).
Test images are chrome-masked full frames with exact px/cm from
build_exact_scale_table.py; OSF benchmark images use their expert GT scale.

Output: <out-root>/<tag>/<images>/<image_id>.npz with keys apo, fasc (float16 HxW),
mm_per_pixel (float), shape (int[2]).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry.sector_crop import mask_chrome
from muscle_arc.infer.predict import predict_prob
from muscle_arc.models.segmentation import build_segmentation_model

Item = tuple[str, np.ndarray, float]


def _mean_prob(
    models: Sequence[nn.Module],
    gray: np.ndarray,
    img_size: int,
    device: torch.device,
    tta_hflip: bool,
    multiscale: bool,
) -> np.ndarray:
    acc = None
    for m in models:
        p = predict_prob(m, gray, img_size, device, tta_hflip, True, multiscale, False)
        acc = p if acc is None else acc + p
    return (acc / len(models)).astype(np.float32)


def cache_items(
    items: Iterable[Item],
    apo_models: Sequence[nn.Module],
    fasc_models: Sequence[nn.Module],
    *,
    img_size: int,
    device: torch.device,
    out_dir: Path,
    tta_hflip: bool = True,
    multiscale: bool = False,
    overwrite: bool = False,
) -> int:
    """Write one npz per item; return how many were written (existing files skipped)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in (*apo_models, *fasc_models):
        m.eval()
    n = 0
    for image_id, gray, mm in items:
        dst = out_dir / f"{image_id}.npz"
        if dst.exists() and not overwrite:
            continue
        apo = _mean_prob(apo_models, gray, img_size, device, tta_hflip, multiscale)
        fasc = _mean_prob(fasc_models, gray, img_size, device, tta_hflip, multiscale)
        np.savez_compressed(
            dst,
            apo=apo.astype(np.float16),
            fasc=fasc.astype(np.float16),
            mm_per_pixel=np.float64(mm),
            shape=np.asarray(gray.shape[:2], dtype=np.int32),
        )
        n += 1
        if n % 25 == 0:
            print(f"  cached {n}", flush=True)
    return n


def _load(paths: Sequence[Path], model_cfg: dict, device: torch.device) -> list[nn.Module]:
    out = []
    for p in paths:
        m = build_segmentation_model(model_cfg).to(device)
        payload = torch.load(p, map_location=device, weights_only=False)
        m.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
        m.eval()
        out.append(m)
    return out


def test_items(cfg: dict, scale_table: Path) -> list[Item]:
    paths = DataPaths.from_config(cfg["data"])
    scale = pd.read_csv(scale_table).set_index("image_id")["mm_per_pixel"].to_dict()
    items = []
    for p in list_images(paths.test_images):
        gray, _ = mask_chrome(read_gray(p))
        items.append((p.name, gray, float(scale[p.name])))
    return items


def osf_items(osf_gt: Path) -> list[Item]:
    df = pd.read_csv(osf_gt)
    items = []
    for _, r in df.iterrows():
        p = Path(str(r["path"]))
        if p.exists() and pd.notna(r.get("mm_per_pixel")):
            items.append((str(r.get("image_id", p.name)), read_gray(p), float(r["mm_per_pixel"])))
    return items


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/dual_v3.yaml"))
    ap.add_argument("--apo-ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--fasc-ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--images", choices=["test", "osf"], default="test")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("experiments/prob_cache"))
    ap.add_argument("--scale-table", type=Path, default=Path("experiments/depth_scale_table_exact.csv"))
    ap.add_argument("--osf-gt", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--multiscale", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    apo = _load(args.apo_ckpt, cfg["model"], device)
    fasc = _load(args.fasc_ckpt, cfg["model"], device)
    items = test_items(cfg, args.scale_table) if args.images == "test" else osf_items(args.osf_gt)
    out = args.out_root / args.tag / args.images
    n = cache_items(items, apo, fasc, img_size=int(cfg["img_size"]), device=device, out_dir=out,
                    multiscale=args.multiscale, overwrite=args.overwrite)
    print(f"wrote {n} new / {len(items)} total -> {out}")


if __name__ == "__main__":
    main()
