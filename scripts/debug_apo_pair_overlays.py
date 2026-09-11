#!/usr/bin/env python3
"""Debug apo-band pairing overlays: 20 console + 20 cropped test frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry.metrics import _apo_bands
from muscle_arc.geometry.sector_crop import classify_kind, find_sector, mask_chrome
from muscle_arc.models.dl_track import load_dl_track


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--dl-track-dir", type=Path, default=Path("data/external/dl_track"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/overlays_apo_pair"))
    ap.add_argument("--n-each", type=int, default=20)
    ap.add_argument("--chrome-mask", type=str, default="true", choices=["true", "false"])
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = DataPaths.from_config(cfg["data"])
    seg = load_dl_track(args.dl_track_dir, img_size=512)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    consoles: list[Path] = []
    cropped: list[Path] = []
    for p in list_images(paths.test_images):
        gray = read_gray(p)
        kind, _ = classify_kind(gray, find_sector(gray))
        if kind == "console" and len(consoles) < args.n_each:
            consoles.append(p)
        elif kind != "console" and len(cropped) < args.n_each:
            cropped.append(p)
        if len(consoles) >= args.n_each and len(cropped) >= args.n_each:
            break

    for tag, batch in (("console", consoles), ("cropped", cropped)):
        for p in batch:
            full = read_gray(p)
            gray = mask_chrome(full)[0] if args.chrome_mask == "true" else full
            apo, fasc, _, _ = seg.predict_masks(gray, apo_thr=0.35, fasc_thr=0.10)
            bands = _apo_bands(apo, fasc_mask=fasc, mm_per_pixel=0.06)
            vis = cv2.cvtColor(full, cv2.COLOR_GRAY2BGR)
            if bands is not None:
                (y0a, y1a), (y0b, y1b) = bands
                vis[y0a:y1a, :, 1] = np.clip(vis[y0a:y1a, :, 1].astype(int) + 80, 0, 255)
                vis[y0b:y1b, :, 2] = np.clip(vis[y0b:y1b, :, 2].astype(int) + 80, 0, 255)
                cv2.putText(
                    vis,
                    f"pair {y0a}-{y1a}/{y0b}-{y1b}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    vis, "NO PAIR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            # fasc overlay in blue tint
            m = fasc > 0
            vis[m, 0] = np.clip(vis[m, 0].astype(int) + 60, 0, 255)
            out = args.out_dir / f"{tag}_{p.stem}.png"
            cv2.imwrite(str(out), vis)
            print("wrote", out)


if __name__ == "__main__":
    main()
