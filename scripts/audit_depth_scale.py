#!/usr/bin/env python3
"""Audit per-image depth OCR / tick scales on test (and optional train) folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.geometry.depth_scale import estimate_depth_scale
from muscle_arc.geometry.sector_crop import classify_kind, find_sector, sector_crop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--folders", nargs="+", default=["test_images_v2"])
    parser.add_argument("--out", type=Path, default=Path("experiments/depth_scale_table.csv"))
    args = parser.parse_args()

    rows = []
    for folder in args.folders:
        d = args.root / folder
        imgs = list_images(d)
        print(f"Scanning {d}: {len(imgs)} images", flush=True)
        for p in imgs:
            gray = read_gray(p)
            sector = find_sector(gray)
            kind, chrome = classify_kind(gray, sector)
            crop = sector_crop(gray)
            est = estimate_depth_scale(gray)
            rows.append(
                {
                    "image_id": p.name,
                    "stem": p.stem,
                    "folder": folder,
                    "h": gray.shape[0],
                    "w": gray.shape[1],
                    "kind": kind,
                    "chrome_ratio": chrome,
                    "crop_applied": crop.applied,
                    "sector": str(est.sector),
                    "depth_cm": est.depth_cm,
                    "px_per_cm": est.px_per_cm,
                    "mm_per_pixel": est.mm_per_pixel,
                    "source": est.source,
                    "confidence": est.confidence,
                    "text": (est.text or "")[:120],
                }
            )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n = len(df)
    n_ok = int(df["mm_per_pixel"].notna().sum())
    print(f"\nWrote {args.out} rows={n} with_scale={n_ok} ({100 * n_ok / max(n, 1):.1f}%)")
    if n_ok:
        print(df.loc[df["mm_per_pixel"].notna(), "mm_per_pixel"].describe().to_string())
        print("sources:\n", df.loc[df["mm_per_pixel"].notna(), "source"].value_counts().to_string())
    print("kinds:\n", df["kind"].value_counts().to_string())
    print(f"crop_applied={int(df['crop_applied'].sum())}/{n}")
    print(f"\nDECISION depth_scale_ok={n_ok}/{n}")
    if n_ok >= 100:
        print("DEPTH_SCALE_OK — prefer per-image OCR/ticks for submit gate")
    elif n_ok > 0:
        print("PARTIAL_DEPTH_SCALE — hybrid OCR + cluster fallback")
    else:
        print("NO_DEPTH_SCALE — prioritize tick/group/OSF lookup; still use sector crop")


if __name__ == "__main__":
    main()
