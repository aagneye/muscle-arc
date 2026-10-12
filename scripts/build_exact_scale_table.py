#!/usr/bin/env python3
"""Write a per-image exact scale + sector-box table for calibrate_predict.py.

Output columns match ``depth_scale.load_depth_scale_table`` (image_id, stem,
mm_per_pixel, px_per_cm, source, confidence) plus the crop box (l, t, r, b)
consumed by ``calibrate_predict.py --crop-table``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from muscle_arc.data.dataset import list_images
from muscle_arc.data.paths import DataPaths
from muscle_arc.geometry.exact_scale import decode_exact_scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("experiments/depth_scale_table_exact.csv"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = DataPaths.from_config(cfg["data"])
    rows = []
    for p in list_images(paths.test_images):
        with Image.open(p) as im:
            arr = __import__("numpy").asarray(im)
        s = decode_exact_scale(arr, p.name)
        l, t, r, b = s.box
        rows.append(
            {
                "image_id": p.name,
                "stem": p.stem,
                "h": int(arr.shape[0]),
                "w": int(arr.shape[1]),
                "mm_per_pixel": s.mm_per_pixel if s.ok else float("nan"),
                "px_per_cm": s.px_per_cm if s.ok else float("nan"),
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "source": s.source,
                "confidence": 1.0 if s.ok else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_ok = int((df["confidence"] > 0).sum())
    print(f"Wrote {args.out}: {n_ok}/{len(df)} images decoded")
    bad = df[df["confidence"] == 0]
    if len(bad):
        print("Undecoded:", bad[["image_id", "h", "w", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
