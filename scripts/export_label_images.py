#!/usr/bin/env python3
"""Export test images as PNG + manifest.json for the manual labelling tool.

Order: images are interleaved round-robin across device layouts (seeded shuffle
within each layout) so any prefix of the queue covers every device. For each
5-frame video clip only the middle frame is in the main queue; the other frames
are marked optional and placed at the end.

Output (never commit; competition data, and the repo is public):
  <out>/images/<image_id>.png
  <out>/manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import cv2
import pandas as pd
import yaml

from muscle_arc.data.dataset import list_images
from muscle_arc.data.paths import DataPaths


def _clips(paths: list[Path]) -> list[list[int]]:
    spec = importlib.util.spec_from_file_location("cal", "scripts/calibrate_predict.py")
    cal = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cal)
    return [g for g in cal.sequence_groups(paths) if len(g) > 1]


def build_order(layouts: list[str], clip_of: dict[int, tuple[int, int, int]], seed: int = 0) -> list[int]:
    """Main queue (round-robin over layouts, clip middles only) then optional clip frames."""
    rng = random.Random(seed)
    main: dict[str, list[int]] = {}
    optional: list[int] = []
    for i, lay in enumerate(layouts):
        if i in clip_of and clip_of[i][1] != clip_of[i][2] // 2:
            optional.append(i)
        else:
            main.setdefault(lay, []).append(i)
    for v in main.values():
        rng.shuffle(v)
    order: list[int] = []
    queues = sorted(main.values(), key=len, reverse=True)
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop())
    return order + optional


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/dual_v3.yaml"))
    ap.add_argument("--scale-table", type=Path, default=Path("experiments/depth_scale_table_exact.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/labeling"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    paths = list_images(DataPaths.from_config(cfg["data"]).test_images)
    scale = pd.read_csv(args.scale_table).set_index("image_id")
    clip_of: dict[int, tuple[int, int, int]] = {}
    for c, g in enumerate(_clips(paths)):
        for pos, i in enumerate(g):
            clip_of[i] = (c, pos, len(g))
    layouts = [str(scale.loc[p.name, "source"]) for p in paths]
    order = build_order(layouts, clip_of)

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    items = []
    for rank, i in enumerate(order):
        p = paths[i]
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        png = f"{p.stem}.png"
        cv2.imwrite(str(args.out / "images" / png), img)
        clip = clip_of.get(i)
        items.append({
            "rank": rank,
            "image_id": p.name,
            "file": png,
            "w": int(img.shape[1]),
            "h": int(img.shape[0]),
            "mm_per_pixel": float(scale.loc[p.name, "mm_per_pixel"]),
            "layout": layouts[i],
            "clip": None if clip is None else int(clip[0]),
            "clip_pos": None if clip is None else int(clip[1]),
            "optional": bool(clip is not None and clip[1] != clip[2] // 2),
        })
    (args.out / "manifest.json").write_text(json.dumps({"images": items}, indent=1))
    n_main = sum(not it["optional"] for it in items)
    print(f"wrote {len(items)} images ({n_main} main + {len(items) - n_main} optional) -> {args.out}")


if __name__ == "__main__":
    main()
