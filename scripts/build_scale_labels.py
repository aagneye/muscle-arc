"""Build scale pseudo-labels from heuristic detector (+ optional OSF GT).

Writes CSV: image_id, path, mm_per_pixel, source, confidence, kind
for training ScaleStripDetector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from muscle_arc.data.dataset import list_images, read_gray
from muscle_arc.geometry.depth_scale import estimate_depth_scale
from muscle_arc.geometry.scale import is_placeholder_dpi_scale
from muscle_arc.geometry.sector_crop import classify_kind, find_sector


def _rows_from_folder(folder: Path, min_conf: float) -> list[dict]:
    rows = []
    imgs = list_images(folder)
    print(f"Labeling {folder}: {len(imgs)} images", flush=True)
    for p in imgs:
        gray = read_gray(p)
        sector = find_sector(gray)
        kind, _ = classify_kind(gray, sector)
        est = estimate_depth_scale(gray)
        if est.mm_per_pixel is None:
            continue
        if is_placeholder_dpi_scale(est.mm_per_pixel):
            continue
        if est.confidence < min_conf:
            continue
        rows.append(
            {
                "image_id": p.name,
                "path": str(p),
                "mm_per_pixel": float(est.mm_per_pixel),
                "source": est.source,
                "confidence": float(est.confidence),
                "kind": kind,
            }
        )
    return rows


def _rows_from_osf(gt_csv: Path) -> list[dict]:
    if not gt_csv.exists():
        return []
    df = pd.read_csv(gt_csv)
    rows = []
    for _, r in df.iterrows():
        p = Path(str(r.get("path", "")))
        mm = r.get("mm_per_pixel")
        if not p.exists() or pd.isna(mm):
            if "scale_pixel_per_cm" in df.columns and pd.notna(r.get("scale_pixel_per_cm")):
                mm = 10.0 / float(r["scale_pixel_per_cm"])
            else:
                continue
        mm = float(mm)
        if is_placeholder_dpi_scale(mm):
            continue
        rows.append(
            {
                "image_id": p.name,
                "path": str(p),
                "mm_per_pixel": mm,
                "source": "osf_gt",
                "confidence": 1.0,
                "kind": "osf",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", type=Path, required=True)
    parser.add_argument("--osf-gt", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    parser.add_argument("--min-conf", type=float, default=0.55)
    parser.add_argument("--out", type=Path, default=Path("experiments/scale_pseudo_labels.csv"))
    args = parser.parse_args()

    rows: list[dict] = []
    for folder in args.folders:
        rows.extend(_rows_from_folder(folder, args.min_conf))
    rows.extend(_rows_from_osf(args.osf_gt))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["path"], keep="last")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} n={len(df)}")
    if len(df):
        print(df["source"].value_counts().to_string())
        print(df["mm_per_pixel"].describe().to_string())


if __name__ == "__main__":
    main()
