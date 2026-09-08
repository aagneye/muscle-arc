#!/usr/bin/env python3
"""Build expert GT CSV (+ known scales) from UMUD OSF architecture benchmarks."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def unpack_zips(zip_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for z in sorted(zip_dir.glob("*.zip")):
        print(f"unpack {z.name}")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out_dir)


def expert_median_row(row: pd.Series) -> tuple[float, float, float]:
    """Median across R1..R7 expert MT/FL/PA; fall back to SMA if needed."""
    mts, fls, pas = [], [], []
    for i in range(1, 8):
        for col, bucket in ((f"R{i}_MT", mts), (f"R{i}_FL", fls), (f"R{i}_PA", pas)):
            if col in row and pd.notna(row[col]):
                bucket.append(float(row[col]))
    if not mts and "SMA_MT" in row and pd.notna(row["SMA_MT"]):
        mts.append(float(row["SMA_MT"]))
    if not fls and "SMA_FL" in row and pd.notna(row["SMA_FL"]):
        fls.append(float(row["SMA_FL"]))
    if not pas and "SMA_PA" in row and pd.notna(row["SMA_PA"]):
        pas.append(float(row["SMA_PA"]))
    return (
        float(np.median(pas)) if pas else float("nan"),
        float(np.median(fls)) if fls else float("nan"),
        float(np.median(mts)) if mts else float("nan"),
    )


def parse_architecture_xlsx(xlsx: Path, images_dir: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=0)
    rows = []
    for _, r in df.iterrows():
        image_id = str(r["ImageID"])
        # Prefer matching .tif next to the spreadsheet
        candidates = [
            images_dir / f"{image_id}.tif",
            images_dir / f"{image_id}.tiff",
            images_dir / f"{image_id}.png",
        ]
        path = next((c for c in candidates if c.exists()), None)
        pa, fl, mt = expert_median_row(r)
        scale_ppc = float(r["Scale_pixel_per_cm"]) if "Scale_pixel_per_cm" in r and pd.notna(r["Scale_pixel_per_cm"]) else np.nan
        mm_per_pixel = (10.0 / scale_ppc) if np.isfinite(scale_ppc) and scale_ppc > 0 else np.nan
        rows.append(
            {
                "image_id": path.name if path else f"{image_id}.tif",
                "stem": image_id,
                "path": str(path) if path else "",
                "pa_deg": pa,
                "fl_mm": fl,
                "mt_mm": mt,
                "scale_pixel_per_cm": scale_ppc,
                "mm_per_pixel": mm_per_pixel,
                "source": "osf_architecture_v0.1.0",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=Path("data/external/umud_osf/ExpertAnalysed"),
    )
    parser.add_argument(
        "--unpacked",
        type=Path,
        default=Path("data/external/umud_osf/unpacked"),
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/osf_expert_gt.csv"))
    parser.add_argument("--skip-unpack", action="store_true")
    args = parser.parse_args()

    if not args.skip_unpack:
        unpack_zips(args.zip_dir, args.unpacked)

    arch_dir = args.unpacked / "benchmark_dataset_architecture_v0.1.0"
    xlsx = arch_dir / "Results_benchmark_architecture_v0.1.0.xlsx"
    if not xlsx.exists():
        raise SystemExit(f"missing {xlsx}")

    gt = parse_architecture_xlsx(xlsx, arch_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(args.out, index=False)
    print(f"Wrote {args.out} rows={len(gt)}")
    print(gt[["pa_deg", "fl_mm", "mt_mm", "mm_per_pixel"]].describe().to_string())
    print("scales:", sorted(gt["mm_per_pixel"].dropna().unique().round(5).tolist()))


if __name__ == "__main__":
    main()
