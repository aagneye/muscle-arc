#!/usr/bin/env python3
"""Download UMUD Challenge data via Kaggle API into data/raw."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory for competition files",
    )
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "Install kaggle and place credentials at ~/.kaggle/kaggle.json"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {COMPETITION} → {out}")
    api.competition_download_files(COMPETITION, path=str(out), quiet=False)

    for zpath in out.glob("*.zip"):
        print(f"Extracting {zpath.name}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out)
        zpath.unlink()

    print("Done. Expected folders: apo_imgs, apo_masks, fasc_imgs, fasc_masks, test_images")


if __name__ == "__main__":
    main()
