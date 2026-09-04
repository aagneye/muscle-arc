#!/usr/bin/env python3
"""Download UMUD Challenge data via Kaggle API (Bearer token or legacy CLI)."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

COMPETITION = "umud-challenge-muscle-architecture-in-ultrasound-data"


def _read_token() -> str | None:
    env = os.environ.get("KAGGLE_API_TOKEN")
    if env:
        return env.strip()
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def download_with_bearer(out: Path, token: str) -> None:
    import urllib.request

    url = f"https://api.kaggle.com/v1/competitions/data/download-all/{COMPETITION}"
    zip_path = out / f"{COMPETITION}.zip"
    print(f"Downloading {COMPETITION} → {zip_path} (Bearer)")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp, zip_path.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    print(f"Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    zip_path.unlink()


def download_with_kaggle_cli(out: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {COMPETITION} → {out}")
    api.competition_download_files(COMPETITION, path=str(out), quiet=False)
    for zpath in out.glob("*.zip"):
        print(f"Extracting {zpath.name}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(out)
        zpath.unlink()


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

    token = _read_token()
    if token:
        download_with_bearer(out, token)
    else:
        # Legacy username/key via ~/.kaggle/kaggle.json
        cfg = Path.home() / ".kaggle" / "kaggle.json"
        if cfg.exists():
            try:
                json.loads(cfg.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    "Malformed ~/.kaggle/kaggle.json; use access_token or valid JSON"
                ) from exc
        download_with_kaggle_cli(out)

    print("Done. Expected top-level folders like apo_imgs_v1, fasc_imgs_v1, test_images_v2")


if __name__ == "__main__":
    main()
