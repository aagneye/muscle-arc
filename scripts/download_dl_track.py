#!/usr/bin/env python3
"""Download DL_Track_US pretrained models from OSF (https://osf.io/7mjsc).

External data — declare on Kaggle. Apache-2.0 weights; prize submissions are GPL-3.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

OSF_NODE = "7mjsc"
OSF_API = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/"


def _walk(url: str, out_dir: Path, depth: int = 0) -> list[Path]:
    saved: list[Path] = []
    if depth > 8:
        return saved
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    data = r.json()
    for item in data.get("data", []):
        attrs = item.get("attributes", {})
        name = attrs.get("name", "unknown")
        kind = attrs.get("kind")
        links = item.get("links", {})
        if kind == "folder":
            rel = links.get("related") or links.get("move") or {}
            href = rel.get("href") if isinstance(rel, dict) else None
            if not href:
                href = item.get("relationships", {}).get("files", {}).get("links", {}).get(
                    "related", {}
                ).get("href")
            if href:
                sub = out_dir / name
                sub.mkdir(parents=True, exist_ok=True)
                saved.extend(_walk(href, sub, depth + 1))
            continue
        # file
        download = links.get("download")
        if not download:
            continue
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip existing {dest}")
            saved.append(dest)
            continue
        print(f"GET {name} → {dest}")
        with requests.get(download, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        saved.append(dest)
    # pagination
    next_url = data.get("links", {}).get("next")
    if next_url:
        saved.extend(_walk(next_url, out_dir, depth))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/external/dl_track"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.out_dir.rglob("*.h5")) + list(args.out_dir.rglob("*.hdf5"))
    if existing:
        print(f"DL_TRACK_OK n_h5={len(existing)} (skip download)")
        for e in existing[:10]:
            print(" ", e)
        return
    notes = args.out_dir / "DOWNLOAD_NOTES.txt"
    notes.write_text(
        "DL_Track_US from https://osf.io/7mjsc\n"
        "Declare as external data on Kaggle.\n"
        "Use with muscle_arc.models.dl_track for Gate0 reference.\n"
    )
    try:
        files = _walk(OSF_API, args.out_dir)
        h5 = [p for p in files if p.suffix.lower() in {".h5", ".hdf5", ".keras"}]
        listing = {
            "n_files": len(files),
            "h5": [str(p) for p in h5],
            "all": [str(p.relative_to(args.out_dir)) for p in files][:200],
        }
        (args.out_dir / "osf_listing.json").write_text(json.dumps(listing, indent=2))
        print(json.dumps(listing, indent=2))
        if h5:
            print(f"DL_TRACK_OK n_h5={len(h5)}")
        else:
            print("DL_TRACK_LISTED — no .h5 found; check OSF UI / zip contents")
    except Exception as exc:  # noqa: BLE001
        print(f"DL_TRACK_SKIP: {exc}")
        print("Place apo/fasc .h5 under data/external/dl_track/ manually.")


if __name__ == "__main__":
    main()
