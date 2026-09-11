#!/usr/bin/env python3
"""Download UMUD OSF expert-analysed benchmark folders for local MAE."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = "https://api.osf.io/v2"
NODE = "xbawc"
EXPERT_FOLDER = "677817635e3d93d497e6e40f"


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.api+json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def list_folder(folder_id: str) -> list[dict]:
    url = f"{ROOT}/nodes/{NODE}/files/osfstorage/{folder_id}/"
    items = []
    while url:
        data = get_json(url)
        items.extend(data.get("data", []))
        url = data.get("links", {}).get("next")
    return items


def download_file(download_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest}")
        return
    print(f"GET {dest.name} ...")
    req = urllib.request.Request(download_url)
    with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


def walk(folder_id: str, rel: Path, out_root: Path) -> None:
    for item in list_folder(folder_id):
        attrs = item["attributes"]
        name = attrs["name"]
        if attrs["kind"] == "folder":
            walk(item["id"], rel / name, out_root)
        else:
            link = item["links"].get("download")
            if not link:
                continue
            download_file(link, out_root / rel / name)


def main() -> None:
    out = Path("data/external/umud_osf")
    out.mkdir(parents=True, exist_ok=True)
    print("Listing Expert Analysed Benchmark Image Datasets...")
    walk(EXPERT_FOLDER, Path("ExpertAnalysed"), out)
    print("Done. Tree:")
    for p in sorted(out.rglob("*"))[:80]:
        if p.is_file():
            print(" ", p.relative_to(out), p.stat().st_size)
    zip_dir = out / "ExpertAnalysed"
    if (zip_dir / "benchmark_dataset_architecture_v0.1.0.zip").exists():
        print("Building experiments/osf_expert_gt.csv ...")
        import subprocess
        import sys

        subprocess.check_call(
            [
                sys.executable,
                "scripts/build_osf_gt_csv.py",
                "--zip-dir",
                str(zip_dir),
                "--out",
                "experiments/osf_expert_gt.csv",
            ]
        )


if __name__ == "__main__":
    main()
