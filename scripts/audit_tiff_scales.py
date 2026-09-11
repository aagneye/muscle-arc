#!/usr/bin/env python3
"""Audit TIFF / image metadata for recoverable mm-per-pixel scales.

Scans train + test image folders, reads X/Y resolution tags and ImageJ
ImageDescription (tag 270), writes experiments/scale_table.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from muscle_arc.data.dataset import list_images
from muscle_arc.geometry.scale import is_placeholder_dpi_scale

# TIFF tag ids
TAG_XRES = 282
TAG_YRES = 283
TAG_RESUNIT = 296  # 2=inch, 3=cm
TAG_IMAGEDESC = 270
TAG_OMEXML = 270  # also used; ImageJ packs into description


def _rational_to_float(val) -> float | None:
    try:
        if hasattr(val, "numerator"):
            return float(val.numerator) / float(val.denominator) if val.denominator else None
        if isinstance(val, tuple) and len(val) == 2:
            return float(val[0]) / float(val[1]) if val[1] else None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _parse_imagej_description(desc: str) -> dict[str, float | str]:
    """Parse ImageJ-style ImageDescription for unit/spacing/pixel width."""
    out: dict[str, float | str] = {}
    if not desc:
        return out
    # unit=mm / unit=cm / unit=um
    m = re.search(r"unit[=:\s]+([a-zA-Zµμ]+)", desc, re.I)
    if m:
        out["unit"] = m.group(1).lower().replace("µ", "u").replace("μ", "u")
    # spacing=0.06 or pixelWidth / pixelHeight
    for key in ("spacing", "pixelwidth", "pixelheight", "x_resolution", "y_resolution"):
        m = re.search(rf"{key}[=:\s]+([0-9.eE+-]+)", desc, re.I)
        if m:
            out[key] = float(m.group(1))
    # ImageJ often has lines like: "x_scale = 0.0625 mm"
    m = re.search(r"([0-9.]+)\s*(mm|cm|um|µm)/?\s*pix", desc, re.I)
    if m:
        out["explicit_mm_hint"] = float(m.group(1))
        out["explicit_unit"] = m.group(2).lower()
    return out


def _unit_to_mm_factor(unit: str | None) -> float | None:
    if unit is None:
        return None
    u = unit.lower().strip()
    if u in ("mm", "millimeter", "millimetre"):
        return 1.0
    if u in ("cm", "centimeter", "centimetre"):
        return 10.0
    if u in ("um", "µm", "micrometer", "micrometre", "micron"):
        return 0.001
    if u in ("m", "meter", "metre"):
        return 1000.0
    if u in ("inch", "in"):
        return 25.4
    return None


def extract_scale(path: Path) -> dict:
    row: dict = {
        "image_id": path.name,
        "stem": path.stem,
        "path": str(path),
        "h": None,
        "w": None,
        "xres": None,
        "yres": None,
        "resunit": None,
        "desc_unit": None,
        "desc_spacing": None,
        "mm_per_pixel_x": None,
        "mm_per_pixel_y": None,
        "mm_per_pixel": None,
        "source": None,
    }
    try:
        with Image.open(path) as im:
            row["w"], row["h"] = im.size
            tags = getattr(im, "tag_v2", None) or {}
            xres = _rational_to_float(tags.get(TAG_XRES)) if tags else None
            yres = _rational_to_float(tags.get(TAG_YRES)) if tags else None
            resunit = int(tags[TAG_RESUNIT]) if tags and TAG_RESUNIT in tags else None
            desc = ""
            if tags and TAG_IMAGEDESC in tags:
                d = tags[TAG_IMAGEDESC]
                desc = d.decode("utf-8", errors="ignore") if isinstance(d, bytes) else str(d)
            elif hasattr(im, "info") and "description" in im.info:
                desc = str(im.info["description"])
            row["xres"] = xres
            row["yres"] = yres
            row["resunit"] = resunit

            parsed = _parse_imagej_description(desc)
            if "unit" in parsed:
                row["desc_unit"] = parsed["unit"]
            if "spacing" in parsed:
                row["desc_spacing"] = parsed["spacing"]
            elif "pixelwidth" in parsed:
                row["desc_spacing"] = parsed["pixelwidth"]

            # Path 1: ImageJ explicit spacing + unit
            if row["desc_spacing"] is not None:
                factor = _unit_to_mm_factor(str(row["desc_unit"]) if row["desc_unit"] else "mm")
                if factor is None:
                    factor = 1.0  # assume mm if spacing present without unit
                mm = float(row["desc_spacing"]) * factor
                if 0.001 < mm < 5.0 and not is_placeholder_dpi_scale(mm, xres):
                    row["mm_per_pixel"] = mm
                    row["mm_per_pixel_x"] = mm
                    row["mm_per_pixel_y"] = mm
                    row["source"] = "imagej_desc"

            # Path 2: TIFF X/Y resolution (reject classic 72 DPI placeholders)
            if row["mm_per_pixel"] is None and xres and xres > 0:
                # Resolution is pixels per unit
                if resunit == 2:  # inch
                    mm_x = 25.4 / xres
                    mm_y = 25.4 / yres if yres and yres > 0 else mm_x
                elif resunit == 3:  # cm
                    mm_x = 10.0 / xres
                    mm_y = 10.0 / yres if yres and yres > 0 else mm_x
                else:
                    # Unknown unit — skip unless values look like mm-ish when inverted
                    mm_x = 1.0 / xres
                    mm_y = 1.0 / yres if yres and yres > 0 else mm_x
                if 0.001 < mm_x < 5.0 and not is_placeholder_dpi_scale(mm_x, xres):
                    row["mm_per_pixel_x"] = mm_x
                    row["mm_per_pixel_y"] = mm_y
                    row["mm_per_pixel"] = float(0.5 * (mm_x + mm_y))
                    row["source"] = f"tiff_resunit_{resunit}"
                elif xres is not None and abs(float(xres) - 72.0) < 0.5:
                    row["source"] = "rejected_72dpi"

            # Path 3: explicit regex hint
            if row["mm_per_pixel"] is None and "explicit_mm_hint" in parsed:
                factor = _unit_to_mm_factor(str(parsed.get("explicit_unit", "mm"))) or 1.0
                mm = float(parsed["explicit_mm_hint"]) * factor
                if 0.001 < mm < 5.0 and not is_placeholder_dpi_scale(mm, xres):
                    row["mm_per_pixel"] = mm
                    row["source"] = "desc_regex"
    except Exception as exc:  # noqa: BLE001
        row["source"] = f"error:{type(exc).__name__}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["test_images_v2", "apo_imgs_v1", "fasc_imgs_v1"],
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/scale_table.csv"))
    args = parser.parse_args()

    rows = []
    for folder in args.folders:
        d = args.root / folder
        imgs = list_images(d)
        print(f"Scanning {d}: {len(imgs)} images")
        for p in imgs:
            r = extract_scale(p)
            r["folder"] = folder
            rows.append(r)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n = len(df)
    n_ok = int(df["mm_per_pixel"].notna().sum())
    n_rej = int((df["source"] == "rejected_72dpi").sum()) if "source" in df.columns else 0
    print(
        f"\nWrote {args.out} rows={n} with_usable_scale={n_ok} "
        f"({100*n_ok/max(n,1):.1f}%) rejected_72dpi={n_rej}"
    )
    if n_ok:
        print(df.loc[df["mm_per_pixel"].notna(), "mm_per_pixel"].describe().to_string())
        print("sources:\n", df.loc[df["mm_per_pixel"].notna(), "source"].value_counts().to_string())
    # Per-folder
    for folder, g in df.groupby("folder"):
        ok = int(g["mm_per_pixel"].notna().sum())
        print(f"  {folder}: {ok}/{len(g)} scaled")
    # Test-only decision metric (usable only — placeholders already nulled)
    test = df[df["folder"].str.contains("test", case=False)]
    if len(test):
        tok = int(test["mm_per_pixel"].notna().sum())
        print(f"\nDECISION test_with_usable_scale={tok}/{len(test)}")
        if tok == 0:
            print("NO_METADATA_SCALE — use Phase A' residual regressor / richer clustering")
        elif tok < 0.5 * len(test):
            print("PARTIAL_METADATA_SCALE — hybrid metadata + cluster fallback")
        else:
            print("METADATA_SCALE_OK — prefer per-image metadata")


if __name__ == "__main__":
    main()
