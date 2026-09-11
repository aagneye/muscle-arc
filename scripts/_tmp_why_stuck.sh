#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/${COMP}/leaderboard/download?type=public" \
  -o /tmp/lb.zip
python3 <<'PY'
import zipfile
z = zipfile.ZipFile("/tmp/lb.zip")
name = z.namelist()[0]
text = z.read(name).decode("utf-8", errors="replace")
lines = text.strip().splitlines()
print("header:", lines[0])
print("--- top 20 ---")
for i, line in enumerate(lines[1:21], 1):
    print(f"{i:3d} {line}")
print("--- our team ---")
for i, line in enumerate(lines[1:], 1):
    low = line.lower()
    if "batman" in low or "aagneye" in low or "0.95357" in line or "0.99973" in line:
        print(f"{i:3d} {line}")
# score needed for top 15
print("--- score at rank 15 ---")
print(lines[15] if len(lines) > 15 else "n/a")
print("total teams:", len(lines)-1)
PY

# Compare our submissions vs thwaits insights
source .venv/bin/activate
python3 <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path

depth = pd.read_csv("experiments/depth_scale_table.csv")
v8 = pd.read_csv("submissions/submission_v8.csv")
v10c = pd.read_csv("submissions/submission_v10c.csv")
v10b = pd.read_csv("submissions/submission_v10b.csv")

print("\n=== OUR SCALE vs THWAITS ANCHOR ===")
print("Thwaits: IMG_00001 depth 4.5cm → ~0.074 mm/px (135 px/cm)")
print("Thwaits: public median FL ~81.5 mm")
d1 = depth[depth.image_id.str.contains("IMG_00001")]
print("ours IMG_00001:", d1[["mm_per_pixel","depth_cm","source"]].to_dict("records"))

print("\nFL/MT/PA medians (need ~81.5 FL public):")
for name,d in [("v8",v8),("v10b",v10b),("v10c",v10c)]:
    print(f"  {name}: PA={d.pa_deg.median():.1f} FL={d.fl_mm.median():.1f} MT={d.mt_mm.median():.1f}  FL_std={d.fl_mm.std():.1f}")

# How many v10b FL clipped at 140?
print("v10b FL at clip 140:", int((v10b.fl_mm>=139.9).sum()), "/309")
print("v10c FL at clip 140:", int((v10c.fl_mm>=139.9).sum()), "/309")

# implied mm from OCR vs cluster
print("\ndepth coverage:", int(depth.mm_per_pixel.notna().sum()), "/309")
print("mm describe:\n", depth.mm_per_pixel.describe().round(4).to_string())
print("sources:\n", depth.loc[depth.mm_per_pixel.notna(),"source"].value_counts().to_string())

# Official score components rough: S=(MAE_PA/6 + MAE_FL/12 + MAE_MT/3)/3
# If score=0.95 and uniform error, rough MAE:
# assume equal contribution: each term ~0.95 → MAE_PA~5.7, MAE_FL~11.4, MAE_MT~2.85
print("\n=== rough error budget for score 0.95 ===")
print("If equal parts: MAE_PA≈5.7°, MAE_FL≈11.4mm, MAE_MT≈2.85mm")
print("Top15 ~0.45 would need ~half that error")
print("Top1 ~0.29 needs ~1/3 our error")
PY
