#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python3 <<'PY'
import pandas as pd

depth = pd.read_csv("experiments/depth_scale_table.csv")
g0 = pd.read_csv("submissions/submission_dltrack_g0.csv")
sof = pd.read_csv("submissions/submission_sof_v1c.csv")
v10 = pd.read_csv("submissions/submission_v10c.csv")

print("depth kind\n", depth.kind.value_counts().to_string())
print("depth source\n", depth.source.value_counts().to_string())
m = g0.merge(
    depth[["image_id", "kind", "source", "mm_per_pixel", "confidence"]],
    on="image_id",
    how="left",
)
print("\nby kind g0 medians:")
print(m.groupby("kind")[["pa_deg", "fl_mm", "mt_mm"]].median())
print("\nby depth source g0 fl/mt:")
print(m.groupby("source")[["fl_mm", "mt_mm"]].agg(["median", "count"]))

for a, b, name in [(g0, sof, "g0-sof"), (g0, v10, "g0-v10c")]:
    print(
        name,
        "dPA",
        round((a.pa_deg - b.pa_deg).abs().mean(), 2),
        "dFL",
        round((a.fl_mm - b.fl_mm).abs().mean(), 2),
        "dMT",
        round((a.mt_mm - b.mt_mm).abs().mean(), 2),
    )

print(
    "\nIMG00001 depth",
    depth[depth.image_id.str.contains("00001")][
        ["image_id", "mm_per_pixel", "source", "confidence", "kind"]
    ].to_string(index=False),
)
print("IMG00001 g0", g0[g0.image_id.str.contains("00001")].to_string(index=False))
print("IMG00001 sof", sof[sof.image_id.str.contains("00001")].to_string(index=False))
print("IMG00001 v10", v10[v10.image_id.str.contains("00001")].to_string(index=False))
PY
