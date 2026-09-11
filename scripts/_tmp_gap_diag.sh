#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate

echo "=== LB top 15 ==="
kaggle competitions leaderboard umud-challenge-muscle-architecture-in-ultrasound-data --show 2>/dev/null | head -20 || true

echo "=== our submissions ==="
kaggle competitions submissions -c umud-challenge-muscle-architecture-in-ultrasound-data 2>/dev/null | head -12 || true

echo "=== gate2 latest ==="
python - <<'PY'
import json
from pathlib import Path
for name in ["gate2_osf_umud.json","gate2b_osf_pred_scale.json","gate1_geometry.json","gate0_dl_track.json"]:
    p = Path("experiments")/name
    if not p.exists():
        print(name, "MISSING")
        continue
    d = json.loads(p.read_text())
    keys = ["umud","pa_mae","fl_mae","mt_mae","scale_mode","masks_from","n","gate2_ok","n_fl_ok","n_fl_bad","n_mt_ok","n_mt_bad","n_pa_ok","n_pa_bad",
            "mt_ratio_median","fl_ratio_median","pa_mae_deg","fl_rel_mae","mt_rel_mae","verdict","gate1_ok"]
    print(name, {k:d.get(k) for k in keys if k in d})
PY

echo "=== score decomposition proxy (sof_v1b vs OSF GT if available) ==="
python - <<'PY'
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Component contribution if MAE known from gate2
g = json.loads(Path("experiments/gate2_osf_umud.json").read_text())
pa, fl, mt = g["pa_mae"], g["fl_mae"], g["mt_mae"]
c_pa, c_fl, c_mt = pa/6, fl/12, mt/3
s = (c_pa+c_fl+c_mt)/3
print("Gate2 GT-scale ours: UMUD=", round(s,4))
print("  PA term", round(c_pa,3), f"({100*c_pa/(c_pa+c_fl+c_mt):.0f}% of sum)")
print("  FL term", round(c_fl,3), f"({100*c_fl/(c_pa+c_fl+c_mt):.0f}% of sum)")
print("  MT term", round(c_mt,3), f"({100*c_mt/(c_pa+c_fl+c_mt):.0f}% of sum)")
print("  To hit 0.40 need each term ~1.2 avg; current avg", round((c_pa+c_fl+c_mt)/3,3))

# Public score gap arithmetic
best = 0.85193
top10 = 0.40
print("public best", best, "top10~", top10, "gap", round(best-top10,3))
# If equal error in 3 terms: each MAE needs to drop by factor best/top10
print("need ~factor", round(best/top10,2), "error reduction if uniform")

# sof_v1b vs v14c vs sof_v1c deltas
for tag in ["sof_v1b","sof_v1c","v14c","v8"]:
    p = Path(f"submissions/submission_{tag}.csv")
    if p.exists():
        d=pd.read_csv(p)
        print(tag, "FL_med", round(d.fl_mm.median(),1), "FL_std", round(d.fl_mm.std(),1),
              "MT_med", round(d.mt_mm.median(),1), "PA_med", round(d.pa_deg.median(),1),
              "fl_nan_proxy_clip140", int((d.fl_mm>=139.9).sum()))

# Compare sof vs v14c absolute diffs
a=pd.read_csv("submissions/submission_sof_v1b.csv").sort_values("image_id")
b=pd.read_csv("submissions/submission_v14c.csv").sort_values("image_id")
c=pd.read_csv("submissions/submission_sof_v1c.csv").sort_values("image_id")
for name,x,y in [("sof_v1b-v14c",a,b),("sof_v1c-v14c",c,b)]:
    print(name,
          "dPA", round((x.pa_deg-y.pa_deg).abs().mean(),2),
          "dFL", round((x.fl_mm-y.fl_mm).abs().mean(),2),
          "dMT", round((x.mt_mm-y.mt_mm).abs().mean(),2))

# Depth coverage
depth=pd.read_csv("experiments/depth_scale_table.csv")
print("depth rows", len(depth), "with_mm", int(depth.mm_per_pixel.notna().sum()) if "mm_per_pixel" in depth else "n/a")
if "source" in depth.columns:
    print(depth.loc[depth.mm_per_pixel.notna(),"source"].value_counts().head(8).to_string())

# GATE_B sample errors from debug if present
dbg=Path("experiments/submission_sof_v1b_debug.csv")
if dbg.exists():
    d=pd.read_csv(dbg)
    print("sof_v1b debug cols", list(d.columns)[:20], "n", len(d))
    for col in d.columns:
        if "err" in col.lower() or col in ("fl_mm","mt_mm","pa_deg","scale_mm","mm_per_pixel","fl_src","scale_src"):
            if d[col].dtype.kind in "fiu":
                print(" ", col, "med", float(np.nanmedian(d[col])), "mean", float(np.nanmean(d[col])))
PY
echo DONE
