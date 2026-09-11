#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python3 <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path

print("=== GATE1 (drawing quality, no ruler) ===")
print(Path("experiments/gate1_geometry.json").read_text())

print("\n=== GATE2 (OSF full pipeline with GT scale) ===")
print(Path("experiments/gate2_osf_umud.json").read_text())
g2 = pd.read_csv("experiments/gate2_osf_umud.csv") if Path("experiments/gate2_osf_umud.csv").exists() else None
if g2 is not None:
    g2["pa_e"] = (g2.pa_deg - g2.pa_gt).abs()
    g2["fl_e"] = (g2.fl_mm - g2.fl_gt).abs()
    g2["mt_e"] = (g2.mt_mm - g2.mt_gt).abs()
    print("OSF error medians PA/FL/MT", g2.pa_e.median(), g2.fl_e.median(), g2.mt_e.median())
    print("OSF pred medians", g2.pa_deg.median(), g2.fl_mm.median(), g2.mt_mm.median())
    print("OSF gt medians", g2.pa_gt.median(), g2.fl_gt.median(), g2.mt_gt.median())
    print("FL bias (pred-gt) mean", (g2.fl_mm - g2.fl_gt).mean())
    print("MT bias (pred-gt) mean", (g2.mt_mm - g2.mt_gt).mean())

print("\n=== v11 vs v10c vs probed FL~81.5 ===")
for n in ["v8","v10c","v11"]:
    d = pd.read_csv(f"submissions/submission_{n}.csv")
    print(n, "FL_med", round(d.fl_mm.median(),1), "MT_med", round(d.mt_mm.median(),1),
          "FL_clip140", int((d.fl_mm>=139.9).sum()), "PA_med", round(d.pa_deg.median(),1))

dbg = Path("experiments/submission_v11_debug.csv")
if dbg.exists():
    d = pd.read_csv(dbg)
    print("\n=== v11 debug ===")
    print("scale_source counts:\n", d.scale_source.value_counts().to_string() if "scale_source" in d else "n/a")
    print("kind:\n", d.kind.value_counts().to_string() if "kind" in d else "n/a")
    if "fl_px" in d and "mt_px" in d:
        print("fl_px med", d.fl_px.median(), "mt_px med", d.mt_px.median())
        # implied mm = fl_mm/fl_px
        m = d[d.fl_px>1]
        print("implied mm from FL", (m.fl_mm/m.fl_px).median())
        print("implied mm from MT", (d.loc[d.mt_px>1,"mt_mm"]/d.loc[d.mt_px>1,"mt_px"]).median())

depth = pd.read_csv("experiments/depth_scale_table.csv")
print("\n=== depth audit ===")
print("with_scale", depth.mm_per_pixel.notna().sum(), "/", len(depth))
print(depth.groupby("kind")["mm_per_pixel"].apply(lambda s: s.notna().mean()).to_string())
print("sources:\n", depth.loc[depth.mm_per_pixel.notna(),"source"].value_counts().to_string())
PY
