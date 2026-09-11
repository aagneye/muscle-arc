#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
python3 <<'PY'
import json
from pathlib import Path
import pandas as pd
for j in ["gate2b_letterbox.json","gate2b_dltrack_win.json","gate2b_dltrack_pred.json"]:
    p=Path("experiments")/j
    if p.exists():
        m=json.loads(p.read_text())
        print(j, "umud", m.get("umud"), "mt_bad", m.get("n_mt_bad"), "n", m.get("n"))
for c in ["gate2b_letterbox.csv","gate2b_dltrack_win.csv","gate2b_dltrack_pred.csv"]:
    p=Path("experiments")/c
    print("---", c, p.exists())
    if not p.exists():
        continue
    df=pd.read_csv(p)
    print(df.columns.tolist())
    if "scale_src" in df.columns:
        print("scale_src", df["scale_src"].value_counts().to_dict())
    bad=df[df["mt_ratio"].notna() & ((df.mt_ratio<0.9)|(df.mt_ratio>1.1))]
    print("bad n", len(bad))
    cols=[x for x in ["image_id","mt_ratio","fl_ratio","mm_per_pixel","scale_src","est_src","est_mm"] if x in df.columns]
    print(bad[cols].head(12).to_string())
    ok=df[df["mt_ratio"].notna() & (df.mt_ratio>=0.9) & (df.mt_ratio<=1.1)]
    print("ok median mm", ok["mm_per_pixel"].median() if "mm_per_pixel" in df else None)
    print("bad median mm", bad["mm_per_pixel"].median() if "mm_per_pixel" in df else None)
PY
# md5 of key modules vs when letterbox ran — show current hashes
md5sum src/muscle_arc/geometry/metrics.py src/muscle_arc/geometry/depth_scale.py src/muscle_arc/geometry/surfaces.py src/muscle_arc/models/dl_track.py
# show if letterbox log recorded scale sources
grep -E 'osf_shape|depth_hyp|UMUD|umud' logs/dltrack_lb_gate2b.log | head -40
