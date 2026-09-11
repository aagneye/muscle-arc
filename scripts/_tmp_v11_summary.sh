#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
cat experiments/gate1_geometry.json
echo ---
python3 <<'PY'
import pandas as pd
for n in ["v8","v10c","v11"]:
    d=pd.read_csv(f"submissions/submission_{n}.csv")
    print(n, "FL_med", round(float(d.fl_mm.median()),1), "FL_std", round(float(d.fl_mm.std()),1),
          "clip140", int((d.fl_mm>=139.9).sum()), "MT_med", round(float(d.mt_mm.median()),1))
PY
