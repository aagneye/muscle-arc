#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
OUT=submissions/submission_dltrack_g0.csv
ls -la "$OUT" submissions/submission_dltrack_g0_debug.csv
python - <<'PY'
import pandas as pd
from pathlib import Path
s=pd.read_csv("submissions/submission_dltrack_g0.csv")
print(s.describe())
print("n", len(s))
dbg=Path("submissions/submission_dltrack_g0_debug.csv")
if dbg.exists():
    d=pd.read_csv(dbg)
    print("debug cols", list(d.columns)[:30])
    if "scale_src" in d.columns:
        print(d["scale_src"].value_counts().head(20))
    if "mm_per_pixel" in d.columns:
        print("mm describe", d["mm_per_pixel"].describe())
PY
bash scripts/vm_submit_kaggle.sh "$OUT" "dltrack-g0-geom-fixed"
