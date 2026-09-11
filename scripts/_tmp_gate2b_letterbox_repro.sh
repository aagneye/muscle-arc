#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
# Reproduce letterbox Gate2b flags as closely as possible
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --geom ours \
  --sector-crop true \
  --chrome-mask false \
  --scale-mode pred \
  --max-images 35 \
  --out experiments/gate2b_dltrack_win.json \
  2>&1 | tee logs/gate2b_sector_true.out
python3 -c "import json; m=json.load(open('experiments/gate2b_dltrack_win.json')); print('UMUD', m['umud'], 'mt_bad', m['n_mt_bad']); print('PASS' if m['umud']<=0.42 else 'FAIL')"
# Compare one BAD row scales
python3 - <<'PY'
import pandas as pd
from pathlib import Path
for name in ["gate2b_letterbox.csv", "gate2b_dltrack_win.csv"]:
    p = Path("experiments")/name
    if not p.exists():
        # letterbox may only have json
        print(name, "missing")
        continue
    df = pd.read_csv(p)
    print(name, "cols", list(df.columns)[:12])
    bad = df[df["mt_ratio"].notna() & ((df["mt_ratio"]<0.9)|(df["mt_ratio"]>1.1))]
    print(" n_bad", len(bad), "scale_src", bad["scale_src"].value_counts().to_dict() if "scale_src" in df.columns else "n/a")
    print(bad[["image_id","mt_ratio","fl_ratio","mm_per_pixel","scale_src"]].head(8).to_string() if "scale_src" in df.columns else bad.head(3))
PY
