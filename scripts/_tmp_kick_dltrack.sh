#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
python3 - <<'PY'
import json
from pathlib import Path
for p in ["experiments/gate2_sof_gt.json","experiments/gate0_reference.json"]:
    path=Path(p)
    if not path.exists():
        print(p, "MISSING")
        continue
    m=json.load(open(path))
    keys=["n","umud","pa_mae","fl_mae","mt_mae","gate2_ok","gate0_ok","n_mt_bad","img_size","masks_from"]
    print(p, {k:m.get(k) for k in keys})
PY
nohup bash scripts/_tmp_infer_dltrack.sh > logs/goal037_dltrack_infer.out 2>&1 &
echo "started pid=$!"
