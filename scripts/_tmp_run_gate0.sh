#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python - <<'PY'
from muscle_arc.models.dl_track import find_dl_track_models
print(find_dl_track_models("data/external/dl_track"))
PY
echo "=== Gate0 ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track --scale-mode gt \
  --sector-crop false \
  --out experiments/gate0_reference.json \
  --gate0-out experiments/gate0_reference.json \
  2>&1 | tee logs/goal037_gate0.log
python - <<'PY'
import json
m=json.load(open("experiments/gate0_reference.json"))
print("--- summary ---")
for k in ("n","umud","pa_mae","fl_mae","mt_mae","gate0_ok","n_mt_bad","n_fl_bad","mt_ratio_median","fl_ratio_median"):
    print(f"{k}: {m.get(k)}")
PY
