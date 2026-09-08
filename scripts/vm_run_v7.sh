#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q

echo "=== geometry validator (gate A) ==="
python scripts/eval_geometry.py --max-samples 100
EVAL_RC=$?

echo "=== inference v7 (extrapolated-line geometry, v3 weights) ==="
export CUDA_VISIBLE_DEVICES=0
python scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints/apo_best.pt \
  --fasc-ckpt experiments/checkpoints/fasc_best.pt \
  --out submissions/submission_v7.csv \
  --thr 0.30 \
  --temporal-smooth false

echo "=== compare vs v3 ==="
python3 - <<'PY'
import pandas as pd
from pathlib import Path
v3 = Path("submissions/submission_v3.csv")
v7 = Path("submissions/submission_v7.csv")
if not v7.exists():
    raise SystemExit("missing v7")
d7 = pd.read_csv(v7)
print("v7 describe:")
print(d7.describe().round(3).to_string())
print("v7 PA median", float(d7.pa_deg.median()), "FL>=139", int((d7.fl_mm>=139).sum()))
if v3.exists():
    d3 = pd.read_csv(v3)
    print("v3 PA median", float(d3.pa_deg.median()), "FL>=139", int((d3.fl_mm>=139).sum()))
    print("delta mean PA", float(d7.pa_deg.mean()-d3.pa_deg.mean()))
    print("delta mean FL", float(d7.fl_mm.mean()-d3.fl_mm.mean()))
PY

exit $EVAL_RC
