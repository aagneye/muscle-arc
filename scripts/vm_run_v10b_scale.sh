#!/usr/bin/env bash
# v10b: tighter tick band + no sample gain + PA-only temporal when OCR dominates
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs experiments submissions

echo "=== re-audit depth OCR/ticks ==="
python -u scripts/audit_depth_scale.py \
  --out experiments/depth_scale_table.csv \
  2>&1 | tee logs/depth_scale_audit_v10b.log

echo "=== v10b infer ==="
export CUDA_VISIBLE_DEVICES=0
python -u scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out submissions/submission_v10b.csv \
  --thr 0.30 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  --depth-scale-table experiments/depth_scale_table.csv \
  --sector-crop true \
  --live-depth-scale true \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee logs/infer_v10b.log

echo "=== local gates ==="
python scripts/eval_umud_local.py --pred submissions/submission_v8.csv --save-baseline || true
python scripts/eval_umud_local.py --pred submissions/submission_v10b.csv || true
python scripts/ensemble_submissions.py \
  --cands submissions/submission_v8.csv submissions/submission_v10.csv submissions/submission_v10b.csv \
  --out submissions/submission_best_local.csv || true

python3 - <<'PY'
import pandas as pd
from pathlib import Path
for name in ["submission_v8.csv","submission_v10.csv","submission_v10b.csv","submission_best_local.csv"]:
    p=Path("submissions")/name
    if not p.exists():
        continue
    d=pd.read_csv(p)
    print(name, "FL_std", round(float(d.fl_mm.std()),2), "MT_std", round(float(d.mt_mm.std()),2),
          "FL_med", round(float(d.fl_mm.median()),2), "MT_med", round(float(d.mt_mm.median()),2))
PY
echo DONE
