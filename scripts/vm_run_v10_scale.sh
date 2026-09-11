#!/usr/bin/env bash
# Top-10: sector crop + depth OCR/ticks → v10 infer (residual off).
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
pip install pytesseract openpyxl -q || true
# system tesseract if missing
sudo apt-get install -y tesseract-ocr >/dev/null 2>&1 || true
mkdir -p logs experiments submissions

echo "=== audit depth OCR/ticks on test ==="
python -u scripts/audit_depth_scale.py \
  --out experiments/depth_scale_table.csv \
  2>&1 | tee logs/depth_scale_audit.log

echo "=== v10 infer: sector crop + depth scales ==="
export CUDA_VISIBLE_DEVICES=0
python -u scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out submissions/submission_v10.csv \
  --thr 0.30 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  --depth-scale-table experiments/depth_scale_table.csv \
  --sector-crop true \
  --live-depth-scale true \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee logs/infer_v10.log

echo "=== local gates ==="
python scripts/eval_umud_local.py --pred submissions/submission_v8.csv --save-baseline || true
python scripts/eval_umud_local.py --pred submissions/submission_v10.csv || true
python scripts/ensemble_submissions.py \
  --cands submissions/submission_v8.csv submissions/submission_v10.csv \
  --out submissions/submission_best_local.csv || true

python3 - <<'PY'
import pandas as pd
from pathlib import Path
for name in ["submission_v8.csv","submission_v10.csv","submission_best_local.csv"]:
    p=Path("submissions")/name
    if not p.exists():
        continue
    d=pd.read_csv(p)
    print(name, "FL_std", float(d.fl_mm.std()), "MT_std", float(d.mt_mm.std()),
          "FL_med", float(d.fl_mm.median()), "MT_med", float(d.mt_mm.median()))
PY
echo DONE
