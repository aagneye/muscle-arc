#!/usr/bin/env bash
# Phase A scale audit + Phase A' residual path (metadata failed → OSF regressor).
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
pip install openpyxl -q
mkdir -p logs experiments submissions data/external

echo "=== TIFF metadata audit (decision A vs A') ==="
python scripts/audit_tiff_scales.py --out experiments/scale_table.csv | tee logs/scale_audit_phaseA.log

echo "=== OSF download (best effort) ==="
python scripts/download_osf_benchmarks.py 2>&1 | tee logs/osf_download.log | tail -30 || true

echo "=== build OSF expert GT CSV ==="
python scripts/build_osf_gt_csv.py \
  --out experiments/osf_expert_gt.csv \
  2>&1 | tee logs/osf_gt_build.log

echo "=== train Phase A' residual models on OSF ==="
python scripts/train_residual_scale.py \
  --gt experiments/osf_expert_gt.csv \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out-dir experiments/residual_models \
  2>&1 | tee logs/train_residual.log

echo "=== v9 infer with residual scales ==="
export CUDA_VISIBLE_DEVICES=0
python scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out submissions/submission_v9.csv \
  --thr 0.30 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  --residual-model-dir experiments/residual_models \
  --residual-prefer scale \
  2>&1 | tee logs/infer_v9.log

echo "=== local MAE gate (sample + OSF if preds exist) ==="
python scripts/eval_umud_local.py \
  --pred submissions/submission_v8.csv \
  --save-baseline || true
python scripts/eval_umud_local.py --pred submissions/submission_v9.csv || true

# Also report OSF holdout-style MAE if we have a pred CSV over OSF paths (optional)
python scripts/eval_umud_local.py \
  --pred submissions/submission_v9.csv \
  --external-gt experiments/osf_expert_gt.csv \
  --baseline experiments/local_mae_baseline_osf.json \
  --save-baseline || true

echo "Phase A/A' done. Review logs/scale_audit_phaseA.log and logs/train_residual.log"
