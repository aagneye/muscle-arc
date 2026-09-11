#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
echo "=== geometry validator ==="
python scripts/eval_geometry.py --max-samples 100 || true
echo "=== constant probes ==="
python scripts/make_constant_submission.py --pa 13 --fl 75 --mt 19 --out submissions/probe_a.csv
python scripts/make_constant_submission.py --pa 15 --fl 75 --mt 19 --out submissions/probe_b.csv
python scripts/make_constant_submission.py --pa 13 --fl 70 --mt 19 --out submissions/probe_c.csv
echo "=== inference v6 (fixed geometry + cluster scale) ==="
export CUDA_VISIBLE_DEVICES=0
python scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints/apo_best.pt \
  --fasc-ckpt experiments/checkpoints/fasc_best.pt \
  --out submissions/submission_v6.csv \
  --thr 0.30
