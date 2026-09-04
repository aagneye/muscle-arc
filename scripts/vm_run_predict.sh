#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
mkdir -p submissions logs
echo "Checkpoints:"
ls -lh experiments/checkpoints/
echo "Running predict..."
python scripts/predict.py \
  --config configs/default.yaml \
  --apo-ckpt experiments/checkpoints/apo_best.pt \
  --fasc-ckpt experiments/checkpoints/fasc_best.pt \
  --out submissions/submission.csv \
  2>&1 | tee logs/predict_gpu0.log
echo "--- submission preview ---"
head -10 submissions/submission.csv
wc -l submissions/submission.csv
