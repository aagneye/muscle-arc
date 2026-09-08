#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
mkdir -p logs submissions
pkill -f 'calibrate_predict.py' 2>/dev/null || true
sleep 1
export CUDA_VISIBLE_DEVICES=0
nohup python scripts/calibrate_predict.py \
  --apo-ckpt experiments/checkpoints/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_fasc_v2/fasc_best.pt \
  --out submissions/submission_v5.csv \
  > logs/infer_v5.log 2>&1 &
echo $! > logs/infer_v5.pid
echo "STARTED_PID=$(cat logs/infer_v5.pid)"
sleep 6
ps -p "$(cat logs/infer_v5.pid)" -o pid,etime,cmd
tail -20 logs/infer_v5.log || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
