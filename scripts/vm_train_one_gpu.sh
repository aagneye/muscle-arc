#!/usr/bin/env bash
# Start apo+fasc training on a single GPU (default: 0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p experiments/checkpoints logs

echo "Training on physical GPU $GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader

nohup python scripts/train.py --config configs/default.yaml --branch both \
  > "logs/train_gpu${GPU}.log" 2>&1 &
echo $! > "logs/train_gpu${GPU}.pid"
echo "PID $(cat logs/train_gpu${GPU}.pid)  log=logs/train_gpu${GPU}.log"
