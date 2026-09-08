#!/usr/bin/env bash
set -euo pipefail
pgrep -af 'train_fasc_tversky|scripts/train.py' || echo NO_TRAIN
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
echo '--- epochs ---'
grep -a '\[fasc_v2\] epoch\|saved\|done best\|Traceback\|Error' /home/azureuser/muscle-arc/logs/train_fasc_v2_gpu1.log 2>/dev/null \
  | tr '\r' '\n' \
  | grep -E 'fasc_v2|saved|done|Traceback|Error' \
  | tail -25 || true
echo '--- tail ---'
tail -c 500 /home/azureuser/muscle-arc/logs/train_fasc_v2_gpu1.log 2>/dev/null | tr '\r' '\n' | tail -10 || true
ls -lh /home/azureuser/muscle-arc/experiments/checkpoints_fasc_v2 2>/dev/null || true
