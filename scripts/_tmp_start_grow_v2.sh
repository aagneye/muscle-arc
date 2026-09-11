#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
mkdir -p logs
sed -i 's/\r$//' scripts/vm_run_grow_v2.sh
chmod +x scripts/vm_run_grow_v2.sh
# stop any leftover grow job
pkill -f 'vm_run_grow|scripts/train.py' 2>/dev/null || true
sleep 2
nohup bash scripts/vm_run_grow_v2.sh > logs/grow_v2_run.log 2>&1 &
echo "PID=$!"
sleep 5
tail -40 logs/grow_v2_run.log || true
ps aux | grep -E 'vm_run_grow_v2|train.py' | grep -v grep || true
