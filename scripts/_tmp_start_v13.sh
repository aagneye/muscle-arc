#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pkill -f 'vm_run_scale_climb|audit_depth_scale|train_scale_detector' 2>/dev/null || true
sleep 1
nohup bash scripts/vm_run_scale_climb.sh > logs/v13_scale_climb_master.log 2>&1 &
echo "PID=$!"
sleep 5
head -50 logs/v13_scale_climb_master.log
