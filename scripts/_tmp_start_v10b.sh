#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
mkdir -p logs
sed -i 's/\r$//' scripts/vm_run_v10b_scale.sh
nohup bash scripts/vm_run_v10b_scale.sh > logs/v10b_scale_run.log 2>&1 &
echo "PID=$!"
sleep 3
ps aux | grep -E 'vm_run_v10b|audit_depth|calibrate_predict' | grep -v grep || echo none
tail -30 logs/v10b_scale_run.log || true
