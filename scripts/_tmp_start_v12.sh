#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
mkdir -p logs
sed -i 's/\r$//' scripts/vm_run_v12_geometry.sh
nohup bash scripts/vm_run_v12_geometry.sh > logs/v12_geometry_run.log 2>&1 &
echo "PID=$!"
sleep 3
ps aux | grep -E 'vm_run_v12|eval_osf|calibrate_predict' | grep -v grep || true
tail -20 logs/v12_geometry_run.log || true
