#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
mkdir -p logs
# strip CRLF if any
sed -i 's/\r$//' scripts/vm_run_v10_scale.sh
nohup bash scripts/vm_run_v10_scale.sh > logs/v10_scale_run.log 2>&1 &
echo "PID=$!"
sleep 3
tail -40 logs/v10_scale_run.log || true
ps aux | grep -E 'vm_run_v10|audit_depth|calibrate_predict' | grep -v grep || true
