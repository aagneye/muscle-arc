#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
mkdir -p logs
sed -i 's/\r$//' scripts/vm_run_v11.sh
# ensure submit helper present
if [[ ! -f /tmp/vm_submit_v8.sh ]]; then
  cp scripts/vm_submit_v8.sh /tmp/vm_submit_v8.sh 2>/dev/null || true
  sed -i 's/\r$//' /tmp/vm_submit_v8.sh 2>/dev/null || true
fi
nohup bash scripts/vm_run_v11.sh > logs/v11_run.log 2>&1 &
echo "PID=$!"
sleep 2
tail -20 logs/v11_run.log || true
