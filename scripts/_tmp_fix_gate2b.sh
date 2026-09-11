#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
# leave calibrate_predict (48434) alone
kill 71752 73925 2>/dev/null || true
sleep 2
source .venv/bin/activate
# confirm metrics is HEAD (no IQR filter)
if grep -q 'arr < 0.42 \* h' src/muscle_arc/geometry/metrics.py; then
  echo "WARN: metrics still has IQR filter"
else
  echo "metrics looks clean"
fi
nohup bash scripts/_tmp_gate2b_only.sh > logs/gate2b_only.out 2>&1 &
echo GATE2B_STARTED
sleep 3
pgrep -af eval_osf_pipeline || true
pgrep -af calibrate_predict || true
