#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
# Kill stale infer (will crash on residual pickle anyway)
pkill -f 'calibrate_predict.py.*dltrack_win' 2>/dev/null || true
pkill -f '_tmp_win_finish.sh' 2>/dev/null || true
sleep 2
nohup bash scripts/_tmp_win_finish.sh > logs/win_finish.out 2>&1 &
echo RESTARTED
sleep 3
pgrep -af 'win_finish|calibrate_predict' | head -5
# Ensure midnight submitter still alive
pgrep -af midnight_submit || {
  nohup bash scripts/_tmp_midnight_submit.sh > logs/midnight_submit.out 2>&1 &
  echo MIDNIGHT_REARMED
}
