#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
# Kill any old midnight waiters
pkill -f '_tmp_midnight_submit' 2>/dev/null || true
pkill -f 'QUOTA_WINDOW' 2>/dev/null || true
# kill python waiters that print remaining_s
sleep 1
cp -f scripts/_tmp_midnight_submit.sh /tmp/midnight_submit_run.sh
sed -i 's/\r$//' /tmp/midnight_submit_run.sh
nohup bash /tmp/midnight_submit_run.sh > logs/midnight_submit.out 2>&1 &
echo MID=$!
sleep 2
pgrep -af midnight_submit || pgrep -af midnight_submit_run || true
head -6 logs/midnight_submit.out
