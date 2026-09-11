#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
# ensure only one waiter
for pid in $(pgrep -f 'midnight_submit_run|/_tmp_midnight_submit' || true); do
  kill "$pid" 2>/dev/null || true
done
sleep 1
cp -f /home/azureuser/muscle-arc/scripts/_tmp_midnight_submit.sh /tmp/midnight_submit_run.sh
sed -i 's/\r$//' /tmp/midnight_submit_run.sh
nohup bash /tmp/midnight_submit_run.sh >> /home/azureuser/muscle-arc/logs/midnight_submit.out 2>&1 &
echo STARTED:$!
sleep 2
pgrep -af midnight_submit_run
tail -5 /home/azureuser/muscle-arc/logs/midnight_submit.out
