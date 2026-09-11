#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
echo "=== gate2b files ==="
for f in experiments/gate2b_dltrack_restore.json experiments/gate2b_dltrack_win.json experiments/gate2b_letterbox.json; do
  if [[ -f "$f" ]]; then
    python3 -c "import json; m=json.load(open('$f')); print('$f', 'umud', m.get('umud'), 'mt_bad', m.get('n_mt_bad'))"
  else
    echo "missing $f"
  fi
done
echo "=== restore log tail ==="
tail -20 logs/gate2b_restore.out || true
echo "=== processes ==="
ps aux | grep -E 'calibrate_predict|eval_osf|win_climb|midnight' | grep -v grep || true
echo "=== win csv ==="
wc -l submissions/submission_dltrack_win.csv 2>/dev/null || echo no_csv
tail -12 logs/win_climb.out || true
