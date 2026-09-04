#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
echo "=== log size ==="
wc -l logs/train_gpu0.log || true
echo "=== epoch lines ==="
grep -a 'epoch\|saved\|Apo\|Fasc\|Error\|Traceback\|pairs' logs/train_gpu0.log | tail -40 || true
echo "=== end of log ==="
tail -c 2000 logs/train_gpu0.log | tr '\r' '\n' | tail -30
echo "=== pid ==="
ps -p 9507 -o pid,etime,pcpu,pmem,cmd || echo dead
