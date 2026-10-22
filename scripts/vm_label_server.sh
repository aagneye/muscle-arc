#!/usr/bin/env bash
# (Re)start the manual-measurement server on the VM, detached, on 127.0.0.1:8765.
# Reach it from the laptop with:  ssh -N -L 8765:127.0.0.1:8765 azureuser@<vm>
set -euo pipefail
cd /home/azureuser/muscle-arc
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p logs
pkill -f "python -u scripts/label_server.py" || true
sleep 1
setsid nohup python -u scripts/label_server.py --port "${PORT:-8765}" > logs/label_server.log 2>&1 < /dev/null &
sleep 2
cat logs/label_server.log
