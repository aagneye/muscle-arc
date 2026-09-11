#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
sed -i 's/\r$//' scripts/_tmp_win_climb.sh scripts/_tmp_midnight_submit.sh scripts/debug_apo_pair_overlays.py || true
chmod +x scripts/_tmp_win_climb.sh scripts/_tmp_midnight_submit.sh
source .venv/bin/activate
pip install -e . -q
pkill -f '_tmp_win_climb.sh' 2>/dev/null || true
pkill -f 'eval_osf_pipeline.py' 2>/dev/null || true
pkill -f 'calibrate_predict.py' 2>/dev/null || true
pkill -f '_tmp_midnight_submit.sh' 2>/dev/null || true
sleep 1
mkdir -p logs
nohup bash scripts/_tmp_win_climb.sh > logs/win_climb.out 2>&1 &
echo "WIN_PID=$!"
nohup bash scripts/_tmp_midnight_submit.sh > logs/midnight_submit.out 2>&1 &
echo "MID_PID=$!"
sleep 2
head -25 logs/win_climb.out || true
echo ---
head -8 logs/midnight_submit.out || true
