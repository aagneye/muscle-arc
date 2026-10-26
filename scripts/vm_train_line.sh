#!/usr/bin/env bash
# (Re)start one 5-fold training line on the VM, detached, with a pid file so a
# later call replaces the queue without pkill (which kills the ssh shell when
# the pattern is on its own command line).
#
# Usage: bash scripts/vm_train_line.sh TAG CONFIG "GPUS" [MIN_FREE_MB]
#   e.g. bash scripts/vm_train_line.sh k768 configs/dual_v3_768.yaml "0 3" 14000
set -uo pipefail
cd /home/azureuser/muscle-arc
TAG=$1; CFG=$2; GPUS=${3:-}; MIN_FREE_MB=${4:-6000}
NAME=$(basename "$CFG" .yaml)
PIDFILE="logs/${TAG}_queue.pid"
mkdir -p logs
if [[ -f $PIDFILE ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  kill "$(cat "$PIDFILE")" && echo "stopped previous $TAG queue $(cat "$PIDFILE")"
  sleep 1
fi
CFG=$CFG SPLITS_ROOT=experiments/splits_k5 CKPT_ROOT="experiments/checkpoints_${NAME}_k5" \
  GPUS=$GPUS MAX_PER_GPU="${MAX_PER_GPU:-1}" MIN_FREE_MB=$MIN_FREE_MB LOG_TAG=$TAG \
  setsid nohup bash scripts/vm_train_folds.sh > "logs/${TAG}_queue.log" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "$TAG queue pid $! (config $CFG, GPUs '${GPUS:-any}', min free ${MIN_FREE_MB} MB)"
