#!/usr/bin/env bash
# Cache fold-averaged probabilities for every checkpoint root given, then build the
# candidate CSV with the best-public geometry (pa_legacy+2 / fl_legacy / mt_legacy).
# Runs detached; never touches other users' processes.
#
# Usage: bash scripts/vm_build_ensemble.sh TAG CKPT_ROOT [CKPT_ROOT ...]
#   e.g. bash scripts/vm_build_ensemble.sh k15 experiments/checkpoints_dual_v3_k5 \
#          experiments/checkpoints_dual_v3_k5_s2 experiments/checkpoints_dual_v3_k5_s3
set -euo pipefail
CFG="${CFG:-configs/dual_v3.yaml}"  # must match the img_size the checkpoints were trained at
cd /home/azureuser/muscle-arc
# shellcheck disable=SC1091
source .venv/bin/activate
TAG=$1; shift
APO=(); FASC=()
for root in "$@"; do
  for f in 0 1 2 3 4; do APO+=("$root/fold$f/apo_best.pt"); FASC+=("$root/fold$f/fasc_best.pt"); done
done
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "$(date -u +%FT%TZ) tag=$TAG models=${#APO[@]} gpu=$GPU"
mkdir -p logs
setsid nohup bash -c "
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/cache_probs.py --config $CFG --apo-ckpt ${APO[*]} --fasc-ckpt ${FASC[*]} --images osf --tag $TAG --device cuda --multiscale &&
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/cache_probs.py --config $CFG --apo-ckpt ${APO[*]} --fasc-ckpt ${FASC[*]} --images test --tag $TAG --device cuda --multiscale &&
  python -u scripts/geometry_sweep.py --tag $TAG --procs 24 --pick pa_legacy+2 'fl_legacy|trig0' mt_legacy --write-candidate submissions/cand_$TAG.csv &&
  echo BUILD_DONE
" > "logs/build_$TAG.log" 2>&1 < /dev/null &
echo "started, log: logs/build_$TAG.log"
