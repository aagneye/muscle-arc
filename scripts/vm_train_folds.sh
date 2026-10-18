#!/usr/bin/env bash
# 5-fold dual U-Net training queue for rogii-gpu.
#
# Builds group-aware K-fold manifests, then launches one train.py job per
# (fold, branch) on whichever GPU has enough free memory, at most MAX_PER_GPU
# of our jobs per GPU. Longest jobs (fasc, ~3 h) are queued first. Never
# touches other users' processes: it only waits for free memory.
#
# Usage: nohup bash scripts/vm_train_folds.sh > logs/k5_queue.log 2>&1 &
set -uo pipefail
cd /home/azureuser/muscle-arc
# shellcheck disable=SC1091
source .venv/bin/activate

CFG="${CFG:-configs/dual_v3.yaml}"
K="${K:-5}"
SPLITS_ROOT="${SPLITS_ROOT:-experiments/splits_k5}"
CKPT_ROOT="${CKPT_ROOT:-experiments/checkpoints_dual_v3_k5}"
MIN_FREE_MB="${MIN_FREE_MB:-6000}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
POLL_S="${POLL_S:-60}"
mkdir -p logs "$SPLITS_ROOT" "$CKPT_ROOT"

for f in $(seq 0 $((K - 1))); do
  if [[ ! -f "$SPLITS_ROOT/fold$f/fasc_splits.json" ]]; then
    python scripts/make_splits.py --config "$CFG" --kfold "$K" --fold "$f" \
      --out-dir "$SPLITS_ROOT/fold$f" | tail -2
  fi
done

JOBS=()
for f in $(seq 0 $((K - 1))); do JOBS+=("fasc:$f"); done
for f in $(seq 0 $((K - 1))); do JOBS+=("apo:$f"); done

declare -A RUNNING_ON  # pid -> gpu index

ours_on_gpu() {
  local gpu=$1 n=0 pid
  for pid in "${!RUNNING_ON[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      [[ "${RUNNING_ON[$pid]}" == "$gpu" ]] && n=$((n + 1))
    else
      unset "RUNNING_ON[$pid]"
    fi
  done
  echo "$n"
}

pick_gpu() {
  local idx free
  while IFS=, read -r idx free; do
    idx=${idx// /}; free=${free// /}
    if (( free >= MIN_FREE_MB )) && (( $(ours_on_gpu "$idx") < MAX_PER_GPU )); then
      echo "$idx"; return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

for job in "${JOBS[@]}"; do
  branch=${job%%:*}; f=${job##*:}
  out="$CKPT_ROOT/fold$f"
  if [[ -f "$out/${branch}_best.pt" && -f "logs/k5_fold${f}_${branch}.done" ]]; then
    echo "$(date -u +%FT%TZ) skip $job (done)"; continue
  fi
  until gpu=$(pick_gpu); do sleep "$POLL_S"; done
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "python -u scripts/train.py --config '$CFG' \
      --branch $branch --split-dir '$SPLITS_ROOT/fold$f' --checkpoint-dir '$out' \
      && touch logs/k5_fold${f}_${branch}.done" > "logs/k5_fold${f}_${branch}.log" 2>&1 &
  RUNNING_ON[$!]=$gpu
  echo "$(date -u +%FT%TZ) launched $job on GPU $gpu (pid $!)"
  sleep 90  # let the job allocate memory before the next free-memory check
done

wait
echo "$(date -u +%FT%TZ) all fold jobs finished"
ls -la "$CKPT_ROOT"/fold*/
