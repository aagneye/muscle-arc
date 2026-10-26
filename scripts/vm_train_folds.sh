#!/usr/bin/env bash
# 5-fold dual U-Net training queue for rogii-gpu.
#
# Builds group-aware K-fold manifests, then launches one train.py job per
# (fold, branch) on whichever allowed GPU has enough free memory, at most
# MAX_PER_GPU of our jobs per GPU. Longest jobs (fasc, ~3 h) are queued first.
# Jobs that die (e.g. OOM because another project grabbed the GPU) are retried
# in up to three rounds. Never touches other users' processes: it only waits
# for free memory.
#
# Usage: nohup bash scripts/vm_train_folds.sh > logs/k5_queue.log 2>&1 &
#   env: CFG, K, SPLITS_ROOT, CKPT_ROOT, MIN_FREE_MB, MAX_PER_GPU, GPUS, LOG_TAG
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
GPUS="${GPUS:-}"  # optional space-separated allow-list, e.g. GPUS="0 3"
POLL_S="${POLL_S:-60}"
LOG_TAG="${LOG_TAG:-k5}"  # log/done prefix, one per training line
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
    if [[ -n "$GPUS" && " $GPUS " != *" $idx "* ]]; then continue; fi
    if (( free >= MIN_FREE_MB )) && (( $(ours_on_gpu "$idx") < MAX_PER_GPU )); then
      echo "$idx"; return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

done_file() { echo "logs/${LOG_TAG}_fold${2}_${1}.done"; }

run_round() {
  local job branch f out gpu
  for job in "${JOBS[@]}"; do
    branch=${job%%:*}; f=${job##*:}
    out="$CKPT_ROOT/fold$f"
    if [[ -f "$out/${branch}_best.pt" && -f "$(done_file "$branch" "$f")" ]]; then
      echo "$(date -u +%FT%TZ) skip $job (done)"; continue
    fi
    until gpu=$(pick_gpu); do sleep "$POLL_S"; done
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "python -u scripts/train.py --config '$CFG' \
        --branch $branch --split-dir '$SPLITS_ROOT/fold$f' --checkpoint-dir '$out' \
        && touch '$(done_file "$branch" "$f")'" > "logs/${LOG_TAG}_fold${f}_${branch}.log" 2>&1 &
    RUNNING_ON[$!]=$gpu
    echo "$(date -u +%FT%TZ) launched $job on GPU $gpu (pid $!)"
    sleep 90  # let the job allocate memory before the next free-memory check
  done
  wait
}

for round in 1 2 3; do
  run_round
  pending=0
  for job in "${JOBS[@]}"; do
    [[ -f "$(done_file "${job%%:*}" "${job##*:}")" ]] || pending=$((pending + 1))
  done
  (( pending == 0 )) && break
  echo "$(date -u +%FT%TZ) round $round: $pending job(s) not finished, retrying in 10 min"
  sleep 600
done
echo "$(date -u +%FT%TZ) all fold jobs finished"
ls -la "$CKPT_ROOT"/fold*/
