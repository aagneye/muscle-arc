#!/usr/bin/env bash
# After vm_train_folds.sh: cache the 5-fold ensemble probabilities on GPU and
# build the candidate with the geometry of the best public file (0.44273):
# legacy PA (Radon blend) +2 deg, legacy FL, legacy MT, 5-frame clip smoothing.
# Does NOT submit; prints a gate summary for review.
#
# Usage: nohup bash scripts/vm_post_k5.sh > logs/post_k5.log 2>&1 &
set -uo pipefail
cd /home/azureuser/muscle-arc
# shellcheck disable=SC1091
source .venv/bin/activate

CKPT_ROOT="${CKPT_ROOT:-experiments/checkpoints_dual_v3_k5}"
K="${K:-5}"
TAG="${TAG:-k5}"

until [[ $(ls logs/k5_fold*_*.done 2>/dev/null | wc -l) -ge $((2 * K)) ]]; do sleep 120; done
echo "$(date -u +%FT%TZ) all $((2 * K)) fold jobs done"

APO=(); FASC=()
for f in $(seq 0 $((K - 1))); do
  APO+=("$CKPT_ROOT/fold$f/apo_best.pt"); FASC+=("$CKPT_ROOT/fold$f/fasc_best.pt")
done

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -nr | head -1 | cut -d, -f1)
for imgs in osf test; do
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/cache_probs.py --config configs/dual_v3.yaml \
    --apo-ckpt "${APO[@]}" --fasc-ckpt "${FASC[@]}" --images "$imgs" --tag "$TAG" \
    --device cuda --multiscale
done

python -u scripts/geometry_sweep.py --tag "$TAG" --procs 24 \
  --pick "pa_legacy+2" "fl_legacy|trig0" "mt_legacy" \
  --write-candidate "submissions/cand_${TAG}.csv"

python - <<PY
import numpy as np, pandas as pd
c = pd.read_csv("submissions/cand_${TAG}.csv")
b = pd.read_csv("submissions/cand_D_pa2_sm.csv").set_index("image_id").reindex(c.image_id).reset_index()
ok = len(c) == 309 and not c.isna().any().any()
print("GATE rows/NaN ok:", ok)
for col in ("pa_deg", "fl_mm", "mt_mm"):
    d = c[col] - b[col]
    print(f"GATE {col}: median {c[col].median():.2f}, std {c[col].std():.2f} (best-file std {b[col].std():.2f}), "
          f"mean shift {d.mean():+.2f}, mean|shift| {d.abs().mean():.2f}")
PY
echo "$(date -u +%FT%TZ) post_k5 done -> submissions/cand_${TAG}.csv"
