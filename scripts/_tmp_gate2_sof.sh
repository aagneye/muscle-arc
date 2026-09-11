#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
echo "=== Fair Gate2 SOF@768 GT scale ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/sof.yaml \
  --sof-ckpt experiments/checkpoints_sof/sof_best.pt \
  --masks-from ours --scale-mode gt \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate2_sof_gt.json \
  2>&1 | tee logs/goal037_gate2_sof.log
python - <<'PY'
import json
m=json.load(open("experiments/gate2_sof_gt.json"))
print("--- SOF Gate2 ---")
for k in ("n","umud","pa_mae","fl_mae","mt_mae","gate2_ok","n_mt_bad","masks_from","img_size","sof_ckpt"):
    print(f"{k}: {m.get(k)}")
PY
