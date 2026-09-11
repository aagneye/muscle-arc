#!/usr/bin/env bash
# Phase D — only after public < 0.40. Console-aug SOF retrain toward Gate2 ≤ 0.37.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate

# Guard: require evidence of public < 0.40
python3 - <<'PY'
import json, sys
from pathlib import Path
p = Path("experiments/post_midnight_scores.txt")
ok = False
if p.exists():
    for line in p.read_text().splitlines():
        parts = line.split()
        for tok in parts:
            try:
                if float(tok) < 0.40:
                    ok = True
            except ValueError:
                pass
best = Path("experiments/best_public.txt")
if best.exists():
    try:
        if float(best.read_text().strip()) < 0.40:
            ok = True
    except ValueError:
        pass
if not ok:
    print("BLOCKED: stretch-sof requires publicScore < 0.40 first")
    sys.exit(2)
print("public < 0.40 confirmed — starting console-aug SOF retrain")
PY

TAG=sof_console_d
CKPT_DIR=experiments/checkpoints_${TAG}
mkdir -p logs "$CKPT_DIR"

# sof.yaml already has aug_strength: console; paste_console_chrome wired in train_multitask
python -u scripts/train_multitask.py \
  --config configs/sof.yaml \
  --checkpoint-dir "$CKPT_DIR" \
  2>&1 | tee logs/${TAG}_train.log

python -u scripts/eval_osf_pipeline.py \
  --config configs/sof.yaml \
  --sof-ckpt "${CKPT_DIR}/sof_best.pt" \
  --scale-mode gt \
  --sector-crop false \
  --chrome-mask true \
  --geom ours \
  --max-images 35 \
  --out experiments/gate2_${TAG}.json \
  2>&1 | tee logs/gate2_${TAG}.log || true

python3 -c "import json; m=json.load(open('experiments/gate2_${TAG}.json')); print('Gate2', m.get('umud')); print('OK' if m.get('umud',9)<=0.37 else 'NEED_MORE')"
echo DONE_STRETCH_SOF
