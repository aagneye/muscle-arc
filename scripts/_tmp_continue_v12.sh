#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
mkdir -p logs submissions/probes

echo "=== Generate MT probes ==="
python scripts/probe_public_median.py --axis mt --values 16 22 28 \
  --pa 15.2 --fl 81.5 --out-dir submissions/probes | tee logs/probe_mt_gen.log

echo "=== Soft Gate2 decision ==="
python3 - <<'PY'
import json
from pathlib import Path
g2 = json.loads(Path("experiments/gate2_osf_umud.json").read_text())
print(g2)
soft = g2.get("umud", 9) < 1.20 and g2.get("n_mt_bad", 99) <= 15
Path("experiments/v12_infer_decision.txt").write_text("INFER\n" if soft else "HOLD\n")
print("INFER_DECISION", "INFER" if soft else "HOLD")
PY

if ! grep -q INFER experiments/v12_infer_decision.txt; then
  echo HOLD > experiments/v12_submit_decision.txt
  echo "HOLD infer"
  exit 0
fi

APO=experiments/checkpoints_phase4/apo_best.pt
FASC=experiments/checkpoints_phase4/fasc_best.pt
FASC2=""
[[ -f experiments/checkpoints_seed2/fasc_best.pt ]] && FASC2="--fasc-ckpt2 experiments/checkpoints_seed2/fasc_best.pt"

echo "=== re-audit depth (cropped ticks) ==="
python -u scripts/audit_depth_scale.py --out experiments/depth_scale_table.csv \
  2>&1 | tee logs/depth_scale_audit_v12.log || true

echo "=== v12 infer ==="
export CUDA_VISIBLE_DEVICES=0
# shellcheck disable=SC2086
python -u scripts/calibrate_predict.py \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" $FASC2 \
  --out submissions/submission_v12.csv \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  --depth-scale-table experiments/depth_scale_table.csv \
  --sector-crop true --live-depth-scale true \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  --debug-csv experiments/submission_v12_debug.csv \
  2>&1 | tee logs/infer_v12.log

python scripts/eval_gate4_dist.py \
  --cand submissions/submission_v12.csv \
  --ref submissions/submission_v8.csv \
  --out experiments/gate4_v12.json || true

python scripts/eval_umud_local.py --pred submissions/submission_v12.csv || true

python3 - <<'PY'
import json
from pathlib import Path
g2 = json.loads(Path("experiments/gate2_osf_umud.json").read_text())
g4 = json.loads(Path("experiments/gate4_v12.json").read_text()) if Path("experiments/gate4_v12.json").exists() else {}
# Strict submit: gate2_ok OR (umud<0.55 and mt_bad<8) AND gate4
submit = bool(g4.get("gate4_ok")) and (
    bool(g2.get("gate2_ok")) or (g2.get("umud", 9) < 0.55 and g2.get("n_mt_bad", 99) < 8)
)
Path("experiments/v12_submit_decision.txt").write_text("SUBMIT\n" if submit else "HOLD\n")
print("SUBMIT_DECISION", "SUBMIT" if submit else "HOLD")
print("gate2_umud", g2.get("umud"), "mt_bad", g2.get("n_mt_bad"), "gate4", g4)
PY
echo DONE
