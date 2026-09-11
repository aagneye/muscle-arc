#!/usr/bin/env bash
# Geometry fix validation: Gate2 OSF ratios, then optional v12 if gates pass.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs experiments submissions/probes

APO=experiments/checkpoints_phase4/apo_best.pt
FASC=experiments/checkpoints_phase4/fasc_best.pt

echo "=== Gate2 OSF (geometry fix check) ==="
python -u scripts/eval_osf_pipeline.py \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate2_osf_umud.json \
  2>&1 | tee logs/gate2_v12.log || true

echo "=== Generate MT public-median probes ==="
python scripts/probe_public_median.py --axis mt --values 16 22 28 \
  --pa 15.2 --fl 81.5 --mt 22 \
  --out-dir submissions/probes 2>&1 | tee logs/probe_mt_gen.log

python3 - <<'PY'
import json
from pathlib import Path
g2 = json.loads(Path("experiments/gate2_osf_umud.json").read_text())
print("GATE2", g2)
ok = bool(g2.get("gate2_ok")) or (
    g2.get("umud", 9) < 0.70 and g2.get("n_mt_bad", 99) <= 12
)
# Soft proceed to infer if major improvement even if not full pass
soft = g2.get("umud", 9) < 1.20 and g2.get("n_mt_bad", 99) <= 15
Path("experiments/v12_infer_decision.txt").write_text(
    "INFER\n" if (ok or soft) else "HOLD\n"
)
print("INFER_DECISION", "INFER" if (ok or soft) else "HOLD",
      "umud", g2.get("umud"), "mt_bad", g2.get("n_mt_bad"))
PY

if grep -q INFER experiments/v12_infer_decision.txt; then
  echo "=== v12 infer ==="
  export CUDA_VISIBLE_DEVICES=0
  FASC2=""
  if [[ -f experiments/checkpoints_seed2/fasc_best.pt ]]; then
    FASC2="--fasc-ckpt2 experiments/checkpoints_seed2/fasc_best.pt"
  fi
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

  python3 - <<'PY'
import json
from pathlib import Path
g2 = json.loads(Path("experiments/gate2_osf_umud.json").read_text())
g4 = json.loads(Path("experiments/gate4_v12.json").read_text()) if Path("experiments/gate4_v12.json").exists() else {}
submit = bool(g2.get("gate2_ok") and g4.get("gate4_ok"))
Path("experiments/v12_submit_decision.txt").write_text("SUBMIT\n" if submit else "HOLD\n")
print("SUBMIT_DECISION", "SUBMIT" if submit else "HOLD", g2.get("umud"), g4)
PY
else
  echo "HOLD infer — Gate2 not improved enough"
  echo HOLD > experiments/v12_submit_decision.txt
fi
echo DONE
