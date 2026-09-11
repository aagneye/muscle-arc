#!/usr/bin/env bash
# Grow v2: keep apo, retrain fasc with Gate1-lite + stronger augs, then report card.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs experiments/checkpoints_grow_v2 submissions

CFG="${CFG:-configs/grow_v2.yaml}"
SPLIT_DIR="${SPLIT_DIR:-experiments/splits}"
# Keep grown apo; warm-start fasc from previous grow (fallback phase4)
APO_SRC="${APO_SRC:-experiments/checkpoints_grow/apo_best.pt}"
FASC_WARM="${FASC_WARM:-experiments/checkpoints_grow/fasc_best.pt}"
if [[ ! -f "$FASC_WARM" ]]; then
  FASC_WARM="experiments/checkpoints_phase4/fasc_best.pt"
fi
CKPT_DIR="${CKPT_DIR:-experiments/checkpoints_grow_v2}"
TAG="${TAG:-grow_v2}"

# Copy apo into v2 dir so gates use one folder
mkdir -p "$CKPT_DIR"
if [[ -f "$APO_SRC" ]]; then
  cp -f "$APO_SRC" "$CKPT_DIR/apo_best.pt"
  echo "Reusing apo from $APO_SRC"
else
  echo "WARN missing $APO_SRC — train apo if needed"
fi

echo "=== 1) Ensure splits exist ==="
if [[ ! -f "$SPLIT_DIR/apo_splits.json" || ! -f "$SPLIT_DIR/fasc_splits.json" ]]; then
  python -u scripts/make_splits.py --config "$CFG" --out-dir "$SPLIT_DIR" \
    2>&1 | tee "logs/${TAG}_splits.log"
else
  echo "Using existing $SPLIT_DIR"
fi

echo "=== 2) Train fasc v2 (Gate1-lite ckpt pick, stronger augs) ==="
python -u scripts/train.py --config "$CFG" --branch fasc --loss tversky \
  --split-dir "$SPLIT_DIR" \
  --warm-start "$FASC_WARM" \
  2>&1 | tee "logs/${TAG}_train_fasc.log"

APO="$CKPT_DIR/apo_best.pt"
FASC="$CKPT_DIR/fasc_best.pt"

echo "=== 3) Gate1 holdout ==="
python -u scripts/eval_gate1_geometry.py \
  --config "$CFG" \
  --split holdout --split-dir "$SPLIT_DIR" \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --baseline experiments/gate1_geometry.json \
  --out "experiments/gate1_${TAG}.json" \
  2>&1 | tee "logs/${TAG}_gate1.log" || true

echo "=== 4) Gate2 OSF ==="
python -u scripts/eval_osf_pipeline.py \
  --config "$CFG" \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out "experiments/gate2_${TAG}.json" \
  2>&1 | tee "logs/${TAG}_gate2.log" || true

cp -f "experiments/gate1_${TAG}.json" experiments/gate1_geometry.json || true
cp -f "experiments/gate2_${TAG}.json" experiments/gate2_osf_umud.json || true

python3 - <<'PY'
import json
from pathlib import Path
g1 = json.loads(Path("experiments/gate1_geometry.json").read_text()) if Path("experiments/gate1_geometry.json").exists() else {}
g2 = json.loads(Path("experiments/gate2_osf_umud.json").read_text()) if Path("experiments/gate2_osf_umud.json").exists() else {}
ok = bool(g1.get("gate1_ok") or g1.get("geom_ok")) and bool(g2.get("gate2_ok"))
soft = bool(g1.get("gate1_ok") or g1.get("geom_ok") or g1.get("vs_baseline_improved")) and (
    g2.get("umud", 9) < 0.70 and g2.get("n_mt_bad", 99) <= 8
)
Path("experiments/grow_infer_decision.txt").write_text("INFER\n" if (ok or soft) else "HOLD\n")
print(
    "INFER_DECISION", "INFER" if (ok or soft) else "HOLD",
    "g1", g1.get("gate1_ok"), g1.get("pa_mae_deg"), g1.get("vs_baseline_improved"),
    "g2", g2.get("gate2_ok"), g2.get("umud"), g2.get("n_mt_bad"),
)
PY

if ! grep -q INFER experiments/grow_infer_decision.txt; then
  echo "HOLD infer — not grown enough"
  echo HOLD > experiments/grow_submit_decision.txt
  python -u scripts/grow_report_card.py \
    --gate1 experiments/gate1_geometry.json \
    --gate2 experiments/gate2_osf_umud.json \
    --gate3 experiments/gate3_dist.json \
    --out experiments/grow_report_card.json || true
  echo DONE_HOLD
  exit 0
fi

echo "=== 5) Infer (no middle blend) ==="
export CUDA_VISIBLE_DEVICES=0
python -u scripts/calibrate_predict.py \
  --config "$CFG" \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --out "submissions/submission_${TAG}.csv" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true \
  --scale-table experiments/scale_table.csv \
  --depth-scale-table experiments/depth_scale_table.csv \
  --sector-crop true --live-depth-scale true \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  --debug-csv "experiments/submission_${TAG}_debug.csv" \
  2>&1 | tee "logs/${TAG}_infer.log"

echo "=== 6) Gate3 distribution ==="
python -u scripts/eval_gate3_dist.py \
  --cand "submissions/submission_${TAG}.csv" \
  --ref submissions/submission_v8.csv \
  --out experiments/gate3_dist.json \
  2>&1 | tee "logs/${TAG}_gate3.log" || true

python -u scripts/grow_report_card.py \
  --gate1 experiments/gate1_geometry.json \
  --gate2 experiments/gate2_osf_umud.json \
  --gate3 experiments/gate3_dist.json \
  --out experiments/grow_report_card.json \
  2>&1 | tee "logs/${TAG}_report_card.log" || true

python3 - <<'PY'
import json
from pathlib import Path
card = json.loads(Path("experiments/grow_report_card.json").read_text())
Path("experiments/grow_submit_decision.txt").write_text(card["submit"] + "\n")
print("SUBMIT_DECISION", card["submit"], "grown=", card.get("grown"))
PY

echo DONE
