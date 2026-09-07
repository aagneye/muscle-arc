#!/usr/bin/env bash
# Curator climb: audit ruler → pseudo-labels → train scale detector → Gate1 → infer gated.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs experiments/checkpoints_scale submissions

CFG="${CFG:-configs/default.yaml}"
APO="${APO:-experiments/checkpoints_phase4/apo_best.pt}"
FASC="${FASC:-experiments/checkpoints_phase4/fasc_best.pt}"
TAG="${TAG:-v13_scale}"

echo "=== 1) Audit depth/OCR/ticks (duty-cycle console) ==="
python -u scripts/audit_depth_scale.py \
  --folders test_images_v2 \
  --out experiments/depth_scale_table.csv \
  2>&1 | tee "logs/${TAG}_audit.log"

echo "=== 2) Build scale pseudo-labels ==="
python -u scripts/build_scale_labels.py \
  --folders data/raw/test_images_v2 data/raw/apo_imgs_v1 \
  --osf-gt experiments/osf_expert_gt.csv \
  --min-conf 0.55 \
  --out experiments/scale_pseudo_labels.csv \
  2>&1 | tee "logs/${TAG}_labels.log"

echo "=== 3) Train ScaleStripDetector ==="
python -u scripts/train_scale_detector.py \
  --labels experiments/scale_pseudo_labels.csv \
  --out experiments/checkpoints_scale/scale_detector.pt \
  --epochs 40 \
  --batch-size 16 \
  2>&1 | tee "logs/${TAG}_train_scale.log"

echo "=== 4) Gate1 geometry (phase4) ==="
python -u scripts/eval_gate1_geometry.py \
  --config "$CFG" \
  --split holdout --split-dir experiments/splits \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate1_geometry.json \
  2>&1 | tee "logs/${TAG}_gate1.log" || true

echo "=== 5) Infer with OCR table + scale detector gaps ==="
python -u scripts/calibrate_predict.py \
  --config "$CFG" \
  --apo-ckpt "$APO" --fasc-ckpt "$FASC" \
  --out "submissions/submission_${TAG}.csv" \
  --thr 0.30 \
  --temporal-smooth true \
  --sector-crop true \
  --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector.pt \
  --residual-prefer off \
  --debug-csv "experiments/submission_${TAG}_debug.csv" \
  2>&1 | tee "logs/${TAG}_infer.log"

echo "=== 6) Gate3 dist vs v8 ==="
python -u scripts/eval_gate3_dist.py \
  --cand "submissions/submission_${TAG}.csv" \
  --ref submissions/submission_v8.csv \
  2>&1 | tee "logs/${TAG}_gate3.log" || true

python -u scripts/eval_umud_local.py --pred "submissions/submission_${TAG}.csv" || true

python3 - <<PY
import pandas as pd
from pathlib import Path
tag = "${TAG}"
cand = Path(f"submissions/submission_{tag}.csv")
ref = Path("submissions/submission_v8.csv")
d = pd.read_csv(cand)
print(tag, "n", len(d), "FL_med", round(float(d.fl_mm.median()),1),
      "FL_std", round(float(d.fl_mm.std()),1),
      "clip140", int((d.fl_mm>=139.9).sum()),
      "MT_med", round(float(d.mt_mm.median()),1))
if ref.exists():
    r = pd.read_csv(ref)
    print("v8", "FL_med", round(float(r.fl_mm.median()),1), "FL_std", round(float(r.fl_mm.std()),1))
print("REVIEW Gate3 log; submit only if no std collapse / clip storm")
PY

echo "=== DONE ${TAG} ==="
ls -la "submissions/submission_${TAG}.csv" experiments/checkpoints_scale/scale_detector.pt experiments/depth_scale_table.csv
