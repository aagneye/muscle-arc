#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
TAG=dltrack_g0b
OUT=submissions/submission_${TAG}.csv

echo "=== Infer ${TAG} sector-crop true ==="
python -u scripts/calibrate_predict.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --dl-track-dir data/external/dl_track \
  --out "$OUT" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true --sector-crop true --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector_native.pt \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee logs/${TAG}_infer.log

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" \
  --ref submissions/submission_v8.csv \
  --out experiments/${TAG}_gate3.json \
  2>&1 | tee logs/${TAG}_gate3.log || true

python - <<PY
import json, pandas as pd
g3=json.load(open("experiments/${TAG}_gate3.json"))
s=pd.read_csv("$OUT")
print("gate3_ok", g3.get("gate3_ok"), "fl_med", g3.get("fl_med"), "mt_med", g3.get("mt_med"))
print(s.describe())
d=pd.read_csv("${OUT}".replace(".csv","_debug.csv"))
print(d["scale_source"].value_counts().head(15))
print(d.groupby("scale_source")[["fl_mm","mt_mm"]].median())
PY

bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-sector-crop-on"
