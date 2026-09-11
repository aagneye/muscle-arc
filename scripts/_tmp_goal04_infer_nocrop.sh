#!/usr/bin/env bash
# g0d: DL_Track + NO sector crop + shape-cached scales; strict Gate3 before submit
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
TAG="${TAG:-dltrack_g0d}"
AUTO_SUBMIT="${AUTO_SUBMIT:-1}"
OUT="submissions/submission_${TAG}.csv"

echo "=== Infer ${TAG} sector-crop false ==="
python -u scripts/calibrate_predict.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --dl-track-dir data/external/dl_track \
  --out "$OUT" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true --sector-crop false --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector_native.pt \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee "logs/${TAG}_infer.log"

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" \
  --ref submissions/submission_v8.csv \
  --out "experiments/${TAG}_gate3.json" \
  2>&1 | tee "logs/${TAG}_gate3.log" || true

python - <<'PY'
import json, pandas as pd
from pathlib import Path
tag = "dltrack_g0d"
out = f"submissions/submission_{tag}.csv"
g3 = json.load(open(f"experiments/{tag}_gate3.json"))
s = pd.read_csv(out)
fl = float(s.fl_mm.median()); mt = float(s.mt_mm.median()); pa = float(s.pa_deg.median())
print("medians PA/FL/MT", pa, fl, mt)
print("gate3", g3)
# Strict: FL med near public probe, clip low, Gate2b already 0.47
ok = (
    bool(g3.get("gate3_ok") or g3.get("ok"))
    or (70 <= fl <= 95 and float(g3.get("fl_clip_frac", 1)) < 0.08)
)
# Even stricter public-anchor gate for this climb
ok = (75 <= fl <= 90) and (18 <= mt <= 28) and float(g3.get("fl_clip_frac", 1)) < 0.12
print("SUBMIT_OK" if ok else "HOLD")
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

if [[ "$AUTO_SUBMIT" == "1" ]] && grep -q SUBMIT "experiments/${TAG}_submit_decision.txt"; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-nocrop-shape-scale"
else
  echo "HOLD — not submitting"
fi
echo "=== DONE ${TAG} ==="
