#!/usr/bin/env bash
# DL_Track + fixed geometry test infer → Gate3 → optional submit.
# Gate0 proved UMUD~0.34 on GT scale; this is the transfer attempt.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
TAG="${TAG:-dltrack_g0}"
AUTO_SUBMIT="${AUTO_SUBMIT:-1}"
OUT="submissions/submission_${TAG}.csv"

echo "=== Gate2b DL_Track + pred scale (transfer estimate) ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track --scale-mode pred \
  --sector-crop false \
  --out experiments/gate2b_dltrack_pred.json \
  2>&1 | tee "logs/${TAG}_gate2b.log" || true
python - <<'PY'
import json
from pathlib import Path
p=Path("experiments/gate2b_dltrack_pred.json")
if p.exists():
    m=json.load(open(p))
    print("Gate2b pred-scale", {k:m.get(k) for k in ("umud","pa_mae","fl_mae","mt_mae","n","gate2_ok")})
PY

echo "=== Infer ${TAG} (DL_Track, no shrink) ==="
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
g3=json.load(open(f"experiments/${TAG}_gate3.json")) if Path(f"experiments/${TAG}_gate3.json").exists() else {}
s=pd.read_csv("${OUT}")
print("gate3", {k:g3.get(k) for k in ("gate3_ok","fl_med","fl_med_near_81","clip_frac","fl_std","mt_std")})
print("pred medians", s.pa_deg.median(), s.fl_mm.median(), s.mt_mm.median(), "n", len(s))
print("GATE3_OK" if g3.get("gate3_ok") else "GATE3_FAIL")
PY

if [[ "$AUTO_SUBMIT" == "1" ]]; then
  echo "=== Submit ${TAG} ==="
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-gate0-geom-fixed" || true
fi
echo "=== DONE ${TAG} ==="
