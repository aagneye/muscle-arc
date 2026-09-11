#!/usr/bin/env bash
# Letterbox DL_Track: Gate0 sanity → Gate2b → nocrop infer → strict submit
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
TAG="${TAG:-dltrack_lb}"
AUTO_SUBMIT="${AUTO_SUBMIT:-1}"
OUT="submissions/submission_${TAG}.csv"

echo "=== Gate0 letterbox DL_Track + GT scale ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track --scale-mode gt \
  --sector-crop false \
  --out experiments/gate0_letterbox.json \
  --gate0-out experiments/gate0_letterbox.json \
  2>&1 | tee "logs/${TAG}_gate0.log" || true
python - <<'PY'
import json
m=json.load(open("experiments/gate0_letterbox.json"))
print("Gate0", {k:m.get(k) for k in ("umud","pa_mae","fl_mae","mt_mae","gate0_ok","n_mt_bad")})
PY

echo "=== Gate2b letterbox + pred scale ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track --scale-mode pred \
  --sector-crop false \
  --out experiments/gate2b_letterbox.json \
  2>&1 | tee "logs/${TAG}_gate2b.log" || true
python - <<'PY'
import json
m=json.load(open("experiments/gate2b_letterbox.json"))
print("Gate2b", {k:m.get(k) for k in ("umud","pa_mae","fl_mae","mt_mae","mt_ratio_median","fl_ratio_median")})
PY

G0=$(python -c "import json; print(json.load(open('experiments/gate0_letterbox.json')).get('umud',9))")
echo "Gate0_umud=$G0"
# Abort infer if letterbox wrecked Gate0
python - <<PY
g0=float("$G0")
assert g0 < 0.55, f"Gate0 regressed to {g0}"
print("Gate0 OK for infer")
PY

echo "=== Infer ${TAG} ==="
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
  --cand "$OUT" --ref submissions/submission_v8.csv \
  --out "experiments/${TAG}_gate3.json" || true

python - <<'PY'
import json, pandas as pd
from pathlib import Path
tag="dltrack_lb"
s=pd.read_csv(f"submissions/submission_{tag}.csv")
g3=json.load(open(f"experiments/{tag}_gate3.json"))
fl=float(s.fl_mm.median()); mt=float(s.mt_mm.median()); pa=float(s.pa_deg.median())
print("medians", pa, fl, mt, "gate3", g3)
# Strict: only submit if FL near probe AND better geometry signals
ok = (78 <= fl <= 88) and (18 <= mt <= 24) and float(g3.get("fl_clip_frac",1)) < 0.06
# Also require Gate0 still good
g0=json.load(open("experiments/gate0_letterbox.json")).get("umud",9)
ok = ok and g0 < 0.50
print("SUBMIT_OK" if ok else "HOLD", "g0", g0)
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

if [[ "$AUTO_SUBMIT" == "1" ]] && grep -q SUBMIT "experiments/${TAG}_submit_decision.txt"; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-letterbox-nocrop"
else
  echo HOLD_no_submit
fi
echo "=== DONE ${TAG} ==="
