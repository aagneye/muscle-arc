#!/usr/bin/env bash
# Goal <0.4: fix pred-scale (shape>hyp) → Gate2b → DL_Track infer auto-crop → submit
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
mkdir -p logs submissions experiments
TAG="${TAG:-dltrack_g0c}"
AUTO_SUBMIT="${AUTO_SUBMIT:-1}"
OUT="submissions/submission_${TAG}.csv"

echo "=== Gate2b DL_Track + pred scale (post shape-lookup fix) ==="
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track --scale-mode pred \
  --sector-crop false \
  --out experiments/gate2b_dltrack_pred.json \
  2>&1 | tee "logs/${TAG}_gate2b.log" || true
python - <<'PY'
import json
m=json.load(open("experiments/gate2b_dltrack_pred.json"))
keys=["umud","pa_mae","fl_mae","mt_mae","n","gate2_ok","mt_ratio_median","fl_ratio_median","n_mt_bad","n_fl_bad"]
print("Gate2b", {k:m.get(k) for k in keys})
PY

echo "=== Infer ${TAG} (DL_Track, sector-crop auto) ==="
python -u scripts/calibrate_predict.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --dl-track-dir data/external/dl_track \
  --out "$OUT" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true --sector-crop auto --live-depth-scale true \
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
tag="${TAG}"
out=f"submissions/submission_{tag}.csv"
g3=json.load(open(f"experiments/{tag}_gate3.json")) if Path(f"experiments/{tag}_gate3.json").exists() else {}
s=pd.read_csv(out)
print("gate3", {k:g3.get(k) for k in ("gate3_ok","fl_med","fl_med_near_81","clip_frac","fl_std","mt_std")})
print("pred medians PA/FL/MT", float(s.pa_deg.median()), float(s.fl_mm.median()), float(s.mt_mm.median()), "n", len(s))
dbg=Path(out.replace(".csv","_debug.csv"))
if dbg.exists():
    d=pd.read_csv(dbg)
    if "scale_source" in d.columns:
        print(d["scale_source"].value_counts().head(12).to_string())
PY

# Only submit if Gate2b improved vs prior 1.48 disaster and medians sane
G2=$(python -c "import json; print(json.load(open('experiments/gate2b_dltrack_pred.json')).get('umud',9))")
FLMED=$(python -c "import pandas as pd; print(pd.read_csv('$OUT').fl_mm.median())")
echo "Gate2b_umud=$G2 FL_med=$FLMED"
python - <<PY
g2=float("$G2"); fl=float("$FLMED")
ok = g2 < 0.70 and 60 <= fl <= 100
print("SUBMIT_OK" if ok else "HOLD", "g2", g2, "fl", fl)
open("experiments/${TAG}_submit_decision.txt","w").write("SUBMIT\n" if ok else "HOLD\n")
PY

if [[ "$AUTO_SUBMIT" == "1" ]] && grep -q SUBMIT "experiments/${TAG}_submit_decision.txt"; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-shape-scale-auto-crop"
fi
echo "=== DONE ${TAG} ==="
