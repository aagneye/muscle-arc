#!/usr/bin/env bash
# Infer-only continuation after Gate2b=0.47
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
TAG="${TAG:-dltrack_g0c}"
AUTO_SUBMIT="${AUTO_SUBMIT:-1}"
OUT="submissions/submission_${TAG}.csv"

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

python - <<PY
import json, pandas as pd
from pathlib import Path
tag = "${TAG}"
out = f"submissions/submission_{tag}.csv"
g3 = json.load(open(f"experiments/{tag}_gate3.json")) if Path(f"experiments/{tag}_gate3.json").exists() else {}
s = pd.read_csv(out)
print("gate3", {k: g3.get(k) for k in ("gate3_ok", "fl_med", "fl_med_near_81", "clip_frac", "fl_std", "mt_std")})
print("pred medians PA/FL/MT", float(s.pa_deg.median()), float(s.fl_mm.median()), float(s.mt_mm.median()), "n", len(s))
dbg = Path(out.replace(".csv", "_debug.csv"))
if dbg.exists():
    d = pd.read_csv(dbg)
    if "scale_source" in d.columns:
        print(d["scale_source"].value_counts().head(12).to_string())
g2 = json.load(open("experiments/gate2b_dltrack_pred.json")).get("umud", 9)
fl = float(s.fl_mm.median())
ok = g2 < 0.70 and 60 <= fl <= 100
print("SUBMIT_OK" if ok else "HOLD", "g2", g2, "fl", fl)
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

if [[ "$AUTO_SUBMIT" == "1" ]] && grep -q SUBMIT "experiments/${TAG}_submit_decision.txt"; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-shape-scale-auto-crop"
fi
echo "=== DONE ${TAG} ==="
