#!/usr/bin/env bash
# Resume: letterbox already proved Gate0=0.287 Gate2b=0.41 — infer + submit
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
TAG=dltrack_lb
OUT=submissions/submission_${TAG}.csv

# Reconfirm Gate0 quickly if json missing
if [[ ! -f experiments/gate0_letterbox.json ]]; then
  python -u scripts/eval_osf_pipeline.py \
    --config configs/default.yaml \
    --masks-from dl_track --scale-mode gt \
    --sector-crop false \
    --out experiments/gate0_letterbox.json \
    --gate0-out experiments/gate0_letterbox.json \
    2>&1 | tee logs/${TAG}_gate0.log || true
fi
python - <<'PY'
import json
from pathlib import Path
for p in ["experiments/gate0_letterbox.json","experiments/gate2b_letterbox.json"]:
    if Path(p).exists():
        m=json.load(open(p))
        print(p, m.get("umud"), m.get("pa_mae"), m.get("fl_mae"), m.get("mt_mae"))
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
  2>&1 | tee logs/${TAG}_infer_resume.log

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" --ref submissions/submission_v8.csv \
  --out experiments/${TAG}_gate3.json || true

python - <<'PY'
import json, pandas as pd
from pathlib import Path
tag="dltrack_lb"
s=pd.read_csv(f"submissions/submission_{tag}.csv")
g3=json.load(open(f"experiments/{tag}_gate3.json"))
fl=float(s.fl_mm.median()); mt=float(s.mt_mm.median()); pa=float(s.pa_deg.median())
print("medians PA/FL/MT", pa, fl, mt)
print("gate3", {k:g3.get(k) for k in ("gate3_ok","fl_med","fl_clip_frac","fl_med_near_81","ok","reasons")})
# Submit if medians sane — Gate2b already 0.41; public is the real test
ok = (70 <= fl <= 95) and (15 <= mt <= 28) and float(g3.get("fl_clip_frac",1)) < 0.15
print("SUBMIT_OK" if ok else "HOLD")
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

if grep -q SUBMIT experiments/${TAG}_submit_decision.txt; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-letterbox-nocrop-Gate0-0.287"
fi
echo DONE
