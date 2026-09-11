#!/usr/bin/env bash
# DL_Track letterbox + chrome-mask (no crop) — full-frame geometry
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
TAG=dltrack_lb_mask
OUT=submissions/submission_${TAG}.csv

python -u scripts/calibrate_predict.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --dl-track-dir data/external/dl_track \
  --out "$OUT" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true --sector-crop false --chrome-mask true \
  --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector_native.pt \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  2>&1 | tee logs/${TAG}_infer.log

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" --ref submissions/submission_v8.csv \
  --out experiments/${TAG}_gate3.json || true

python - <<'PY'
import json, pandas as pd
from pathlib import Path
tag="dltrack_lb_mask"
s=pd.read_csv(f"submissions/submission_{tag}.csv")
d=pd.read_csv("experiments/depth_scale_table.csv")
g3=json.load(open(f"experiments/{tag}_gate3.json"))
m=s.merge(d[["image_id","kind"]], on="image_id", how="left")
print("overall", float(s.pa_deg.median()), float(s.fl_mm.median()), float(s.mt_mm.median()))
print(m.groupby("kind")[["fl_mm","mt_mm","pa_deg"]].median())
print("gate3_ok", g3.get("gate3_ok"), "fl_clip", g3.get("fl_clip_frac"), "reasons", g3.get("reasons"))
fl=float(s.fl_mm.median()); mt=float(s.mt_mm.median())
# Submit if looks like a real transfer candidate (near probe, low clip)
ok = (76 <= fl <= 88) and (17 <= mt <= 24) and float(g3.get("fl_clip_frac",1)) < 0.08
# Also beat sof median proximity isn't enough — require Gate3
ok = ok and bool(g3.get("gate3_ok") or g3.get("ok"))
print("SUBMIT_OK" if ok else "HOLD")
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

if grep -q SUBMIT experiments/${TAG}_submit_decision.txt; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-letterbox-chrome-mask"
fi
echo DONE
