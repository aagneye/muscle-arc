#!/usr/bin/env bash
# Letterbox DL_Track + adaptive sector crop (console only)
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
TAG=dltrack_lb_crop
OUT=submissions/submission_${TAG}.csv

echo "=== Infer ${TAG} sector-crop auto + letterbox ==="
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
  2>&1 | tee logs/${TAG}_infer.log

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" --ref submissions/submission_v8.csv \
  --out experiments/${TAG}_gate3.json || true

python - <<'PY'
import json, pandas as pd
from pathlib import Path
tag="dltrack_lb_crop"
s=pd.read_csv(f"submissions/submission_{tag}.csv")
d=pd.read_csv("experiments/depth_scale_table.csv")
g3=json.load(open(f"experiments/{tag}_gate3.json"))
m=s.merge(d[["image_id","kind","confidence","source"]], on="image_id", how="left")
print("overall med", float(s.pa_deg.median()), float(s.fl_mm.median()), float(s.mt_mm.median()))
print(m.groupby("kind")[["pa_deg","fl_mm","mt_mm"]].median())
print("gate3", g3)
fl=float(s.fl_mm.median()); mt=float(s.mt_mm.median())
# Only submit if better than sof_v1c proxies: FL near 81 AND not collapsed like g0c
ok = (75 <= fl <= 90) and (17 <= mt <= 25) and float(g3.get("fl_clip_frac",1)) < 0.08
print("SUBMIT_OK" if ok else "HOLD")
Path(f"experiments/{tag}_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

# Also build hybrid: lb_crop on cropped frames, sof_v1c on console
python - <<'PY'
import pandas as pd
from pathlib import Path
depth=pd.read_csv("experiments/depth_scale_table.csv")
sof=pd.read_csv("submissions/submission_sof_v1c.csv").sort_values("image_id").reset_index(drop=True)
lb=pd.read_csv("submissions/submission_dltrack_lb_crop.csv").sort_values("image_id").reset_index(drop=True)
kind=depth.set_index("image_id")["kind"].to_dict()
rows=[]
n_lb=n_sof=0
for i,iid in enumerate(sof.image_id):
    k=kind.get(iid, "console")
    if k=="cropped":
        rows.append(lb.iloc[i]); n_lb+=1
    else:
        rows.append(sof.iloc[i]); n_sof+=1
out=pd.DataFrame(rows)[["image_id","pa_deg","fl_mm","mt_mm"]]
out.to_csv("submissions/submission_hybrid_lb_sof.csv", index=False)
print("hybrid n_lb", n_lb, "n_sof", n_sof, "FL", float(out.fl_mm.median()), "MT", float(out.mt_mm.median()))
PY

python -u scripts/eval_gate3_dist.py \
  --cand submissions/submission_hybrid_lb_sof.csv \
  --ref submissions/submission_v8.csv \
  --out experiments/hybrid_lb_sof_gate3.json || true

# Prefer submitting hybrid if full lb_crop HOLDs; always submit hybrid if FL sane
python - <<'PY'
import json, pandas as pd
from pathlib import Path
s=pd.read_csv("submissions/submission_hybrid_lb_sof.csv")
g3=json.load(open("experiments/hybrid_lb_sof_gate3.json"))
fl=float(s.fl_mm.median()); mt=float(s.mt_mm.median())
ok = (75 <= fl <= 90) and (18 <= mt <= 26)
print("hybrid", "SUBMIT_OK" if ok else "HOLD", fl, mt, g3.get("gate3_ok"))
Path("experiments/hybrid_lb_sof_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

if grep -q SUBMIT experiments/dltrack_lb_crop_submit_decision.txt; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-letterbox-autocrop"
fi
if grep -q SUBMIT experiments/hybrid_lb_sof_submit_decision.txt; then
  bash scripts/vm_submit_kaggle.sh submissions/submission_hybrid_lb_sof.csv "hybrid-lb-cropped-sof-console"
fi
echo DONE
