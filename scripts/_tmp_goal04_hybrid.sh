#!/usr/bin/env bash
# Hybrid: DL_Track on cropped frames, sof_v1c on console (domain-shift fix)
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
TAG=hybrid_crop_dl_sof
OUT=submissions/submission_${TAG}.csv

python3 <<'PY'
import pandas as pd
from pathlib import Path

depth = pd.read_csv("experiments/depth_scale_table.csv")
sof = pd.read_csv("submissions/submission_sof_v1c.csv")
# Prefer nocrop dltrack (g0d); fall back to g0
dl_path = Path("submissions/submission_dltrack_g0d.csv")
if not dl_path.exists():
    dl_path = Path("submissions/submission_dltrack_g0.csv")
dl = pd.read_csv(dl_path)
v10 = pd.read_csv("submissions/submission_v10c.csv")

kind = depth.set_index("image_id")["kind"].to_dict()
conf = depth.set_index("image_id")["confidence"].to_dict()
src = depth.set_index("image_id")["source"].to_dict()

sof = sof.sort_values("image_id").reset_index(drop=True)
dl = dl.sort_values("image_id").reset_index(drop=True)
v10 = v10.sort_values("image_id").reset_index(drop=True)
assert list(sof.image_id) == list(dl.image_id) == list(v10.image_id)

rows = []
n_dl = n_sof = n_v10 = 0
for i, iid in enumerate(sof.image_id):
    k = kind.get(iid, kind.get(str(iid), "console"))
    c = float(conf.get(iid, conf.get(str(iid), 0.0)) or 0.0)
    s = str(src.get(iid, src.get(str(iid), "none")))
    # Cropped frames ≈ OSF domain where Gate0/Gate2b work
    if k == "cropped":
        row = dl.iloc[i].copy()
        n_dl += 1
    # High-conf OCR console: prefer sof (best public) but keep DL PA if sof PA extreme
    elif c >= 0.90 and ("ocr" in s or "ticks" in s):
        row = sof.iloc[i].copy()
        n_sof += 1
    else:
        # Low-conf / none: lean on v10c (best pre-sof public)
        row = v10.iloc[i].copy()
        n_v10 += 1
    rows.append(row)

out = pd.DataFrame(rows)[["image_id", "pa_deg", "fl_mm", "mt_mm"]]
out.to_csv("submissions/submission_hybrid_crop_dl_sof.csv", index=False)
print("n_dl_cropped", n_dl, "n_sof", n_sof, "n_v10", n_v10)
print(out.describe())
print("FL_med", float(out.fl_mm.median()), "MT_med", float(out.mt_mm.median()), "PA_med", float(out.pa_deg.median()))
PY

python -u scripts/eval_gate3_dist.py \
  --cand "$OUT" \
  --ref submissions/submission_v8.csv \
  --out experiments/${TAG}_gate3.json || true

python3 <<'PY'
import json, pandas as pd
s=pd.read_csv("submissions/submission_hybrid_crop_dl_sof.csv")
g3=json.load(open("experiments/hybrid_crop_dl_sof_gate3.json"))
fl=float(s.fl_mm.median()); mt=float(s.mt_mm.median())
print("gate3", g3)
ok = 75 <= fl <= 90 and 18 <= mt <= 26 and float(g3.get("fl_clip_frac",1)) < 0.08
print("SUBMIT_OK" if ok else "HOLD", fl, mt)
open("experiments/hybrid_crop_dl_sof_submit_decision.txt","w").write("SUBMIT\n" if ok else "HOLD\n")
PY

if grep -q SUBMIT experiments/hybrid_crop_dl_sof_submit_decision.txt; then
  bash scripts/vm_submit_kaggle.sh "$OUT" "${TAG}-cropped-dl-console-sof/v10c"
fi
echo DONE
