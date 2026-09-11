#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python3 <<'PY'
import pandas as pd
from pathlib import Path
import yaml
from muscle_arc.infer.predict import load_sample_submission

depth = pd.read_csv("experiments/depth_scale_table.csv")
print("=== depth for sample-like ===")
for s in ["IMG_00001", "IMG_00002"]:
    m = depth[depth.stem.str.contains(s) | depth.image_id.str.contains(s)]
    cols = [c for c in ["image_id","kind","depth_cm","mm_per_pixel","source","confidence","crop_applied"] if c in m.columns]
    print(m[cols].to_string() if len(m) else f"missing {s}")

cfg = yaml.safe_load(open("configs/default.yaml"))
sample = load_sample_submission(Path(cfg["data"]["root"]) / cfg["data"]["sample_submission"], sep=";")
print("\nsample:")
print(sample.to_string())

v8 = pd.read_csv("submissions/submission_v8.csv")
v10 = pd.read_csv("submissions/submission_v10.csv")
print("\n=== dists ===")
for name, d in [("v8", v8), ("v10", v10)]:
    print(
        name,
        "FL med/std", round(float(d.fl_mm.median()), 2), round(float(d.fl_mm.std()), 2),
        "MT med/std", round(float(d.mt_mm.median()), 2), round(float(d.mt_mm.std()), 2),
        "PA med", round(float(d.pa_deg.median()), 2),
    )

print("\ndepth mm by source:")
print(depth.groupby("source")["mm_per_pixel"].describe().round(4).to_string())

# Compare sample rows
id_col = [c for c in sample.columns if "id" in c.lower()][0]
print("\n=== sample pred compare ===")
for _, srow in sample.iterrows():
    rid = str(srow[id_col])
    r8 = v8[v8.image_id == rid]
    r10 = v10[v10.image_id == rid]
    if r8.empty:
        r8 = v8[v8.image_id.map(lambda x: Path(x).stem) == Path(rid).stem]
    if r10.empty:
        r10 = v10[v10.image_id.map(lambda x: Path(x).stem) == Path(rid).stem]
    print(rid, "GT", float(srow.pa_deg), float(srow.fl_mm), float(srow.mt_mm))
    if len(r8):
        print("  v8 ", float(r8.iloc[0].pa_deg), float(r8.iloc[0].fl_mm), float(r8.iloc[0].mt_mm))
    if len(r10):
        print("  v10", float(r10.iloc[0].pa_deg), float(r10.iloc[0].fl_mm), float(r10.iloc[0].mt_mm))
    dm = depth[depth.image_id == rid]
    if dm.empty:
        dm = depth[depth.stem == Path(rid).stem]
    if len(dm):
        print("  depth", dm.iloc[0][["mm_per_pixel","source","depth_cm","kind"]].to_dict())

osf = pd.read_csv("experiments/osf_expert_gt.csv")
print("\nOSF cols", osf.columns.tolist())
print("OSF n", len(osf))
PY
echo "=== recent submits ==="
ls -lt submissions/submission_v*.csv | head -10
