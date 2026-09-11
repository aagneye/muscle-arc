#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python - <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path
from muscle_arc.geometry.depth_scale import estimate_depth_scale
from muscle_arc.data.dataset import read_gray

gt = pd.read_csv("experiments/osf_expert_gt.csv")
rows=[]
for _,s in gt.iterrows():
    p=Path(str(s.get("path","")))
    if not p.exists():
        continue
    gt_mm=float(s["mm_per_pixel"]) if pd.notna(s.get("mm_per_pixel")) else None
    if gt_mm is None or gt_mm<=0:
        continue
    full=read_gray(p)
    est=estimate_depth_scale(full)
    pred=est.mm_per_pixel
    ratio=(pred/gt_mm) if pred else np.nan
    rows.append({
        "image_id": s.get("image_id", p.name),
        "gt_mm": gt_mm,
        "pred_mm": pred,
        "ratio": ratio,
        "source": est.source,
        "conf": est.confidence,
        "h": full.shape[0],
        "w": full.shape[1],
    })
df=pd.DataFrame(rows)
print("n", len(df), "with_pred", int(df.pred_mm.notna().sum()))
print(df.groupby("source")["ratio"].describe())
print("overall ratio median", df.ratio.median(), "mean", df.ratio.mean())
print(df[["image_id","gt_mm","pred_mm","ratio","source","conf"]].to_string(index=False))
# If we multiply pred by 1/median_ratio, what happens to Gate2b terms roughly
med=df.ratio.median()
print("suggested global scale_gain", 1.0/med if med and med==med else None)
df.to_csv("experiments/osf_scale_diag.csv", index=False)
PY
