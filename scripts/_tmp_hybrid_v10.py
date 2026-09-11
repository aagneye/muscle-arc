#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python3 <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from muscle_arc.infer.predict import load_sample_submission

v8 = pd.read_csv("submissions/submission_v8.csv").sort_values("image_id").reset_index(drop=True)
v10b = pd.read_csv("submissions/submission_v10b.csv").sort_values("image_id").reset_index(drop=True)
assert list(v8.image_id) == list(v10b.image_id)

variants = {}

# mean
m = v8.copy()
for c in ("pa_deg","fl_mm","mt_mm"):
    m[c] = 0.5 * (v8[c] + v10b[c])
variants["mean_v8_v10b"] = m

# PA from v10b, FL/MT from v8
p = v8.copy()
p["pa_deg"] = v10b["pa_deg"]
variants["pa_v10b_rest_v8"] = p

# shrink v10b FL/MT toward v8 medians
s = v10b.copy()
for c in ("fl_mm","mt_mm"):
    scale = float(v8[c].median() / max(v10b[c].median(), 1e-3))
    scale = float(np.clip(scale, 0.7, 1.0))
    s[c] = v10b[c] * scale
    print(f"shrink {c} x{scale:.3f}")
variants["v10b_shrink_to_v8med"] = s

# 0.35 OCR + 0.65 v8 on FL/MT, PA blend
h = v8.copy()
h["pa_deg"] = 0.5 * (v8["pa_deg"] + v10b["pa_deg"])
h["fl_mm"] = 0.35 * v10b["fl_mm"] + 0.65 * v8["fl_mm"]
h["mt_mm"] = 0.35 * v10b["mt_mm"] + 0.65 * v8["mt_mm"]
variants["blend35_ocr"] = h

cfg = yaml.safe_load(open("configs/default.yaml"))
sample = load_sample_submission(Path(cfg["data"]["root"]) / cfg["data"]["sample_submission"], sep=";")
id_col = [c for c in sample.columns if "id" in c.lower()][0]

def combo(pred):
    pa_e, fl_e, mt_e = [], [], []
    for _, srow in sample.iterrows():
        rid = str(srow[id_col])
        m = pred[pred.image_id == rid]
        if m.empty:
            m = pred[pred.image_id.map(lambda x: Path(x).stem) == Path(rid).stem]
        if m.empty:
            continue
        pa_e.append(abs(float(m.iloc[0].pa_deg) - float(srow.pa_deg)))
        fl_e.append(abs(float(m.iloc[0].fl_mm) - float(srow.fl_mm)))
        mt_e.append(abs(float(m.iloc[0].mt_mm) - float(srow.mt_mm)))
    return {
        "pa": float(np.mean(pa_e)),
        "fl": float(np.mean(fl_e)),
        "mt": float(np.mean(mt_e)),
        "combo": float(np.mean([np.mean(pa_e)/5, np.mean(fl_e)/10, np.mean(mt_e)/3])),
        "fl_std": float(pred.fl_mm.std()),
        "mt_std": float(pred.mt_mm.std()),
        "fl_med": float(pred.fl_mm.median()),
        "mt_med": float(pred.mt_mm.median()),
    }

print("v8", combo(v8))
print("v10b", combo(v10b))
best_name, best_c, best_df = None, 9e9, None
for name, df in variants.items():
    c = combo(df)
    print(name, c)
    out = Path("submissions") / f"submission_{name}.csv"
    df[["image_id","pa_deg","fl_mm","mt_mm"]].to_csv(out, index=False)
    if c["combo"] < best_c:
        best_c, best_name, best_df = c["combo"], name, df

print("BEST_HYBRID", best_name, best_c)
best_df[["image_id","pa_deg","fl_mm","mt_mm"]].to_csv("submissions/submission_v10c_hybrid.csv", index=False)
PY
