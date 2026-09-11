#!/usr/bin/env bash
# Build confidence-gated hybrids from v14/v13 + v10c/v8; Gate3; submit best.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate

python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

from muscle_arc.geometry.umud_metric import gate4_dist_ok

v8 = pd.read_csv("submissions/submission_v8.csv").sort_values("image_id").reset_index(drop=True)
v10c = pd.read_csv("submissions/submission_v10c.csv").sort_values("image_id").reset_index(drop=True)
v13 = pd.read_csv("submissions/submission_v13_scale.csv").sort_values("image_id").reset_index(drop=True)
v14 = pd.read_csv("submissions/submission_v14_scale.csv").sort_values("image_id").reset_index(drop=True)
assert list(v8.image_id) == list(v14.image_id) == list(v10c.image_id)

depth = pd.read_csv("experiments/depth_scale_table.csv")
conf = {str(r.image_id): float(r.confidence) for _, r in depth.iterrows()}
src = {str(r.image_id): str(r.source) for _, r in depth.iterrows()}
mm = {
    str(r.image_id): float(r.mm_per_pixel)
    for _, r in depth.iterrows()
    if pd.notna(r.mm_per_pixel)
}


def stats(df, name):
    return {
        "name": name,
        "FL_med": round(float(df.fl_mm.median()), 2),
        "FL_std": round(float(df.fl_mm.std()), 2),
        "clip140": int((df.fl_mm >= 139.9).sum()),
        "MT_med": round(float(df.mt_mm.median()), 2),
        "MT_std": round(float(df.mt_mm.std()), 2),
        "PA_med": round(float(df.pa_deg.median()), 2),
    }


def gate3(df):
    ok, info = gate4_dist_ok(df, v8, max_clip_frac=0.05)
    near = 70.0 <= info["fl_med"] <= 95.0
    info["fl_med_near_81"] = near
    info["gate3_ok"] = bool(ok and near)
    return info


variants = {}

# A) trust fused OCR only; else v10c
a = v10c.copy()
n_a = 0
for i, row in a.iterrows():
    iid = str(row.image_id)
    c = conf.get(iid, 0.0)
    s = src.get(iid, "none")
    if c >= 0.90 and (s.startswith("ocr+") or s == "ocr+sector"):
        a.at[i, "fl_mm"] = float(v14.loc[i, "fl_mm"])
        a.at[i, "mt_mm"] = float(v14.loc[i, "mt_mm"])
        a.at[i, "pa_deg"] = float(v14.loc[i, "pa_deg"])
        n_a += 1
variants["v14c_fused90"] = a
print("v14c_fused90 replaced", n_a)

# B) trust any conf>=0.85 ticks/ocr; else v10c; soft-clip FL
b = v10c.copy()
n_b = 0
for i, row in b.iterrows():
    iid = str(row.image_id)
    c = conf.get(iid, 0.0)
    s = src.get(iid, "none")
    if c >= 0.85 and s != "none" and iid in mm:
        b.at[i, "fl_mm"] = float(np.clip(v14.loc[i, "fl_mm"], 30, 130))
        b.at[i, "mt_mm"] = float(np.clip(v14.loc[i, "mt_mm"], 8, 40))
        b.at[i, "pa_deg"] = float(v14.loc[i, "pa_deg"])
        n_b += 1
variants["v14c_conf85_clip"] = b
print("v14c_conf85_clip replaced", n_b)

# C) classic blend: 0.35 v14 + 0.65 v10c on FL/MT (PA half)
c = v10c.copy()
c["pa_deg"] = 0.5 * (v10c["pa_deg"] + v14["pa_deg"])
c["fl_mm"] = 0.35 * v14["fl_mm"] + 0.65 * v10c["fl_mm"]
c["mt_mm"] = 0.35 * v14["mt_mm"] + 0.65 * v10c["mt_mm"]
variants["v14c_blend35"] = c

# D) same with v13 (milder overshoot)
d = v10c.copy()
d["pa_deg"] = 0.5 * (v10c["pa_deg"] + v13["pa_deg"])
d["fl_mm"] = 0.35 * v13["fl_mm"] + 0.65 * v10c["fl_mm"]
d["mt_mm"] = 0.35 * v13["mt_mm"] + 0.65 * v10c["mt_mm"]
variants["v13c_blend35"] = d

# E) shrink v14 FL/MT toward public FL median 81.5
e = v14.copy()
fl_scale = float(np.clip(81.5 / max(float(v14.fl_mm.median()), 1e-3), 0.75, 1.05))
e["fl_mm"] = (v14["fl_mm"] * fl_scale).clip(30, 130)
e["mt_mm"] = (v14["mt_mm"] * fl_scale).clip(8, 40)  # same mm/px error scales both
e["pa_deg"] = 0.5 * (v10c["pa_deg"] + v14["pa_deg"])
variants["v14_shrink_fl81"] = e
print("v14_shrink_fl81 scale", round(fl_scale, 4))

# F) blend35 of shrink with v10c
f = v10c.copy()
f["pa_deg"] = e["pa_deg"]
f["fl_mm"] = 0.40 * e["fl_mm"] + 0.60 * v10c["fl_mm"]
f["mt_mm"] = 0.40 * e["mt_mm"] + 0.60 * v10c["mt_mm"]
variants["v14c_shrink_blend40"] = f

best = None
results = []
for name, df in variants.items():
    out = Path("submissions") / f"submission_{name}.csv"
    df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].to_csv(out, index=False)
    g = gate3(df)
    st = stats(df, name)
    st.update({"gate3_ok": g["gate3_ok"], "reasons": g.get("reasons", []), "fl_clip_frac": g["fl_clip_frac"]})
    results.append(st)
    print(json.dumps(st))
    if g["gate3_ok"]:
        # Prefer FL_med closest to 81.5 among Gate3 passers
        score = abs(st["FL_med"] - 81.5) + 0.1 * st["clip140"]
        if best is None or score < best[0]:
            best = (score, name, df, g, st)

if best is None:
    print("NO_GATE3_PASS — falling back to v10c")
    best = (0.0, "v10c", v10c, gate3(v10c), stats(v10c, "v10c"))
    Path("experiments/v14c_submit_decision.txt").write_text("HOLD\n")
else:
    Path("experiments/v14c_submit_decision.txt").write_text("SUBMIT\n")

_, name, df, g, st = best
df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].to_csv(
    "submissions/submission_v14c.csv", index=False
)
Path("experiments/gate3_dist.json").write_text(json.dumps({**g, "chosen": name, **st}, indent=2))
print("CHOSEN", name, st, "gate3", g["gate3_ok"])
print("WROTE submissions/submission_v14c.csv")
PY

DECISION="$(tr -d '\n' < experiments/v14c_submit_decision.txt || echo HOLD)"
echo "Decision=${DECISION}"
if [[ "$DECISION" == "SUBMIT" ]]; then
  bash scripts/vm_submit_kaggle.sh \
    submissions/submission_v14c.csv \
    "v14c: confidence-gated hybrid of scale climb + v10c (Gate3 pass)"
else
  echo "HOLD — submitting v10c only if forced"
fi
