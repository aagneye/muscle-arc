#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q

TAG=sof_v1b
echo "=== Infer with SOF ckpt ==="
python -u scripts/calibrate_predict.py \
  --config configs/sof.yaml \
  --sof-ckpt experiments/checkpoints_sof/sof_best.pt \
  --out "submissions/submission_${TAG}.csv" \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --temporal-smooth true --sector-crop true --live-depth-scale true \
  --depth-scale-table experiments/depth_scale_table.csv \
  --scale-detector experiments/checkpoints_scale/scale_detector_native.pt \
  --residual-prefer off \
  --osf-gt experiments/osf_expert_gt.csv \
  --debug-csv "experiments/submission_${TAG}_debug.csv" \
  2>&1 | tee "logs/${TAG}_infer.log"

python -u scripts/eval_osf_pipeline.py \
  --config configs/sof.yaml \
  --masks-from ours --scale-mode gt \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --out experiments/gate2_osf_umud.json \
  2>&1 | tee "logs/${TAG}_gate2.log" || true

# Note: Gate2 still uses phase4 unless we add SOF to eval_osf — quick local geom on test stats:
python - <<'PY'
import pandas as pd
from muscle_arc.geometry.umud_metric import gate4_dist_ok
import json
from pathlib import Path
tag = "sof_v1b"
d = pd.read_csv(f"submissions/submission_{tag}.csv")
ref = pd.read_csv("submissions/submission_v8.csv")
ok, info = gate4_dist_ok(d, ref)
info["fl_med_near_81"] = 70 <= info["fl_med"] <= 95
info["gate3_ok"] = bool(ok)
print(json.dumps(info, indent=2))
Path("experiments/gate3_dist.json").write_text(json.dumps(info, indent=2))
print(tag, "FL_med", round(float(d.fl_mm.median()),1), "clip140", int((d.fl_mm>=139.9).sum()),
      "MT_med", round(float(d.mt_mm.median()),1), "PA_med", round(float(d.pa_deg.median()),1))
print("GATE3", "OK" if ok else "FAIL")
PY

# Hybrid with v14c if needed for Gate3
python - <<'PY'
import pandas as pd
import numpy as np
from pathlib import Path
from muscle_arc.geometry.umud_metric import gate4_dist_ok
import json

v8 = pd.read_csv("submissions/submission_v8.csv").sort_values("image_id").reset_index(drop=True)
v14c = pd.read_csv("submissions/submission_v14c.csv").sort_values("image_id").reset_index(drop=True)
sof = pd.read_csv("submissions/submission_sof_v1b.csv").sort_values("image_id").reset_index(drop=True)
# blend 40% sof + 60% v14c on FL/MT; PA half
h = v14c.copy()
h["pa_deg"] = 0.5 * (v14c["pa_deg"] + sof["pa_deg"])
h["fl_mm"] = 0.40 * sof["fl_mm"] + 0.60 * v14c["fl_mm"]
h["mt_mm"] = 0.40 * sof["mt_mm"] + 0.60 * v14c["mt_mm"]
h.to_csv("submissions/submission_sof_v1c.csv", index=False)
ok, info = gate4_dist_ok(h, v8)
info["gate3_ok"] = bool(ok)
print("sof_v1c", json.dumps({k: info[k] for k in ("fl_med","fl_clip_frac","ok","reasons")}, indent=2))
print("FL_med", float(h.fl_mm.median()), "clip", int((h.fl_mm>=139.9).sum()))
if ok:
    Path("experiments/sof_v1c_submit_decision.txt").write_text("SUBMIT\n")
else:
    Path("experiments/sof_v1c_submit_decision.txt").write_text("HOLD\n")
PY

DECISION=$(tr -d '\n' < experiments/sof_v1c_submit_decision.txt || echo HOLD)
echo "Decision=${DECISION}"
if [[ "$DECISION" == "SUBMIT" ]]; then
  bash scripts/_tmp_submit_v14c.sh submissions/submission_sof_v1c.csv 'sof_v1c-SOF-masks-blend-v14c-Gate3'
fi
echo DONE
