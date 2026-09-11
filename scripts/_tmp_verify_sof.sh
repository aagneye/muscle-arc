#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
python - <<'PY'
from muscle_arc.geometry.metrics import FL_BLEND_TRIG, fascicle_length_px
from muscle_arc.geometry.surfaces import extract_apo_surfaces
from muscle_arc.geometry.orientation import radon_orientation_field
from muscle_arc.models.multitask import build_sof_model
from muscle_arc.models.scale_detector import ScaleNativeCropDetector
from muscle_arc.data.labels import apo_three_class
print("FL_BLEND", FL_BLEND_TRIG)
print("imports_ok")
PY
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from ours --scale-mode gt \
  --apo-ckpt experiments/checkpoints_phase4/apo_best.pt \
  --fasc-ckpt experiments/checkpoints_phase4/fasc_best.pt \
  --apo-thr 0.35 --fasc-thr 0.10 \
  --out experiments/gate2_osf_umud.json \
  2>&1 | tee logs/sof_gate2_gt_smoke.log | tail -50
