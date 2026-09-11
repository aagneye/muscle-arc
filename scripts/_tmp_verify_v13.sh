#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python - <<'PY'
from muscle_arc.geometry.depth_scale import estimate_depth_scale
from muscle_arc.models.scale_detector import ScaleStripDetector
print("imports_ok")
PY
python -m py_compile scripts/build_scale_labels.py scripts/train_scale_detector.py scripts/audit_depth_scale.py scripts/calibrate_predict.py
echo COMPILE_OK
file scripts/vm_run_scale_climb.sh
wc -c src/muscle_arc/geometry/depth_scale.py src/muscle_arc/models/scale_detector.py scripts/calibrate_predict.py
