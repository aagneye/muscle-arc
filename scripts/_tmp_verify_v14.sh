#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -e . -q
python - <<'PY'
from muscle_arc.geometry.depth_scale import load_depth_scale_table, share_scales_in_groups
d, c = load_depth_scale_table("experiments/depth_scale_table.csv", return_confidence=True)
print("scales", len(d), "conf", len(c))
print("ok")
PY
