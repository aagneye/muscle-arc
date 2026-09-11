#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
# Repair ABI breakage from DL-Track-US pin churn
pip install -q 'numpy>=1.26,<2' 'opencv-python-headless>=4.8,<5' pandas matplotlib pillow scikit-image scikit-learn tqdm openpyxl
python - <<'PY'
import cv2, numpy, tensorflow
print("cv2", cv2.__version__, "numpy", numpy.__version__, "tf", tensorflow.__version__)
PY
python -u scripts/infer_dltrack_official.py \
  --out submissions/submission_dltrack_official.csv \
  2>&1 | tee logs/goal04_official.out

python -u scripts/eval_gate3_dist.py \
  --cand submissions/submission_dltrack_official.csv \
  --ref submissions/submission_v8.csv \
  --out experiments/dltrack_official_gate3.json || true

python - <<'PY'
import json
from pathlib import Path
import pandas as pd
s = pd.read_csv("submissions/submission_dltrack_official.csv")
print(s.describe())
g3 = json.load(open("experiments/dltrack_official_gate3.json")) if Path("experiments/dltrack_official_gate3.json").exists() else {}
fl = float(s.fl_mm.median()); mt = float(s.mt_mm.median())
print("med", fl, mt, "gate3", g3.get("gate3_ok"), g3.get("fl_clip_frac"))
ok = (70 <= fl <= 95) and (15 <= mt <= 28)
print("SUBMIT_OK" if ok else "HOLD")
Path("experiments/dltrack_official_submit_decision.txt").write_text("SUBMIT\n" if ok else "HOLD\n")
PY

if grep -q SUBMIT experiments/dltrack_official_submit_decision.txt; then
  bash scripts/vm_submit_kaggle.sh submissions/submission_dltrack_official.csv "dltrack-official-doCalculations"
fi
echo DONE
