#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python3 <<'PY'
import json
from pathlib import Path
import pandas as pd
g3=json.load(open("experiments/dltrack_win_gate3.json"))
print("gate3", json.dumps(g3, indent=2)[:800])
s=pd.read_csv("submissions/submission_dltrack_win.csv")
print(s.describe())
print("soft", Path("experiments/dltrack_win_soft_submit.txt").read_text())
print("hard", Path("experiments/dltrack_win_submit_decision.txt").read_text())
print("chrome", Path("experiments/chrome_mask_coverage.json").read_text())
PY
