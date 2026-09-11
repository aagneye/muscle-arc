#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
FILE=submissions/submission_dltrack_official.csv
MSG="dltrack-official-doCalculations-phaseA"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
python - <<'PY'
import json
from pathlib import Path
home = Path.home() / ".kaggle"
token = (home / "access_token").read_text().strip()
(home / "kaggle.json").write_text(json.dumps({"username": "aagneye", "key": token}))
(home / "kaggle.json").chmod(0o600)
print("auth ok", len(token))
PY
test -f "$FILE" || { echo "MISSING $FILE"; exit 1; }
cp "$FILE" submissions/submission.csv
kaggle competitions submit -c "$COMP" -f submissions/submission.csv -m "$MSG"
sleep 30
bash scripts/vm_list_scores.sh
