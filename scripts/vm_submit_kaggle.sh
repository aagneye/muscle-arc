#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
# shellcheck disable=SC1091
source .venv/bin/activate
FILE="${1:-submissions/submission_v3.csv}"
MSG="${2:-v3 dual-scale FL/MT calibration + fasc mask fix}"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
export KAGGLE_API_TOKEN="$TOKEN"
cp "$FILE" submissions/submission.csv
wc -l submissions/submission.csv
python <<'PY'
import json
from pathlib import Path
home = Path.home() / ".kaggle"
token = (home / "access_token").read_text().strip()
(home / "kaggle.json").write_text(json.dumps({"username": "aagneye", "key": token}))
(home / "kaggle.json").chmod(0o600)
PY
kaggle competitions submit -c "$COMP" -f submissions/submission.csv -m "$MSG"
sleep 20
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=3" \
  | python -c "import sys,json; d=json.load(sys.stdin); 
[print(x.get('date'), x.get('publicScore'), x.get('status'), x.get('description','')[:60]) for x in d[:3]]"
