#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
export KAGGLE_API_TOKEN="$TOKEN"
python <<'PY'
import json
from pathlib import Path
home = Path.home() / ".kaggle"
token = (home / "access_token").read_text().strip()
(home / "kaggle.json").write_text(json.dumps({"username": "aagneye", "key": token}))
(home / "kaggle.json").chmod(0o600)
print("wrote kaggle.json")
PY
# Retry submit a few times — intermittent 401s observed
for i in 1 2 3; do
  echo "attempt $i"
  if kaggle competitions submit \
      -c umud-challenge-muscle-architecture-in-ultrasound-data \
      -f submissions/submission_v3.csv \
      -m "v3 dual-scale FL/MT + fasc mask fix"; then
    echo SUBMIT_OK
    break
  fi
  sleep 5
done
sleep 20
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/umud-challenge-muscle-architecture-in-ultrasound-data?pageSize=3" \
  | python3 -c "import sys,json; d=json.load(sys.stdin);
[print(x.get('date'), x.get('publicScore'), x.get('status'), (x.get('description') or '')[:70]) for x in d[:3]]"
