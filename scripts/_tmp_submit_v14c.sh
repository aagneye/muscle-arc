#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
export KAGGLE_API_TOKEN="$TOKEN"
FILE="${1:-submissions/submission_v14c.csv}"
MSG="${2:-v14c scale climb hybrid Gate3}"
cp "$FILE" submissions/submission.csv
wc -l submissions/submission.csv

python - <<'PY'
import json
from pathlib import Path
home = Path.home() / ".kaggle"
token = (home / "access_token").read_text().strip()
(home / "kaggle.json").write_text(json.dumps({"username": "aagneye", "key": token}))
(home / "kaggle.json").chmod(0o600)
print("wrote kaggle.json")
PY

ok=0
if kaggle competitions submit -c "$COMP" -f submissions/submission.csv -m "$MSG"; then
  echo SUBMIT_OK_CLI
  ok=1
else
  echo CLI_FAIL
fi

if [[ "$ok" -ne 1 ]]; then
  RESP=$(curl -sS -w "\nHTTP:%{http_code}" -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -F "SubmissionDescription=${MSG}" \
    -F "submissionFile=@submissions/submission.csv" \
    "https://api.kaggle.com/v1/competitions/submissions/submit/${COMP}")
  echo "$RESP" | tail -20
fi

echo "waiting for score..."
sleep 45
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=6" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
items = d if isinstance(d, list) else d.get('submissions') or d.get('data') or []
for x in items[:6]:
    print(x.get('date') or x.get('dateSubmitted'), '|', x.get('publicScore'), '|', x.get('status'), '|', (x.get('description') or '')[:90])
"
