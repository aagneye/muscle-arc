#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
FILE="${1:-submissions/submission_v7.csv}"
MSG="${2:-v7 extrapolated-line geometry + apo-referenced PA (v3 weights)}"
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
print("wrote kaggle.json key_prefix=", token[:8], "len=", len(token))
PY

ok=0
for i in 1 2 3 4 5; do
  echo "CLI attempt $i"
  if kaggle competitions submit -c "$COMP" -f submissions/submission.csv -m "$MSG"; then
    echo SUBMIT_OK
    ok=1
    break
  fi
  sleep 4
done

if [ "$ok" -ne 1 ]; then
  echo "CLI failed; trying Bearer multipart submit..."
  RESP=$(curl -sS -w "\nHTTP:%{http_code}" -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -F "SubmissionDescription=${MSG}" \
    -F "submissionFile=@submissions/submission.csv" \
    "https://api.kaggle.com/v1/competitions/submissions/submit/${COMP}")
  echo "$RESP" | tail -8
fi

echo "waiting for score..."
sleep 30
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=5" \
  | python3 -c "import sys,json; d=json.load(sys.stdin);
items=d if isinstance(d,list) else d.get('submissions') or d.get('data') or []
[print(x.get('date') or x.get('dateSubmitted'), x.get('publicScore'), x.get('status'), (x.get('description') or '')[:90]) for x in items[:5]]"
