#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
FILE="${1:-submissions/submission_v3.csv}"
MSG="${2:-v3 dual-scale FL/MT + fasc mask fix}"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
cp "$FILE" submissions/submission.csv

# Get upload URL via API then PUT/POST file
# Try direct submit endpoint used earlier
RESP=$(curl -sS -w "\nHTTP:%{http_code}" -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "SubmissionDescription=${MSG}" \
  -F "submissionFile=@submissions/submission.csv" \
  "https://api.kaggle.com/v1/competitions/submissions/submit/${COMP}")
echo "$RESP" | tail -5

# Also try www host
RESP2=$(curl -sS -w "\nHTTP:%{http_code}" -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "submissionDescription=${MSG}" \
  -F "file=@submissions/submission.csv" \
  "https://www.kaggle.com/api/v1/competitions/submissions/submit/${COMP}")
echo "$RESP2" | tail -8

sleep 15
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=3" \
  | python3 -c "import sys,json; d=json.load(sys.stdin);
[print(x.get('date'), x.get('publicScore'), x.get('status'), (x.get('description') or '')[:70]) for x in d[:3]]"
