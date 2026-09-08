#!/usr/bin/env bash
set -euo pipefail
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=6" \
  | python3 -c "import sys,json; d=json.load(sys.stdin);
items=d if isinstance(d,list) else d.get('submissions') or d.get('data') or []
for x in items[:6]:
  print(x.get('date') or x.get('dateSubmitted'), '|', x.get('publicScore'), '|', x.get('status'), '|', (x.get('description') or '')[:90])"
