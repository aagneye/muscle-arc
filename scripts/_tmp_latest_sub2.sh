#!/usr/bin/env bash
TOKEN="$(tr -d '\n' < /home/azureuser/.kaggle/access_token)"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=1" \
  -o /tmp/sub.json
python3 -c '
import json
d=json.load(open("/tmp/sub.json"))
items=d if isinstance(d,list) else d.get("submissions") or d.get("data") or []
x=items[0]
print("ref", x.get("ref") or x.get("id"))
print("score", x.get("publicScore"))
print("status", x.get("status"))
print("desc", (x.get("description") or "")[:100])
print("keys", sorted(x.keys()))
'
echo "https://www.kaggle.com/competitions/${COMP}/submissions"
