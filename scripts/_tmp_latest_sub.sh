#!/usr/bin/env bash
cd /home/azureuser/muscle-arc
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=1" \
  | python3 - <<'PY'
import sys, json
raw = sys.stdin.read()
d = json.loads(raw)
items = d if isinstance(d, list) else d.get("submissions") or d.get("data") or []
x = items[0]
print("id:", x.get("ref") or x.get("id"))
print("score:", x.get("publicScore"))
print("status:", x.get("status"))
print("desc:", (x.get("description") or "")[:120])
print("date:", x.get("date") or x.get("dateSubmitted"))
for k,v in sorted(x.items()):
    if "url" in k.lower() or k in ("ref","id","fileName"):
        print(f"{k}: {v}")
PY
echo "PAGE: https://www.kaggle.com/competitions/${COMP}/submissions"
