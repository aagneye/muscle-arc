#!/usr/bin/env bash
set -euo pipefail
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/${COMP}/leaderboard/download?type=public" \
  -o /tmp/lb.zip
python3 - <<'PY'
import zipfile
z = zipfile.ZipFile("/tmp/lb.zip")
print("files:", z.namelist())
name = z.namelist()[0]
text = z.read(name).decode("utf-8", errors="replace")
lines = text.strip().splitlines()
print("header:", lines[0])
print("--- top 20 ---")
for line in lines[1:21]:
    print(line)
print("--- around score 1.71 ---")
# find teams near our score
for i, line in enumerate(lines[1:], start=1):
    parts = line.split(",")
    # public score often last numeric-ish field
    if "1.71" in line or "1.7" in line or "aagneye" in line.lower() or "muscle" in line.lower():
        print(i, line)
print("total rows:", len(lines)-1)
PY
