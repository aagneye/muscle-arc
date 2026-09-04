#!/usr/bin/env bash
set -euo pipefail
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
OUT="/home/azureuser/muscle-arc/data/raw"
COMP="umud-challenge-muscle-architecture-in-ultrasound-data"
mkdir -p "$OUT"
cd "$OUT"

echo "=== file list ==="
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/data/list/${COMP}?pageSize=100" \
  -o /tmp/umud_files.json
/home/azureuser/muscle-arc/.venv/bin/python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/tmp/umud_files.json").read_text())
files = data.get("files") or []
print("n_files", len(files))
for f in files:
    print(f.get("totalBytesNullable") or f.get("totalBytes") or "?", "\t", f.get("nameNullable") or f.get("name"))
PY

echo "=== download-all (follow redirects) ==="
# -C - resumes; long timeout for ~6GB
curl -L --fail --retry 5 --retry-delay 5 \
  -H "Authorization: Bearer ${TOKEN}" \
  -o "${COMP}.zip" \
  "https://api.kaggle.com/v1/competitions/data/download-all/${COMP}"

ls -lh "${COMP}.zip"
file "${COMP}.zip"
echo "=== extract ==="
unzip -o "${COMP}.zip"
rm -f "${COMP}.zip"
echo "=== layout ==="
ls -la
du -sh * | sort -h
