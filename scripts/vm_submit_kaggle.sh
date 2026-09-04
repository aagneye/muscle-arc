#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate

FILE=submissions/submission.csv
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
MSG="unet-b4 seg+geometry baseline v1 (309 images)"

if [ ! -f "$FILE" ]; then
  echo "Missing $FILE"
  exit 1
fi
wc -l "$FILE"
head -3 "$FILE"

TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
export KAGGLE_API_TOKEN="$TOKEN"

# Ensure kaggle.json is valid for CLI (username + token as key)
python <<'PY'
import json
from pathlib import Path
home = Path.home() / ".kaggle"
token = (home / "access_token").read_text().strip()
cfg_path = home / "kaggle.json"
username = "aagneye"
if cfg_path.exists():
    try:
        data = json.loads(cfg_path.read_text())
        username = data.get("username") or username
    except Exception:
        pass
cfg_path.write_text(json.dumps({"username": username, "key": token}))
cfg_path.chmod(0o600)
print("auth ready for", username)
PY

echo "Submitting via kaggle CLI..."
if kaggle competitions submit -c "$COMP" -f "$FILE" -m "$MSG"; then
  echo "CLI submit OK"
else
  echo "CLI failed; trying HTTP upload..."
  # Fallback: competition submit API
  curl -sS -X POST \
    -H "Authorization: Bearer ${TOKEN}" \
    -F "file=@${FILE}" \
    -F "submissionDescription=${MSG}" \
    "https://www.kaggle.com/api/v1/competitions/submissions/submit/${COMP}" \
    | tee /tmp/kaggle_submit_resp.json
  echo
fi

echo "--- recent submissions ---"
kaggle competitions submissions -c "$COMP" 2>&1 | head -15 || true
