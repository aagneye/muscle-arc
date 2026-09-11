#!/usr/bin/env bash
set -euo pipefail
# Refresh kaggle.json from access_token (Bearer-style KGAT_ tokens)
python3 - <<'PY'
import json
from pathlib import Path
home = Path.home() / ".kaggle"
token = (home / "access_token").read_text().strip()
(home / "kaggle.json").write_text(json.dumps({"username": "aagneye", "key": token}))
(home / "kaggle.json").chmod(0o600)
print("auth_ok", len(token))
PY
export KAGGLE_API_TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
