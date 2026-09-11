#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
pip install -q "DL-Track-US==0.3.1" 2>&1 | tail -30 || pip install -q DL-Track-US 2>&1 | tail -30
python - <<'PY'
import DL_Track_US, os, pkgutil
print("file", DL_Track_US.__file__)
root = os.path.dirname(DL_Track_US.__file__)
print("top", os.listdir(root)[:40])
for m in pkgutil.walk_packages([root], prefix="DL_Track_US."):
    print(m.name)
PY
