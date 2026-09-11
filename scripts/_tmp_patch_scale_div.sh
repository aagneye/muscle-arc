#!/usr/bin/env bash
set -euo pipefail
pkill -f infer_dltrack_official.py 2>/dev/null || true
sleep 1
python3 <<'PY'
from pathlib import Path
p = Path("/home/azureuser/muscle-arc/src/muscle_arc/models/dl_track_official/do_calculations.py")
t = p.read_text()
old = """        if calib_dist:
            fasc_l = fasc_l / (calib_dist / int(spacing))
            midthick = midthick / (calib_dist / int(spacing))
            unit = "mm"
"""
new = """        if calib_dist:
            fasc_l = list(np.asarray(fasc_l, dtype=float) / (calib_dist / float(spacing)))
            midthick = float(np.asarray(midthick).reshape(-1)[0]) / (
                calib_dist / float(spacing)
            )
            unit = "mm"
"""
if old not in t:
    raise SystemExit("pattern not found")
p.write_text(t.replace(old, new))
print("patched")
PY
cd /home/azureuser/muscle-arc
source .venv/bin/activate
nohup bash scripts/_tmp_goal04_official.sh > logs/goal04_official_master3.out 2>&1 &
echo PID=$!
