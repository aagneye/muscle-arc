#!/usr/bin/env bash
# Wait until next UTC midnight (quota reset), then submit Phase A + win candidate.
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
bash scripts/_tmp_refresh_kaggle_auth.sh || true

python3 - <<'PY'
import time
from datetime import datetime, timezone, timedelta
now = datetime.now(timezone.utc)
nxt = (now + timedelta(days=1)).replace(hour=0, minute=2, second=0, microsecond=0)
# If already past today's reset window and we haven't submitted today, don't wait full day
# Wait until next 00:02 UTC
wait = (nxt - now).total_seconds()
if wait > 86400:
    wait = 120
print(f"now_utc={now.isoformat()} wait_s={wait:.0f} until={nxt.isoformat()}", flush=True)
# Cap wait logging; sleep in chunks
left = wait
while left > 0:
    step = min(300, left)
    time.sleep(step)
    left -= step
    print(f"remaining_s={left:.0f}", flush=True)
print("QUOTA_WINDOW_OPEN", flush=True)
PY

OFFICIAL=submissions/submission_dltrack_official.csv
WIN=submissions/submission_dltrack_win.csv

# Wait up to 3h for win CSV if infer still running at midnight
python3 - <<'PY'
import time
from pathlib import Path
deadline = time.time() + 3 * 3600
p = Path("submissions/submission_dltrack_win.csv")
while not p.exists() and time.time() < deadline:
    print("waiting for win csv...", flush=True)
    time.sleep(60)
print("win_csv", p.exists(), flush=True)
PY

echo "=== Phase A official ==="
bash scripts/vm_submit_kaggle.sh "$OFFICIAL" "dltrack-official-doCalculations-phaseA" || \
  bash scripts/vm_submit_bearer.sh "$OFFICIAL" "dltrack-official-doCalculations-phaseA" || true
sleep 30

if [[ -f "$WIN" ]] && [[ -f experiments/dltrack_win_submit_decision.txt ]] && grep -q SUBMIT experiments/dltrack_win_submit_decision.txt; then
  echo "=== Win candidate (Gate2b+Gate3 green) ==="
  bash scripts/vm_submit_kaggle.sh "$WIN" "dltrack-win-chrome-letterbox-ours-geom" || \
    bash scripts/vm_submit_bearer.sh "$WIN" "dltrack-win-chrome-letterbox-ours-geom" || true
elif [[ -f "$WIN" ]] && [[ -f experiments/dltrack_win_soft_submit.txt ]] && grep -q SUBMIT experiments/dltrack_win_soft_submit.txt; then
  echo "=== Soft win candidate (Gate2b ok) ==="
  bash scripts/vm_submit_kaggle.sh "$WIN" "dltrack-win-chrome-letterbox-soft" || \
    bash scripts/vm_submit_bearer.sh "$WIN" "dltrack-win-chrome-letterbox-soft" || true
elif [[ -f "$WIN" ]]; then
  echo "=== Win CSV present; submit as late candidate ==="
  bash scripts/vm_submit_kaggle.sh "$WIN" "dltrack-win-chrome-letterbox-late" || \
    bash scripts/vm_submit_bearer.sh "$WIN" "dltrack-win-chrome-letterbox-late" || true
else
  echo "HOLD win — no CSV; Phase A only"
fi

sleep 45
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
curl -sS -H "Authorization: Bearer ${TOKEN}" \
  "https://api.kaggle.com/v1/competitions/submissions/list/umud-challenge-muscle-architecture-in-ultrasound-data?pageSize=5" \
  | python3 -c "import sys,json; d=json.load(sys.stdin);
[print(x.get('date'), x.get('publicScore'), x.get('status'), (x.get('description') or '')[:80]) for x in d[:5]]" \
  | tee experiments/post_midnight_scores.txt

echo DONE_MIDNIGHT_SUBMIT
