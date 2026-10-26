#!/usr/bin/env bash
# Submit a batch of candidates to Kaggle at a UTC time (HH:MM, next occurrence; the
# daily quota of 5 resets at 00:00 UTC), 40 s apart, then print the scored
# submission list twice so the log holds the public scores. Missing files are skipped.
#
# Usage: setsid nohup bash scripts/vm_submit_batch.sh 00:05 "FILE|MESSAGE" ["FILE|MESSAGE" ...] \
#          > logs/submit_batch.log 2>&1 < /dev/null &
set -uo pipefail
cd /home/azureuser/muscle-arc
COMP=umud-challenge-muscle-architecture-in-ultrasound-data
AT=$1; shift
now=$(date -u +%s)
target=$(date -u -d "today $AT" +%s)
(( target <= now )) && target=$(date -u -d "tomorrow $AT" +%s)
echo "$(date -u +%FT%TZ) waiting $((target - now)) s until $(date -u -d "@$target" +%FT%TZ) for ${#} file(s)"
sleep $((target - now))
n=0
for spec in "$@"; do
  file=${spec%%|*}; msg=${spec#*|}
  if [[ ! -f $file ]]; then echo "$(date -u +%FT%TZ) MISSING $file, skipped"; continue; fi
  echo "$(date -u +%FT%TZ) submitting $file  ($msg)"
  bash scripts/vm_submit_kaggle.sh "$file" "$msg" 2>&1 | tail -3
  n=$((n + 1))
  sleep 40
done
# shellcheck disable=SC1091
source .venv/bin/activate
TOKEN="$(tr -d '\n' < ~/.kaggle/access_token)"
for wait_s in 180 300; do
  sleep "$wait_s"
  echo "$(date -u +%FT%TZ) last $((n + 1)) submissions:"
  curl -sS -H "Authorization: Bearer ${TOKEN}" \
    "https://api.kaggle.com/v1/competitions/submissions/list/${COMP}?pageSize=$((n + 1))" \
    | python -c "import sys,json; d=json.load(sys.stdin); [print(' ', x.get('date'), x.get('publicScore'), x.get('status'), x.get('description','')[:70]) for x in d]"
done
echo "$(date -u +%FT%TZ) BATCH_DONE"
