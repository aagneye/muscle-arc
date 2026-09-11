#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
python -u scripts/eval_osf_pipeline.py \
  --config configs/default.yaml \
  --masks-from dl_track \
  --geom ours \
  --sector-crop false \
  --chrome-mask false \
  --scale-mode pred \
  --max-images 35 \
  --out experiments/gate2b_dltrack_win.json \
  2>&1 | tee logs/gate2b_win_restore2.out
python3 -c "import json; m=json.load(open('experiments/gate2b_dltrack_win.json')); print('UMUD', m['umud'], 'mt_bad', m['n_mt_bad']); print('OK' if m['umud']<=0.42 else 'FAIL')"
