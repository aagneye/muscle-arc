#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc

# Stop bloated full-OSF download / prior climb
pkill -f "download_dl_track.py" 2>/dev/null || true
pkill -f "vm_run_sof_climb.sh" 2>/dev/null || true
sleep 2

# Make download a no-op when .h5 already present (avoid multi-GB re-pull)
python3 - <<'PY'
from pathlib import Path
p = Path("scripts/download_dl_track.py")
text = p.read_text()
needle = "args.out_dir.mkdir(parents=True, exist_ok=True)"
patch = '''args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.out_dir.rglob("*.h5")) + list(args.out_dir.rglob("*.hdf5"))
    if existing:
        print(f"DL_TRACK_OK n_h5={len(existing)} (skip download)")
        for e in existing[:10]:
            print(" ", e)
        return
'''
if "skip download" not in text:
    if needle not in text:
        raise SystemExit("patch point missing")
    p.write_text(text.replace(needle, patch, 1))
    print("patched download_dl_track.py")
else:
    print("download already patched")
PY

mkdir -p logs
nohup env SKIP_SOF_TRAIN=0 AUTO_SUBMIT=0 TAG=sof_v1 bash scripts/vm_run_sof_climb.sh \
  > logs/sof_v1_climb.out 2>&1 &
echo "started pid=$!"
sleep 3
ps aux | grep -E "vm_run_sof|download_dl|eval_osf|train_multitask" | grep -v grep || true
tail -n 30 logs/sof_v1_climb.out
