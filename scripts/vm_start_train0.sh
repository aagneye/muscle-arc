#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
git pull --ff-only
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e . -q
python - <<'PY'
from pathlib import Path
from muscle_arc.data.dataset import pair_images_masks
root = Path("data/raw")
apo = pair_images_masks(root / "apo_imgs_v1", root / "apo_masks_v1")
fasc = pair_images_masks(root / "fasc_imgs_v1", root / "fasc_masks_v1")
print("apo_pairs", len(apo), "fasc_pairs", len(fasc))
PY
chmod +x scripts/vm_train_one_gpu.sh
# avoid double-start
if [ -f logs/train_gpu0.pid ] && kill -0 "$(cat logs/train_gpu0.pid)" 2>/dev/null; then
  echo "Training already running pid=$(cat logs/train_gpu0.pid)"
else
  bash scripts/vm_train_one_gpu.sh 0
fi
sleep 12
nvidia-smi
echo "--- log ---"
tail -50 logs/train_gpu0.log || true
