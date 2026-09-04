#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser

if [ ! -d muscle-arc/.git ]; then
  git clone https://github.com/aagneye/muscle-arc.git
else
  git -C muscle-arc pull --ff-only
fi

cd muscle-arc
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
pip install -e .
python -c "import torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())"

echo "=== downloading competition data ==="
python scripts/download_data.py --out data/raw

echo "=== data layout ==="
ls -la data/raw | head -40
du -sh data/raw/*
