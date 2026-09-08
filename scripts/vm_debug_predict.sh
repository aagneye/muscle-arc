#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
echo "cwd=$(pwd)"
ls -la data/raw | head
wc -l data/raw/sample_submission.csv submissions/submission.csv || true
echo "test files=$(find data/raw/test_images_v2 -type f | wc -l)"
head -5 data/raw/sample_submission.csv
echo "---"
python <<'PY'
from pathlib import Path
from muscle_arc.data.dataset import list_images
root = Path("data/raw/test_images_v2")
imgs = list_images(root)
print("listed", len(imgs))
print("sample", imgs[:5])
print("stems", [p.stem for p in imgs[:5]])
# show sample submission ids
import pandas as pd
df = pd.read_csv("data/raw/sample_submission.csv", sep=";", encoding="utf-8-sig")
print("sample cols", df.columns.tolist(), "rows", len(df))
print(df.head())
print("sample ids", df.iloc[:,0].astype(str).head().tolist())
PY
