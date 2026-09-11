#!/usr/bin/env bash
set -euo pipefail
cd /home/azureuser/muscle-arc
source .venv/bin/activate
mkdir -p src/muscle_arc/models/dl_track_official
cp -f .venv/lib/python3.10/site-packages/DL_Track_US/gui_helpers/do_calculations.py \
  src/muscle_arc/models/dl_track_official/do_calculations.py
python3 <<'PY'
from pathlib import Path
p = Path("src/muscle_arc/models/dl_track_official/do_calculations.py")
t = p.read_text()
t = t.replace("import tkinter as tk\n", "")
t = t.replace(
    'tk.messagebox.showerror("Information", "Aponeurosis length threshold too big.")',
    'raise ValueError("Aponeurosis length threshold too big.")',
)
# Make importable as package submodule without DL_Track_US deps
Path("src/muscle_arc/models/dl_track_official/__init__.py").write_text("")
p.write_text(t)
print("ok", len(t.splitlines()))
PY
