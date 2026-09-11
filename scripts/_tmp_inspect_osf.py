#!/usr/bin/env python3
"""Inspect OSF architecture xlsx + sample_submission GT rows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from muscle_arc.infer.predict import load_sample_submission


def peek_xlsx(path: Path) -> None:
    print("===", path)
    if not path.exists():
        print("missing")
        return
    xl = pd.ExcelFile(path)
    print("sheets", xl.sheet_names)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        print("---", sheet, df.shape, list(df.columns))
        print(df.head(12).to_string())
        print()


def main() -> None:
    root = Path("data/external/umud_osf/unpacked")
    for p in sorted(root.rglob("*.xlsx")):
        peek_xlsx(p)
    readme = root / "benchmark_dataset_architecture_v0.1.0" / "Readme.md"
    if readme.exists():
        print("README:\n", readme.read_text()[:2500])

    sample = Path("data/raw/sample_submission.csv")
    if sample.exists():
        s = load_sample_submission(sample, sep=";")
        print("sample shape", s.shape, list(s.columns))
        print(s.head(10).to_string())
        uniq = s.drop_duplicates(["pa_deg", "fl_mm", "mt_mm"])
        print("n_unique_triples", len(uniq))
        print(uniq.head(30).to_string())
        # rows that differ from mode
        mode = s.groupby(["pa_deg", "fl_mm", "mt_mm"]).size().sort_values(ascending=False)
        print("top triples:\n", mode.head(10))


if __name__ == "__main__":
    main()
