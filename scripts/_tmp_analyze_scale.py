#!/usr/bin/env python3
"""One-shot analysis of experiments/scale_table.csv (decision A vs A')."""
from __future__ import annotations

import pandas as pd
from muscle_arc.geometry.scale import load_scale_table


def main() -> None:
    df = pd.read_csv("experiments/scale_table.csv")
    print("raw_with_scale", int(df["mm_per_pixel"].notna().sum()), "/", len(df))
    for f, g in df.groupby("folder"):
        print("folder", f, int(g["mm_per_pixel"].notna().sum()), "/", len(g))
    print("sources", df.loc[df.mm_per_pixel.notna(), "source"].value_counts().to_dict())
    udf = load_scale_table("experiments/scale_table.csv")
    print("usable", int(udf["scale_usable"].sum()), "/", len(udf))
    test = udf[udf["folder"].astype(str).str.contains("test", case=False)]
    print("TEST_usable", int(test["scale_usable"].sum()), "/", len(test))
    print("TEST_raw", int(test["mm_per_pixel"].notna().sum()))
    t = test[test.mm_per_pixel.notna()]
    if len(t):
        print(t[["image_id", "xres", "resunit", "mm_per_pixel", "source", "scale_usable"]].head(20).to_string())
        print("mm_unique", sorted(t["mm_per_pixel"].unique().tolist())[:30])
        print("usable_by_source", t.groupby("source")["scale_usable"].sum().to_dict())
    # Decision
    tok = int(test["scale_usable"].sum())
    n = len(test)
    print(f"\nDECISION test_with_usable_scale={tok}/{n}")
    if tok == 0:
        print("NO_METADATA_SCALE — use Phase A' residual regressor / richer clustering")
    elif tok < 0.5 * n:
        print("PARTIAL_METADATA_SCALE — hybrid metadata + cluster fallback")
    else:
        print("METADATA_SCALE_OK — prefer per-image metadata")


if __name__ == "__main__":
    main()
