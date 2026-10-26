"""scripts/fuse_measurements.py: per-image median of several runs' picked variants."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods():
    return _load("fuse_measurements"), _load("geometry_sweep")


def _measures(seed: int, n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "image_id": [f"IMG_{i:05d}.tif" for i in range(n)], "mm_per_pixel": 0.07,
        "mt_legacy": rng.uniform(15, 25, n), "mt_host": rng.uniform(15, 25, n),
        "pa_legacy": rng.uniform(12, 25, n), "pa_noradon": rng.uniform(12, 25, n),
        "fl_legacy": rng.uniform(50, 80, n), "fl_contour": rng.uniform(50, 80, n),
        "fl_le3": rng.uniform(50, 80, n), "fl_le5": rng.uniform(50, 80, n),
    })


def test_fuse_is_per_image_median_of_picked_variants(mods):
    fm, sw = mods
    runs = [_measures(s) for s in (0, 1, 2)]
    pick = {"pa_deg": "pa_legacy+2", "fl_mm": "fl_legacy|trig0", "mt_mm": "mt_legacy"}
    per_run = fm.picked_series(runs, pick, None, sw.build_variants)
    fused, spread = fm.fuse(per_run, "median")
    expect_pa = np.median(np.stack([sw.build_variants(r, None)["pa_deg"]["pa_legacy+2"].to_numpy() for r in runs]), axis=0)
    assert np.allclose(fused["pa_deg"].to_numpy(), expect_pa)
    expect_mt = np.median(np.stack([r["mt_legacy"].to_numpy() for r in runs]), axis=0)
    assert np.allclose(fused["mt_mm"].to_numpy(), expect_mt)
    assert (spread >= 0).all().all()
    assert list(fused.index) == runs[0]["image_id"].tolist()


def test_runs_are_aligned_by_image_id(mods):
    fm, sw = mods
    a = _measures(0)
    b = _measures(0).iloc[::-1].reset_index(drop=True)  # same values, reversed order
    per_run = fm.picked_series([a, b], {"pa_deg": "pa_legacy+0", "fl_mm": "fl_contour|trig0", "mt_mm": "mt_host"},
                               None, sw.build_variants)
    fused, spread = fm.fuse(per_run, "mean")
    assert np.allclose(spread.to_numpy(), 0.0)
    assert np.allclose(fused["mt_mm"].to_numpy(), a["mt_host"].to_numpy())
