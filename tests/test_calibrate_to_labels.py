"""scripts/calibrate_to_labels.py: pooled per-device calibration gated by leave-one-out."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_to_labels.py"


@pytest.fixture(scope="module")
def cal():
    spec = importlib.util.spec_from_file_location("calibrate_to_labels", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _world(n_lab: int = 40, n_all: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = [f"IMG_{i:05d}.tif" for i in range(n_all)]
    layout = np.where(np.arange(n_all) % 2 == 0, "siemens", "ge")
    truth = pd.DataFrame({"image_id": ids, "pa_deg": rng.uniform(12, 28, n_all),
                          "fl_mm": rng.uniform(45, 85, n_all), "mt_mm": rng.uniform(12, 28, n_all)})
    pred = truth.copy()
    pred["pa_deg"] += np.where(layout == "siemens", 2.5, -1.0) + rng.normal(0, 0.3, n_all)  # device bias
    pred["fl_mm"] += rng.normal(0, 0.5, n_all)  # already calibrated: must stay untouched
    pred["mt_mm"] += 0.8 + rng.normal(0, 0.1, n_all)  # global bias
    gt = truth.iloc[:n_lab].reset_index(drop=True)
    groups = pd.Series(layout, index=ids)
    return pred, gt, groups, truth


def test_calibrate_fixes_biased_parameters_and_leaves_the_rest(cal):
    pred, gt, groups, truth = _world()
    out, maps, report = cal.calibrate(pred, gt, groups, shrink=0.0, pool=5.0)
    r = report.set_index("param")
    assert r.loc["pa_deg", "applied"] and r.loc["mt_mm", "applied"]
    assert not r.loc["fl_mm", "applied"]
    assert report.attrs["umud_after_loo"] < report.attrs["umud_before"]
    unl = out.iloc[40:].merge(truth, on="image_id", suffixes=("", "_t"))  # unlabelled images improve too
    assert np.mean(np.abs(unl["pa_deg"] - unl["pa_deg_t"])) < 0.6
    assert np.mean(np.abs(unl["mt_mm"] - unl["mt_mm_t"])) < 0.25
    assert out["fl_mm"].equals(pred["fl_mm"])
    assert set(maps) == {"pa_deg", "mt_mm"} and {"siemens", "ge", "__global__"} <= set(maps["pa_deg"])


def test_calibrate_needs_enough_labels(cal):
    pred, gt, groups, _ = _world(n_lab=3)
    with pytest.raises(ValueError):
        cal.calibrate(pred, gt, groups, shrink=0.5, pool=10.0)
