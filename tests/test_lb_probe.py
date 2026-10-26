"""scripts/lb_probe.py: probe files and exact recovery of MAE / bias from public scores."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from muscle_arc.geometry.umud_metric import umud_from_errors

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lb_probe.py"
PARAMS = ("pa_deg", "fl_mm", "mt_mm")


@pytest.fixture(scope="module")
def lp():
    spec = importlib.util.spec_from_file_location("lb_probe", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _world(n: int = 120, seed: int = 0):
    """Test set, predictions, and the public subset the board scores (about a third)."""
    rng = np.random.default_rng(seed)
    ids = [f"IMG_{i:05d}.tif" for i in range(n)]
    gt = pd.DataFrame({"image_id": ids, "pa_deg": rng.uniform(10, 30, n),
                       "fl_mm": rng.uniform(40, 90, n), "mt_mm": rng.uniform(10, 30, n)})
    pred = gt.copy()
    pred["pa_deg"] += rng.normal(0.4, 2.0, n)
    pred["fl_mm"] += rng.normal(-1.5, 8.0, n)
    pred["mt_mm"] += rng.normal(-0.2, 0.5, n)
    public = sorted(rng.choice(ids, n // 3, replace=False))
    return gt, pred, public


def _score(pred: pd.DataFrame, gt: pd.DataFrame, public: list[str]) -> float:
    """Board score: official formula on the public subset, rounded like the board."""
    m = gt[gt["image_id"].isin(public)].merge(pred, on="image_id", suffixes=("_gt", ""))
    e = {p: (m[p] - m[f"{p}_gt"]).abs().to_numpy() for p in PARAMS}
    return round(umud_from_errors(e["pa_deg"], e["fl_mm"], e["mt_mm"])["umud"], 5)


def test_make_probe_shifts_only_subset(lp):
    _, pred, _ = _world()
    ids = list(pred["image_id"][:10])
    probe = lp.make_probe(pred, "fl_mm", 1000.0, ids)
    assert np.allclose(probe["fl_mm"][:10] - pred["fl_mm"][:10], 1000.0)
    assert probe["fl_mm"][10:].equals(pred["fl_mm"][10:])
    assert probe[["pa_deg", "mt_mm"]].equals(pred[["pa_deg", "mt_mm"]])


def test_select_ids_where(lp):
    table = pd.DataFrame({"image_id": ["a", "b", "c"], "source": ["x", "y", "x"], "clip": [1, None, 2]})
    assert lp.select_ids(table, "source=x") == ["a", "c"]
    assert lp.select_ids(table, "source!=x") == ["b"]
    assert lp.select_ids(table, "clip=*") == ["a", "c"]


def test_decode_recovers_full_set_and_subgroup_stats(lp):
    gt, pred, public = _world()
    n = len(public)
    s0 = _score(pred, gt, public)
    sub = list(pred["image_id"][:40])  # a "device layout"; only some of it is public
    pairs, singles = [], []
    for param, k in (("pa_deg", 100.0), ("fl_mm", 1000.0)):
        sp = _score(lp.make_probe(pred, param, k, None), gt, public)
        sm = _score(lp.make_probe(pred, param, -k, None), gt, public)
        pairs.append((param, k, sp, sm, n, "all"))
    sp = _score(lp.make_probe(pred, "fl_mm", 1e4, sub), gt, public)
    sm = _score(lp.make_probe(pred, "fl_mm", -1e4, sub), gt, public)
    pairs.append(("fl_mm", 1e4, sp, sm, None, "sub"))  # public size unknown -> recovered
    singles.append(("mt_mm", 100.0, _score(lp.make_probe(pred, "mt_mm", 100.0, None), gt, public), None, "all"))

    df = lp.decode(n, s0, pairs, singles).set_index(["param", "subset"])
    m = gt[gt["image_id"].isin(public)].merge(pred, on="image_id", suffixes=("_gt", ""))
    err = {p: (m[p] - m[f"{p}_gt"]).to_numpy() for p in PARAMS}
    for p in ("pa_deg", "fl_mm"):
        assert df.loc[(p, "all"), "mae"] == pytest.approx(np.abs(err[p]).mean(), abs=0.02)
        assert df.loc[(p, "all"), "bias"] == pytest.approx(err[p].mean(), abs=0.02)
    in_sub = m["image_id"].isin(sub).to_numpy()
    e_sub = err["fl_mm"][in_sub]
    assert df.loc[("fl_mm", "sub"), "n"] == in_sub.sum()
    assert df.loc[("fl_mm", "sub"), "mae"] == pytest.approx(np.abs(e_sub).mean(), abs=0.05)
    assert df.loc[("fl_mm", "sub"), "bias"] == pytest.approx(e_sub.mean(), abs=0.05)
    # MT closes from the base score; its bias follows from the single probe
    assert df.loc[("mt_mm", "all (from S0)"), "mae"] == pytest.approx(np.abs(err["mt_mm"]).mean(), abs=0.02)
    assert df.loc[("mt_mm", "all"), "bias"] == pytest.approx(err["mt_mm"].mean(), abs=0.02)
    full = df.loc[[("pa_deg", "all"), ("fl_mm", "all"), ("mt_mm", "all (from S0)")], "share"]
    assert full.sum() == pytest.approx(1.0)


def test_full_set_decode_is_independent_of_public_size(lp):
    """A shift on every image cancels N, so the wrong N still gives the right MAE / bias."""
    gt, pred, public = _world()
    s0 = _score(pred, gt, public)
    sp = _score(lp.make_probe(pred, "pa_deg", 100.0, None), gt, public)
    sm = _score(lp.make_probe(pred, "pa_deg", -100.0, None), gt, public)
    _, mae_a, bias_a = lp.pair_stats("pa_deg", 100.0, sp, sm, s0, 102, 102)
    _, mae_b, bias_b = lp.pair_stats("pa_deg", 100.0, sp, sm, s0, 309, 309)
    assert mae_a == pytest.approx(mae_b) and bias_a == pytest.approx(bias_b)
