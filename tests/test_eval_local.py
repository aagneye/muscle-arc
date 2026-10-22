"""scripts/eval_local.py: official UMUD score against manual test measurements."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_local.py"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_local", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write(tmp: Path) -> tuple[Path, Path]:
    labels = {
        # MT 20 mm, FL 50 mm, PA 45 deg at 0.1 mm/px
        "A.tif": {"mt": [[[0, 0], [0, 200]]], "fl": [[[0, 0], [300, 400]]],
                  "pa": [{"fasc": [[0, 0], [10, -10]], "apo": [[0, 0], [10, 0]]}], "skip": False},
        "B.tif": {"mt": [], "fl": [], "pa": [], "skip": True},  # skipped -> excluded
        "C.tif": {"mt": [], "fl": [], "pa": [], "skip": False},  # blank -> excluded
    }
    manifest = {"images": [
        {"image_id": i, "mm_per_pixel": 0.1, "clip": None} for i in ("A.tif", "B.tif", "C.tif")
    ]}
    lp, mp = tmp / "labels.json", tmp / "manifest.json"
    lp.write_text(json.dumps(labels))
    mp.write_text(json.dumps(manifest))
    return lp, mp


def test_load_manual_labels_excludes_skipped_and_blank(tmp_path: Path, ev) -> None:
    gt = ev.load_manual_labels(*_write(tmp_path))
    assert list(gt["image_id"]) == ["A.tif"]
    row = gt.iloc[0]
    assert (row.pa_deg, row.fl_mm, row.mt_mm) == pytest.approx((45.0, 50.0, 20.0))


def test_score_matches_official_formula(tmp_path: Path, ev) -> None:
    gt = ev.load_manual_labels(*_write(tmp_path))
    # off by exactly one tolerance on each parameter -> UMUD 1.0; B is not labelled, so ignored
    pred = pd.DataFrame({
        "image_id": ["A.tif", "B.tif"],
        "pa_deg": [51.0, 1.0],
        "fl_mm": [62.0, 1.0],
        "mt_mm": [23.0, 1.0],
    })
    s = ev.score(pred, gt)
    assert s["n"] == 1
    assert s["umud"] == pytest.approx((6 / 6 + 12 / 12 + 3 / 3) / 3)


def test_score_fails_on_missing_prediction(tmp_path: Path, ev) -> None:
    gt = ev.load_manual_labels(*_write(tmp_path))
    with pytest.raises(ValueError):
        ev.score(pd.DataFrame({"image_id": ["Z.tif"], "pa_deg": [1.0], "fl_mm": [1.0], "mt_mm": [1.0]}), gt)
