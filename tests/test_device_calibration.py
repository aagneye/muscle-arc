"""scripts/apply_device_calibration.py: per-layout constants applied to the final CSV."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("apply_device_calibration", _ROOT / "scripts" / "apply_device_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dc():
    return _load()


def _cand() -> tuple[pd.DataFrame, pd.Series]:
    cand = pd.DataFrame({"image_id": ["a.tif", "b.tif", "c.png", "d.tif"],
                         "pa_deg": [17.0, 15.0, 22.0, 20.0], "fl_mm": [80.0, 90.0, 60.0, 70.0],
                         "mt_mm": [20.0, 21.0, 19.0, 18.0]})
    layouts = pd.Series({"a.tif": "siemens_right_ruler", "b.tif": "left_ruler_5cm", "c.png": "png_left_ruler"})
    return cand, layouts


def test_corrections_apply_to_their_layout_only(dc):
    cand, layouts = _cand()
    out = dc.apply_calibration(cand, layouts, {"siemens_right_ruler": {"fl_mul": 0.9},
                                               "png_left_ruler": {"fl_mul": 1.1, "pa_add": -3.0}})
    assert out["fl_mm"].tolist() == pytest.approx([72.0, 90.0, 66.0, 70.0])
    assert out["pa_deg"].tolist() == pytest.approx([17.0, 15.0, 19.0, 20.0])
    assert out["mt_mm"].tolist() == cand["mt_mm"].tolist()
    assert cand["fl_mm"].tolist() == [80.0, 90.0, 60.0, 70.0]  # input untouched


def test_image_without_layout_is_unchanged(dc):
    cand, layouts = _cand()
    out = dc.apply_calibration(cand, layouts, {"siemens_right_ruler": {"fl_add": -8.0, "mt_add": 0.5}})
    assert out.loc[3].tolist() == cand.loc[3].tolist()
    assert out.loc[0, "fl_mm"] == pytest.approx(72.0) and out.loc[0, "mt_mm"] == pytest.approx(20.5)


def test_unknown_correction_is_rejected(dc):
    cand, layouts = _cand()
    with pytest.raises(KeyError):
        dc.apply_calibration(cand, layouts, {"siemens_right_ruler": {"fl_scale": 0.9}})


def test_shipped_config_uses_known_corrections(dc):
    calib = yaml.safe_load((_ROOT / "configs" / "device_calibration.yaml").read_text())["layouts"]
    assert calib and all(set(c) <= set(dc.OPS) for c in calib.values())
