"""scripts/cache_probs.py writes one float16 probability npz per image."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cache_probs.py"


@pytest.fixture(scope="module")
def cp():
    spec = importlib.util.spec_from_file_location("cache_probs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class _ConstLogit(nn.Module):
    """Returns a constant logit map so the averaged probability is predictable."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.value)


def test_cache_roundtrip_and_model_averaging(tmp_path: Path, cp) -> None:
    items = [
        ("IMG_00001.tif", np.full((120, 160), 90, np.uint8), 0.0686),
        ("IMG_00300.png", np.full((512, 465), 60, np.uint8), 0.1285),
    ]
    # two apo "folds": sigmoid(0)=0.5 and sigmoid(2)=0.881 -> mean 0.6905
    apo = [_ConstLogit(0.0), _ConstLogit(2.0)]
    fasc = [_ConstLogit(-2.0)]
    n = cp.cache_items(items, apo, fasc, img_size=64, device=torch.device("cpu"),
                       out_dir=tmp_path, tta_hflip=True, multiscale=False)
    assert n == 2
    for image_id, gray, mm in items:
        d = np.load(tmp_path / f"{image_id}.npz")
        assert d["apo"].dtype == np.float16 and d["fasc"].dtype == np.float16
        assert tuple(d["shape"]) == gray.shape == d["apo"].shape
        assert abs(float(d["mm_per_pixel"]) - mm) < 1e-9
        assert abs(float(d["apo"].mean()) - 0.6905) < 2e-3
        assert abs(float(d["fasc"].mean()) - 0.1192) < 2e-3


def test_existing_files_are_skipped(tmp_path: Path, cp) -> None:
    items = [("IMG_00001.tif", np.full((40, 40), 10, np.uint8), 0.07)]
    (tmp_path / "IMG_00001.tif.npz").write_bytes(b"placeholder")
    n = cp.cache_items(items, [_ConstLogit(0.0)], [_ConstLogit(0.0)], img_size=32,
                       device=torch.device("cpu"), out_dir=tmp_path,
                       tta_hflip=False, multiscale=False, overwrite=False)
    assert n == 0
