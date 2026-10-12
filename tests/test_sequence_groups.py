"""calibrate_predict.sequence_groups must only link frames of one video clip."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_predict.py"


@pytest.fixture(scope="module")
def cal():
    spec = importlib.util.spec_from_file_location("calibrate_predict", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, img: np.ndarray) -> Path:
    assert cv2.imwrite(str(path), img)
    return path


def _speckle(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    base = rng.integers(20, 200, size=shape, dtype=np.uint8)
    return cv2.GaussianBlur(base, (0, 0), 3)


def test_clip_frames_group_and_unrelated_frames_stay_singletons(tmp_path: Path, cal) -> None:
    rng = np.random.default_rng(0)
    shape = (120, 160)
    clip = _speckle(rng, shape)
    paths: list[Path] = []
    # IMG_01..IMG_03: three unrelated standalone images with consecutive numbers
    for k in range(1, 4):
        paths.append(_write(tmp_path / f"IMG_{k:02d}.png", _speckle(rng, shape)))
    # IMG_04..IMG_08: one 5-frame clip (tiny per-frame noise), then IMG_09..IMG_11 a 3-frame clip
    for k in range(4, 9):
        jitter = rng.integers(-1, 2, size=shape).astype(np.int16)
        paths.append(_write(tmp_path / f"IMG_{k:02d}.png", np.clip(clip.astype(np.int16) + jitter, 0, 255).astype(np.uint8)))
    clip2 = _speckle(rng, shape)
    for k in range(9, 12):
        jitter = rng.integers(-1, 2, size=shape).astype(np.int16)
        paths.append(_write(tmp_path / f"IMG_{k:02d}.png", np.clip(clip2.astype(np.int16) + jitter, 0, 255).astype(np.uint8)))

    groups = cal.sequence_groups(paths)
    sizes = [len(g) for g in groups]
    assert sizes == [1, 1, 1, 5, 3]
    assert groups[3] == [3, 4, 5, 6, 7]
    assert sorted(i for g in groups for i in g) == list(range(len(paths)))


def test_long_clip_is_capped_at_max_run(tmp_path: Path, cal) -> None:
    rng = np.random.default_rng(1)
    shape = (120, 160)
    clip = _speckle(rng, shape)
    paths = [_write(tmp_path / f"IMG_{k:02d}.png", clip) for k in range(1, 13)]
    groups = cal.sequence_groups(paths, max_run=5)
    assert [len(g) for g in groups] == [5, 5, 2]


def test_shape_change_or_number_gap_breaks_a_run(tmp_path: Path, cal) -> None:
    rng = np.random.default_rng(2)
    clip = _speckle(rng, (120, 160))
    paths = [
        _write(tmp_path / "IMG_01.png", clip),
        _write(tmp_path / "IMG_02.png", clip),
        _write(tmp_path / "IMG_04.png", clip),  # number gap
        _write(tmp_path / "IMG_05.png", cv2.resize(clip, (200, 150))),  # shape change
    ]
    groups = cal.sequence_groups(paths)
    assert [len(g) for g in groups] == [2, 1, 1]
