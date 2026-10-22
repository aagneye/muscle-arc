"""scripts/label_server.py: atomic label saves, backups, and safe image paths."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "label_server.py"


@pytest.fixture(scope="module")
def ls():
    spec = importlib.util.spec_from_file_location("label_server", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_save_and_load_roundtrip(tmp_path: Path, ls) -> None:
    store = ls.LabelStore(tmp_path / "labels.json", backup_every=2)
    assert store.load() == {}
    rec = {"mt": [[[1, 2], [1, 50]]], "fl": [], "pa": [], "skip": False, "seconds": 12}
    store.save("IMG_00001.tif", rec)
    store.save("IMG_00002.tif", {"mt": [], "fl": [], "pa": [], "skip": True})
    data = json.loads((tmp_path / "labels.json").read_text())
    assert data["IMG_00001.tif"]["mt"] == rec["mt"]
    assert data["IMG_00002.tif"]["skip"] is True
    assert "saved_at" in data["IMG_00001.tif"]
    assert len(list((tmp_path / "backups").glob("labels_*.json"))) == 1  # every 2nd save


def test_save_overwrites_same_image(tmp_path: Path, ls) -> None:
    store = ls.LabelStore(tmp_path / "labels.json")
    store.save("IMG_00001.tif", {"mt": [], "fl": [], "pa": [], "skip": False})
    store.save("IMG_00001.tif", {"mt": [[[0, 0], [0, 10]]], "fl": [], "pa": [], "skip": False})
    assert len(store.load()["IMG_00001.tif"]["mt"]) == 1


def test_rejects_unknown_image_id(tmp_path: Path, ls) -> None:
    store = ls.LabelStore(tmp_path / "labels.json", allowed_ids={"IMG_00001.tif"})
    with pytest.raises(ValueError):
        store.save("../../etc/passwd", {"mt": [], "fl": [], "pa": [], "skip": False})


def test_safe_child_blocks_traversal(tmp_path: Path, ls) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"x")
    assert ls.safe_child(tmp_path / "images", "a.png") == (tmp_path / "images" / "a.png").resolve()
    assert ls.safe_child(tmp_path / "images", "../secret.txt") is None
    assert ls.safe_child(tmp_path / "images", "missing.png") is None
