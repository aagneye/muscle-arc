"""Unit tests for muscle_arc.data.splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from muscle_arc.data.splits import assign_groups_to_splits, dedupe_pairs, group_ids_for_stems


def _write_pair(tmp_path: Path, stem: str, img_bytes: bytes, mask_bytes: bytes) -> tuple[Path, Path]:
    img_dir = tmp_path / "imgs"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    img_path = img_dir / f"{stem}.tif"
    mask_path = mask_dir / f"{stem}.tif"
    img_path.write_bytes(img_bytes)
    mask_path.write_bytes(mask_bytes)
    return img_path, mask_path


def test_dedupe_drops_exact_duplicate_pair(tmp_path: Path) -> None:
    p1 = _write_pair(tmp_path, "IMG0001", b"same-image-bytes", b"same-mask-bytes")
    p2 = _write_pair(tmp_path, "IMG0002", b"same-image-bytes", b"same-mask-bytes")
    p3 = _write_pair(tmp_path, "IMG0003", b"different-image-bytes", b"different-mask-bytes")

    kept, n_dropped, dup_groups = dedupe_pairs([p1, p2, p3])

    assert n_dropped == 1
    kept_stems = {img.stem for img, _ in kept}
    assert kept_stems == {"IMG0001", "IMG0003"}
    assert dup_groups == {"IMG0001": ["IMG0002"]}


def test_dedupe_no_duplicates_is_noop(tmp_path: Path) -> None:
    p1 = _write_pair(tmp_path, "IMG0001", b"a-image", b"a-mask")
    p2 = _write_pair(tmp_path, "IMG0002", b"b-image", b"b-mask")

    kept, n_dropped, dup_groups = dedupe_pairs([p1, p2])

    assert n_dropped == 0
    assert dup_groups == {}
    assert len(kept) == 2


def test_dedupe_handles_duplicate_group_of_three(tmp_path: Path) -> None:
    p1 = _write_pair(tmp_path, "IMG0001", b"x", b"y")
    p2 = _write_pair(tmp_path, "IMG0002", b"x", b"y")
    p3 = _write_pair(tmp_path, "IMG0003", b"x", b"y")

    kept, n_dropped, dup_groups = dedupe_pairs([p1, p2, p3])

    assert n_dropped == 2
    assert len(kept) == 1
    assert kept[0][0].stem == "IMG0001"
    assert dup_groups == {"IMG0001": ["IMG0002", "IMG0003"]}


def test_no_hash_collision_across_splits_after_dedupe(tmp_path: Path) -> None:
    """Regression guard: once deduped, no content-identical pair should be
    able to leak into two different splits (train/val/holdout)."""
    pairs = [
        _write_pair(tmp_path, "IMG0001", b"same", b"same-mask"),
        _write_pair(tmp_path, "IMG0002", b"same", b"same-mask"),  # dup of 0001
        _write_pair(tmp_path, "IMG0010", b"unique-a", b"mask-a"),
        _write_pair(tmp_path, "IMG0020", b"unique-b", b"mask-b"),
        _write_pair(tmp_path, "IMG0030", b"unique-c", b"mask-c"),
    ]
    kept, n_dropped, _ = dedupe_pairs(pairs)
    assert n_dropped == 1

    stems = [img.stem for img, _ in kept]
    splits = assign_groups_to_splits(stems, seed=0, train_frac=0.4, val_frac=0.3, holdout_frac=0.3)

    for a, b in (("train", "val"), ("train", "holdout"), ("val", "holdout")):
        ga = group_ids_for_stems(splits[a])
        gb = group_ids_for_stems(splits[b])
        assert not (ga & gb)
