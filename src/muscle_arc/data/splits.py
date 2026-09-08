"""Sequence-safe train / val / holdout splits for ultrasound mask pairs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_TRAIL_NUM = re.compile(r"^(.*?)(\d+)$")


def parse_stem(stem: str) -> tuple[str, int] | None:
    m = _TRAIL_NUM.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def group_key(stem: str) -> str:
    """Stable group id: prefix + run-start frame for consecutive numeric stems.

    Prefer calling ``build_sequence_groups`` for split assignment; this helper is
    for leakage checks on already-grouped stems (falls back to prefix).
    """
    parsed = parse_stem(stem)
    if parsed is None:
        return stem
    return f"{parsed[0]}#{parsed[1]}"


def build_sequence_groups(stems: list[str], max_run: int = 5) -> dict[str, str]:
    """Map each stem → group_id using contiguous consecutive frame runs.

    Same idea as ``calibrate_predict.sequence_groups``: frames that share a
    prefix and consecutive integers stay together so a short clip cannot leak
    across train/val/holdout.

    Long consecutive ID ranges (common in training dumps) are chunked into
    windows of ``max_run`` so we still get many independent groups.
    """
    ordered = sorted(stems)
    stem_to_group: dict[str, str] = {}
    i = 0
    while i < len(ordered):
        s = ordered[i]
        parsed = parse_stem(s)
        if parsed is None:
            stem_to_group[s] = s
            i += 1
            continue
        prefix, start_n = parsed
        run = [s]
        expect = start_n + 1
        j = i + 1
        while j < len(ordered):
            p2 = parse_stem(ordered[j])
            if p2 is None or p2[0] != prefix or p2[1] != expect:
                break
            run.append(ordered[j])
            expect += 1
            j += 1
        # Chunk long consecutive ranges into max_run windows
        for offset in range(0, len(run), max_run):
            chunk = run[offset : offset + max_run]
            first = parse_stem(chunk[0])
            assert first is not None
            gid = f"{first[0]}#{first[1]}"
            for member in chunk:
                stem_to_group[member] = gid
        i = j
    return stem_to_group


def assign_groups_to_splits(
    stems: list[str],
    *,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    holdout_frac: float = 0.15,
) -> dict[str, list[str]]:
    """Partition stems into train/val/holdout by sequence group (no leakage)."""
    if abs(train_frac + val_frac + holdout_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + holdout_frac must sum to 1")

    stem_to_group = build_sequence_groups(stems)
    by_group: dict[str, list[str]] = {}
    for s in stems:
        by_group.setdefault(stem_to_group[s], []).append(s)

    groups = sorted(by_group.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_hold = max(1, int(round(n * holdout_frac))) if n >= 3 else 0
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else max(1, n // 5)
    n_train = n - n_val - n_hold
    if n_train < 1:
        n_train = 1
        n_val = max(0, n - n_train - n_hold)

    hold_g = set(groups[:n_hold])
    val_g = set(groups[n_hold : n_hold + n_val])

    out: dict[str, list[str]] = {"train": [], "val": [], "holdout": []}
    for g, members in by_group.items():
        if g in hold_g:
            out["holdout"].extend(members)
        elif g in val_g:
            out["val"].extend(members)
        else:
            out["train"].extend(members)

    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def write_split_manifest(
    out_dir: Path,
    branch: str,
    splits: dict[str, list[str]],
    *,
    seed: int,
    meta: dict | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_to_group = build_sequence_groups([s for ss in splits.values() for s in ss])
    payload = {
        "branch": branch,
        "seed": seed,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_holdout": len(splits["holdout"]),
        "n_groups": len(set(stem_to_group.values())),
        "splits": splits,
        "meta": meta or {},
    }
    path = out_dir / f"{branch}_splits.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_split_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def filter_pairs_by_stems(
    pairs: list[tuple[Path, Path]],
    stems: set[str] | list[str],
) -> list[tuple[Path, Path]]:
    allow = set(stems)
    return [(i, m) for i, m in pairs if i.stem in allow]


def group_ids_for_stems(stems: list[str]) -> set[str]:
    m = build_sequence_groups(stems)
    return {m[s] for s in stems}
