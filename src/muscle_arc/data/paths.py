from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Resolved Kaggle UMUD challenge directories under ``root``."""

    root: Path
    apo_imgs: Path
    apo_masks: Path
    fasc_imgs: Path
    fasc_masks: Path
    test_images: Path
    sample_submission: Path

    @classmethod
    def from_config(cls, data_cfg: dict) -> "DataPaths":
        root = Path(data_cfg["root"])
        return cls(
            root=root,
            apo_imgs=root / data_cfg["apo_imgs"],
            apo_masks=root / data_cfg["apo_masks"],
            fasc_imgs=root / data_cfg["fasc_imgs"],
            fasc_masks=root / data_cfg["fasc_masks"],
            test_images=root / data_cfg["test_images"],
            sample_submission=root / data_cfg["sample_submission"],
        )

    def assert_train_present(self) -> None:
        missing = [
            p
            for p in (self.apo_imgs, self.apo_masks, self.fasc_imgs, self.fasc_masks)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing training folders (run scripts/download_data.py): "
                + ", ".join(str(p) for p in missing)
            )

    def assert_test_present(self) -> None:
        if not self.test_images.exists():
            raise FileNotFoundError(
                f"Missing test images at {self.test_images} (run scripts/download_data.py)"
            )
