"""Integrity checks for image-classification dataset splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from cats_dogs.data.dataset import CLASS_TO_IDX, IMAGE_EXTENSIONS

SPLITS = ("train", "val", "test")


@dataclass
class DatasetValidationReport:
    """Results from checking expected folders, image integrity, and leakage."""

    counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            split: {class_name: 0 for class_name in CLASS_TO_IDX}
            for split in SPLITS
        }
    )
    missing_directories: list[str] = field(default_factory=list)
    empty_class_directories: list[str] = field(default_factory=list)
    unexpected_files: list[str] = field(default_factory=list)
    invalid_images: list[str] = field(default_factory=list)
    duplicates_across_splits: list[tuple[str, str]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            not self.missing_directories
            and not self.empty_class_directories
            and not self.unexpected_files
            and not self.invalid_images
            and not self.duplicates_across_splits
            and all(
                count > 0
                for split_counts in self.counts.values()
                for count in split_counts.values()
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(root: str | Path) -> DatasetValidationReport:
    """Check all expected images under ``root/{train,val,test}/{cats,dogs}``.

    Byte-identical image files in different splits are reported because they
    can leak evaluation samples into training or validation.
    """
    root = Path(root)
    report = DatasetValidationReport()
    image_hashes: dict[str, list[tuple[str, Path]]] = {}

    for split in SPLITS:
        for class_name in CLASS_TO_IDX:
            class_root = root / split / class_name
            if not class_root.is_dir():
                report.missing_directories.append(str(class_root))
                continue

            readable_count = report.counts[split][class_name]
            for path in sorted(class_root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    report.unexpected_files.append(str(path))
                    continue

                try:
                    with Image.open(path) as image:
                        image.load()
                    digest = _sha256(path)
                except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
                    report.invalid_images.append(str(path))
                    continue

                report.counts[split][class_name] += 1
                image_hashes.setdefault(digest, []).append((split, path))
            if report.counts[split][class_name] == readable_count:
                report.empty_class_directories.append(str(class_root))

    for locations in image_hashes.values():
        first_by_split: dict[str, Path] = {}
        for split, path in locations:
            for other_split, other_path in first_by_split.items():
                if split != other_split:
                    report.duplicates_across_splits.append(
                        (str(other_path), str(path))
                    )
            first_by_split.setdefault(split, path)

    return report
