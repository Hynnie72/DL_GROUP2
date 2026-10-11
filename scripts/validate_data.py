"""Validate dataset splits before starting model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cats_dogs.data.validation import SPLITS, validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check dataset folders, image readability, and split leakage."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "datastet" / "raw",
        help="Dataset root containing train, val, and test folders.",
    )
    args = parser.parse_args()

    report = validate_dataset(args.root)
    print(f"Dataset root: {args.root.resolve()}")
    print("Verified readable images:")
    for split in SPLITS:
        counts = report.counts[split]
        print(f"  {split:5} cats={counts['cats']:5} dogs={counts['dogs']:5}")

    problems = (
        ("Missing class directories", report.missing_directories),
        ("Class directories with no readable images", report.empty_class_directories),
        ("Unexpected non-image files", report.unexpected_files),
        ("Unreadable or corrupt images", report.invalid_images),
        ("Byte-identical images across splits", report.duplicates_across_splits),
    )
    for title, entries in problems:
        print(f"{title}: {len(entries)}")
        for entry in entries[:10]:
            print(f"  {entry}")
        if len(entries) > 10:
            print(f"  ... and {len(entries) - 10} more")

    if not report.is_valid:
        print("Dataset validation failed; fix the issues above before training.")
        return 1

    print("Dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
