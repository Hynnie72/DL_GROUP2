"""Tests for image loading, transforms, and dataset integrity checks."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cats_dogs.data import CLASS_TO_IDX, CatsDogsDataset, get_transforms, validate_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        for split_index, split in enumerate(("train", "val", "test")):
            for class_index, class_name in enumerate(CLASS_TO_IDX):
                image_path = self.root / split / class_name / f"{class_name}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                color = (split_index * 50, class_index * 100, 25)
                Image.new("RGB", (48, 32), color).save(image_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_dataset_loads_rgb_images_with_stable_class_labels(self) -> None:
        Image.new("L", (48, 32), 128).save(self.root / "train" / "cats" / "cats.png")
        dataset = CatsDogsDataset(root=self.root, split="train")

        self.assertEqual(dataset.class_to_idx, {"cats": 0, "dogs": 1})
        self.assertEqual(len(dataset), 2)
        cat_image, cat_label = dataset[0]
        dog_image, dog_label = dataset[1]
        self.assertIsInstance(cat_image, Image.Image)
        self.assertIsInstance(dog_image, Image.Image)
        self.assertEqual(cat_image.mode, "RGB")
        self.assertEqual((cat_label, dog_label), (0, 1))

    def test_model_transforms_return_expected_rgb_tensor_sizes(self) -> None:
        image = Image.new("RGB", (48, 32), (120, 80, 40))

        lenet_image = get_transforms("lenet", "val")(image)
        resnet_image = get_transforms("resnet18", "val")(image)

        self.assertIsInstance(lenet_image, Tensor)
        self.assertEqual(tuple(lenet_image.shape), (3, 32, 32))
        self.assertEqual(tuple(resnet_image.shape), (3, 224, 224))
        self.assertTrue(
            (get_transforms("lenet", "val")(image) == lenet_image).all().item()
        )

    def test_train_augmentation_can_be_disabled_and_validation_is_not_augmented(self) -> None:
        train_transform = get_transforms("lenet", "train", augment=False)
        val_transform = get_transforms("lenet", "val", augment=True)
        image = Image.new("RGB", (48, 32), (120, 80, 40))

        self.assertTrue((train_transform(image) == val_transform(image)).all().item())

    def test_rejects_missing_or_empty_dataset_splits(self) -> None:
        with self.assertRaises(ValueError):
            CatsDogsDataset(root=self.root, split="other")  # type: ignore[arg-type]

        (self.root / "val" / "dogs" / "dogs.png").unlink()
        with self.assertRaises(ValueError):
            CatsDogsDataset(root=self.root, split="val")

    def test_validation_detects_corrupt_files_and_cross_split_duplicates(self) -> None:
        train_cat = self.root / "train" / "cats" / "cats.png"
        val_cat = self.root / "val" / "cats" / "cats.png"
        shutil.copyfile(train_cat, val_cat)
        corrupt_image = self.root / "train" / "dogs" / "broken.jpg"
        corrupt_image.write_bytes(b"not an image")

        report = validate_dataset(self.root)

        self.assertFalse(report.is_valid)
        self.assertIn(str(corrupt_image), report.invalid_images)
        self.assertTrue(report.duplicates_across_splits)

    def test_validation_passes_for_complete_readable_nonoverlapping_splits(self) -> None:
        report = validate_dataset(self.root)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.counts["train"], {"cats": 1, "dogs": 1})

    def test_validation_detects_empty_class_directories(self) -> None:
        (self.root / "test" / "dogs" / "dogs.png").unlink()

        report = validate_dataset(self.root)

        self.assertFalse(report.is_valid)
        self.assertIn(str(self.root / "test" / "dogs"), report.empty_class_directories)


if __name__ == "__main__":
    unittest.main()
