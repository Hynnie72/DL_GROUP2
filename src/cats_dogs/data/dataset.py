"""Dataset loading for cat-and-dog image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]
CLASS_TO_IDX = {"cats": 0, "dogs": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CatsDogsDataset(Dataset[tuple[Image.Image | Tensor, int]]):
    """Load images from ``root/<split>/{cats,dogs}``."""

    def __init__(
        self,
        root: str | Path | None = None,
        split: Split = "train",
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split {split!r}; use 'train', 'val', or 'test'.")

        if root is None:
            root = Path(__file__).resolve().parents[3] / "datastet" / "raw"
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.classes = tuple(CLASS_TO_IDX)
        self.class_to_idx = CLASS_TO_IDX.copy()
        self.samples: list[tuple[Path, int]] = []

        split_root = self.root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"Dataset split directory does not exist: {split_root}")

        for class_name, class_index in self.class_to_idx.items():
            class_root = split_root / class_name
            if not class_root.is_dir():
                raise FileNotFoundError(f"Class directory does not exist: {class_root}")

            class_images = sorted(
                path
                for path in class_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not class_images:
                raise ValueError(f"No supported images found in class directory: {class_root}")
            self.samples.extend((path, class_index) for path in class_images)

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image | Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label
