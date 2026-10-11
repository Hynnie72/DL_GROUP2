"""Image transformations for LeNet and ResNet18 classification."""

from __future__ import annotations

from typing import Literal

from torchvision import transforms

from cats_dogs.data.dataset import Split

ModelName = Literal["lenet", "resnet18"]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(
    model: ModelName,
    split: Split,
    augment: bool = True,
) -> transforms.Compose:
    """Build deterministic evaluation or training transforms for one model.

    Both models receive RGB tensors. The LeNet implementation should therefore
    accept three input channels.
    """
    if model not in ("lenet", "resnet18"):
        raise ValueError(f"Unsupported model {model!r}; use 'lenet' or 'resnet18'.")
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unsupported split {split!r}; use 'train', 'val', or 'test'.")

    operations: list[transforms.Transform] = []
    if split == "train" and augment:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))

    if model == "lenet":
        operations.extend(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    else:
        operations.extend(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    return transforms.Compose(operations)
