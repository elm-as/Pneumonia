"""Medical image transforms and augmentation pipelines for chest radiographs."""

from typing import Tuple
from torchvision import transforms


def build_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Construct data augmentation pipeline for training radiographs.

    Why: Pediatric chest radiographs feature subtle morphological variations.
    Mild affine rotations (+/- 8 degrees), slight zoom, and controlled contrast adjustments
    prevent overfitting while preserving critical lung field opacities.
    Vertical flips are avoided because thoracic anatomy has strict vertical orientation.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t[:3]),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_eval_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Construct deterministic transform pipeline for validation and test inference.

    Deterministic resizing, grayscale channel replication, and ImageNet standardization.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.repeat(3, 1, 1) if t.shape[0] == 1 else t[:3]),
        transforms.Normalize(mean=mean, std=std),
    ])

