"""PyTorch Dataset and DataLoader constructors for chest radiographs."""

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.config import PipelineConfig
from src.core.types import DatasetManifest, DiagnosisClass, ImageRecord, SplitName
from src.data_pipeline.image_transforms import build_eval_transforms, build_train_transforms


class ChestXRayDataset(Dataset):
    """PyTorch Dataset representing clinical chest radiographs.

    Loads images on-demand using PIL to keep memory footprint bounded.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Optional[Callable] = None,
    ) -> None:
        self._records = list(records)
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        record = self._records[idx]
        image_path = record.path

        try:
            with Image.open(image_path) as img:
                # Convert to RGB to ensure consistent 3-channel layout
                rgb_image = img.convert("RGB")
                tensor = self._transform(rgb_image) if self._transform else rgb_image
        except Exception as err:
            raise IOError(f"Failed to read radiograph at {image_path}: {err}") from err

        label_val = int(record.label.value)
        return tensor, label_val, str(image_path)

    @property
    def records(self) -> Sequence[ImageRecord]:
        """Access underlying record metadata."""
        return self._records


def create_split_loaders(
    manifest: DatasetManifest,
    config: PipelineConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiate train, validation, and test PyTorch DataLoaders.

    Args:
        manifest: Pre-built DatasetManifest containing partitioned records.
        config: Pipeline configuration object specifying batch size and workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_records = [r for r in manifest.records if r.split == SplitName.TRAIN]
    val_records = [r for r in manifest.records if r.split == SplitName.VAL]
    test_records = [r for r in manifest.records if r.split == SplitName.TEST]

    train_tf = build_train_transforms(config.image_size, config.mean, config.std)
    eval_tf = build_eval_transforms(config.image_size, config.mean, config.std)

    train_ds = ChestXRayDataset(train_records, transform=train_tf)
    val_ds = ChestXRayDataset(val_records, transform=eval_tf)
    test_ds = ChestXRayDataset(test_records, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader

