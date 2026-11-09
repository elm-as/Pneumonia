"""Dataset manifest generation and stratification for chest radiographs."""

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.core.types import DatasetManifest, DiagnosisClass, EtiologyType, ImageRecord, SplitName


def _parse_etiology(filename: str, label: DiagnosisClass) -> EtiologyType:
    """Determine fine-grained etiology from filename and diagnosis class.

    Why: Pediatric pneumonia manifests differently radiologically depending on whether
    it is bacterial (focal lobar consolidation) or viral (diffuse interstitial patterns).
    Filenames in Kermany et al. contain explicit 'bacteria' or 'virus' tags.
    """
    if label == DiagnosisClass.NORMAL:
        return EtiologyType.NORMAL

    name_lower = filename.lower()
    if "bacteria" in name_lower:
        return EtiologyType.BACTERIA
    if "virus" in name_lower:
        return EtiologyType.VIRUS

    return EtiologyType.UNKNOWN


def _scan_split_folder(data_dir: Path, split_name: SplitName) -> List[ImageRecord]:
    """Scan a single split folder for Normal and Pneumonia images."""
    split_dir = data_dir / split_name.value
    if not split_dir.exists():
        return []

    records: List[ImageRecord] = []
    classes = [
        ("NORMAL", DiagnosisClass.NORMAL),
        ("PNEUMONIA", DiagnosisClass.PNEUMONIA),
    ]

    for class_folder, label in classes:
        target_dir = split_dir / class_folder
        if not target_dir.exists():
            continue

        for entry in os.scandir(target_dir):
            if not entry.is_file():
                continue
            if not entry.name.lower().endswith((".jpeg", ".jpg", ".png")):
                continue

            etiology = _parse_etiology(entry.name, label)
            records.append(
                ImageRecord(
                    path=Path(entry.path),
                    label=label,
                    etiology=etiology,
                    split=split_name,
                    filename=entry.name,
                )
            )

    return records


def _stratified_repartition(
    train_records: List[ImageRecord],
    val_records: List[ImageRecord],
    val_ratio: float,
    random_seed: int,
) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Create a statistically sound stratified train/val split from combined records.

    Why: The original Kermany et al. benchmark contains only 16 images in val (8 normal,
    8 pneumonia), which is too small for statistical significance. We combine raw train
    and val, then stratify by etiology (Normal, Bacterial, Viral) to produce a reliable
    validation set while preserving the 624-sample holdout test partition untouched.
    """
    combined = train_records + val_records
    if not combined or val_ratio <= 0.0 or val_ratio >= 1.0:
        return train_records, val_records

    rng = np.random.default_rng(random_seed)
    # Group by etiology for balanced representation
    grouped: dict[EtiologyType, List[ImageRecord]] = {}
    for rec in combined:
        grouped.setdefault(rec.etiology, []).append(rec)

    new_train: List[ImageRecord] = []
    new_val: List[ImageRecord] = []

    for _, group in grouped.items():
        indices = np.arange(len(group))
        rng.shuffle(indices)
        split_idx = int(round(len(group) * val_ratio))
        val_idx = set(indices[:split_idx])

        for i, rec in enumerate(group):
            assigned_split = SplitName.VAL if i in val_idx else SplitName.TRAIN
            reassigned = ImageRecord(
                path=rec.path,
                label=rec.label,
                etiology=rec.etiology,
                split=assigned_split,
                filename=rec.filename,
            )
            if assigned_split == SplitName.VAL:
                new_val.append(reassigned)
            else:
                new_train.append(reassigned)

    return new_train, new_val


def build_dataset_manifest(
    data_dir: Path,
    use_resplit: bool = True,
    val_ratio: float = 0.10,
    random_seed: int = 42,
) -> DatasetManifest:
    """Assemble the comprehensive DatasetManifest from raw folder structure.

    Args:
        data_dir: Root dataset folder containing train/, val/, test/ subdirectories.
        use_resplit: Whether to re-stratify the 16-sample val anomaly into a reliable 10% split.
        val_ratio: Fraction of training records to allocate for validation.
        random_seed: Deterministic random seed for reproducible stratification.

    Returns:
        Immutable DatasetManifest with verified record tuples and summary counts.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {data_dir}")

    raw_train = _scan_split_folder(data_dir, SplitName.TRAIN)
    raw_val = _scan_split_folder(data_dir, SplitName.VAL)
    raw_test = _scan_split_folder(data_dir, SplitName.TEST)

    if not raw_train and not raw_test:
        raise ValueError(f"No valid radiographic images found in {data_dir}")

    if use_resplit:
        train_records, val_records = _stratified_repartition(
            raw_train, raw_val, val_ratio, random_seed
        )
    else:
        train_records, val_records = raw_train, raw_val

    all_records = tuple(train_records + val_records + raw_test)

    normal_count = sum(1 for r in all_records if r.label == DiagnosisClass.NORMAL)
    pneumonia_count = sum(1 for r in all_records if r.label == DiagnosisClass.PNEUMONIA)
    bacteria_count = sum(1 for r in all_records if r.etiology == EtiologyType.BACTERIA)
    virus_count = sum(1 for r in all_records if r.etiology == EtiologyType.VIRUS)

    return DatasetManifest(
        records=all_records,
        train_count=len(train_records),
        val_count=len(val_records),
        test_count=len(raw_test),
        normal_count=normal_count,
        pneumonia_count=pneumonia_count,
        bacteria_count=bacteria_count,
        virus_count=virus_count,
    )

