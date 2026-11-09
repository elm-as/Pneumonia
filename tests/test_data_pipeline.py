"""Unit tests for dataset manifest and transform pipelines."""

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from src.core.types import DiagnosisClass, EtiologyType, ImageRecord, SplitName
from src.data_pipeline.dataset_manifest import _parse_etiology, build_dataset_manifest
from src.data_pipeline.image_transforms import build_eval_transforms, build_train_transforms


def test_parse_etiology_normal() -> None:
    """Ensure normal diagnosis assigns NORMAL etiology."""
    etiology = _parse_etiology("IM-0115-0001.jpeg", DiagnosisClass.NORMAL)
    assert etiology == EtiologyType.NORMAL


def test_parse_etiology_pneumonia_subtypes() -> None:
    """Ensure bacterial and viral patterns are accurately extracted."""
    bact = _parse_etiology("person1266_bacteria_3224.jpeg", DiagnosisClass.PNEUMONIA)
    assert bact == EtiologyType.BACTERIA

    vir = _parse_etiology("person1266_virus_2158.jpeg", DiagnosisClass.PNEUMONIA)
    assert vir == EtiologyType.VIRUS


def test_image_transforms_shapes() -> None:
    """Ensure transforms output expected (3, 224, 224) standardized tensors."""
    dummy_img = Image.fromarray(np.uint8(np.random.rand(300, 400) * 255), mode="L")
    transform = build_eval_transforms(image_size=224)
    tensor = transform(dummy_img.convert("RGB"))

    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_manifest_builder_with_real_data() -> None:
    """Verify that dataset manifest loads real radiograph files properly."""
    data_dir = Path(__file__).resolve().parents[1] / "data"
    if not data_dir.exists():
        pytest.skip("Data directory not present on test runner.")

    manifest = build_dataset_manifest(data_dir=data_dir, use_resplit=True, val_ratio=0.10)
    assert manifest.train_count > 0
    assert manifest.val_count > 0
    assert manifest.test_count == 624
    assert manifest.normal_count > 0
    assert manifest.bacteria_count > 0
    assert manifest.virus_count > 0

