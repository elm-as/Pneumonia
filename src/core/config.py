"""Configuration management for the Pneumonia deep learning diagnostic pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import torch


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable global pipeline configuration.

    Adheres to ElmasCore standards: zero global state mutation, clear typed paths,
    and sensible medical imaging hyperparameters.
    """
    # Directory paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    artifacts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts")
    checkpoints_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints")
    reports_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts" / "reports")
    images_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "images")

    # Image preprocessing parameters
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Modeling and training hyperparameters
    architecture: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    dropout_rate: float = 0.3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 6
    num_workers: int = 0
    pos_weight_multiplier: float = 1.0  # Multiplier for handling class imbalance (Normal vs Pneumonia)

    # Validation strategy
    # The original Kermany et al. dataset has only 16 images in val (8 normal, 8 pneumonia).
    # Setting use_resplit=True samples 10% from train into validation for reliable metric tracking.
    use_resplit: bool = True
    val_ratio: float = 0.10
    random_seed: int = 42

    # Hardware target
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_default_config() -> PipelineConfig:
    """Instantiate standard production pipeline configuration."""
    return PipelineConfig()

