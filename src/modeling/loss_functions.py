"""Loss functions for class-imbalanced medical image classification."""

from typing import Optional
import torch
import torch.nn as nn

from src.core.types import DatasetManifest


def build_loss_criterion(
    manifest: Optional[DatasetManifest] = None,
    device: str = "cpu",
    pos_weight_multiplier: float = 1.0,
) -> nn.CrossEntropyLoss:
    """Construct an inverse-frequency weighted CrossEntropyLoss.

    Why: The training cohort exhibits a notable class imbalance (~2.9:1 ratio of Pneumonia
    to Normal). Inverse class weighting prevents the model from under-penalizing the minority
    class while maintaining balanced gradient updates during backpropagation.

    Args:
        manifest: Pre-computed dataset statistics containing class counts.
        device: Target hardware device ("cpu" or "cuda").
        pos_weight_multiplier: Fine-tuning multiplier for clinical sensitivity tradeoff.

    Returns:
        Configured PyTorch CrossEntropyLoss.
    """
    if manifest is None or manifest.normal_count == 0 or manifest.pneumonia_count == 0:
        return nn.CrossEntropyLoss().to(device)

    total_samples = manifest.normal_count + manifest.pneumonia_count
    # Compute inverse frequency weights
    weight_normal = total_samples / (2.0 * manifest.normal_count)
    weight_pneumonia = (total_samples / (2.0 * manifest.pneumonia_count)) * pos_weight_multiplier

    weights = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32).to(device)
    return nn.CrossEntropyLoss(weight=weights)

