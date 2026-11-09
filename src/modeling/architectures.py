"""Deep neural network architectures for pediatric chest radiography classification."""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.core.config import PipelineConfig


class PneumoniaResNet(nn.Module):
    """Transfer-learning convolutional architecture based on ResNet18.

    Features a specialized clinical classification head with dropout regularization
    and batch normalization to prevent overfitting on radiograph patterns.
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        # Replace default fc with clinical multi-layer perceptron head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating raw class logits."""
        return self.backbone(x)

    def get_target_layer_for_cam(self) -> nn.Module:
        """Expose the final residual convolutional block for Grad-CAM explainability."""
        return self.backbone.layer4[-1].conv2


def build_classifier(config: PipelineConfig) -> PneumoniaResNet:
    """Factory creating and configuring the neural network model.

    Args:
        config: Immutable PipelineConfig instance.

    Returns:
        Configured PneumoniaResNet on target device.
    """
    model = PneumoniaResNet(
        pretrained=config.pretrained,
        dropout_rate=config.dropout_rate,
        num_classes=config.num_classes,
    )
    return model.to(config.device)

