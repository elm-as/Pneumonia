"""Unit tests for neural architectures and loss criteria."""

import torch
from src.core.config import PipelineConfig
from src.modeling.architectures import PneumoniaResNet, build_classifier
from src.modeling.loss_functions import build_loss_criterion


def test_model_forward_pass() -> None:
    """Verify that model accepts batch tensors and returns correct logit dimensions."""
    model = PneumoniaResNet(pretrained=False, dropout_rate=0.2, num_classes=2)
    model.eval()

    dummy_batch = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_batch)

    assert logits.shape == (4, 2)
    assert not torch.isnan(logits).any()


def test_gradcam_target_layer_exists() -> None:
    """Verify target convolutional layer is exposed for activation mapping."""
    model = PneumoniaResNet(pretrained=False, dropout_rate=0.2, num_classes=2)
    target_layer = model.get_target_layer_for_cam()

    assert isinstance(target_layer, torch.nn.Conv2d)


def test_loss_criterion_weighting() -> None:
    """Verify weighted cross entropy handles class balancing."""
    criterion = build_loss_criterion(manifest=None, device="cpu")
    dummy_logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
    dummy_labels = torch.tensor([0, 1])

    loss = criterion(dummy_logits, dummy_labels)
    assert loss.item() > 0.0

