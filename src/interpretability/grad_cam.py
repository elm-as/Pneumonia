"""Gradient-weighted Class Activation Mapping (Grad-CAM) implementation."""

from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM heatmaps for interpreting neural network predictions.

    Why: In clinical medical imaging, black-box predictions are unacceptable.
    Grad-CAM highlights the exact anatomical regions (e.g., right upper lobe consolidation
    or diffuse interstitial patterns) that led the network to diagnose pneumonia.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target convolutional layer."""
        def forward_hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            self.activations = out.detach()

        def backward_hook(module: nn.Module, grad_in: Tuple[torch.Tensor, ...], grad_out: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = 1,
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM heatmap for a single image tensor.

        Args:
            input_tensor: Image tensor of shape (1, 3, H, W).
            target_class: Target class index (1 for Pneumonia).

        Returns:
            2D numpy array with normalized heatmap values in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations during backward pass.")

        # Global average pooling of gradients: weights alpha_k
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of activation maps
        activations = self.activations[0]
        for i, weight in enumerate(pooled_gradients):
            activations[i] *= weight

        heatmap = torch.sum(activations, dim=0)
        # Apply ReLU to focus solely on positive contributions towards the class
        heatmap = F.relu(heatmap).cpu().numpy()

        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def release(self) -> None:
        """Remove hooks to avoid memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

