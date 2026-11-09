"""Grad-CAM diagnostic gallery comparing Normal, Bacterial, and Viral cases."""

from pathlib import Path
from typing import List, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.core.types import ImageRecord
from src.interpretability.grad_cam import GradCAM
from src.interpretability.saliency_overlay import overlay_cam_on_image


def generate_gradcam_comparative_gallery(
    records: Sequence[ImageRecord],
    grad_cam: GradCAM,
    eval_transform,
    device: str,
    save_path: Path,
) -> None:
    """Produce publication-grade multi-case Grad-CAM explainability panel.

    Args:
        records: Selected sample records (Normal, Bacterial, Viral).
        grad_cam: Active GradCAM instance bound to the model.
        eval_transform: Deterministic image preprocessing transform.
        device: Target execution device.
        save_path: Destination PNG file path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_cases = len(records)
    if n_cases == 0:
        return

    fig, axes = plt.subplots(n_cases, 2, figsize=(10, 4 * n_cases))
    if n_cases == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, record in enumerate(records):
        with Image.open(record.path) as raw_img:
            rgb_img = raw_img.convert("RGB")
            tensor = eval_transform(rgb_img).unsqueeze(0).to(device)

            # Generate CAM activation
            heatmap = grad_cam.generate_cam(tensor, target_class=1)
            overlay = overlay_cam_on_image(rgb_img, heatmap, alpha=0.45)

        # Left: Original radiograph
        axes[i, 0].imshow(rgb_img, cmap="gray")
        axes[i, 0].set_title(f"Original X-Ray: {record.etiology.value}\n({record.filename})", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # Right: Grad-CAM overlay
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Grad-CAM Pathological Activation\n(Target: Pneumonia)", fontsize=11, fontweight="bold")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

