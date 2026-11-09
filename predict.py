"""Standalone clinical inference and Grad-CAM explainability CLI.

Allows clinicians and researchers to screen a single anterior-posterior
chest radiograph, generate probability calibration, and export Grad-CAM saliency maps.

Adheres strictly to ElmasCore standards:
- File under 200 lines
- Guard clauses and early returns
- Typed dataclasses and clear error handling
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

from src.core.config import PipelineConfig, get_default_config
from src.core.types import DiagnosisClass, InferencePrediction
from src.data_pipeline.image_transforms import build_eval_transforms
from src.interpretability.grad_cam import GradCAM
from src.interpretability.saliency_overlay import overlay_cam_on_image
from src.modeling.architectures import build_classifier


def load_inference_model(
    checkpoint_path: Path,
    config: PipelineConfig,
) -> torch.nn.Module:
    """Load pre-trained weights into the classification model."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    model = build_classifier(config)
    state_dict = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_single_radiograph(
    image_path: Path,
    model: torch.nn.Module,
    config: PipelineConfig,
) -> Tuple[InferencePrediction, Optional[np.ndarray]]:
    """Execute forward inference and Grad-CAM activation on a single image.

    Args:
        image_path: Absolute or relative path to radiograph file.
        model: Loaded PyTorch classifier.
        config: Pipeline configuration object.

    Returns:
        Tuple of (InferencePrediction dataclass, 2D Grad-CAM heatmap array).
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Radiograph image not found: {image_path}")

    with Image.open(image_path) as raw_img:
        rgb_img = raw_img.convert("RGB")
        eval_tf = build_eval_transforms(config.image_size, config.mean, config.std)
        input_tensor = eval_tf(rgb_img).unsqueeze(0).to(config.device)

    # Compute prediction probability
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pneumonia_prob = float(probs[0, 1].item())

    predicted_label = (
        DiagnosisClass.PNEUMONIA
        if pneumonia_prob >= 0.5
        else DiagnosisClass.NORMAL
    )

    # Compute Grad-CAM heatmap for Pneumonia class
    target_layer = model.get_target_layer_for_cam()
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_cam(input_tensor, target_class=1)
    grad_cam.release()

    prediction = InferencePrediction(
        image_path=image_path,
        predicted_label=predicted_label,
        pneumonia_probability=pneumonia_prob,
        true_label=None,
        etiology=None,
    )

    return prediction, heatmap


def export_cam_visualization(
    image_path: Path,
    heatmap: np.ndarray,
    output_path: Path,
) -> None:
    """Blend Grad-CAM activation over original radiograph and save to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        overlay = overlay_cam_on_image(img, heatmap, alpha=0.45)
        bgr_overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr_overlay)


def parse_cli_args() -> argparse.Namespace:
    """Parse inference command line flags."""
    parser = argparse.ArgumentParser(description="Pneumonia Radiograph Inference Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to chest radiograph image")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best_model.pth")
    parser.add_argument("--save-cam", type=str, default="artifacts/reports/prediction_cam.png")
    return parser.parse_args()


def main() -> None:
    """Inference CLI entrypoint."""
    args = parse_cli_args()
    config = get_default_config()

    print("=" * 64)
    print(" PNEUMONIA INFERENCE ENGINE — CLINICAL RADIOGRAPH SCREENING")
    print("=" * 64)

    image_path = Path(args.image).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    save_cam_path = Path(args.save_cam).resolve()

    print(f"[+] Loading model from: {checkpoint_path}")
    model = load_inference_model(checkpoint_path, config)

    print(f"[+] Screening radiograph: {image_path.name}")
    prediction, heatmap = predict_single_radiograph(image_path, model, config)

    conf_pct = prediction.pneumonia_probability * 100.0
    diagnosis_name = prediction.predicted_label.name

    print("-" * 64)
    print(f" DIAGNOSIS PREDICTION : {diagnosis_name}")
    print(f" PNEUMONIA PROBABILITY: {conf_pct:.2f}%")

    if prediction.predicted_label == DiagnosisClass.PNEUMONIA:
        print(" CLINICAL TRIAGE      : [!] URGENT: Evidence of pulmonary opacification.")
        print("                        Inspect Grad-CAM overlay for focal vs diffuse pattern.")
    else:
        print(" CLINICAL TRIAGE      : [✓] CLEAR: No dominant consolidation detected.")

    if heatmap is not None:
        export_cam_visualization(image_path, heatmap, save_cam_path)
        print(f"[+] Grad-CAM Saliency Overlay saved to: {save_cam_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()

