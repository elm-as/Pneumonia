"""Visualization utilities for clinical diagnostic reporting."""

from src.visualization.clinical_plots import (
    plot_confusion_matrix,
    plot_dataset_distribution,
    plot_roc_and_pr_curves,
    plot_training_history,
)
from src.visualization.gradcam_gallery import generate_gradcam_comparative_gallery

__all__ = [
    "plot_confusion_matrix",
    "plot_dataset_distribution",
    "plot_roc_and_pr_curves",
    "plot_training_history",
    "generate_gradcam_comparative_gallery",
]

