"""Publication-grade visualization utilities for clinical model reporting."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless CLI / server execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.core.types import ClinicalEvaluationResult, DatasetManifest, TrainingHistory


def plot_dataset_distribution(manifest: DatasetManifest, save_path: Path) -> None:
    """Generate demographic & etiology distribution chart of the radiograph cohort."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Split breakdown
    split_names = ["Train", "Validation", "Test Holdout"]
    split_counts = [manifest.train_count, manifest.val_count, manifest.test_count]
    colors = ["#2563EB", "#10B981", "#F59E0B"]

    bars1 = ax1.bar(split_names, split_counts, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_title("Dataset Cohort Partitioning", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Radiographs Count", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 50, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")

    # Pathological Etiology breakdown
    etiology_names = ["Healthy Normal", "Bacterial Pneumonia", "Viral Pneumonia"]
    etiology_counts = [manifest.normal_count, manifest.bacteria_count, manifest.virus_count]
    et_colors = ["#059669", "#DC2626", "#7C3AED"]

    bars2 = ax2.bar(etiology_names, etiology_counts, color=et_colors, edgecolor="black", alpha=0.85)
    ax2.set_title("Pathological Etiology Breakdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Total Cases", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 50, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eval_result: ClinicalEvaluationResult,
    save_path: Path,
) -> None:
    """Export publication-grade ROC and Precision-Recall diagnostic curves."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax1.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ResNet-18 (ROC-AUC = {eval_result.roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax2.plot(recall, precision, color="#10B981", lw=2.5, label=f"ResNet-18 (PR-AUC = {eval_result.pr_auc:.4f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(eval_result: ClinicalEvaluationResult, save_path: Path) -> None:
    """Generate annotated confusion matrix highlighting clinical error severity."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.array(eval_result.confusion_matrix)

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = ["Normal (Healthy)", "Pneumonia"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_title("Clinical Test Cohort Confusion Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Pathology", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_history(history: TrainingHistory, save_path: Path) -> None:
    """Plot cross-epoch loss convergence and validation AUC trajectory."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [e.epoch for e in history.epochs]
    t_losses = [e.train_loss for e in history.epochs]
    v_losses = [e.val_loss for e in history.epochs]
    v_aucs = [e.val_auc for e in history.epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss convergence
    ax1.plot(epochs, t_losses, "o-", color="#2563EB", label="Train Loss", lw=2)
    ax1.plot(epochs, v_losses, "s-", color="#DC2626", label="Validation Loss", lw=2)
    ax1.set_title("Training & Validation Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Weighted Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    # Validation AUC
    ax2.plot(epochs, v_aucs, "d-", color="#10B981", label="Validation ROC-AUC", lw=2)
    ax2.axvline(history.best_epoch, color="#F59E0B", linestyle="--", label=f"Best Model (Epoch {history.best_epoch})")
    ax2.set_title("Validation ROC-AUC Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("ROC-AUC Score", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

