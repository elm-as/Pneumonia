"""Clinical diagnostic metrics evaluation with Youden's J optimization."""

from typing import Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)

from src.core.types import ClinicalEvaluationResult


def find_optimal_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Identify optimal classification threshold maximizing Youden's J statistic.

    Why: J = Sensitivity + Specificity - 1. In clinical settings, a standard 0.5 threshold
    is often suboptimal under class imbalance. Youden's index objectively balances false
    negatives (missed pneumonia) and false positives (unnecessary hospitalization).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # J = TPR - FPR
    youden_indices = tpr - fpr
    best_idx = int(np.argmax(youden_indices))
    best_thresh = float(thresholds[best_idx])
    # Bound between 0.05 and 0.95 to avoid degenerate thresholds
    return float(np.clip(best_thresh, 0.05, 0.95))


def evaluate_clinical_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> ClinicalEvaluationResult:
    """Compute publication-grade diagnostic metrics on predicted probabilities.

    Args:
        y_true: Ground truth binary labels (0 for Normal, 1 for Pneumonia).
        y_prob: Predicted probabilities for the Pneumonia class.
        threshold: Decision threshold for discrete classification.

    Returns:
        ClinicalEvaluationResult dataclass with sensitivity, specificity, and ROC-AUC.
    """
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty ground truth array.")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (Recall on positive class)
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    # Specificity (True negative rate)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true_arr)) > 1:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        p_curve, r_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        # Numerical trapezoidal PR-AUC integration
        pr_auc = float(-np.trapz(p_curve, r_curve))
        optimal_thresh = find_optimal_youden_threshold(y_true_arr, y_prob_arr)
    else:
        roc_auc = 0.5
        pr_auc = 0.5
        optimal_thresh = 0.5

    return ClinicalEvaluationResult(
        accuracy=acc,
        sensitivity=sens,
        specificity=spec,
        precision=prec,
        f1_score=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        optimal_threshold=optimal_thresh,
        confusion_matrix=((int(tn), int(fp)), (int(fn), int(tp))),
        n_samples=len(y_true_arr),
    )

