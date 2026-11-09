"""Unit tests for diagnostic evaluation metrics and Youden thresholding."""

import numpy as np
import pytest
from src.validation.clinical_metrics import (
    evaluate_clinical_performance,
    find_optimal_youden_threshold,
)


def test_youden_threshold_optimization() -> None:
    """Ensure Youden index identifies optimal cutoff balancing sensitivity/specificity."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    best_thresh = find_optimal_youden_threshold(y_true, y_prob)
    assert 0.4 <= best_thresh <= 0.6


def test_perfect_classification_metrics() -> None:
    """Verify metrics under perfect synthetic predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.9, 0.95])

    eval_result = evaluate_clinical_performance(y_true, y_prob, threshold=0.5)

    assert eval_result.accuracy == 1.0
    assert eval_result.sensitivity == 1.0
    assert eval_result.specificity == 1.0
    assert eval_result.roc_auc == 1.0
    assert eval_result.confusion_matrix == ((2, 0), (0, 2))


def test_empty_array_guard_clause() -> None:
    """Verify that empty inputs raise ValueError immediately."""
    with pytest.raises(ValueError):
        evaluate_clinical_performance(np.array([]), np.array([]))

