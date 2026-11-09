"""Pathological etiology analysis and clinical error categorization."""

from typing import Dict, List, Sequence
import numpy as np

from src.core.types import ClinicalEvaluationResult, DiagnosisClass, EtiologyType, ImageRecord


def analyze_etiology_breakdown(
    records: Sequence[ImageRecord],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Compute diagnostic accuracy broken down by biological etiology.

    Why: In Kermany et al., bacterial pneumonia typically presents with focal lobar
    consolidation, whereas viral pneumonia presents with diffuse interstitial patterns.
    Clinicians need to know if the model is disproportionately missing viral vs bacterial cases.
    """
    results: Dict[str, Dict[str, float]] = {}
    etiology_groups: Dict[EtiologyType, List[Tuple[int, float]]] = {}

    for record, prob in zip(records, probabilities):
        label_val = int(record.label.value)
        etiology_groups.setdefault(record.etiology, []).append((label_val, float(prob)))

    for etiology, items in etiology_groups.items():
        if not items:
            continue
        y_true = np.array([x[0] for x in items])
        y_prob = np.array([x[1] for x in items])
        y_pred = (y_prob >= threshold).astype(int)

        total = len(y_true)
        correct = int(np.sum(y_true == y_pred))
        accuracy = float(correct / total) if total > 0 else 0.0

        results[etiology.value] = {
            "total": float(total),
            "correct": float(correct),
            "accuracy": accuracy,
            "mean_confidence": float(np.mean(y_prob)),
        }

    return results


def format_clinical_audit_table(
    eval_result: ClinicalEvaluationResult,
    etiology_stats: Dict[str, Dict[str, float]],
) -> str:
    """Format comprehensive clinical validation summary table for reporting."""
    cm = eval_result.confusion_matrix
    tn, fp = cm[0]
    fn, tp = cm[1]

    lines = [
        "=" * 68,
        " CLINICAL RADIOGRAPHIC DIAGNOSTIC AUDIT (TEST COHORT)",
        "=" * 68,
        f" Total Test Cohort Patients : {eval_result.n_samples}",
        f" Diagnostic Accuracy        : {eval_result.accuracy * 100:.2f}%",
        f" Sensitivity (Recall)       : {eval_result.sensitivity * 100:.2f}% (Pneumonia Detection)",
        f" Specificity (TNR)          : {eval_result.specificity * 100:.2f}% (Healthy Cleared)",
        f" Precision (PPV)            : {eval_result.precision * 100:.2f}%",
        f" F1-Score                   : {eval_result.f1_score:.4f}",
        f" ROC-AUC Score              : {eval_result.roc_auc:.4f}",
        f" PR-AUC Score               : {eval_result.pr_auc:.4f}",
        f" Brier Calibration Score    : {eval_result.brier_score:.4f}",
        f" Optimal Youden Threshold   : {eval_result.optimal_threshold:.4f}",
        "-" * 68,
        " CONFUSION MATRIX BREAKDOWN:",
        f"   - True Negatives  (Normal -> Normal)       : {tn}",
        f"   - False Positives (Normal -> Pneumonia)    : {fp} (False Alarm)",
        f"   - False Negatives (Pneumonia -> Normal)    : {fn} (Critical Miss)",
        f"   - True Positives  (Pneumonia -> Pneumonia) : {tp}",
        "-" * 68,
        " PATHOLOGICAL ETIOLOGY RESOLUTION:",
    ]

    for key, stats in etiology_stats.items():
        lines.append(
            f"   - {key:<10}: {int(stats['correct'])}/{int(stats['total'])} correct "
            f"({stats['accuracy'] * 100:.1f}%) | Avg Conf: {stats['mean_confidence']:.3f}"
        )

    lines.append("=" * 68)
    return "\n".join(lines)

