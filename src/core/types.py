"""Core type definitions and domain entities for the Pneumonia classification system."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class DiagnosisClass(int, Enum):
    """Binary clinical diagnosis classification."""
    NORMAL = 0
    PNEUMONIA = 1


class EtiologyType(str, Enum):
    """Fine-grained pathological etiology of the chest radiograph."""
    NORMAL = "NORMAL"
    BACTERIA = "BACTERIA"
    VIRUS = "VIRUS"
    UNKNOWN = "UNKNOWN"


class SplitName(str, Enum):
    """Dataset partition splits."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class ImageRecord:
    """Individual chest radiograph metadata entry.

    Attributes:
        path: Absolute path to the image file.
        label: Binary diagnosis class (0 for normal, 1 for pneumonia).
        etiology: Detailed pathology source (Normal, Bacteria, Virus).
        split: Assigned partition split.
        filename: Base name of the image.
    """
    path: Path
    label: DiagnosisClass
    etiology: EtiologyType
    split: SplitName
    filename: str


@dataclass(frozen=True)
class DatasetManifest:
    """Complete collection of parsed radiograph records with summary statistics."""
    records: Tuple[ImageRecord, ...]
    train_count: int
    val_count: int
    test_count: int
    normal_count: int
    pneumonia_count: int
    bacteria_count: int
    virus_count: int


@dataclass(frozen=True)
class ClinicalEvaluationResult:
    """Comprehensive clinical diagnostic evaluation metrics.

    Focuses on sensitivity (recall) to minimize critical false-negative pneumonia cases,
    along with specificity and threshold calibration via Youden's J statistic.
    """
    accuracy: float
    sensitivity: float  # Recall on Pneumonia class (crucial in clinical practice)
    specificity: float  # True Negative Rate on Normal class
    precision: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    optimal_threshold: float
    confusion_matrix: Tuple[Tuple[int, int], Tuple[int, int]]
    n_samples: int


@dataclass(frozen=True)
class TrainingEpochLog:
    """Summary metrics of a single training epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    val_sensitivity: float
    val_specificity: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingHistory:
    """Training progression history over multiple epochs."""
    epochs: Tuple[TrainingEpochLog, ...]
    best_epoch: int
    best_val_auc: float
    total_duration_seconds: float


@dataclass(frozen=True)
class InferencePrediction:
    """Individual inference result with class probabilities and saliency metadata."""
    image_path: Path
    predicted_label: DiagnosisClass
    pneumonia_probability: float
    true_label: Optional[DiagnosisClass]
    etiology: Optional[EtiologyType]

