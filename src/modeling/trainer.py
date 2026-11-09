"""Training orchestrator with validation tracking and checkpointing."""

import time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.core.config import PipelineConfig
from src.core.types import TrainingEpochLog, TrainingHistory


def _compute_quick_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute validation AUC, Sensitivity, and Specificity without heavy imports."""
    from sklearn.metrics import roc_auc_score

    # Guard against single-class batch
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.0

    auc = float(roc_auc_score(y_true, y_prob))
    preds = (y_prob >= 0.5).astype(int)

    # Sensitivity = TP / (TP + FN)
    tp = np.sum((y_true == 1) & (preds == 1))
    fn = np.sum((y_true == 1) & (preds == 0))
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    tn = np.sum((y_true == 0) & (preds == 0))
    fp = np.sum((y_true == 0) & (preds == 1))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return auc, sensitivity, specificity


class PneumoniaTrainer:
    """Manages the full lifecycle of neural network training and validation."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        config: PipelineConfig,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = config.device

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6,
        )

    def train_epoch(self, loader: DataLoader) -> float:
        """Execute one complete training epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        self.scheduler.step()
        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, float]:
        """Compute validation loss, ROC-AUC, sensitivity, and specificity."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        all_labels: List[int] = []
        all_probs: List[float] = []

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        avg_loss = total_loss / max(n_samples, 1)
        auc, sens, spec = _compute_quick_metrics(np.array(all_labels), np.array(all_probs))
        return avg_loss, auc, sens, spec

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingHistory:
        """Train model across epochs with best checkpoint preservation."""
        self.config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = self.config.checkpoints_dir / "best_model.pth"

        best_auc = -1.0
        best_epoch = 1
        epoch_logs: List[TrainingEpochLog] = []
        start_time = time.time()

        for epoch in range(1, self.config.num_epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss, v_auc, v_sens, v_spec = self.evaluate(val_loader)
            current_lr = self.scheduler.get_last_lr()[0]

            log = TrainingEpochLog(
                epoch=epoch,
                train_loss=t_loss,
                val_loss=v_loss,
                val_auc=v_auc,
                val_sensitivity=v_sens,
                val_specificity=v_spec,
                learning_rate=current_lr,
            )
            epoch_logs.append(log)

            print(
                f"Epoch [{epoch:02d}/{self.config.num_epochs:02d}] "
                f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
                f"Val AUC: {v_auc:.4f} | Sens: {v_sens:.3f} | Spec: {v_spec:.3f}"
            )

            if v_auc > best_auc:
                best_auc = v_auc
                best_epoch = epoch
                torch.save(self.model.state_dict(), best_checkpoint_path)

        # Restore weights from best performing epoch
        if best_checkpoint_path.exists():
            self.model.load_state_dict(torch.load(best_checkpoint_path, map_location=self.device))

        duration = time.time() - start_time
        return TrainingHistory(
            epochs=tuple(epoch_logs),
            best_epoch=best_epoch,
            best_val_auc=best_auc,
            total_duration_seconds=duration,
        )

