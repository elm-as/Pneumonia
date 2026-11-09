"""Main CLI entrypoint and orchestrator for the Pneumonia Diagnostic System.

Adheres strictly to ElmasCore standards:
- Explicit separation of concerns
- Guard clauses and early returns
- Comprehensive console logging with clinical tabular formatting
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.config import PipelineConfig, get_default_config
from src.core.types import (
    ClinicalEvaluationResult,
    DatasetManifest,
    DiagnosisClass,
    EtiologyType,
    ImageRecord,
    SplitName,
    TrainingHistory,
)
from src.data_pipeline import (
    build_dataset_manifest,
    build_eval_transforms,
    create_split_loaders,
)
from src.interpretability import GradCAM
from src.modeling import PneumoniaTrainer, build_classifier, build_loss_criterion
from src.validation import (
    analyze_etiology_breakdown,
    evaluate_clinical_performance,
    format_clinical_audit_table,
)
from src.visualization import (
    generate_gradcam_comparative_gallery,
    plot_confusion_matrix,
    plot_dataset_distribution,
    plot_roc_and_pr_curves,
    plot_training_history,
)


def _display_banner() -> None:
    """Print project header banner."""
    print("=" * 76)
    print(" PNEUMONIA: Pediatric Chest Radiography Deep Diagnostic System")
    print(" ElmasCore Production Architecture | Cell 2018 Kermany Benchmark")
    print("=" * 76)


def _log_dataset_statistics(manifest: DatasetManifest) -> None:
    """Log dataset distribution to stdout."""
    print("\n[1/5] Dataset Partitioning & Cohort Summary:")
    print(f"    - Total Radiographs     : {len(manifest.records)}")
    print(f"    - Training Partition    : {manifest.train_count} images")
    print(f"    - Validation Partition  : {manifest.val_count} images (Stratified)")
    print(f"    - Test Holdout Partition: {manifest.test_count} images (Cell 2018 Test)")
    print(f"    - Healthy Normal Cases  : {manifest.normal_count}")
    print(f"    - Pneumonia Cases       : {manifest.pneumonia_count} "
          f"(Bacteria: {manifest.bacteria_count}, Virus: {manifest.virus_count})")


def _collect_test_predictions(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run deterministic inference over holdout test loader."""
    model.eval()
    all_targets: list[int] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_targets.extend(labels.cpu().tolist())
            all_probabilities.extend(probs.cpu().tolist())

    return np.array(all_targets), np.array(all_probabilities)


def _select_sample_records_for_cam(manifest: DatasetManifest) -> Sequence[ImageRecord]:
    """Pick representative Normal, Bacterial, and Viral test cases for Grad-CAM."""
    test_records = [r for r in manifest.records if r.split == SplitName.TEST]
    selected: list[ImageRecord] = []

    # Normal sample
    normals = [r for r in test_records if r.etiology == EtiologyType.NORMAL]
    if normals:
        selected.append(normals[0])

    # Bacterial sample
    bacterias = [r for r in test_records if r.etiology == EtiologyType.BACTERIA]
    if bacterias:
        selected.append(bacterias[0])

    # Viral sample
    viruses = [r for r in test_records if r.etiology == EtiologyType.VIRUS]
    if viruses:
        selected.append(viruses[0])

    return selected


def run_pipeline(
    config: PipelineConfig,
    skip_train: bool = False,
    skip_cam: bool = False,
) -> ClinicalEvaluationResult:
    """Execute end-to-end medical deep learning diagnostic workflow."""
    _display_banner()

    # Step 1: Assemble Dataset Manifest
    manifest = build_dataset_manifest(
        data_dir=config.data_dir,
        use_resplit=config.use_resplit,
        val_ratio=config.val_ratio,
        random_seed=config.random_seed,
    )
    _log_dataset_statistics(manifest)

    # Step 2: Build PyTorch DataLoaders
    print("\n[2/5] Building PyTorch DataLoaders & Augmentation Pipelines...")
    train_loader, val_loader, test_loader = create_split_loaders(manifest, config)
    print(f"    - Train Batches: {len(train_loader)} (Batch Size: {config.batch_size})")
    print(f"    - Val Batches  : {len(val_loader)}")
    print(f"    - Test Batches : {len(test_loader)}")

    # Step 3: Model & Criterion Setup
    print(f"\n[3/5] Instantiating Deep Architecture ({config.architecture})...")
    model = build_classifier(config)
    criterion = build_loss_criterion(manifest, config.device, config.pos_weight_multiplier)
    trainer = PneumoniaTrainer(model, criterion, config)

    checkpoint_path = config.checkpoints_dir / "best_model.pth"
    history: TrainingHistory | None = None

    if skip_train and checkpoint_path.exists():
        print(f"    - Skipping training: Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    else:
        print(f"    - Commencing training for {config.num_epochs} epochs on {config.device}...")
        history = trainer.fit(train_loader, val_loader)
        print(f"    - Training completed in {history.total_duration_seconds:.1f}s. "
              f"Best Val AUC: {history.best_val_auc:.4f} at epoch {history.best_epoch}.")

    # Step 4: Evaluate Test Cohort
    print("\n[4/5] Evaluating on Independent Test Holdout (624 patients)...")
    y_true, y_prob = _collect_test_predictions(model, test_loader, config.device)
    eval_result = evaluate_clinical_performance(y_true, y_prob, threshold=0.5)

    test_records = [r for r in manifest.records if r.split == SplitName.TEST]
    etiology_stats = analyze_etiology_breakdown(test_records, y_prob.tolist(), threshold=0.5)

    audit_table = format_clinical_audit_table(eval_result, etiology_stats)
    print("\n" + audit_table)

    # Save audit report to artifacts
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = config.reports_dir / "clinical_audit_report.txt"
    report_file.write_text(audit_table, encoding="utf-8")
    print(f"\n[+] Clinical audit saved to {report_file}")

    # Step 5: Export Publication Visualizations & Grad-CAM Explainability
    print("\n[5/5] Exporting Diagnostic Charts & Grad-CAM Saliency Maps...")
    config.images_dir.mkdir(parents=True, exist_ok=True)

    plot_dataset_distribution(manifest, config.images_dir / "dataset_cohort_distribution.png")
    plot_roc_and_pr_curves(y_true, y_prob, eval_result, config.images_dir / "diagnostic_roc_pr_curves.png")
    plot_confusion_matrix(eval_result, config.images_dir / "clinical_confusion_matrix.png")

    if history is not None:
        plot_training_history(history, config.images_dir / "training_validation_history.png")

    if not skip_cam:
        cam_samples = _select_sample_records_for_cam(manifest)
        target_layer = model.get_target_layer_for_cam()
        grad_cam = GradCAM(model, target_layer)
        eval_tf = build_eval_transforms(config.image_size, config.mean, config.std)

        generate_gradcam_comparative_gallery(
            records=cam_samples,
            grad_cam=grad_cam,
            eval_transform=eval_tf,
            device=config.device,
            save_path=config.images_dir / "gradcam_interpretability_panel.png",
        )
        grad_cam.release()
        print(f"    - Grad-CAM visual gallery exported to {config.images_dir / 'gradcam_interpretability_panel.png'}")

    print("\n[✓] Diagnostic pipeline execution finished successfully.")
    return eval_result


def parse_cli_args() -> argparse.Namespace:
    """Parse command line flags."""
    parser = argparse.ArgumentParser(description="Pneumonia Radiography Diagnostic System")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for loaders")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and use existing checkpoint")
    parser.add_argument("--skip-cam", action="store_true", help="Skip Grad-CAM visualization generation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    cfg = PipelineConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    run_pipeline(cfg, skip_train=args.skip_train, skip_cam=args.skip_cam)

