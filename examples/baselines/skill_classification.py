#!/usr/bin/env python3
"""
Skill Tier Classification Baseline
===================================

4-class classification: beginner, intermediate, advanced, professional

Model: Wav2Vec2-base encoder with classification head
Training: AdamW, cosine LR scheduling, early stopping
Metrics: accuracy, balanced accuracy, per-class F1, confusion matrix

Usage:
    # Quick test run
    python -m examples.baselines.skill_classification --preset fast_debug --data-dir output/dataset

    # Full baseline training
    python -m examples.baselines.skill_classification --preset baseline --data-dir output/dataset

    # Custom configuration
    python -m examples.baselines.skill_classification \\
        --data-dir output/dataset \\
        --epochs 100 \\
        --lr 5e-5 \\
        --batch-size 32

Output:
    checkpoints/skill_classification/
    ├── config.json          # Training configuration
    ├── best.pt              # Best model checkpoint
    ├── last.pt              # Last epoch checkpoint
    ├── history.json         # Training history
    ├── test_results.json    # Final test metrics
    └── tensorboard/         # TensorBoard logs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.pytorch_dataloader import (
    SOUSADataset,
    create_dataloader,
    SKILL_TIER_LABELS,
    ID_TO_SKILL_TIER,
)
from examples.baselines.config import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    get_preset,
    save_config,
)
from examples.baselines.models import Wav2Vec2Classifier
from examples.baselines.training import Trainer, seed_everything


def compute_class_weights(dataset: SOUSADataset) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced skill tiers.

    The SOUSA dataset has significant class imbalance:
    - professional: ~8% (underrepresented)
    - beginner: ~26%
    - advanced: ~32%
    - intermediate: ~34%

    Using class weights is STRONGLY RECOMMENDED for skill tier classification.
    """
    from collections import Counter

    labels = [dataset[i]["label"] for i in range(len(dataset))]
    class_counts = Counter(labels)

    num_classes = len(class_counts)
    total_samples = len(labels)

    # Inverse frequency weighting
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


def compute_detailed_metrics(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int = 4,
) -> dict:
    """Compute detailed classification metrics including confusion matrix."""
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch["waveforms"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(waveforms, attention_mask)
            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = outputs["logits"].argmax(dim=-1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Basic metrics
    accuracy = (all_predictions == all_labels).mean()

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_predictions, all_labels):
        confusion[label, pred] += 1

    # Per-class metrics
    per_class_metrics = {}
    for c in range(num_classes):
        class_name = SKILL_TIER_LABELS[c]
        true_positives = confusion[c, c]
        false_positives = confusion[:, c].sum() - true_positives
        false_negatives = confusion[c, :].sum() - true_positives
        true_negatives = confusion.sum() - true_positives - false_positives - false_negatives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[c, :].sum()),
        }

    # Balanced accuracy
    class_recalls = [per_class_metrics[SKILL_TIER_LABELS[c]]["recall"] for c in range(num_classes)]
    balanced_accuracy = np.mean(class_recalls)

    # Macro F1
    class_f1s = [per_class_metrics[SKILL_TIER_LABELS[c]]["f1"] for c in range(num_classes)]
    macro_f1 = np.mean(class_f1s)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion.tolist(),
        "class_labels": SKILL_TIER_LABELS,
        "per_class": per_class_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train skill tier classification baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/dataset",
        help="Path to SOUSA dataset directory",
    )

    # Config arguments
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast_debug", "baseline", "full"],
        default="baseline",
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/skill_classification",
        help="Output directory for checkpoints and logs",
    )

    # Override arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--freeze-layers", type=int, help="Number of encoder layers to freeze")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Execution arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one batch only (for testing)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run evaluation on test set",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weights (NOT recommended due to class imbalance)",
    )

    args = parser.parse_args()

    # Load preset configuration
    config = get_preset(args.preset)

    # Apply overrides
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.freeze_layers is not None:
        config.model.freeze_encoder_layers = args.freeze_layers
    if args.seed:
        config.seed = args.seed

    # Set task-specific config
    config.task_type = "classification"
    config.loss_fn = "cross_entropy"
    config.data.target = "skill_tier"
    config.data.data_dir = args.data_dir
    config.save_dir = args.output_dir

    # Seed everything
    seed_everything(config.seed, config.deterministic)

    print("=" * 60)
    print("SOUSA Skill Tier Classification Baseline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Frozen encoder layers: {config.model.freeze_encoder_layers}")
    print("=" * 60)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = SOUSADataset(
        data_dir=args.data_dir,
        split="train",
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    val_dataset = SOUSADataset(
        data_dir=args.data_dir,
        split="validation",
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    test_dataset = SOUSADataset(
        data_dir=args.data_dir,
        split="test",
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")

    # Compute class weights (enabled by default due to class imbalance)
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_dataset)
        config.class_weights = class_weights.tolist()
        print(f"Class weights enabled: min={class_weights.min():.2f}, max={class_weights.max():.2f}")
    else:
        print("WARNING: Class weights disabled - expect biased predictions toward majority class")

    # Create dataloaders
    fixed_length = config.data.max_length_samples
    # Use num_workers=0 for dry run to avoid pickle issues with lambda
    num_workers = 0 if args.dry_run else config.data.num_workers

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        fixed_length=fixed_length,
        pin_memory=config.data.pin_memory,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        fixed_length=fixed_length,
        pin_memory=config.data.pin_memory,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        fixed_length=fixed_length,
        pin_memory=config.data.pin_memory,
    )

    # Dry run check
    if args.dry_run:
        print("\nDry run: testing one batch...")
        batch = next(iter(train_loader))
        print(f"Batch waveforms shape: {batch['waveforms'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print("Dry run successful!")
        return

    # Create model
    print("\nCreating model...")
    model = Wav2Vec2Classifier(config.model, num_classes=4)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(model, config, output_dir=args.output_dir)

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Test only mode
    if args.test_only:
        print("\nRunning test evaluation...")
        device = torch.device(config.device)

        # Load best model
        best_path = Path(args.output_dir) / "best.pt"
        if best_path.exists():
            trainer.load_checkpoint(best_path)

        detailed_metrics = compute_detailed_metrics(model, test_loader, device)

        # Print results
        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        print(f"Accuracy: {detailed_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {detailed_metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {detailed_metrics['macro_f1']:.4f}")
        print("\nPer-class metrics:")
        for class_name, metrics in detailed_metrics['per_class'].items():
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1: {metrics['f1']:.4f}")
            print(f"    Support: {metrics['support']}")

        print("\nConfusion Matrix:")
        print("Predicted →")
        print("Actual ↓")
        cm = np.array(detailed_metrics['confusion_matrix'])
        header = "         " + " ".join(f"{l[:4]:>6}" for l in SKILL_TIER_LABELS)
        print(header)
        for i, label in enumerate(SKILL_TIER_LABELS):
            row = f"{label[:8]:<8} " + " ".join(f"{cm[i, j]:6d}" for j in range(len(SKILL_TIER_LABELS)))
            print(row)

        # Save detailed results
        results_path = Path(args.output_dir) / "detailed_test_results.json"
        with open(results_path, "w") as f:
            json.dump(detailed_metrics, f, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

        return

    # Train
    print("\nStarting training...")
    history = trainer.fit(train_loader, val_loader)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    device = torch.device(config.device)

    # Load best model for final evaluation
    best_path = Path(args.output_dir) / "best.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path)

    detailed_metrics = compute_detailed_metrics(model, test_loader, device)

    print(f"Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {detailed_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {detailed_metrics['macro_f1']:.4f}")

    # Save detailed results
    results_path = Path(args.output_dir) / "detailed_test_results.json"
    with open(results_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    # Save to benchmarks directory
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)
    benchmark_path = benchmarks_dir / "skill_classification.json"
    with open(benchmark_path, "w") as f:
        json.dump({
            "task": "skill_classification",
            "model": "wav2vec2-base",
            "preset": args.preset,
            "metrics": {
                "accuracy": detailed_metrics["accuracy"],
                "balanced_accuracy": detailed_metrics["balanced_accuracy"],
                "macro_f1": detailed_metrics["macro_f1"],
            },
            "per_class": detailed_metrics["per_class"],
            "config": {
                "epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.data.batch_size,
                "frozen_layers": config.model.freeze_encoder_layers,
            },
        }, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - {benchmark_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
