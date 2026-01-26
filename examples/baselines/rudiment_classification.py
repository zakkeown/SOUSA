#!/usr/bin/env python3
"""
Rudiment Classification Baseline
=================================

40-class classification of all PAS drum rudiments.

Model: Wav2Vec2-base encoder with classification head
Training: AdamW, cosine LR, weighted cross-entropy or focal loss
Metrics: top-1 accuracy, top-5 accuracy, per-class F1, confusion matrix

Usage:
    # Quick test run
    python -m examples.baselines.rudiment_classification --preset fast_debug --data-dir output/dataset

    # Full baseline training
    python -m examples.baselines.rudiment_classification --preset baseline --data-dir output/dataset

    # With focal loss (better for imbalanced classes)
    python -m examples.baselines.rudiment_classification --preset baseline --loss focal

Output:
    checkpoints/rudiment_classification/
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
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.pytorch_dataloader import SOUSADataset, create_dataloader
from examples.baselines.config import get_preset
from examples.baselines.models import Wav2Vec2Classifier
from examples.baselines.training import Trainer, seed_everything


def compute_class_weights(dataset: SOUSADataset) -> torch.Tensor:
    """Compute inverse frequency class weights for imbalanced classes."""
    labels = [dataset.rudiment_to_id[r] for r in dataset.data["rudiment_slug"]]
    counts = Counter(labels)
    num_classes = len(dataset.rudiments)

    # Inverse frequency weighting
    total = len(labels)
    weights = torch.zeros(num_classes)
    for cls, count in counts.items():
        weights[cls] = total / (num_classes * count)

    # Normalize so mean weight is 1
    weights = weights / weights.mean()

    return weights


def compute_detailed_metrics(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    id_to_rudiment: dict,
    num_classes: int,
) -> dict:
    """Compute detailed classification metrics for 40-class problem."""
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
    top1_accuracy = (all_predictions == all_labels).mean()

    # Top-5 accuracy
    top5_correct = 0
    for i, label in enumerate(all_labels):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if label in top5_indices:
            top5_correct += 1
    top5_accuracy = top5_correct / len(all_labels)

    # Confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_predictions, all_labels):
        confusion[label, pred] += 1

    # Per-class metrics
    per_class_metrics = {}
    for c in range(num_classes):
        class_name = id_to_rudiment[c]
        true_positives = confusion[c, c]
        false_positives = confusion[:, c].sum() - true_positives
        false_negatives = confusion[c, :].sum() - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[c, :].sum()),
        }

    # Balanced accuracy and macro F1
    class_recalls = [m["recall"] for m in per_class_metrics.values()]
    class_f1s = [m["f1"] for m in per_class_metrics.values()]
    balanced_accuracy = np.mean(class_recalls)
    macro_f1 = np.mean(class_f1s)

    # Weighted F1 (by support)
    total_support = sum(m["support"] for m in per_class_metrics.values())
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total_support

    # Find most confused pairs
    confusion_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                confusion_pairs.append({
                    "true": id_to_rudiment[i],
                    "predicted": id_to_rudiment[j],
                    "count": int(confusion[i, j]),
                    "rate": float(confusion[i, j] / confusion[i, :].sum()),
                })
    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "top1_accuracy": float(top1_accuracy),
        "top5_accuracy": float(top5_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_classes": num_classes,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class_metrics,
        "top_confusions": confusion_pairs[:20],  # Top 20 confused pairs
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train rudiment classification baseline",
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
        default="checkpoints/rudiment_classification",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["cross_entropy", "focal"],
        default="cross_entropy",
        help="Loss function (focal helps with class imbalance)",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse frequency class weights",
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
    config.loss_fn = args.loss
    config.data.target = "rudiment_slug"
    config.data.data_dir = args.data_dir
    config.save_dir = args.output_dir

    # Use larger hidden dim for 40 classes
    config.model.hidden_dim = 512

    # Seed everything
    seed_everything(config.seed, config.deterministic)

    print("=" * 60)
    print("SOUSA Rudiment Classification Baseline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Loss function: {args.loss}")
    print(f"Use class weights: {args.use_class_weights}")
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
        target="rudiment_slug",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    val_dataset = SOUSADataset(
        data_dir=args.data_dir,
        split="validation",
        target="rudiment_slug",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    test_dataset = SOUSADataset(
        data_dir=args.data_dir,
        split="test",
        target="rudiment_slug",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )

    num_classes = train_dataset.num_classes
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of rudiments: {num_classes}")

    # Show class distribution
    rudiment_counts = train_dataset.data["rudiment_slug"].value_counts()
    print(f"Samples per rudiment: min={rudiment_counts.min()}, max={rudiment_counts.max()}, median={rudiment_counts.median():.0f}")

    # Compute class weights if requested
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset)
        config.class_weights = class_weights.tolist()
        print(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")

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
        print(f"Unique labels in batch: {batch['labels'].unique().tolist()}")
        print("Dry run successful!")
        return

    # Create model
    print("\nCreating model...")
    model = Wav2Vec2Classifier(config.model, num_classes=num_classes)

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

    device = torch.device(config.device)
    id_to_rudiment = train_dataset.id_to_rudiment

    # Test only mode
    if args.test_only:
        print("\nRunning test evaluation...")

        # Load best model
        best_path = Path(args.output_dir) / "best.pt"
        if best_path.exists():
            trainer.load_checkpoint(best_path)

        detailed_metrics = compute_detailed_metrics(
            model, test_loader, device, id_to_rudiment, num_classes
        )

        # Print results
        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        print(f"Top-1 Accuracy: {detailed_metrics['top1_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {detailed_metrics['top5_accuracy']:.4f}")
        print(f"Balanced Accuracy: {detailed_metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {detailed_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {detailed_metrics['weighted_f1']:.4f}")

        print("\nTop 10 most confused pairs:")
        for conf in detailed_metrics['top_confusions'][:10]:
            print(f"  {conf['true']:<25} → {conf['predicted']:<25} ({conf['count']} times, {conf['rate']*100:.1f}%)")

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

    # Load best model for final evaluation
    best_path = Path(args.output_dir) / "best.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path)

    detailed_metrics = compute_detailed_metrics(
        model, test_loader, device, id_to_rudiment, num_classes
    )

    print(f"Top-1 Accuracy: {detailed_metrics['top1_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {detailed_metrics['top5_accuracy']:.4f}")
    print(f"Balanced Accuracy: {detailed_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {detailed_metrics['macro_f1']:.4f}")

    print("\nTop 5 most confused pairs:")
    for conf in detailed_metrics['top_confusions'][:5]:
        print(f"  {conf['true']:<25} → {conf['predicted']:<25} ({conf['count']} times)")

    # Save detailed results
    results_path = Path(args.output_dir) / "detailed_test_results.json"
    with open(results_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    # Save to benchmarks directory
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)
    benchmark_path = benchmarks_dir / "rudiment_classification.json"
    with open(benchmark_path, "w") as f:
        json.dump({
            "task": "rudiment_classification",
            "model": "wav2vec2-base",
            "preset": args.preset,
            "loss": args.loss,
            "use_class_weights": args.use_class_weights,
            "num_classes": num_classes,
            "metrics": {
                "top1_accuracy": detailed_metrics["top1_accuracy"],
                "top5_accuracy": detailed_metrics["top5_accuracy"],
                "balanced_accuracy": detailed_metrics["balanced_accuracy"],
                "macro_f1": detailed_metrics["macro_f1"],
                "weighted_f1": detailed_metrics["weighted_f1"],
            },
            "top_confusions": detailed_metrics["top_confusions"][:10],
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
