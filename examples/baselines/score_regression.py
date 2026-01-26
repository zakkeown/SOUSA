#!/usr/bin/env python3
"""
Score Regression Baseline
=========================

Predict overall_score (0-100) from audio.

Model: Wav2Vec2-base encoder with regression head
Training: AdamW, cosine LR scheduling, MSE loss
Metrics: MAE, RMSE, R², Spearman correlation

Usage:
    # Quick test run
    python -m examples.baselines.score_regression --preset fast_debug --data-dir output/dataset

    # Full baseline training
    python -m examples.baselines.score_regression --preset baseline --data-dir output/dataset

    # With Huber loss (more robust to outliers)
    python -m examples.baselines.score_regression --preset baseline --loss huber

Output:
    checkpoints/score_regression/
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

from examples.pytorch_dataloader import SOUSADataset, create_dataloader
from examples.baselines.config import (
    TrainingConfig,
    get_preset,
)
from examples.baselines.models import Wav2Vec2Regressor
from examples.baselines.training import Trainer, seed_everything


def compute_detailed_metrics(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """Compute detailed regression metrics including calibration."""
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch["waveforms"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(waveforms, attention_mask)
            preds = outputs["predictions"]

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Scale back to 0-100 range
    all_predictions_scaled = all_predictions * 100
    all_labels_scaled = all_labels * 100

    # Basic metrics
    errors = all_predictions_scaled - all_labels_scaled
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((all_labels_scaled - np.mean(all_labels_scaled)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Spearman correlation
    try:
        from scipy.stats import spearmanr, pearsonr
        spearman, spearman_p = spearmanr(all_predictions_scaled, all_labels_scaled)
        pearson, pearson_p = pearsonr(all_predictions_scaled, all_labels_scaled)
    except ImportError:
        spearman, spearman_p = 0.0, 1.0
        pearson, pearson_p = 0.0, 1.0

    # Calibration: predictions within X points of true
    within_5 = np.mean(np.abs(errors) <= 5)
    within_10 = np.mean(np.abs(errors) <= 10)
    within_15 = np.mean(np.abs(errors) <= 15)

    # Error distribution by score range
    error_by_range = {}
    ranges = [(0, 25), (25, 50), (50, 75), (75, 100)]
    for low, high in ranges:
        mask = (all_labels_scaled >= low) & (all_labels_scaled < high)
        if mask.sum() > 0:
            range_errors = errors[mask]
            error_by_range[f"{low}-{high}"] = {
                "mae": float(np.mean(np.abs(range_errors))),
                "bias": float(np.mean(range_errors)),  # Systematic over/under prediction
                "count": int(mask.sum()),
            }

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "spearman": float(spearman) if not np.isnan(spearman) else 0.0,
        "spearman_pvalue": float(spearman_p),
        "pearson": float(pearson) if not np.isnan(pearson) else 0.0,
        "pearson_pvalue": float(pearson_p),
        "within_5_points": float(within_5),
        "within_10_points": float(within_10),
        "within_15_points": float(within_15),
        "error_by_range": error_by_range,
        "prediction_stats": {
            "mean": float(np.mean(all_predictions_scaled)),
            "std": float(np.std(all_predictions_scaled)),
            "min": float(np.min(all_predictions_scaled)),
            "max": float(np.max(all_predictions_scaled)),
        },
        "label_stats": {
            "mean": float(np.mean(all_labels_scaled)),
            "std": float(np.std(all_labels_scaled)),
            "min": float(np.min(all_labels_scaled)),
            "max": float(np.max(all_labels_scaled)),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train score regression baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/dataset",
        help="Path to SOUSA dataset directory",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="overall_score",
        help="Score column to predict (overall_score, timing_score, dynamics_score, etc.)",
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
        default="checkpoints/score_regression",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "huber"],
        default="mse",
        help="Loss function",
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
    config.task_type = "regression"
    config.loss_fn = args.loss
    config.data.target = args.target
    config.data.data_dir = args.data_dir
    config.save_dir = args.output_dir

    # Seed everything
    seed_everything(config.seed, config.deterministic)

    print("=" * 60)
    print("SOUSA Score Regression Baseline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target: {args.target}")
    print(f"Preset: {args.preset}")
    print(f"Loss function: {args.loss}")
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
        target=args.target,
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    val_dataset = SOUSADataset(
        data_dir=args.data_dir,
        split="validation",
        target=args.target,
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    test_dataset = SOUSADataset(
        data_dir=args.data_dir,
        split="test",
        target=args.target,
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

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
        print(f"Label range: [{batch['labels'].min():.3f}, {batch['labels'].max():.3f}]")
        print("Dry run successful!")
        return

    # Create model
    print("\nCreating model...")
    model = Wav2Vec2Regressor(config.model, num_outputs=1)

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
        print(f"RMSE: {detailed_metrics['rmse']:.2f} points")
        print(f"MAE: {detailed_metrics['mae']:.2f} points")
        print(f"R²: {detailed_metrics['r2']:.4f}")
        print(f"Spearman ρ: {detailed_metrics['spearman']:.4f}")
        print(f"Pearson r: {detailed_metrics['pearson']:.4f}")
        print(f"\nCalibration:")
        print(f"  Within 5 points: {detailed_metrics['within_5_points']*100:.1f}%")
        print(f"  Within 10 points: {detailed_metrics['within_10_points']*100:.1f}%")
        print(f"  Within 15 points: {detailed_metrics['within_15_points']*100:.1f}%")

        print(f"\nError by score range:")
        for range_name, stats in detailed_metrics['error_by_range'].items():
            print(f"  {range_name}: MAE={stats['mae']:.2f}, bias={stats['bias']:+.2f} (n={stats['count']})")

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

    print(f"RMSE: {detailed_metrics['rmse']:.2f} points")
    print(f"MAE: {detailed_metrics['mae']:.2f} points")
    print(f"R²: {detailed_metrics['r2']:.4f}")
    print(f"Spearman ρ: {detailed_metrics['spearman']:.4f}")

    # Save detailed results
    results_path = Path(args.output_dir) / "detailed_test_results.json"
    with open(results_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    # Save to benchmarks directory
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)
    benchmark_path = benchmarks_dir / "score_regression.json"
    with open(benchmark_path, "w") as f:
        json.dump({
            "task": "score_regression",
            "target": args.target,
            "model": "wav2vec2-base",
            "preset": args.preset,
            "loss": args.loss,
            "metrics": {
                "rmse": detailed_metrics["rmse"],
                "mae": detailed_metrics["mae"],
                "r2": detailed_metrics["r2"],
                "spearman": detailed_metrics["spearman"],
                "pearson": detailed_metrics["pearson"],
            },
            "calibration": {
                "within_5_points": detailed_metrics["within_5_points"],
                "within_10_points": detailed_metrics["within_10_points"],
                "within_15_points": detailed_metrics["within_15_points"],
            },
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
