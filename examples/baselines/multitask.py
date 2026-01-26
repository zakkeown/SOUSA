#!/usr/bin/env python3
"""
Multi-Task Learning Baseline
============================

Joint prediction: skill tier + overall score + rudiment classification.

Model: Shared Wav2Vec2 encoder with task-specific heads
Training: Weighted sum of task losses
Metrics: Compare to single-task baselines to measure transfer effects

Usage:
    # Quick test run
    python -m examples.baselines.multitask --preset fast_debug --data-dir output/dataset

    # Full baseline training
    python -m examples.baselines.multitask --preset baseline --data-dir output/dataset

    # Custom task weights
    python -m examples.baselines.multitask --preset baseline \\
        --skill-weight 1.0 --rudiment-weight 0.5 --score-weight 1.0

Output:
    checkpoints/multitask/
    ├── config.json          # Training configuration
    ├── best.pt              # Best model checkpoint
    ├── last.pt              # Last epoch checkpoint
    ├── history.json         # Training history
    ├── test_results.json    # Final test metrics (all tasks)
    └── tensorboard/         # TensorBoard logs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.pytorch_dataloader import (
    SOUSADataset,
    SKILL_TIER_LABELS,
    SKILL_TIER_TO_ID,
)
from examples.baselines.config import get_preset
from examples.baselines.models import MultiTaskModel
from examples.baselines.training import seed_everything, MetricTracker

try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


class MultiTaskDataset(SOUSADataset):
    """Extended dataset that returns all targets for multi-task learning."""

    def __init__(self, *args, **kwargs):
        # Force target to None since we handle it ourselves
        kwargs["target"] = "overall_score"  # Dummy, we override _get_label
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]

        # Load audio
        waveform = self._load_audio(row["audio_path"])

        # Apply transform if provided
        if self.transform is not None:
            waveform = self.transform(waveform)

        # Get all labels
        skill_label = SKILL_TIER_TO_ID[row["skill_tier"]]
        rudiment_label = self.rudiment_to_id[row["rudiment_slug"]]
        score_label = row["overall_score"] / 100.0  # Normalize to [0, 1]

        return {
            "waveform": waveform,
            "skill_label": skill_label,
            "rudiment_label": rudiment_label,
            "score_label": torch.tensor(score_label, dtype=torch.float32),
            "sample_id": row["sample_id"],
            "skill_tier": row["skill_tier"],
            "rudiment_slug": row["rudiment_slug"],
            "tempo_bpm": row["tempo_bpm"],
        }


def collate_multitask(batch: list[dict], target_length: int) -> dict:
    """Collate function for multi-task learning."""
    waveforms = []
    lengths = []

    for sample in batch:
        waveform = sample["waveform"]
        orig_len = len(waveform)
        lengths.append(min(orig_len, target_length))

        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        elif len(waveform) < target_length:
            padding = target_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding), value=0.0)

        waveforms.append(waveform)

    waveforms = torch.stack(waveforms)
    lengths = torch.tensor(lengths)
    attention_mask = torch.arange(target_length).expand(len(batch), -1) < lengths.unsqueeze(1)

    return {
        "waveforms": waveforms,
        "lengths": lengths,
        "attention_mask": attention_mask.float(),
        "skill_labels": torch.tensor([s["skill_label"] for s in batch]),
        "rudiment_labels": torch.tensor([s["rudiment_label"] for s in batch]),
        "score_labels": torch.stack([s["score_label"] for s in batch]),
        "sample_ids": [s["sample_id"] for s in batch],
    }


class MultiTaskTrainer:
    """Specialized trainer for multi-task learning."""

    def __init__(
        self,
        model: MultiTaskModel,
        config,
        task_weights: dict[str, float],
        output_dir: str | Path,
    ):
        self.model = model
        self.config = config
        self.task_weights = task_weights
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision and self.device.type == "cuda" and AMP_AVAILABLE else None

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.history = {"train": [], "val": []}

    def _train_epoch(self, train_loader: DataLoader) -> dict:
        """Run one training epoch."""
        self.model.train()
        metrics = MetricTracker()

        progress = tqdm(train_loader, desc=f"Epoch {self.current_epoch}", leave=False)

        for batch in progress:
            waveforms = batch["waveforms"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            skill_labels = batch["skill_labels"].to(self.device)
            rudiment_labels = batch["rudiment_labels"].to(self.device)
            score_labels = batch["score_labels"].to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.scaler is not None):
                outputs = self.model(
                    waveforms,
                    attention_mask,
                    skill_labels=skill_labels,
                    rudiment_labels=rudiment_labels,
                    score_labels=score_labels,
                    task_weights=self.task_weights,
                )
                loss = outputs["loss"]

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Track metrics
            batch_metrics = {
                "loss": loss.item(),
                "skill_loss": outputs["losses"].get("skill_loss", torch.tensor(0.0)).item(),
                "rudiment_loss": outputs["losses"].get("rudiment_loss", torch.tensor(0.0)).item(),
                "score_loss": outputs["losses"].get("score_loss", torch.tensor(0.0)).item(),
            }
            metrics.update(batch_metrics, batch_size=waveforms.size(0))

            progress.set_postfix(loss=f"{loss.item():.4f}")
            self.global_step += 1

        return metrics.compute()

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> dict:
        """Run validation."""
        self.model.eval()

        all_skill_preds = []
        all_skill_labels = []
        all_rudiment_preds = []
        all_rudiment_labels = []
        all_score_preds = []
        all_score_labels = []
        total_loss = 0.0
        num_samples = 0

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            waveforms = batch["waveforms"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            skill_labels = batch["skill_labels"].to(self.device)
            rudiment_labels = batch["rudiment_labels"].to(self.device)
            score_labels = batch["score_labels"].to(self.device)

            outputs = self.model(
                waveforms,
                attention_mask,
                skill_labels=skill_labels,
                rudiment_labels=rudiment_labels,
                score_labels=score_labels,
                task_weights=self.task_weights,
            )

            total_loss += outputs["loss"].item() * waveforms.size(0)
            num_samples += waveforms.size(0)

            # Collect predictions
            all_skill_preds.append(outputs["skill_logits"].argmax(dim=-1).cpu())
            all_skill_labels.append(skill_labels.cpu())
            all_rudiment_preds.append(outputs["rudiment_logits"].argmax(dim=-1).cpu())
            all_rudiment_labels.append(rudiment_labels.cpu())
            all_score_preds.append(outputs["score_prediction"].cpu())
            all_score_labels.append(score_labels.cpu())

        # Concatenate
        all_skill_preds = torch.cat(all_skill_preds).numpy()
        all_skill_labels = torch.cat(all_skill_labels).numpy()
        all_rudiment_preds = torch.cat(all_rudiment_preds).numpy()
        all_rudiment_labels = torch.cat(all_rudiment_labels).numpy()
        all_score_preds = torch.cat(all_score_preds).numpy()
        all_score_labels = torch.cat(all_score_labels).numpy()

        # Compute metrics
        skill_accuracy = (all_skill_preds == all_skill_labels).mean()
        rudiment_accuracy = (all_rudiment_preds == all_rudiment_labels).mean()

        # Score metrics (scaled back to 0-100)
        score_errors = (all_score_preds - all_score_labels) * 100
        score_rmse = np.sqrt(np.mean(score_errors ** 2))
        score_mae = np.mean(np.abs(score_errors))

        return {
            "loss": total_loss / num_samples,
            "skill_accuracy": float(skill_accuracy),
            "rudiment_accuracy": float(rudiment_accuracy),
            "score_rmse": float(score_rmse),
            "score_mae": float(score_mae),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> dict:
        """Train the model."""
        print(f"Training for {num_epochs} epochs")
        print(f"Task weights: {self.task_weights}")

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_loader)
            self.history["train"].append(train_metrics)

            # Validate
            val_metrics = self._validate(val_loader)
            self.history["val"].append(val_metrics)

            # Print summary
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"skill_acc={val_metrics['skill_accuracy']:.4f}, "
                f"rudiment_acc={val_metrics['rudiment_accuracy']:.4f}, "
                f"score_rmse={val_metrics['score_rmse']:.2f}"
            )

            # Save checkpoints
            self._save_checkpoint("last.pt")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.best_metric = val_metrics.copy()
                self._save_checkpoint("best.pt")

        # Save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _save_checkpoint(self, filename: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_metric": self.best_metric,
        }
        torch.save(checkpoint, self.output_dir / filename)

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_metric = checkpoint.get("best_metric")


def compute_detailed_metrics(
    model: MultiTaskModel,
    dataloader: DataLoader,
    device: torch.device,
    id_to_rudiment: dict,
) -> dict:
    """Compute detailed metrics for all tasks."""
    model.eval()

    all_skill_preds = []
    all_skill_labels = []
    all_rudiment_preds = []
    all_rudiment_labels = []
    all_score_preds = []
    all_score_labels = []

    with torch.no_grad():
        for batch in dataloader:
            waveforms = batch["waveforms"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(waveforms, attention_mask)

            all_skill_preds.append(outputs["skill_logits"].argmax(dim=-1).cpu())
            all_skill_labels.append(batch["skill_labels"])
            all_rudiment_preds.append(outputs["rudiment_logits"].argmax(dim=-1).cpu())
            all_rudiment_labels.append(batch["rudiment_labels"])
            all_score_preds.append(outputs["score_prediction"].cpu())
            all_score_labels.append(batch["score_labels"])

    # Concatenate
    all_skill_preds = torch.cat(all_skill_preds).numpy()
    all_skill_labels = torch.cat(all_skill_labels).numpy()
    all_rudiment_preds = torch.cat(all_rudiment_preds).numpy()
    all_rudiment_labels = torch.cat(all_rudiment_labels).numpy()
    all_score_preds = torch.cat(all_score_preds).numpy()
    all_score_labels = torch.cat(all_score_labels).numpy()

    # Skill metrics
    skill_accuracy = (all_skill_preds == all_skill_labels).mean()

    # Per-class skill accuracy
    skill_per_class = {}
    for i, tier in enumerate(SKILL_TIER_LABELS):
        mask = all_skill_labels == i
        if mask.sum() > 0:
            skill_per_class[tier] = float((all_skill_preds[mask] == i).mean())

    # Rudiment metrics
    rudiment_accuracy = (all_rudiment_preds == all_rudiment_labels).mean()

    # Top-5 rudiment accuracy
    # (Would need logits stored, simplified here)

    # Score metrics
    score_errors = (all_score_preds - all_score_labels) * 100
    score_rmse = np.sqrt(np.mean(score_errors ** 2))
    score_mae = np.mean(np.abs(score_errors))

    # R-squared
    ss_res = np.sum(score_errors ** 2)
    ss_tot = np.sum((all_score_labels * 100 - np.mean(all_score_labels * 100)) ** 2)
    score_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "skill": {
            "accuracy": float(skill_accuracy),
            "per_class": skill_per_class,
        },
        "rudiment": {
            "accuracy": float(rudiment_accuracy),
        },
        "score": {
            "rmse": float(score_rmse),
            "mae": float(score_mae),
            "r2": float(score_r2),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-task learning baseline",
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
        default="checkpoints/multitask",
        help="Output directory for checkpoints and logs",
    )

    # Task weight arguments
    parser.add_argument("--skill-weight", type=float, default=1.0, help="Weight for skill loss")
    parser.add_argument("--rudiment-weight", type=float, default=1.0, help="Weight for rudiment loss")
    parser.add_argument("--score-weight", type=float, default=1.0, help="Weight for score loss")

    # Override arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--freeze-layers", type=int, help="Number of encoder layers to freeze")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Execution arguments
    parser.add_argument("--dry-run", action="store_true", help="Run one batch only")
    parser.add_argument("--test-only", action="store_true", help="Only run test evaluation")

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

    task_weights = {
        "skill_tier": args.skill_weight,
        "rudiment": args.rudiment_weight,
        "overall_score": args.score_weight,
    }

    seed_everything(config.seed, config.deterministic)

    print("=" * 60)
    print("SOUSA Multi-Task Learning Baseline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Task weights: {task_weights}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.data.batch_size}")
    print("=" * 60)

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = MultiTaskDataset(
        data_dir=args.data_dir,
        split="train",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    val_dataset = MultiTaskDataset(
        data_dir=args.data_dir,
        split="validation",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    test_dataset = MultiTaskDataset(
        data_dir=args.data_dir,
        split="test",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )

    num_rudiments = len(train_dataset.rudiments)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of rudiments: {num_rudiments}")

    # Create dataloaders
    fixed_length = config.data.max_length_samples
    collate_fn = lambda batch: collate_multitask(batch, fixed_length)
    # Use num_workers=0 for dry run to avoid pickle issues with lambda
    num_workers = 0 if args.dry_run else config.data.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory,
    )

    # Dry run check
    if args.dry_run:
        print("\nDry run: testing one batch...")
        batch = next(iter(train_loader))
        print(f"Batch waveforms shape: {batch['waveforms'].shape}")
        print(f"Skill labels shape: {batch['skill_labels'].shape}")
        print(f"Rudiment labels shape: {batch['rudiment_labels'].shape}")
        print(f"Score labels shape: {batch['score_labels'].shape}")
        print("Dry run successful!")
        return

    # Create model
    print("\nCreating model...")
    model = MultiTaskModel(
        config.model,
        num_skill_classes=4,
        num_rudiment_classes=num_rudiments,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = MultiTaskTrainer(
        model, config, task_weights, output_dir=args.output_dir
    )

    device = torch.device(config.device)

    # Test only mode
    if args.test_only:
        print("\nRunning test evaluation...")
        best_path = Path(args.output_dir) / "best.pt"
        if best_path.exists():
            trainer.load_checkpoint(best_path)

        detailed_metrics = compute_detailed_metrics(
            model, test_loader, device, train_dataset.id_to_rudiment
        )

        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        print(f"Skill accuracy: {detailed_metrics['skill']['accuracy']:.4f}")
        print(f"Rudiment accuracy: {detailed_metrics['rudiment']['accuracy']:.4f}")
        print(f"Score RMSE: {detailed_metrics['score']['rmse']:.2f}")
        print(f"Score MAE: {detailed_metrics['score']['mae']:.2f}")
        print(f"Score R²: {detailed_metrics['score']['r2']:.4f}")

        results_path = Path(args.output_dir) / "detailed_test_results.json"
        with open(results_path, "w") as f:
            json.dump(detailed_metrics, f, indent=2)

        return

    # Train
    print("\nStarting training...")
    history = trainer.fit(train_loader, val_loader, config.num_epochs)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    best_path = Path(args.output_dir) / "best.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path)

    detailed_metrics = compute_detailed_metrics(
        model, test_loader, device, train_dataset.id_to_rudiment
    )

    print(f"Skill accuracy: {detailed_metrics['skill']['accuracy']:.4f}")
    print(f"Rudiment accuracy: {detailed_metrics['rudiment']['accuracy']:.4f}")
    print(f"Score RMSE: {detailed_metrics['score']['rmse']:.2f}")
    print(f"Score R²: {detailed_metrics['score']['r2']:.4f}")

    # Save results
    results_path = Path(args.output_dir) / "detailed_test_results.json"
    with open(results_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)

    # Save benchmark
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)
    benchmark_path = benchmarks_dir / "multitask.json"
    with open(benchmark_path, "w") as f:
        json.dump({
            "task": "multitask",
            "model": "wav2vec2-base",
            "preset": args.preset,
            "task_weights": task_weights,
            "metrics": {
                "skill_accuracy": detailed_metrics["skill"]["accuracy"],
                "rudiment_accuracy": detailed_metrics["rudiment"]["accuracy"],
                "score_rmse": detailed_metrics["score"]["rmse"],
                "score_mae": detailed_metrics["score"]["mae"],
                "score_r2": detailed_metrics["score"]["r2"],
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
