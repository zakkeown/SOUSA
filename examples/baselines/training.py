"""
Training Infrastructure for SOUSA Baselines
=============================================

Production-ready training loop with all the bells and whistles:
- Train/validate/test loops with metrics
- Learning rate scheduling (cosine with warmup)
- Early stopping with patience
- Gradient clipping and accumulation
- Mixed precision training (AMP)
- Checkpointing (best, last, periodic)
- TensorBoard/WandB logging
- Reproducibility (seed everything)

Usage:
    from examples.baselines.training import Trainer
    from examples.baselines.config import get_preset
    from examples.baselines.models import Wav2Vec2Classifier

    config = get_preset("baseline")
    model = Wav2Vec2Classifier(config.model, num_classes=4)
    trainer = Trainer(model, config)

    trainer.fit(train_loader, val_loader)
    results = trainer.test(test_loader)
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LambdaLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import TrainingConfig


def seed_everything(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping with patience and minimum delta."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: Literal["min", "max"] = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class MetricTracker:
    """Track and compute running metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = defaultdict(list)

    def update(self, metrics: dict[str, float], batch_size: int = 1):
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.values[key].append((value, batch_size))

    def compute(self) -> dict[str, float]:
        results = {}
        for key, value_list in self.values.items():
            total = sum(v * n for v, n in value_list)
            count = sum(n for _, n in value_list)
            results[key] = total / count if count > 0 else 0.0
        return results


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute classification metrics."""
    # Accuracy
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    # Per-class accuracy for balanced accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for pred, label in zip(predictions, labels):
        class_total[label.item()] += 1
        if pred == label:
            class_correct[label.item()] += 1

    class_accuracies = []
    for c in range(num_classes):
        if class_total[c] > 0:
            class_accuracies.append(class_correct[c] / class_total[c])

    balanced_accuracy = np.mean(class_accuracies) if class_accuracies else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def compute_regression_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute regression metrics."""
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    label_np = labels.cpu().numpy()

    # MSE and MAE
    mse = np.mean((pred_np - label_np) ** 2)
    mae = np.mean(np.abs(pred_np - label_np))
    rmse = np.sqrt(mse)

    # R-squared
    ss_res = np.sum((label_np - pred_np) ** 2)
    ss_tot = np.sum((label_np - np.mean(label_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Spearman correlation
    try:
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(pred_np, label_np)
    except ImportError:
        spearman = 0.0

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": spearman if not np.isnan(spearman) else 0.0,
    }


class Trainer:
    """
    Training orchestrator for SOUSA baseline models.

    Handles the complete training pipeline including optimization,
    scheduling, checkpointing, and logging.

    Args:
        model: PyTorch model to train
        config: TrainingConfig with hyperparameters
        output_dir: Directory for checkpoints and logs (overrides config.save_dir)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        output_dir: str | Path | None = None,
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir or config.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup loss function
        self.loss_fn = self._create_loss_fn()

        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision and self.device.type == "cuda" else None

        # Early stopping
        # Note: Always use "max" mode; we negate the value in fit() for min-is-better metrics
        self.early_stopping = None
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.patience,
                min_delta=config.min_delta,
                mode="max",
            )

        # Logging
        self.writer = None
        if config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.output_dir / "tensorboard")

        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config),
            )

        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.history = defaultdict(list)

        # Set seeds
        seed_everything(config.seed, config.deterministic)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if self.config.optimizer == "adamw":
            return AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps,
            )
        elif self.config.optimizer == "adam":
            return Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps,
            )
        elif self.config.optimizer == "sgd":
            return SGD(
                param_groups,
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_loss_fn(self) -> Callable:
        """Create loss function based on config."""
        if self.config.loss_fn == "cross_entropy":
            weights = None
            if self.config.class_weights:
                weights = torch.tensor(self.config.class_weights, device=self.device)
            return nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=self.config.label_smoothing,
            )
        elif self.config.loss_fn == "focal":
            return FocalLoss(gamma=2.0)
        elif self.config.loss_fn == "mse":
            return nn.MSELoss()
        elif self.config.loss_fn == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_fn}")

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler == "cosine":
            # Warmup then cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=self.config.min_lr,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
        elif self.config.scheduler == "linear":
            # Warmup then linear decay
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                return max(
                    self.config.min_lr / self.config.learning_rate,
                    1 - (step - warmup_steps) / (num_training_steps - warmup_steps),
                )
            return LambdaLR(self.optimizer, lr_lambda)
        elif self.config.scheduler == "constant":
            # Just warmup
            return LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        scheduler,
    ) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        metric_tracker = MetricTracker()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )

        accumulated_loss = 0.0
        accumulated_steps = 0

        for batch_idx, batch in enumerate(progress):
            # Move batch to device
            waveforms = batch["waveforms"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                outputs = self.model(
                    input_values=waveforms,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()
            accumulated_steps += 1

            # Gradient step
            if accumulated_steps >= self.config.gradient_accumulation_steps:
                if self.config.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                scheduler.step()
                self.optimizer.zero_grad()

                # Update metrics
                metric_tracker.update(
                    {"loss": accumulated_loss * self.config.gradient_accumulation_steps},
                    batch_size=waveforms.size(0),
                )

                accumulated_loss = 0.0
                accumulated_steps = 0
                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    progress.set_postfix(
                        loss=f"{metric_tracker.compute()['loss']:.4f}",
                        lr=f"{lr:.2e}",
                    )

                    if self.writer:
                        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                        self.writer.add_scalar("train/lr", lr, self.global_step)

                    if WANDB_AVAILABLE and self.config.use_wandb:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "global_step": self.global_step,
                        })

        return metric_tracker.compute()

    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        prefix: str = "val",
    ) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        metric_tracker = MetricTracker()

        all_predictions = []
        all_labels = []

        for batch in tqdm(val_loader, desc="Validating", leave=False):
            waveforms = batch["waveforms"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast(enabled=self.scaler is not None):
                outputs = self.model(
                    input_values=waveforms,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            metric_tracker.update({"loss": outputs["loss"]}, batch_size=waveforms.size(0))

            # Collect predictions
            if self.config.task_type == "classification":
                preds = outputs["logits"].argmax(dim=-1)
            else:
                preds = outputs["predictions"]

            all_predictions.append(preds.cpu())
            all_labels.append(labels.cpu())

        # Compute metrics
        metrics = metric_tracker.compute()
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        if self.config.task_type == "classification":
            # Infer num_classes from logits
            num_classes = outputs["logits"].size(-1)
            extra_metrics = compute_classification_metrics(
                all_predictions, all_labels, num_classes
            )
        else:
            extra_metrics = compute_regression_metrics(all_predictions, all_labels)

        metrics.update(extra_metrics)

        # Log
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, self.current_epoch)

        if WANDB_AVAILABLE and self.config.use_wandb:
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()})

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        num_epochs: int | None = None,
    ) -> dict[str, list]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Override config.num_epochs

        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.num_epochs

        # Calculate total training steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        num_training_steps = steps_per_epoch * num_epochs

        # Create scheduler
        scheduler = self._create_scheduler(num_training_steps)

        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2, default=str)

        print(f"Training for {num_epochs} epochs ({num_training_steps} steps)")
        print(f"Output directory: {self.output_dir}")

        best_metric_value = None
        # Determine metric for early stopping and best model selection
        # Default: balanced_accuracy for classification (handles class imbalance)
        # Default: rmse for regression
        if self.config.early_stopping_metric:
            metric_key = self.config.early_stopping_metric
        else:
            metric_key = "balanced_accuracy" if self.config.task_type == "classification" else "rmse"

        # Determine if higher is better for this metric
        higher_is_better = metric_key in ("accuracy", "balanced_accuracy", "r2")
        print(f"Tracking metric: {metric_key} ({'higher' if higher_is_better else 'lower'} is better)")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_loader, scheduler)
            self.history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate(val_loader, prefix="val")
                for key, value in val_metrics.items():
                    self.history[f"val_{key}"].append(value)

            # Print epoch summary
            summary = f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}"
            if val_metrics:
                summary += f", val_loss={val_metrics['loss']:.4f}"
                if metric_key in val_metrics:
                    summary += f", val_{metric_key}={val_metrics[metric_key]:.4f}"
            print(summary)

            # Save checkpoints
            if self.config.save_last:
                self._save_checkpoint("last.pt")

            if self.config.save_every_n_epochs and (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(f"epoch_{epoch}.pt")

            # Best model saving
            if val_metrics and metric_key in val_metrics:
                current_value = val_metrics[metric_key]

                is_better = False
                if best_metric_value is None:
                    is_better = True
                elif higher_is_better:
                    is_better = current_value > best_metric_value
                else:
                    is_better = current_value < best_metric_value

                if is_better:
                    best_metric_value = current_value
                    self.best_metric = val_metrics.copy()
                    if self.config.save_best:
                        self._save_checkpoint("best.pt")

                # Early stopping
                # Note: EarlyStopping is configured with mode based on metric direction
                if self.early_stopping is not None:
                    # For "max" mode metrics, pass value directly
                    # For "min" mode metrics, pass negated value (EarlyStopping expects higher=better in max mode)
                    stop_value = current_value if higher_is_better else -current_value
                    if self.early_stopping(stop_value):
                        print(f"Early stopping at epoch {epoch}")
                        break

        # Save final history
        history_path = self.output_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(dict(self.history), f, indent=2)

        if self.writer:
            self.writer.close()

        return dict(self.history)

    def test(self, test_loader: DataLoader) -> dict[str, float]:
        """
        Evaluate on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Test metrics dictionary
        """
        # Load best model if available
        best_path = self.output_dir / "best.pt"
        if best_path.exists():
            self.load_checkpoint(best_path)
            print("Loaded best model for testing")

        metrics = self._validate(test_loader, prefix="test")

        # Save test results
        results_path = self.output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Test results: {metrics}")
        return metrics

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": asdict(self.config),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = self.output_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric")

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def resume(self, checkpoint_path: str | Path | None = None):
        """Resume training from checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / "last.pt"

        if not Path(checkpoint_path).exists():
            print(f"No checkpoint found at {checkpoint_path}, starting fresh")
            return

        self.load_checkpoint(checkpoint_path)

        # Reset early stopping
        if self.early_stopping:
            self.early_stopping.reset()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        return focal_loss.mean()


# Example usage
if __name__ == "__main__":
    from .config import get_preset, ModelConfig
    from .models import Wav2Vec2Classifier

    print("Testing training infrastructure...")

    # Create a dummy model
    config = get_preset("fast_debug")
    model_config = config.model

    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")

    # This would normally be:
    # model = Wav2Vec2Classifier(model_config, num_classes=4)
    # trainer = Trainer(model, config)
    # trainer.fit(train_loader, val_loader)

    print("\nTraining infrastructure ready!")
