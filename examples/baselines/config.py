"""
Configuration Management for SOUSA Baselines
=============================================

Dataclass-based configurations with YAML serialization,
preset configs, and CLI override support.

RECOMMENDED SCORE TARGETS
-------------------------
Based on correlation analysis, use these scores to maximize signal while avoiding redundancy:

1. overall_score - Primary target for general quality assessment
2. tempo_stability - Independent signal about rushing/dragging tendencies

For multi-output regression: ["overall_score", "tempo_stability"]
For single-output regression: "overall_score"

AVOID using multiple correlated scores as separate targets:
- timing_accuracy, timing_consistency, overall_score are highly correlated (r > 0.85)
- Using all three won't provide additional learning signal

NOTE: hand_balance has a ceiling effect (mean ~88, most samples near 100)
and is excluded from recommended targets due to low discriminative power.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import yaml


# Recommended score targets based on correlation analysis
# Use these to avoid redundancy in multi-target learning
RECOMMENDED_SCORES = ["overall_score", "tempo_stability"]
PRIMARY_SCORE = "overall_score"  # Best single target for regression


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Base model
    model_name: str = "facebook/wav2vec2-base"
    pretrained: bool = True

    # Freezing strategy
    freeze_feature_extractor: bool = True
    freeze_encoder_layers: int = 6  # Freeze first N transformer layers

    # Classification head
    hidden_dim: int = 256
    num_hidden_layers: int = 1
    dropout: float = 0.1
    activation: str = "gelu"

    # For multi-task
    task_hidden_dims: dict = field(default_factory=lambda: {
        "skill_tier": 256,
        "rudiment": 512,
        "overall_score": 128,
    })

    def __post_init__(self):
        # Validate
        assert self.freeze_encoder_layers >= 0
        assert 0 <= self.dropout < 1


@dataclass
class DataConfig:
    """Data loading configuration."""

    # Dataset paths
    data_dir: str = "output/dataset"
    cache_dir: str | None = None

    # Audio processing
    sample_rate: int = 16000
    max_length_sec: float = 5.0
    normalize_audio: bool = True

    # Target configuration
    target: str = "skill_tier"  # skill_tier, overall_score, rudiment_slug
    num_classes: int | None = None  # Set automatically for classification

    # Data loading
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True

    # Augmentation
    use_spec_augment: bool = True
    time_mask_param: int = 80
    freq_mask_param: int = 27
    num_time_masks: int = 2
    num_freq_masks: int = 2

    @property
    def max_length_samples(self) -> int:
        return int(self.max_length_sec * self.sample_rate)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Model and data
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Training
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Optimizer
    optimizer: str = "adamw"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Scheduler
    scheduler: str = "cosine"  # cosine, linear, constant
    min_lr: float = 1e-6

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    # Metric for early stopping and best model selection
    # For classification with imbalanced classes, use "balanced_accuracy" (default)
    # Options: "accuracy", "balanced_accuracy", "loss", "rmse", "mae", "r2"
    early_stopping_metric: str | None = None  # None = auto-select based on task

    # Checkpointing
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_last: bool = True
    save_every_n_epochs: int | None = None

    # Logging
    log_every_n_steps: int = 50
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "sousa-baselines"
    wandb_run_name: str | None = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Hardware
    device: str = "auto"  # auto, cuda, cpu, mps
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

    # Task-specific
    task_type: Literal["classification", "regression", "multitask"] = "classification"
    loss_fn: str = "cross_entropy"  # cross_entropy, focal, mse, huber
    label_smoothing: float = 0.0
    class_weights: list[float] | None = None

    # Multi-task specific
    task_weights: dict = field(default_factory=lambda: {
        "skill_tier": 1.0,
        "rudiment": 1.0,
        "overall_score": 1.0,
    })

    def __post_init__(self):
        # Resolve device
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

    @property
    def effective_batch_size(self) -> int:
        return self.data.batch_size * self.gradient_accumulation_steps

    @property
    def warmup_steps(self) -> int:
        # Approximate - actual depends on dataset size
        return int(1000 * self.warmup_ratio)

    def get_config_hash(self) -> str:
        """Get a hash of the configuration for experiment tracking."""
        config_dict = asdict(self)
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None


# Preset configurations
PRESETS = {
    "fast_debug": TrainingConfig(
        model=ModelConfig(
            freeze_encoder_layers=12,  # Freeze everything
            hidden_dim=64,
        ),
        data=DataConfig(
            batch_size=4,
            max_length_sec=2.0,
            num_workers=0,
        ),
        num_epochs=2,
        learning_rate=1e-3,
        early_stopping=False,
        log_every_n_steps=10,
        use_tensorboard=False,
        mixed_precision=False,
    ),
    "baseline": TrainingConfig(
        model=ModelConfig(
            freeze_encoder_layers=6,
            hidden_dim=256,
        ),
        data=DataConfig(
            batch_size=16,
            max_length_sec=5.0,
        ),
        num_epochs=50,
        learning_rate=1e-4,
        patience=10,
    ),
    "full": TrainingConfig(
        model=ModelConfig(
            freeze_encoder_layers=4,  # Unfreeze more layers
            hidden_dim=512,
            num_hidden_layers=2,
        ),
        data=DataConfig(
            batch_size=32,
            max_length_sec=8.0,
        ),
        num_epochs=100,
        learning_rate=5e-5,
        patience=15,
        gradient_accumulation_steps=2,
    ),
}


def load_config(path: str | Path) -> TrainingConfig:
    """Load configuration from YAML file."""
    path = Path(path)

    with open(path) as f:
        data = yaml.safe_load(f)

    # Handle nested configs
    if "model" in data:
        data["model"] = ModelConfig(**data["model"])
    if "data" in data:
        data["data"] = DataConfig(**data["data"])

    return TrainingConfig(**data)


def save_config(config: TrainingConfig, path: str | Path):
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_preset(name: str) -> TrainingConfig:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def merge_config_with_overrides(
    config: TrainingConfig,
    overrides: dict,
) -> TrainingConfig:
    """
    Merge a config with command-line overrides.

    Supports nested keys like "model.hidden_dim=512".
    """
    config_dict = asdict(config)

    for key, value in overrides.items():
        parts = key.split(".")
        d = config_dict
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = value

    # Reconstruct nested configs
    if "model" in config_dict:
        config_dict["model"] = ModelConfig(**config_dict["model"])
    if "data" in config_dict:
        config_dict["data"] = DataConfig(**config_dict["data"])

    return TrainingConfig(**config_dict)


# Example usage
if __name__ == "__main__":
    # Show preset configs
    print("Available presets:")
    for name, config in PRESETS.items():
        print(f"\n{name}:")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  LR: {config.learning_rate}")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Frozen layers: {config.model.freeze_encoder_layers}")

    # Save example config
    config = get_preset("baseline")
    save_config(config, "configs/baseline.yaml")
    print("\nSaved baseline config to configs/baseline.yaml")

    # Show config hash
    print(f"Config hash: {config.get_config_hash()}")
