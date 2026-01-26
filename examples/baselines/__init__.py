"""
SOUSA Baseline Models
=====================

Production-ready baseline models for drum rudiment analysis.

Models:
- Wav2Vec2Classifier: Pretrained Wav2Vec2 with classification head
- Wav2Vec2Regressor: Pretrained Wav2Vec2 with regression head
- MultiTaskModel: Shared encoder with task-specific heads
- AudioSpectrogramTransformer: AST-style model for spectrograms

Training:
- Trainer: Full training loop with validation, checkpointing, logging
- EarlyStopping: Early stopping callback
- MetricTracker: Track and compute running metrics

Config:
- TrainingConfig: Dataclass-based configuration
- ModelConfig: Model architecture configuration
- DataConfig: Data loading configuration
- Presets: fast_debug, baseline, full configurations
"""

from .config import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    load_config,
    save_config,
    get_preset,
    merge_config_with_overrides,
    PRESETS,
)
from .models import (
    Wav2Vec2Classifier,
    Wav2Vec2Regressor,
    MultiTaskModel,
    AudioSpectrogramTransformer,
    create_model,
    ClassificationHead,
    RegressionHead,
)
from .training import (
    Trainer,
    EarlyStopping,
    MetricTracker,
    FocalLoss,
    seed_everything,
    compute_classification_metrics,
    compute_regression_metrics,
)

__all__ = [
    # Config
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "load_config",
    "save_config",
    "get_preset",
    "merge_config_with_overrides",
    "PRESETS",
    # Models
    "Wav2Vec2Classifier",
    "Wav2Vec2Regressor",
    "MultiTaskModel",
    "AudioSpectrogramTransformer",
    "create_model",
    "ClassificationHead",
    "RegressionHead",
    # Training
    "Trainer",
    "EarlyStopping",
    "MetricTracker",
    "FocalLoss",
    "seed_everything",
    "compute_classification_metrics",
    "compute_regression_metrics",
]
