"""Audio augmentation pipeline for realistic recording simulation."""

from dataset_gen.audio_aug.room import (
    RoomSimulator,
    RoomType,
    RoomConfig,
)
from dataset_gen.audio_aug.mic import (
    MicSimulator,
    MicType,
    MicConfig,
)
from dataset_gen.audio_aug.chain import (
    RecordingChain,
    ChainConfig,
)
from dataset_gen.audio_aug.degradation import (
    AudioDegrader,
    DegradationConfig,
)
from dataset_gen.audio_aug.pipeline import (
    AugmentationPipeline,
    AugmentationConfig,
    augment_audio,
)

__all__ = [
    # Room
    "RoomSimulator",
    "RoomType",
    "RoomConfig",
    # Mic
    "MicSimulator",
    "MicType",
    "MicConfig",
    # Chain
    "RecordingChain",
    "ChainConfig",
    # Degradation
    "AudioDegrader",
    "DegradationConfig",
    # Pipeline
    "AugmentationPipeline",
    "AugmentationConfig",
    "augment_audio",
]
