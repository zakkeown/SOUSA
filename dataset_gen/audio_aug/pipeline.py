"""
Unified audio augmentation pipeline.

This module combines all augmentation stages into a single pipeline
with preset configurations for common recording scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np

from dataset_gen.audio_aug.room import (
    RoomSimulator,
    RoomType,
    RoomConfig,
)
from dataset_gen.audio_aug.mic import (
    MicSimulator,
    MicType,
    MicPosition,
    MicConfig,
)
from dataset_gen.audio_aug.chain import (
    RecordingChain,
    ChainConfig,
    PreampConfig,
    PreampType,
    CompressorConfig,
    EQConfig,
)
from dataset_gen.audio_aug.degradation import (
    AudioDegrader,
    DegradationConfig,
    NoiseConfig,
    NoiseType,
    BitDepthConfig,
)


class AugmentationPreset(str, Enum):
    """Preset augmentation configurations."""

    # Clean configurations
    CLEAN_STUDIO = "clean_studio"  # Professional studio recording
    CLEAN_CLOSE = "clean_close"  # Close-miked, minimal room

    # Room variations
    PRACTICE_ROOM = "practice_room"  # Small practice space
    CONCERT_HALL = "concert_hall"  # Large reverberant space
    GYM = "gym"  # Gymnasium (marching band typical)
    GARAGE = "garage"  # Garage rehearsal space

    # Vintage/degraded
    VINTAGE_TAPE = "vintage_tape"  # Tape-style with hiss and warmth
    LO_FI = "lo_fi"  # Low fidelity recording
    PHONE_RECORDING = "phone_recording"  # Phone/voice memo quality

    # Competition/performance
    MARCHING_FIELD = "marching_field"  # Outdoor marching field
    INDOOR_COMPETITION = "indoor_competition"  # Indoor venue


@dataclass
class AugmentationConfig:
    """Complete augmentation pipeline configuration."""

    # Room settings
    room_enabled: bool = True
    room_type: RoomType = RoomType.STUDIO
    room_wet_dry: float | None = None  # None = use default for room type
    ir_path: Path | None = None  # Custom IR file

    # Mic settings
    mic_enabled: bool = True
    mic_type: MicType = MicType.CONDENSER
    mic_position: MicPosition = MicPosition.CENTER
    mic_distance: float = 0.5

    # Recording chain
    chain_enabled: bool = True
    preamp_type: PreampType = PreampType.CLEAN
    preamp_drive: float = 0.0
    compression_enabled: bool = True
    compression_ratio: float = 4.0
    compression_threshold_db: float = -12.0

    # EQ
    eq_enabled: bool = True
    highpass_freq: float = 40.0
    lowpass_freq: float = 18000.0

    # Degradation
    degradation_enabled: bool = False
    noise_type: NoiseType = NoiseType.PINK
    noise_level_db: float = -50.0
    bit_depth: int | None = None  # None = no reduction
    wow_flutter: float = 0.0

    # Output settings
    normalize: bool = True
    target_peak_db: float = -1.0  # Peak level after normalization

    # Tracking
    preset_name: str | None = None  # Name of preset if created from one

    @classmethod
    def from_preset(cls, preset: AugmentationPreset) -> "AugmentationConfig":
        """Create configuration from preset."""
        import copy

        config = copy.copy(PRESET_CONFIGS[preset])
        config.preset_name = preset.value
        return config


# Preset configurations
PRESET_CONFIGS: dict[AugmentationPreset, AugmentationConfig] = {
    AugmentationPreset.CLEAN_STUDIO: AugmentationConfig(
        room_type=RoomType.STUDIO,
        room_wet_dry=0.2,
        mic_type=MicType.CONDENSER,
        mic_distance=0.4,
        preamp_type=PreampType.CLEAN,
        compression_ratio=2.0,
        compression_threshold_db=-18.0,
        noise_level_db=-60.0,
    ),
    AugmentationPreset.CLEAN_CLOSE: AugmentationConfig(
        room_enabled=False,
        mic_type=MicType.DYNAMIC,
        mic_distance=0.15,
        preamp_type=PreampType.CLEAN,
        compression_ratio=1.5,
        noise_level_db=-70.0,
    ),
    AugmentationPreset.PRACTICE_ROOM: AugmentationConfig(
        room_type=RoomType.PRACTICE_ROOM,
        room_wet_dry=0.15,
        mic_type=MicType.DYNAMIC,
        mic_distance=0.5,
        preamp_type=PreampType.CLEAN,
        compression_ratio=2.0,
        compression_threshold_db=-18.0,
        degradation_enabled=True,
        noise_level_db=-45.0,
    ),
    AugmentationPreset.CONCERT_HALL: AugmentationConfig(
        room_type=RoomType.CONCERT_HALL,
        room_wet_dry=0.4,
        mic_type=MicType.CONDENSER,
        mic_position=MicPosition.DISTANT,
        mic_distance=3.0,
        preamp_type=PreampType.CLEAN,
        compression_enabled=False,
    ),
    AugmentationPreset.GYM: AugmentationConfig(
        room_type=RoomType.GYM,
        room_wet_dry=0.35,
        mic_type=MicType.DYNAMIC,
        mic_position=MicPosition.OVERHEAD,
        mic_distance=2.0,
        preamp_type=PreampType.CLEAN,
        compression_ratio=2.5,
        compression_threshold_db=-15.0,
        degradation_enabled=True,
        noise_type=NoiseType.HVAC,
        noise_level_db=-40.0,
    ),
    AugmentationPreset.GARAGE: AugmentationConfig(
        room_type=RoomType.GARAGE,
        room_wet_dry=0.3,
        mic_type=MicType.DYNAMIC,
        mic_distance=0.8,
        preamp_type=PreampType.AGGRESSIVE,
        preamp_drive=0.2,
        compression_ratio=2.5,
        compression_threshold_db=-15.0,
        degradation_enabled=True,
        noise_level_db=-35.0,
    ),
    AugmentationPreset.VINTAGE_TAPE: AugmentationConfig(
        room_type=RoomType.STUDIO,
        room_wet_dry=0.25,
        mic_type=MicType.RIBBON,
        mic_distance=0.6,
        preamp_type=PreampType.WARM,
        preamp_drive=0.3,
        compression_ratio=6.0,
        compression_threshold_db=-10.0,
        degradation_enabled=True,
        noise_type=NoiseType.TAPE_HISS,
        noise_level_db=-35.0,
        wow_flutter=0.2,
        highpass_freq=80.0,
        lowpass_freq=12000.0,
    ),
    AugmentationPreset.LO_FI: AugmentationConfig(
        room_type=RoomType.BEDROOM,
        room_wet_dry=0.1,
        mic_type=MicType.PIEZO,
        mic_distance=0.3,
        preamp_type=PreampType.AGGRESSIVE,
        preamp_drive=0.4,
        compression_ratio=8.0,
        compression_threshold_db=-8.0,
        degradation_enabled=True,
        noise_level_db=-25.0,
        bit_depth=12,
        highpass_freq=200.0,
        lowpass_freq=6000.0,
    ),
    AugmentationPreset.PHONE_RECORDING: AugmentationConfig(
        room_type=RoomType.BEDROOM,
        room_wet_dry=0.15,
        mic_type=MicType.PIEZO,
        mic_position=MicPosition.DISTANT,
        mic_distance=1.5,
        preamp_type=PreampType.AGGRESSIVE,
        preamp_drive=0.3,
        compression_ratio=10.0,
        compression_threshold_db=-6.0,
        degradation_enabled=True,
        noise_type=NoiseType.ROOM_TONE,
        noise_level_db=-30.0,
        bit_depth=16,
        highpass_freq=300.0,
        lowpass_freq=4000.0,
    ),
    AugmentationPreset.MARCHING_FIELD: AugmentationConfig(
        room_type=RoomType.OUTDOOR,
        room_wet_dry=0.05,
        mic_type=MicType.DYNAMIC,
        mic_position=MicPosition.DISTANT,
        mic_distance=5.0,
        preamp_type=PreampType.CLEAN,
        compression_enabled=False,
        degradation_enabled=True,
        noise_type=NoiseType.PINK,
        noise_level_db=-35.0,
        lowpass_freq=10000.0,
    ),
    AugmentationPreset.INDOOR_COMPETITION: AugmentationConfig(
        room_type=RoomType.GYM,
        room_wet_dry=0.3,
        mic_type=MicType.CONDENSER,
        mic_position=MicPosition.OVERHEAD,
        mic_distance=2.5,
        preamp_type=PreampType.CLEAN,
        compression_ratio=2.5,
        degradation_enabled=True,
        noise_type=NoiseType.ROOM_TONE,
        noise_level_db=-40.0,
    ),
}


class AugmentationPipeline:
    """
    Complete audio augmentation pipeline.

    Processes audio through: Room -> Mic -> Chain -> Degradation

    Processing order is designed to simulate the real recording signal flow.
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        preset: AugmentationPreset | None = None,
        sample_rate: int = 44100,
        ir_directory: Path | str | None = None,
        noise_directory: Path | str | None = None,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration (overrides preset)
            preset: Use a preset configuration
            sample_rate: Audio sample rate
            ir_directory: Directory containing impulse response files
            noise_directory: Directory containing noise profile files
        """
        if config is not None:
            self.config = config
        elif preset is not None:
            self.config = AugmentationConfig.from_preset(preset)
        else:
            self.config = AugmentationConfig()

        self.sample_rate = sample_rate
        self.ir_directory = Path(ir_directory) if ir_directory else None
        self.noise_directory = Path(noise_directory) if noise_directory else None

        # Initialize processors
        self._room: RoomSimulator | None = None
        self._mic: MicSimulator | None = None
        self._chain: RecordingChain | None = None
        self._degrader: AudioDegrader | None = None

    def process(
        self,
        audio: np.ndarray,
        config: AugmentationConfig | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Process audio through the augmentation pipeline.

        Args:
            audio: Input audio (shape: [samples] or [samples, channels])
            config: Optional override configuration
            seed: Random seed for reproducible augmentations

        Returns:
            Augmented audio
        """
        cfg = config or self.config
        rng = np.random.default_rng(seed)

        # Ensure 2D
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
            was_mono = True
        else:
            was_mono = False

        output = audio.copy()

        # Stage 1: Room simulation
        if cfg.room_enabled:
            output = self._process_room(output, cfg)

        # Stage 2: Mic simulation
        if cfg.mic_enabled:
            output = self._process_mic(output, cfg)

        # Stage 3: Recording chain
        if cfg.chain_enabled:
            output = self._process_chain(output, cfg)

        # Stage 4: Degradation
        if cfg.degradation_enabled:
            output = self._process_degradation(output, cfg, rng)

        # Final normalization
        if cfg.normalize:
            output = self._normalize(output, cfg.target_peak_db)

        if was_mono:
            output = output.flatten()

        return output

    def _process_room(
        self,
        audio: np.ndarray,
        config: AugmentationConfig,
    ) -> np.ndarray:
        """Apply room simulation."""
        room_config = RoomConfig(
            room_type=config.room_type,
            wet_dry_mix=config.room_wet_dry or 0.3,
            ir_path=config.ir_path,
        )

        if self._room is None:
            self._room = RoomSimulator(
                config=room_config,
                ir_directory=self.ir_directory,
                sample_rate=self.sample_rate,
            )

        # Get default wet/dry if not specified
        if config.room_wet_dry is None:
            room_config.wet_dry_mix = self._room.get_default_wet_dry(config.room_type)

        return self._room.process(audio, room_config)

    def _process_mic(
        self,
        audio: np.ndarray,
        config: AugmentationConfig,
    ) -> np.ndarray:
        """Apply mic simulation."""
        mic_config = MicConfig(
            mic_type=config.mic_type,
            position=config.mic_position,
            distance_meters=config.mic_distance,
        )

        if self._mic is None:
            self._mic = MicSimulator(sample_rate=self.sample_rate)

        return self._mic.process(audio, mic_config)

    def _process_chain(
        self,
        audio: np.ndarray,
        config: AugmentationConfig,
    ) -> np.ndarray:
        """Apply recording chain."""
        chain_config = ChainConfig(
            preamp=PreampConfig(
                preamp_type=config.preamp_type,
                drive=config.preamp_drive,
            ),
            compressor=CompressorConfig(
                enabled=config.compression_enabled,
                ratio=config.compression_ratio,
                threshold_db=config.compression_threshold_db,
            ),
            eq=EQConfig(
                enabled=config.eq_enabled,
                highpass_freq=config.highpass_freq,
                lowpass_freq=config.lowpass_freq,
            ),
        )

        if self._chain is None:
            self._chain = RecordingChain(sample_rate=self.sample_rate)

        return self._chain.process(audio, chain_config)

    def _process_degradation(
        self,
        audio: np.ndarray,
        config: AugmentationConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply degradations."""
        deg_config = DegradationConfig(
            noise=NoiseConfig(
                enabled=config.noise_level_db > -80,
                noise_type=config.noise_type,
                level_db=config.noise_level_db,
            ),
            bit_depth=BitDepthConfig(
                enabled=config.bit_depth is not None,
                bit_depth=config.bit_depth or 16,
            ),
            wow_flutter=config.wow_flutter,
        )

        if self._degrader is None:
            self._degrader = AudioDegrader(
                sample_rate=self.sample_rate,
                noise_directory=self.noise_directory,
            )

        return self._degrader.process(audio, deg_config, seed=int(rng.integers(0, 2**31)))

    def _normalize(
        self,
        audio: np.ndarray,
        target_peak_db: float,
    ) -> np.ndarray:
        """Normalize audio to target peak level."""
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return audio

        target_linear = 10 ** (target_peak_db / 20)
        gain = target_linear / peak

        return audio * gain

    def get_augmentation_params(self) -> dict:
        """Get current augmentation parameters as a dictionary."""
        cfg = self.config
        return {
            "room_type": cfg.room_type.value if cfg.room_enabled else None,
            "room_wet_dry": cfg.room_wet_dry,
            "mic_type": cfg.mic_type.value if cfg.mic_enabled else None,
            "mic_distance": cfg.mic_distance,
            "mic_position": cfg.mic_position.value if cfg.mic_enabled else None,
            "preamp_type": cfg.preamp_type.value if cfg.chain_enabled else None,
            "preamp_drive": cfg.preamp_drive,
            "compression_ratio": cfg.compression_ratio if cfg.compression_enabled else None,
            "compression_threshold_db": (
                cfg.compression_threshold_db if cfg.compression_enabled else None
            ),
            "noise_type": cfg.noise_type.value if cfg.degradation_enabled else None,
            "noise_level_db": cfg.noise_level_db if cfg.degradation_enabled else None,
            "bit_depth": cfg.bit_depth,
        }


def augment_audio(
    audio: np.ndarray,
    preset: AugmentationPreset | None = None,
    config: AugmentationConfig | None = None,
    sample_rate: int = 44100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Convenience function to augment audio.

    Args:
        audio: Input audio array
        preset: Use a preset configuration
        config: Custom configuration (overrides preset)
        sample_rate: Audio sample rate
        seed: Random seed

    Returns:
        Augmented audio
    """
    pipeline = AugmentationPipeline(
        config=config,
        preset=preset,
        sample_rate=sample_rate,
    )

    return pipeline.process(audio, seed=seed)


def random_augmentation(
    audio: np.ndarray,
    sample_rate: int = 44100,
    seed: int | None = None,
) -> tuple[np.ndarray, AugmentationConfig]:
    """
    Apply random augmentation from available presets.

    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        seed: Random seed

    Returns:
        Tuple of (augmented_audio, config_used)
    """
    rng = np.random.default_rng(seed)

    # Random preset (use index to avoid numpy string conversion)
    presets = list(AugmentationPreset)
    preset_idx = rng.choice(len(presets))
    preset = presets[preset_idx]
    config = AugmentationConfig.from_preset(preset)

    # Add some randomization to the preset
    config.room_wet_dry = (config.room_wet_dry or 0.3) * rng.uniform(0.8, 1.2)
    config.mic_distance = config.mic_distance * rng.uniform(0.8, 1.2)
    config.noise_level_db = config.noise_level_db + rng.uniform(-5, 5)

    pipeline = AugmentationPipeline(config=config, sample_rate=sample_rate)
    augmented = pipeline.process(audio, seed=int(rng.integers(0, 2**31)))

    return augmented, config
