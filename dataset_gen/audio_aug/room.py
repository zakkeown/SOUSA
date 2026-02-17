"""
Room simulation using convolution reverb.

This module applies realistic room acoustics to audio using
impulse responses (IRs) from various room types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
from scipy import signal
import soundfile as sf


class RoomType(str, Enum):
    """Types of rooms/spaces for acoustic simulation."""

    PRACTICE_ROOM = "practice_room"  # Small, dry practice space
    STUDIO = "studio"  # Professional recording studio
    CONCERT_HALL = "concert_hall"  # Large concert venue
    GYM = "gym"  # Large gymnasium (common for marching band)
    OUTDOOR = "outdoor"  # Minimal reflections
    BEDROOM = "bedroom"  # Small bedroom/home studio
    GARAGE = "garage"  # Typical garage rehearsal space
    CHURCH = "church"  # High ceilings, long reverb


@dataclass
class RoomConfig:
    """Configuration for room simulation."""

    room_type: RoomType = RoomType.STUDIO
    wet_dry_mix: float = 0.3  # 0 = dry only, 1 = wet only
    ir_path: Path | None = None  # Custom IR path (overrides room_type)

    # Synthetic reverb parameters (used when no IR available)
    decay_time_sec: float = 0.5  # RT60 approximation
    early_reflection_delay_ms: float = 20.0
    pre_delay_ms: float = 10.0

    # Modification parameters
    ir_trim_sec: float | None = None  # Trim IR to this length
    ir_gain: float = 1.0  # Scale IR amplitude


# Default room characteristics for synthetic reverb
ROOM_CHARACTERISTICS = {
    RoomType.PRACTICE_ROOM: {
        "decay_time_sec": 0.3,
        "early_reflection_delay_ms": 8,
        "pre_delay_ms": 2,
        "wet_dry_default": 0.15,
    },
    RoomType.STUDIO: {
        "decay_time_sec": 0.5,
        "early_reflection_delay_ms": 15,
        "pre_delay_ms": 5,
        "wet_dry_default": 0.25,
    },
    RoomType.CONCERT_HALL: {
        "decay_time_sec": 2.0,
        "early_reflection_delay_ms": 40,
        "pre_delay_ms": 20,
        "wet_dry_default": 0.4,
    },
    RoomType.GYM: {
        "decay_time_sec": 1.5,
        "early_reflection_delay_ms": 50,
        "pre_delay_ms": 15,
        "wet_dry_default": 0.35,
    },
    RoomType.OUTDOOR: {
        "decay_time_sec": 0.1,
        "early_reflection_delay_ms": 100,
        "pre_delay_ms": 0,
        "wet_dry_default": 0.05,
    },
    RoomType.BEDROOM: {
        "decay_time_sec": 0.25,
        "early_reflection_delay_ms": 5,
        "pre_delay_ms": 1,
        "wet_dry_default": 0.1,
    },
    RoomType.GARAGE: {
        "decay_time_sec": 0.8,
        "early_reflection_delay_ms": 25,
        "pre_delay_ms": 8,
        "wet_dry_default": 0.3,
    },
    RoomType.CHURCH: {
        "decay_time_sec": 3.0,
        "early_reflection_delay_ms": 60,
        "pre_delay_ms": 30,
        "wet_dry_default": 0.5,
    },
}


class RoomSimulator:
    """
    Apply room acoustics simulation to audio.

    Uses convolution reverb with impulse responses when available,
    or generates synthetic reverb based on room characteristics.
    """

    def __init__(
        self,
        config: RoomConfig | None = None,
        ir_directory: Path | str | None = None,
        sample_rate: int = 44100,
    ):
        """
        Initialize room simulator.

        Args:
            config: Room configuration
            ir_directory: Directory containing impulse response files
            sample_rate: Audio sample rate
        """
        self.config = config or RoomConfig()
        self.ir_directory = Path(ir_directory) if ir_directory else None
        self.sample_rate = sample_rate

        self._ir_cache: dict[str, np.ndarray] = {}

    def process(
        self,
        audio: np.ndarray,
        config: RoomConfig | None = None,
    ) -> np.ndarray:
        """
        Apply room simulation to audio.

        Args:
            audio: Input audio (shape: [samples] or [samples, channels])
            config: Optional override configuration

        Returns:
            Processed audio with room acoustics applied
        """
        cfg = config or self.config

        # Handle mono/stereo
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
            was_mono = True
        else:
            was_mono = False

        # Get or generate impulse response
        ir = self._get_impulse_response(cfg)

        # Apply convolution reverb
        wet = self._convolve(audio, ir)

        # Mix dry and wet signals
        output = self._mix(audio, wet, cfg.wet_dry_mix)

        if was_mono:
            output = output.flatten()

        return output

    def _get_impulse_response(self, config: RoomConfig) -> np.ndarray:
        """Get impulse response for the configured room."""
        # Check for custom IR path
        if config.ir_path is not None:
            return self._load_ir(config.ir_path, config)

        # Check for IR in directory
        if self.ir_directory is not None:
            ir_path = self._find_ir_for_room(config.room_type)
            if ir_path is not None:
                return self._load_ir(ir_path, config)

        # Generate synthetic IR
        return self._generate_synthetic_ir(config)

    def _find_ir_for_room(self, room_type: RoomType) -> Path | None:
        """Find an IR file matching the room type."""
        if self.ir_directory is None or not self.ir_directory.exists():
            return None

        # Look for files matching room type name
        patterns = [
            f"{room_type.value}*.wav",
            f"{room_type.value}*.flac",
            f"*{room_type.value}*.wav",
            f"*{room_type.value}*.flac",
        ]

        for pattern in patterns:
            matches = list(self.ir_directory.glob(pattern))
            if matches:
                return matches[0]

        return None

    def _load_ir(self, path: Path, config: RoomConfig) -> np.ndarray:
        """Load and process an impulse response file."""
        cache_key = f"{path}_{config.ir_trim_sec}_{config.ir_gain}"

        if cache_key in self._ir_cache:
            return self._ir_cache[cache_key]

        # Load IR file
        ir, ir_sr = sf.read(str(path))

        # Resample if needed
        if ir_sr != self.sample_rate:
            num_samples = int(len(ir) * self.sample_rate / ir_sr)
            ir = signal.resample(ir, num_samples)

        # Ensure 2D
        if ir.ndim == 1:
            ir = ir.reshape(-1, 1)

        # Trim if requested
        if config.ir_trim_sec is not None:
            max_samples = int(config.ir_trim_sec * self.sample_rate)
            if len(ir) > max_samples:
                # Apply fade out to avoid clicks
                fade_len = min(1000, max_samples // 10)
                fade = np.linspace(1, 0, fade_len)
                ir = ir[:max_samples]
                ir[-fade_len:] *= fade.reshape(-1, 1) if ir.ndim == 2 else fade

        # Apply gain
        ir = ir * config.ir_gain

        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-8)

        self._ir_cache[cache_key] = ir
        return ir

    def _generate_synthetic_ir(self, config: RoomConfig) -> np.ndarray:
        """Generate a synthetic impulse response."""
        characteristics = ROOM_CHARACTERISTICS.get(
            config.room_type,
            ROOM_CHARACTERISTICS[RoomType.STUDIO],
        )

        decay_time = config.decay_time_sec or characteristics["decay_time_sec"]
        early_delay = (
            config.early_reflection_delay_ms or characteristics["early_reflection_delay_ms"]
        )
        pre_delay = config.pre_delay_ms or characteristics["pre_delay_ms"]

        # IR length based on decay time (4x RT60 for full decay)
        ir_length_sec = decay_time * 4
        ir_length_samples = int(ir_length_sec * self.sample_rate)

        # Create time array
        t = np.arange(ir_length_samples) / self.sample_rate

        # Generate exponential decay envelope
        decay_rate = -6.91 / decay_time  # -60dB at RT60
        envelope = np.exp(decay_rate * t)

        # Generate noise-based reverb tail
        noise = np.random.randn(ir_length_samples) * envelope

        # Add early reflections
        ir = np.zeros(ir_length_samples)

        # Pre-delay
        pre_delay_samples = int(pre_delay / 1000 * self.sample_rate)

        # Direct sound at sample 0 (no pre-delay offset)
        ir[0] = 0.8

        # Early reflections start after pre-delay (relative to direct sound)
        early_samples = int(early_delay / 1000 * self.sample_rate)
        n_early = 6  # Number of early reflections
        for i in range(n_early):
            delay = pre_delay_samples + int((i + 1) * early_samples / n_early)
            amplitude = 0.5 * (1 - i / n_early) * np.random.uniform(0.8, 1.0)
            if delay < ir_length_samples:
                ir[delay] += amplitude * np.random.choice([-1, 1])

        # Diffuse reverb tail starts after early reflections
        tail_start = pre_delay_samples + early_samples
        if tail_start < ir_length_samples:
            ir[tail_start:] += noise[tail_start:] * 0.3

        # Lowpass filter for more natural sound
        b, a = signal.butter(2, 8000 / (self.sample_rate / 2), btype="low")
        ir = signal.filtfilt(b, a, ir)

        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-8)

        # Apply config gain
        ir = ir * config.ir_gain

        # Make stereo
        ir = np.column_stack([ir, ir])

        return ir

    def _convolve(self, audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
        """Convolve audio with impulse response."""
        n_channels = audio.shape[1]
        ir_channels = ir.shape[1] if ir.ndim == 2 else 1

        output_length = len(audio) + len(ir) - 1
        output = np.zeros((output_length, n_channels))

        for ch in range(n_channels):
            ir_ch = ir[:, ch % ir_channels] if ir.ndim == 2 else ir
            output[:, ch] = signal.fftconvolve(audio[:, ch], ir_ch, mode="full")[:output_length]

        # Trim to original length
        output = output[: len(audio)]

        return output

    def _mix(
        self,
        dry: np.ndarray,
        wet: np.ndarray,
        wet_amount: float,
    ) -> np.ndarray:
        """Mix dry and wet signals."""
        # Ensure same length
        min_len = min(len(dry), len(wet))
        dry = dry[:min_len]
        wet = wet[:min_len]

        # Linear crossfade
        output = dry * (1 - wet_amount) + wet * wet_amount

        # Prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        return output

    def get_default_wet_dry(self, room_type: RoomType) -> float:
        """Get default wet/dry mix for a room type."""
        return ROOM_CHARACTERISTICS.get(
            room_type,
            ROOM_CHARACTERISTICS[RoomType.STUDIO],
        )["wet_dry_default"]


def apply_room(
    audio: np.ndarray,
    room_type: RoomType = RoomType.STUDIO,
    wet_dry_mix: float | None = None,
    sample_rate: int = 44100,
    ir_directory: Path | str | None = None,
) -> np.ndarray:
    """
    Convenience function to apply room simulation.

    Args:
        audio: Input audio array
        room_type: Type of room to simulate
        wet_dry_mix: Wet/dry mix (None = use default for room type)
        sample_rate: Audio sample rate
        ir_directory: Directory containing IR files

    Returns:
        Audio with room acoustics applied
    """
    simulator = RoomSimulator(ir_directory=ir_directory, sample_rate=sample_rate)

    if wet_dry_mix is None:
        wet_dry_mix = simulator.get_default_wet_dry(room_type)

    config = RoomConfig(
        room_type=room_type,
        wet_dry_mix=wet_dry_mix,
    )

    return simulator.process(audio, config)
