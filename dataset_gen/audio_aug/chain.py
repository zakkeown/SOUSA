"""
Recording chain simulation (preamp, compression, EQ).

This module simulates the analog recording chain including
preamp coloration, compression, and master EQ.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import signal

try:
    import pedalboard

    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False


class PreampType(str, Enum):
    """Types of preamp character."""

    CLEAN = "clean"  # Transparent, minimal coloration
    WARM = "warm"  # Tube-style warmth, soft saturation
    AGGRESSIVE = "aggressive"  # Solid-state, harder clipping
    VINTAGE = "vintage"  # Transformer saturation, frequency bumps


@dataclass
class PreampConfig:
    """Configuration for preamp simulation."""

    preamp_type: PreampType = PreampType.CLEAN
    gain_db: float = 0.0  # Input gain
    drive: float = 0.0  # Saturation amount (0-1)
    output_gain_db: float = 0.0  # Output level


@dataclass
class CompressorConfig:
    """Configuration for dynamics compression."""

    enabled: bool = True
    threshold_db: float = -12.0  # Threshold level
    ratio: float = 4.0  # Compression ratio (e.g., 4:1)
    attack_ms: float = 10.0  # Attack time
    release_ms: float = 100.0  # Release time
    knee_db: float = 6.0  # Soft knee width
    makeup_gain_db: float = 0.0  # Makeup gain


@dataclass
class EQConfig:
    """Configuration for master EQ."""

    enabled: bool = True

    # High-pass filter
    highpass_freq: float = 40.0  # Hz
    highpass_enabled: bool = True

    # Low-pass filter
    lowpass_freq: float = 18000.0  # Hz
    lowpass_enabled: bool = True

    # Parametric bands (freq_hz, gain_db, q)
    low_shelf: tuple[float, float] = (100.0, 0.0)  # freq, gain
    high_shelf: tuple[float, float] = (8000.0, 0.0)  # freq, gain


@dataclass
class ChainConfig:
    """Full recording chain configuration."""

    preamp: PreampConfig | None = None
    compressor: CompressorConfig | None = None
    eq: EQConfig | None = None

    def __post_init__(self):
        if self.preamp is None:
            self.preamp = PreampConfig()
        if self.compressor is None:
            self.compressor = CompressorConfig()
        if self.eq is None:
            self.eq = EQConfig()


class RecordingChain:
    """
    Simulate a recording chain with preamp, compression, and EQ.

    Processing order: Input -> Preamp -> Compressor -> EQ -> Output
    """

    def __init__(
        self,
        config: ChainConfig | None = None,
        sample_rate: int = 44100,
    ):
        """
        Initialize recording chain.

        Args:
            config: Chain configuration
            sample_rate: Audio sample rate
        """
        self.config = config or ChainConfig()
        self.sample_rate = sample_rate

    def process(
        self,
        audio: np.ndarray,
        config: ChainConfig | None = None,
    ) -> np.ndarray:
        """
        Process audio through the recording chain.

        Args:
            audio: Input audio (shape: [samples] or [samples, channels])
            config: Optional override configuration

        Returns:
            Processed audio
        """
        cfg = config or self.config

        # Handle mono/stereo
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
            was_mono = True
        else:
            was_mono = False

        output = audio.copy()

        # Apply preamp
        if cfg.preamp is not None:
            output = self._apply_preamp(output, cfg.preamp)

        # Apply compression
        if cfg.compressor is not None and cfg.compressor.enabled:
            output = self._apply_compression(output, cfg.compressor)

        # Apply EQ
        if cfg.eq is not None and cfg.eq.enabled:
            output = self._apply_eq(output, cfg.eq)

        # Prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.99

        if was_mono:
            output = output.flatten()

        return output

    def _apply_preamp(
        self,
        audio: np.ndarray,
        config: PreampConfig,
    ) -> np.ndarray:
        """Apply preamp processing with optional saturation."""
        # Apply input gain
        gain_linear = 10 ** (config.gain_db / 20)
        audio = audio * gain_linear

        # Apply drive/saturation based on preamp type
        if config.drive > 0:
            audio = self._apply_saturation(audio, config.preamp_type, config.drive)

        # Apply output gain
        output_gain = 10 ** (config.output_gain_db / 20)
        audio = audio * output_gain

        return audio

    def _apply_saturation(
        self,
        audio: np.ndarray,
        preamp_type: PreampType,
        drive: float,
    ) -> np.ndarray:
        """Apply saturation based on preamp type."""
        if preamp_type == PreampType.CLEAN:
            # Minimal saturation even at high drive
            return self._soft_clip(audio, drive * 0.3)

        elif preamp_type == PreampType.WARM:
            # Tube-style saturation: soft, asymmetric clipping
            # Add even harmonics (warmth)
            saturated = self._tube_saturation(audio, drive)

            # Slight low-mid bump characteristic of tube preamps
            b, a = signal.butter(1, [100, 400], btype="band", fs=self.sample_rate)
            bump = signal.filtfilt(b, a, audio, axis=0)
            saturated = saturated + bump * drive * 0.1

            return saturated

        elif preamp_type == PreampType.AGGRESSIVE:
            # Solid-state: harder clipping, more odd harmonics
            return self._hard_clip(audio, drive)

        elif preamp_type == PreampType.VINTAGE:
            # Transformer saturation: soft limiting + frequency response
            saturated = self._soft_clip(audio, drive)

            # Vintage transformer frequency character
            # Slight bass roll-off and high-mid presence
            b_low, a_low = signal.butter(1, 80 / (self.sample_rate / 2), btype="high")
            saturated = signal.filtfilt(b_low, a_low, saturated, axis=0)

            return saturated

        return audio

    def _soft_clip(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Soft clipping saturation (tanh-based)."""
        if amount <= 0:
            return audio

        # Scale input based on drive amount
        drive_scale = 1 + amount * 3
        driven = audio * drive_scale

        # Apply tanh soft clipping
        clipped = np.tanh(driven)

        # Mix dry/wet based on amount
        output = audio * (1 - amount) + clipped * amount

        return output

    def _hard_clip(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Hard clipping saturation."""
        if amount <= 0:
            return audio

        # Calculate clipping threshold (lower = more clipping)
        threshold = 1.0 - amount * 0.7

        # Apply hard clipping
        clipped = np.clip(audio * (1 + amount * 2), -threshold, threshold)

        # Normalize back
        clipped = clipped / threshold

        # Mix with original
        output = audio * (1 - amount * 0.5) + clipped * amount * 0.5

        return output

    def _tube_saturation(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Asymmetric tube-style saturation."""
        if amount <= 0:
            return audio

        drive_scale = 1 + amount * 4

        # Asymmetric waveshaping (adds even harmonics)
        positive = np.maximum(audio * drive_scale, 0)
        negative = np.minimum(audio * drive_scale, 0)

        # Different curves for positive/negative half
        positive = np.tanh(positive)
        negative = np.tanh(negative * 0.9) * 1.1  # Slight asymmetry

        saturated = positive + negative

        # Mix
        output = audio * (1 - amount) + saturated * amount

        return output

    def _apply_compression(
        self,
        audio: np.ndarray,
        config: CompressorConfig,
    ) -> np.ndarray:
        """Apply dynamics compression."""
        if PEDALBOARD_AVAILABLE:
            return self._apply_compression_pedalboard(audio, config)
        else:
            return self._apply_compression_simple(audio, config)

    def _apply_compression_pedalboard(
        self,
        audio: np.ndarray,
        config: CompressorConfig,
    ) -> np.ndarray:
        """Apply compression using pedalboard library."""
        compressor = pedalboard.Compressor(
            threshold_db=config.threshold_db,
            ratio=config.ratio,
            attack_ms=config.attack_ms,
            release_ms=config.release_ms,
        )

        # Pedalboard expects float32 and (samples, channels) shape
        audio_f32 = audio.astype(np.float32)

        # Process
        compressed = compressor(audio_f32, self.sample_rate)

        # Apply makeup gain
        makeup = 10 ** (config.makeup_gain_db / 20)
        compressed = compressed * makeup

        return compressed

    def _apply_compression_simple(
        self,
        audio: np.ndarray,
        config: CompressorConfig,
    ) -> np.ndarray:
        """Simple envelope-following compression (fallback)."""
        threshold = 10 ** (config.threshold_db / 20)
        ratio = config.ratio

        # Time constants
        attack_samples = int(config.attack_ms * self.sample_rate / 1000)
        release_samples = int(config.release_ms * self.sample_rate / 1000)

        attack_coef = np.exp(-1 / max(attack_samples, 1))
        release_coef = np.exp(-1 / max(release_samples, 1))

        # Process each channel
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            channel = audio[:, ch]

            # Envelope follower
            envelope = np.zeros(len(channel))
            env_val = 0.0

            for i, sample in enumerate(np.abs(channel)):
                if sample > env_val:
                    env_val = attack_coef * env_val + (1 - attack_coef) * sample
                else:
                    env_val = release_coef * env_val + (1 - release_coef) * sample
                envelope[i] = env_val

            # Calculate gain reduction
            gain = np.ones(len(envelope))
            above_threshold = envelope > threshold

            if config.knee_db > 0:
                # Soft knee
                knee_start = threshold * 10 ** (-config.knee_db / 40)
                knee_end = threshold * 10 ** (config.knee_db / 40)
                in_knee = (envelope >= knee_start) & (envelope <= knee_end)

                # Linear interpolation in knee region
                knee_ratio = (envelope[in_knee] - knee_start) / (knee_end - knee_start + 1e-10)
                knee_compression = 1 + knee_ratio * (ratio - 1)

                gain[in_knee] = (threshold / envelope[in_knee]) ** (1 - 1 / knee_compression)

            # Above knee
            hard_above = envelope > threshold * 10 ** (config.knee_db / 40)
            gain[hard_above] = (threshold / envelope[hard_above]) ** (1 - 1 / ratio)

            output[:, ch] = channel * gain

        # Apply makeup gain
        makeup = 10 ** (config.makeup_gain_db / 20)
        output = output * makeup

        return output

    def _apply_eq(self, audio: np.ndarray, config: EQConfig) -> np.ndarray:
        """Apply master EQ."""
        output = audio.copy()

        # High-pass filter
        if config.highpass_enabled and config.highpass_freq > 0:
            b, a = signal.butter(2, config.highpass_freq / (self.sample_rate / 2), btype="high")
            for ch in range(output.shape[1]):
                output[:, ch] = signal.filtfilt(b, a, output[:, ch])

        # Low-pass filter
        if config.lowpass_enabled and config.lowpass_freq < self.sample_rate / 2:
            b, a = signal.butter(2, config.lowpass_freq / (self.sample_rate / 2), btype="low")
            for ch in range(output.shape[1]):
                output[:, ch] = signal.filtfilt(b, a, output[:, ch])

        # Low shelf
        if config.low_shelf[1] != 0:
            output = self._apply_shelf(
                output, config.low_shelf[0], config.low_shelf[1], shelf_type="low"
            )

        # High shelf
        if config.high_shelf[1] != 0:
            output = self._apply_shelf(
                output, config.high_shelf[0], config.high_shelf[1], shelf_type="high"
            )

        return output

    def _apply_shelf(
        self,
        audio: np.ndarray,
        freq: float,
        gain_db: float,
        shelf_type: str = "low",
    ) -> np.ndarray:
        """Apply shelving EQ."""
        # Simple shelf implementation using lowpass/highpass and mixing
        w0 = freq / (self.sample_rate / 2)
        gain_linear = 10 ** (gain_db / 20)

        if shelf_type == "low":
            b, a = signal.butter(1, w0, btype="low")
        else:
            b, a = signal.butter(1, w0, btype="high")

        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            band = signal.filtfilt(b, a, audio[:, ch])
            # Boost/cut the band content
            output[:, ch] = audio[:, ch] + band * (gain_linear - 1)

        return output


def apply_chain(
    audio: np.ndarray,
    preamp_type: PreampType = PreampType.CLEAN,
    drive: float = 0.0,
    compression_ratio: float = 4.0,
    compression_threshold_db: float = -12.0,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Convenience function to apply recording chain.

    Args:
        audio: Input audio array
        preamp_type: Type of preamp character
        drive: Preamp drive/saturation (0-1)
        compression_ratio: Compressor ratio
        compression_threshold_db: Compressor threshold
        sample_rate: Audio sample rate

    Returns:
        Processed audio
    """
    config = ChainConfig(
        preamp=PreampConfig(
            preamp_type=preamp_type,
            drive=drive,
        ),
        compressor=CompressorConfig(
            ratio=compression_ratio,
            threshold_db=compression_threshold_db,
        ),
    )

    chain = RecordingChain(config, sample_rate)
    return chain.process(audio)
