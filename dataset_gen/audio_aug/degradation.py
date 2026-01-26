"""
Audio degradation for simulating various recording conditions.

This module adds realistic degradations including noise,
bit depth reduction, sample rate artifacts, and more.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
from scipy import signal
import soundfile as sf


class NoiseType(str, Enum):
    """Types of background noise."""

    WHITE = "white"  # Flat spectrum noise
    PINK = "pink"  # 1/f spectrum, natural sounding
    BROWN = "brown"  # 1/f^2 spectrum, rumble
    HVAC = "hvac"  # HVAC/air conditioning hum
    ROOM_TONE = "room_tone"  # General room ambience
    TAPE_HISS = "tape_hiss"  # Tape-style hiss
    VINYL = "vinyl"  # Record crackle and pops


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""

    enabled: bool = True
    noise_type: NoiseType = NoiseType.PINK
    level_db: float = -40.0  # Noise level relative to signal
    noise_file: Path | None = None  # Optional external noise file


@dataclass
class BitDepthConfig:
    """Configuration for bit depth reduction."""

    enabled: bool = False
    bit_depth: int = 16  # Target bit depth (24, 16, 12, 8)
    dither: bool = True  # Apply dither before quantization


@dataclass
class SampleRateConfig:
    """Configuration for sample rate degradation."""

    enabled: bool = False
    target_rate: int = 22050  # Target sample rate
    anti_alias: bool = True  # Apply anti-aliasing filter


@dataclass
class DegradationConfig:
    """Full degradation configuration."""

    noise: NoiseConfig | None = None
    bit_depth: BitDepthConfig | None = None
    sample_rate: SampleRateConfig | None = None

    # Additional degradations
    dc_offset: float = 0.0  # DC offset to add (-1 to 1)
    phase_shift: float = 0.0  # Phase shift between channels (0-1)
    wow_flutter: float = 0.0  # Tape-style pitch variation (0-1)
    dropout_probability: float = 0.0  # Probability of brief dropouts

    def __post_init__(self):
        if self.noise is None:
            self.noise = NoiseConfig(enabled=False)
        if self.bit_depth is None:
            self.bit_depth = BitDepthConfig()
        if self.sample_rate is None:
            self.sample_rate = SampleRateConfig()


class AudioDegrader:
    """
    Apply various degradations to audio.

    Simulates imperfect recording conditions, equipment limitations,
    and transmission artifacts.
    """

    def __init__(
        self,
        config: DegradationConfig | None = None,
        sample_rate: int = 44100,
        noise_directory: Path | str | None = None,
    ):
        """
        Initialize audio degrader.

        Args:
            config: Degradation configuration
            sample_rate: Audio sample rate
            noise_directory: Directory containing noise profile files
        """
        self.config = config or DegradationConfig()
        self.sample_rate = sample_rate
        self.noise_directory = Path(noise_directory) if noise_directory else None

        self._noise_cache: dict[str, np.ndarray] = {}

    def process(
        self,
        audio: np.ndarray,
        config: DegradationConfig | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Apply degradations to audio.

        Args:
            audio: Input audio (shape: [samples] or [samples, channels])
            config: Optional override configuration
            seed: Random seed for reproducible degradations

        Returns:
            Degraded audio
        """
        cfg = config or self.config
        rng = np.random.default_rng(seed)

        # Handle mono/stereo
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
            was_mono = True
        else:
            was_mono = False

        output = audio.copy()

        # Apply wow and flutter (before other processing)
        if cfg.wow_flutter > 0:
            output = self._apply_wow_flutter(output, cfg.wow_flutter, rng)

        # Apply dropouts
        if cfg.dropout_probability > 0:
            output = self._apply_dropouts(output, cfg.dropout_probability, rng)

        # Apply sample rate degradation
        if cfg.sample_rate.enabled:
            output = self._apply_sample_rate_degradation(output, cfg.sample_rate)

        # Apply bit depth reduction
        if cfg.bit_depth.enabled:
            output = self._apply_bit_depth_reduction(output, cfg.bit_depth, rng)

        # Add noise
        if cfg.noise.enabled:
            output = self._add_noise(output, cfg.noise, rng)

        # Apply DC offset
        if cfg.dc_offset != 0:
            output = output + cfg.dc_offset

        # Apply phase shift between channels
        if cfg.phase_shift > 0 and output.shape[1] > 1:
            output = self._apply_phase_shift(output, cfg.phase_shift)

        # Clip to valid range
        output = np.clip(output, -1.0, 1.0)

        if was_mono:
            output = output.flatten()

        return output

    def _add_noise(
        self,
        audio: np.ndarray,
        config: NoiseConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add background noise to audio."""
        # Generate or load noise
        if config.noise_file is not None:
            noise = self._load_noise_file(config.noise_file, len(audio))
        else:
            noise = self._generate_noise(config.noise_type, audio.shape, rng)

        # Calculate noise level
        signal_rms = np.sqrt(np.mean(audio**2)) + 1e-10
        noise_level = signal_rms * (10 ** (config.level_db / 20))

        # Normalize and scale noise
        noise_rms = np.sqrt(np.mean(noise**2)) + 1e-10
        noise = noise / noise_rms * noise_level

        return audio + noise

    def _generate_noise(
        self,
        noise_type: NoiseType,
        shape: tuple,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate synthetic noise."""
        if noise_type == NoiseType.WHITE:
            return rng.standard_normal(shape)

        elif noise_type == NoiseType.PINK:
            return self._generate_pink_noise(shape, rng)

        elif noise_type == NoiseType.BROWN:
            return self._generate_brown_noise(shape, rng)

        elif noise_type == NoiseType.HVAC:
            return self._generate_hvac_noise(shape, rng)

        elif noise_type == NoiseType.ROOM_TONE:
            return self._generate_room_tone(shape, rng)

        elif noise_type == NoiseType.TAPE_HISS:
            return self._generate_tape_hiss(shape, rng)

        elif noise_type == NoiseType.VINYL:
            return self._generate_vinyl_noise(shape, rng)

        return rng.standard_normal(shape)

    def _generate_pink_noise(
        self,
        shape: tuple,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        white = rng.standard_normal(shape)

        # Voss-McCartney algorithm approximation
        num_rows = 16
        array = np.zeros((num_rows, shape[0]))
        for i in range(num_rows):
            array[i] = rng.standard_normal(shape[0])

        pink = np.zeros(shape)
        for ch in range(shape[1] if len(shape) > 1 else 1):
            running_sum = np.zeros(shape[0])
            for i in range(num_rows):
                # Update random row
                mask = rng.random(shape[0]) < 1 / (2**i)
                array[i] = np.where(mask, rng.standard_normal(shape[0]), array[i])
                running_sum += array[i]

            if len(shape) > 1:
                pink[:, ch] = running_sum / num_rows
            else:
                pink = running_sum / num_rows

        return pink

    def _generate_brown_noise(
        self,
        shape: tuple,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate brown noise (1/f^2 spectrum, random walk)."""
        white = rng.standard_normal(shape)

        # Integrate white noise
        brown = np.cumsum(white, axis=0)

        # Remove DC and normalize
        brown = brown - np.mean(brown, axis=0)
        brown = brown / (np.max(np.abs(brown)) + 1e-10)

        return brown

    def _generate_hvac_noise(
        self,
        shape: tuple,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate HVAC-style noise (low frequency hum + broadband)."""
        # Low frequency hum (50/60 Hz and harmonics)
        t = np.arange(shape[0]) / self.sample_rate
        hum = np.sin(2 * np.pi * 60 * t) * 0.3
        hum += np.sin(2 * np.pi * 120 * t) * 0.1
        hum += np.sin(2 * np.pi * 180 * t) * 0.05

        # Add broadband component
        broadband = self._generate_pink_noise(shape, rng) * 0.5

        # Lowpass filter for rumble character
        b, a = signal.butter(2, 500 / (self.sample_rate / 2), btype="low")

        noise = np.zeros(shape)
        for ch in range(shape[1] if len(shape) > 1 else 1):
            filtered = signal.filtfilt(b, a, broadband[:, ch] if len(shape) > 1 else broadband)
            if len(shape) > 1:
                noise[:, ch] = hum + filtered
            else:
                noise = hum + filtered

        return noise

    def _generate_room_tone(
        self,
        shape: tuple,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate room tone (low-level ambient noise)."""
        # Combination of filtered noises
        pink = self._generate_pink_noise(shape, rng)

        # Bandpass for typical room frequencies
        b, a = signal.butter(2, [100, 4000], btype="band", fs=self.sample_rate)

        noise = np.zeros(shape)
        for ch in range(shape[1] if len(shape) > 1 else 1):
            if len(shape) > 1:
                noise[:, ch] = signal.filtfilt(b, a, pink[:, ch])
            else:
                noise = signal.filtfilt(b, a, pink)

        return noise

    def _generate_tape_hiss(
        self,
        shape: tuple,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate tape hiss (high-frequency emphasis)."""
        white = rng.standard_normal(shape)

        # Highpass + slight high shelf
        b, a = signal.butter(2, 2000 / (self.sample_rate / 2), btype="high")

        noise = np.zeros(shape)
        for ch in range(shape[1] if len(shape) > 1 else 1):
            if len(shape) > 1:
                noise[:, ch] = signal.filtfilt(b, a, white[:, ch])
            else:
                noise = signal.filtfilt(b, a, white)

        return noise

    def _generate_vinyl_noise(
        self,
        shape: tuple,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate vinyl noise (crackle and pops)."""
        # Base rumble
        rumble = self._generate_brown_noise(shape, rng) * 0.3

        # Random pops and crackles
        pops = np.zeros(shape)
        n_pops = int(shape[0] / self.sample_rate * 10)  # ~10 pops per second

        for _ in range(n_pops):
            pos = rng.integers(0, shape[0])
            amplitude = rng.uniform(0.1, 0.5)
            duration = rng.integers(10, 50)

            # Create pop shape
            end_pos = min(pos + duration, shape[0])
            pop_shape = np.exp(-np.arange(end_pos - pos) / 5)

            if len(shape) > 1:
                for ch in range(shape[1]):
                    pops[pos:end_pos, ch] += pop_shape * amplitude * rng.choice([-1, 1])
            else:
                pops[pos:end_pos] += pop_shape * amplitude * rng.choice([-1, 1])

        return rumble + pops

    def _load_noise_file(self, path: Path, target_length: int) -> np.ndarray:
        """Load and prepare external noise file."""
        cache_key = str(path)

        if cache_key not in self._noise_cache:
            noise, sr = sf.read(str(path))

            # Resample if needed
            if sr != self.sample_rate:
                num_samples = int(len(noise) * self.sample_rate / sr)
                noise = signal.resample(noise, num_samples)

            # Ensure 2D
            if noise.ndim == 1:
                noise = noise.reshape(-1, 1)

            self._noise_cache[cache_key] = noise

        noise = self._noise_cache[cache_key]

        # Loop or trim to target length
        if len(noise) >= target_length:
            noise = noise[:target_length]
        else:
            # Loop noise
            repeats = int(np.ceil(target_length / len(noise)))
            noise = np.tile(noise, (repeats, 1))[:target_length]

        return noise

    def _apply_bit_depth_reduction(
        self,
        audio: np.ndarray,
        config: BitDepthConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Reduce bit depth with optional dithering."""
        bit_depth = config.bit_depth
        max_val = 2 ** (bit_depth - 1) - 1

        # Add dither before quantization
        if config.dither:
            # TPDF dither
            dither = (rng.random(audio.shape) + rng.random(audio.shape) - 1) / max_val
            audio = audio + dither

        # Quantize
        quantized = np.round(audio * max_val) / max_val

        return quantized

    def _apply_sample_rate_degradation(
        self,
        audio: np.ndarray,
        config: SampleRateConfig,
    ) -> np.ndarray:
        """Apply sample rate degradation."""
        target_rate = config.target_rate

        if target_rate >= self.sample_rate:
            return audio

        # Downsample
        if config.anti_alias:
            # Anti-aliasing filter
            nyquist = target_rate / 2
            b, a = signal.butter(5, nyquist / (self.sample_rate / 2), btype="low")
            audio = signal.filtfilt(b, a, audio, axis=0)

        # Resample down and back up (introduces aliasing artifacts)
        down_samples = int(len(audio) * target_rate / self.sample_rate)
        downsampled = signal.resample(audio, down_samples, axis=0)
        upsampled = signal.resample(downsampled, len(audio), axis=0)

        return upsampled

    def _apply_wow_flutter(
        self,
        audio: np.ndarray,
        amount: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply wow and flutter (pitch/time variation)."""
        # Generate slow modulation (wow) + fast modulation (flutter)
        t = np.arange(len(audio)) / self.sample_rate

        # Wow: 0.5-2 Hz variation
        wow_freq = rng.uniform(0.5, 2)
        wow = np.sin(2 * np.pi * wow_freq * t) * amount * 0.002

        # Flutter: 5-15 Hz variation
        flutter_freq = rng.uniform(5, 15)
        flutter = np.sin(2 * np.pi * flutter_freq * t) * amount * 0.0005

        # Combined modulation
        modulation = 1 + wow + flutter

        # Apply via resampling
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            # Create warped time index
            warped_idx = np.cumsum(modulation)
            warped_idx = warped_idx / warped_idx[-1] * (len(audio) - 1)
            output[:, ch] = np.interp(np.arange(len(audio)), warped_idx, audio[:, ch])

        return output

    def _apply_dropouts(
        self,
        audio: np.ndarray,
        probability: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply brief audio dropouts."""
        output = audio.copy()

        # Average dropout length: 10-50ms
        dropout_samples_min = int(0.01 * self.sample_rate)
        dropout_samples_max = int(0.05 * self.sample_rate)

        # Number of potential dropouts
        n_windows = int(len(audio) / dropout_samples_min)

        for _ in range(n_windows):
            if rng.random() < probability:
                start = rng.integers(0, len(audio) - dropout_samples_max)
                length = rng.integers(dropout_samples_min, dropout_samples_max)
                end = min(start + length, len(audio))

                # Fade out/in for smooth dropout
                fade_len = min(100, length // 4)
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = np.linspace(0, 1, fade_len)

                output[start : start + fade_len] *= fade_out.reshape(-1, 1)
                output[start + fade_len : end - fade_len] = 0
                output[end - fade_len : end] *= fade_in.reshape(-1, 1)

        return output

    def _apply_phase_shift(
        self,
        audio: np.ndarray,
        shift: float,
    ) -> np.ndarray:
        """Apply phase shift between stereo channels."""
        if audio.shape[1] < 2:
            return audio

        # Shift in samples
        shift_samples = int(shift * 100)  # Max ~100 samples at shift=1

        output = audio.copy()

        # Delay right channel
        if shift_samples > 0:
            output[shift_samples:, 1] = audio[:-shift_samples, 1]
            output[:shift_samples, 1] = 0

        return output


def add_noise(
    audio: np.ndarray,
    noise_type: NoiseType = NoiseType.PINK,
    level_db: float = -40.0,
    sample_rate: int = 44100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Convenience function to add noise to audio.

    Args:
        audio: Input audio array
        noise_type: Type of noise to add
        level_db: Noise level in dB relative to signal
        sample_rate: Audio sample rate
        seed: Random seed

    Returns:
        Audio with noise added
    """
    config = DegradationConfig(
        noise=NoiseConfig(
            enabled=True,
            noise_type=noise_type,
            level_db=level_db,
        ),
    )

    degrader = AudioDegrader(config, sample_rate)
    return degrader.process(audio, seed=seed)
