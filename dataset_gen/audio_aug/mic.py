"""
Microphone simulation for realistic recording characteristics.

This module simulates different microphone types, distances,
and positions to add realistic recording characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import signal


class MicType(str, Enum):
    """Types of microphones with different frequency characteristics."""

    DYNAMIC = "dynamic"  # SM57-style, presence peak, rolled-off highs
    CONDENSER = "condenser"  # Flat response, detailed highs
    RIBBON = "ribbon"  # Warm, rolled-off highs, figure-8 pattern
    PIEZO = "piezo"  # Contact mic, harsh mids, limited frequency response


class MicPosition(str, Enum):
    """Microphone position relative to sound source."""

    CENTER = "center"  # On-axis, direct sound
    OFF_AXIS = "off_axis"  # Slightly off-axis, reduced highs
    OVERHEAD = "overhead"  # Above source, more room sound
    DISTANT = "distant"  # Far from source, ambient capture


@dataclass
class MicConfig:
    """Configuration for microphone simulation."""

    mic_type: MicType = MicType.CONDENSER
    position: MicPosition = MicPosition.CENTER
    distance_meters: float = 0.5  # Distance from source

    # Proximity effect (bass boost at close distances)
    proximity_effect: bool = True

    # High frequency rolloff for distance
    distance_rolloff: bool = True

    # Self-noise level (dB below signal)
    self_noise_db: float = -80.0

    # Output gain adjustment
    output_gain: float = 1.0


# Frequency response characteristics for each mic type
# Defined as (frequency, gain_db) points for interpolation
MIC_FREQUENCY_RESPONSES = {
    MicType.DYNAMIC: [
        (20, -6),
        (80, -3),
        (200, 0),
        (1000, 0),
        (3000, 3),  # Presence peak
        (5000, 2),
        (8000, 0),
        (12000, -3),
        (16000, -8),
        (20000, -15),
    ],
    MicType.CONDENSER: [
        (20, -3),
        (50, 0),
        (200, 0),
        (1000, 0),
        (5000, 1),  # Slight air boost
        (10000, 2),
        (15000, 1),
        (20000, -2),
    ],
    MicType.RIBBON: [
        (20, -2),
        (100, 0),
        (500, 1),  # Slight warmth
        (1000, 0),
        (3000, -1),
        (5000, -2),
        (8000, -4),
        (12000, -8),
        (16000, -15),
        (20000, -25),
    ],
    MicType.PIEZO: [
        (20, -15),
        (100, -8),
        (300, -2),
        (800, 3),  # Harsh mid presence
        (2000, 5),
        (4000, 2),
        (6000, -2),
        (10000, -8),
        (16000, -15),
        (20000, -25),
    ],
}


class MicSimulator:
    """
    Simulate microphone characteristics on audio.

    Applies frequency response curves, proximity effect,
    distance-based filtering, and mic self-noise.
    """

    def __init__(
        self,
        config: MicConfig | None = None,
        sample_rate: int = 44100,
    ):
        """
        Initialize microphone simulator.

        Args:
            config: Microphone configuration
            sample_rate: Audio sample rate
        """
        self.config = config or MicConfig()
        self.sample_rate = sample_rate

        # Pre-compute filter for default config
        self._filter_cache: dict[str, tuple] = {}

    def process(
        self,
        audio: np.ndarray,
        config: MicConfig | None = None,
    ) -> np.ndarray:
        """
        Apply microphone simulation to audio.

        Args:
            audio: Input audio (shape: [samples] or [samples, channels])
            config: Optional override configuration

        Returns:
            Audio with microphone characteristics applied
        """
        cfg = config or self.config

        # Handle mono/stereo
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
            was_mono = True
        else:
            was_mono = False

        output = audio.copy()

        # Apply mic type frequency response
        output = self._apply_frequency_response(output, cfg.mic_type)

        # Apply position-based filtering
        output = self._apply_position_effect(output, cfg.position)

        # Apply proximity effect for close mics
        if cfg.proximity_effect and cfg.distance_meters < 0.3:
            output = self._apply_proximity_effect(output, cfg.distance_meters)

        # Apply distance rolloff
        if cfg.distance_rolloff:
            output = self._apply_distance_rolloff(output, cfg.distance_meters)

        # Add self-noise
        if cfg.self_noise_db > -100:
            output = self._add_self_noise(output, cfg.self_noise_db)

        # Apply output gain
        output = output * cfg.output_gain

        # Prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        if was_mono:
            output = output.flatten()

        return output

    def _apply_frequency_response(
        self,
        audio: np.ndarray,
        mic_type: MicType,
    ) -> np.ndarray:
        """Apply mic type frequency response curve."""
        response = MIC_FREQUENCY_RESPONSES[mic_type]
        freqs = np.array([f for f, g in response])
        gains_db = np.array([g for f, g in response])

        # Create FIR filter from frequency response
        filter_order = 255
        nyquist = self.sample_rate / 2

        # Interpolate to get response at filter frequencies
        filter_freqs = np.linspace(0, nyquist, filter_order // 2 + 1)

        # Clip frequencies to valid range
        freqs_clipped = np.clip(freqs, 0, nyquist)
        gains_interp = np.interp(filter_freqs, freqs_clipped, gains_db)

        # Convert dB to linear
        gains_linear = 10 ** (gains_interp / 20)

        # Create symmetric frequency response for FIR
        freq_response = np.concatenate([gains_linear, gains_linear[-2:0:-1]])

        # Design FIR filter via inverse FFT
        impulse = np.fft.irfft(freq_response, n=filter_order)
        impulse = np.fft.fftshift(impulse)

        # Apply window
        window = signal.windows.hann(filter_order)
        impulse = impulse * window

        # Apply filter to each channel
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            output[:, ch] = signal.filtfilt(impulse, 1, audio[:, ch])

        return output

    def _apply_position_effect(
        self,
        audio: np.ndarray,
        position: MicPosition,
    ) -> np.ndarray:
        """Apply effects based on mic position."""
        if position == MicPosition.CENTER:
            # On-axis: no additional processing
            return audio

        elif position == MicPosition.OFF_AXIS:
            # Off-axis: reduce high frequencies
            b, a = signal.butter(2, 10000 / (self.sample_rate / 2), btype="low")
            output = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                output[:, ch] = signal.filtfilt(b, a, audio[:, ch])
            return output * 0.95  # Slight level reduction

        elif position == MicPosition.OVERHEAD:
            # Overhead: more room, less direct attack
            # Gentle high-shelf cut + slight delay simulation
            b, a = signal.butter(1, 5000 / (self.sample_rate / 2), btype="low")
            output = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                output[:, ch] = signal.filtfilt(b, a, audio[:, ch])
            return output * 0.85

        elif position == MicPosition.DISTANT:
            # Distant: significant high rolloff, reduced transients
            b, a = signal.butter(2, 3000 / (self.sample_rate / 2), btype="low")
            output = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                output[:, ch] = signal.filtfilt(b, a, audio[:, ch])
            return output * 0.6

        return audio

    def _apply_proximity_effect(
        self,
        audio: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        """Apply bass boost for close-miking (proximity effect)."""
        # More boost the closer the mic
        boost_db = max(0, (0.3 - distance) * 20)  # Up to 6dB at 0m

        if boost_db <= 0:
            return audio

        # Low shelf filter
        boost_linear = 10 ** (boost_db / 20)

        # Create low shelf at ~200Hz
        w0 = 200 / (self.sample_rate / 2)
        b, a = signal.butter(1, w0, btype="low")

        # Apply filter and add to original
        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            low = signal.filtfilt(b, a, audio[:, ch])
            output[:, ch] = audio[:, ch] + low * (boost_linear - 1)

        return output

    def _apply_distance_rolloff(
        self,
        audio: np.ndarray,
        distance: float,
    ) -> np.ndarray:
        """Apply high frequency rolloff based on distance."""
        if distance < 0.3:
            return audio  # Close mic, minimal rolloff

        # Calculate rolloff frequency based on distance
        # Further = more rolloff
        rolloff_freq = 20000 / (1 + distance * 2)
        rolloff_freq = max(3000, min(15000, rolloff_freq))

        b, a = signal.butter(1, rolloff_freq / (self.sample_rate / 2), btype="low")

        output = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            output[:, ch] = signal.filtfilt(b, a, audio[:, ch])

        # Apply inverse square law for amplitude
        amplitude_factor = 1 / (1 + distance**1.5)
        output = output * amplitude_factor

        return output

    def _add_self_noise(
        self,
        audio: np.ndarray,
        noise_db: float,
    ) -> np.ndarray:
        """Add microphone self-noise."""
        # Calculate noise level relative to signal
        signal_rms = np.sqrt(np.mean(audio**2))
        noise_level = signal_rms * (10 ** (noise_db / 20))

        # Generate pink noise (more realistic than white)
        noise = self._generate_pink_noise(audio.shape)
        noise = noise * noise_level

        return audio + noise

    def _generate_pink_noise(self, shape: tuple) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        # Generate white noise
        white = np.random.randn(*shape)

        # Apply 1/f filter
        # Simple approximation using cascaded first-order filters
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])

        pink = np.zeros_like(white)
        for ch in range(shape[1] if len(shape) > 1 else 1):
            if len(shape) > 1:
                pink[:, ch] = signal.lfilter(b, a, white[:, ch])
            else:
                pink = signal.lfilter(b, a, white)

        return pink


def apply_mic(
    audio: np.ndarray,
    mic_type: MicType = MicType.CONDENSER,
    position: MicPosition = MicPosition.CENTER,
    distance: float = 0.5,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    Convenience function to apply mic simulation.

    Args:
        audio: Input audio array
        mic_type: Type of microphone
        position: Mic position relative to source
        distance: Distance in meters
        sample_rate: Audio sample rate

    Returns:
        Audio with mic characteristics applied
    """
    config = MicConfig(
        mic_type=mic_type,
        position=position,
        distance_meters=distance,
    )

    simulator = MicSimulator(config, sample_rate)
    return simulator.process(audio)
