"""
Sample-based drum renderer for SFZ instruments.

Loads WAV/FLAC samples on demand, applies velocity-based gain,
and mixes multiple note events into a stereo output buffer.
Handles sequential and random round-robin sample selection.
"""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import soundfile as sf

from dataset_gen.audio_synth.sfz_parser import SfzInstrument, SfzRegion

logger = logging.getLogger(__name__)


class SfzSampleCache:
    """
    Lazy-loading cache for audio samples.

    Loads WAV/FLAC files on first access via soundfile, converts mono to stereo,
    and caches the float32 numpy arrays in memory.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._cache: dict[str, np.ndarray] = {}
        self._failed: set[str] = set()

    def get(self, path: Path) -> np.ndarray | None:
        """Load and return a sample as float32 stereo array, or None if unavailable."""
        key = str(path)

        if key in self._failed:
            return None
        if key in self._cache:
            return self._cache[key]

        if not path.exists():
            logger.warning(f"Sample not found: {path}")
            self._failed.add(key)
            return None

        try:
            data, file_sr = sf.read(str(path), dtype="float32")
        except Exception as e:
            logger.warning(f"Failed to read sample {path}: {e}")
            self._failed.add(key)
            return None

        # Convert mono to stereo
        if data.ndim == 1:
            data = np.column_stack([data, data])

        # Simple resample if sample rates don't match (nearest-neighbor)
        if file_sr != self.sample_rate:
            ratio = self.sample_rate / file_sr
            num_output = int(len(data) * ratio)
            indices = np.clip((np.arange(num_output) / ratio).astype(int), 0, len(data) - 1)
            data = data[indices]

        self._cache[key] = data
        return data

    def clear(self) -> None:
        """Clear the sample cache to free memory."""
        self._cache.clear()
        self._failed.clear()

    @property
    def num_cached(self) -> int:
        return len(self._cache)


class SfzDrumRenderer:
    """
    Renders MIDI note events through an SFZ instrument into audio.

    For each event:
    1. Find matching regions (velocity layer + round robin)
    2. Load sample(s) from cache
    3. Apply velocity-based gain
    4. Mix into output buffer at the correct time position
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.cache = SfzSampleCache(sample_rate)
        # Per-note sequential round-robin counters
        self._seq_counters: dict[int, int] = {}
        self._rng = np.random.default_rng()

    def render_events(
        self,
        instrument: SfzInstrument,
        events: list[tuple[float, int, int]],
        total_duration: float,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Render a list of note events to audio.

        Args:
            instrument: Parsed SFZ instrument
            events: List of (time_sec, midi_note, velocity) tuples
            total_duration: Total output duration in seconds
            seed: Random seed for round-robin selection

        Returns:
            Stereo float32 numpy array, shape (num_samples, 2)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset round-robin counters for each render call
        self._seq_counters.clear()

        # Allocate output buffer with 0.5s tail for sample decay
        num_samples = int((total_duration + 0.5) * self.sample_rate)
        output = np.zeros((num_samples, 2), dtype=np.float64)

        for time_sec, note, velocity in events:
            if velocity == 0:
                continue  # Note-off, skip for drums

            self._render_single_event(instrument, output, time_sec, note, velocity)

        # Normalize to float32, clip to [-1, 1]
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak
        return output.astype(np.float32)

    def _render_single_event(
        self,
        instrument: SfzInstrument,
        output: np.ndarray,
        time_sec: float,
        note: int,
        velocity: int,
    ) -> None:
        """Render a single note event into the output buffer."""
        # Get round-robin values
        seq_counter = self._seq_counters.get(note, 0)
        rand_value = self._rng.random()

        # Find matching regions
        regions = instrument.get_regions_for_note(note, velocity, seq_counter, rand_value)

        if not regions:
            return

        # Advance sequential counter
        self._seq_counters[note] = seq_counter + 1

        # Calculate start position in output buffer
        start_sample = int(time_sec * self.sample_rate)
        if start_sample < 0:
            start_sample = 0

        for region in regions:
            sample_path = instrument.resolve_sample_path(region)
            sample_data = self.cache.get(sample_path)
            if sample_data is None:
                continue

            # Calculate velocity gain
            gain = self._compute_velocity_gain(region, velocity)

            # Apply region volume (dB)
            if region.volume != 0.0:
                gain *= 10.0 ** (region.volume / 20.0)

            # Mix sample into output at the correct position
            end_sample = min(start_sample + len(sample_data), len(output))
            sample_len = end_sample - start_sample
            if sample_len > 0:
                output[start_sample:end_sample] += sample_data[:sample_len] * gain

    def _compute_velocity_gain(self, region: SfzRegion, velocity: int) -> float:
        """
        Compute gain factor from velocity using amp_veltrack and optional velcurve.

        SFZ formula: gain = (1 - amp_veltrack/100) + (amp_veltrack/100) * curve(vel)
        Where curve(vel) defaults to vel/127 (linear) if no amp_velcurve is defined.

        Special case: when amp_veltrack=0 (e.g. SM Drums uses CC-controlled volume),
        we override to 100 so velocity naturally controls amplitude, since we don't
        send MIDI CCs.
        """
        veltrack = region.amp_veltrack

        # Override CC-controlled mode: if veltrack is 0, use full velocity sensitivity
        if veltrack == 0.0:
            veltrack = 100.0

        veltrack_frac = veltrack / 100.0

        # Compute curve value
        if region.amp_velcurve:
            curve_val = self._interpolate_velcurve(region.amp_velcurve, velocity)
        else:
            # Default linear curve
            curve_val = velocity / 127.0

        gain = (1.0 - veltrack_frac) + veltrack_frac * curve_val
        return max(0.0, gain)

    def _interpolate_velcurve(self, velcurve: dict[int, float], velocity: int) -> float:
        """Interpolate a custom velocity curve at the given velocity."""
        if velocity in velcurve:
            return velcurve[velocity]

        # Find surrounding points
        keys = sorted(velcurve.keys())
        if not keys:
            return velocity / 127.0

        if velocity <= keys[0]:
            return velcurve[keys[0]]
        if velocity >= keys[-1]:
            return velcurve[keys[-1]]

        # Linear interpolation between surrounding points
        for i in range(len(keys) - 1):
            if keys[i] <= velocity <= keys[i + 1]:
                lo_vel, hi_vel = keys[i], keys[i + 1]
                lo_val, hi_val = velcurve[lo_vel], velcurve[hi_vel]
                t = (velocity - lo_vel) / (hi_vel - lo_vel)
                return lo_val + t * (hi_val - lo_val)

        return velocity / 127.0

    def close(self) -> None:
        """Free cached samples."""
        self.cache.clear()
        self._seq_counters.clear()
