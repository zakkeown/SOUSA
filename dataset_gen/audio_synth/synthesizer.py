"""
FluidSynth-based audio synthesizer for MIDI rendering.

This module provides a wrapper around FluidSynth for converting
MIDI performances to audio using various soundfonts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tempfile
import numpy as np

try:
    import fluidsynth

    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False

import soundfile as sf


@dataclass
class SynthConfig:
    """Configuration for audio synthesis."""

    sample_rate: int = 44100
    channels: int = 2  # Stereo
    gain: float = 0.5  # Master gain (0.0 - 1.0)

    # Reverb settings (built-in FluidSynth reverb)
    reverb_enabled: bool = False
    reverb_room_size: float = 0.2
    reverb_damping: float = 0.5
    reverb_width: float = 0.5
    reverb_level: float = 0.5

    # Chorus settings (built-in FluidSynth chorus)
    chorus_enabled: bool = False
    chorus_depth: float = 8.0
    chorus_level: float = 2.0
    chorus_nr: int = 3
    chorus_speed: float = 0.3


@dataclass
class SoundfontInfo:
    """Information about a loaded soundfont."""

    path: Path
    name: str
    preset_count: int
    sfont_id: int


class AudioSynthesizer:
    """
    Synthesize audio from MIDI using FluidSynth.

    This class wraps FluidSynth to provide easy MIDI-to-audio conversion
    with support for multiple soundfonts.
    """

    def __init__(
        self,
        soundfont_path: Path | str | None = None,
        config: SynthConfig | None = None,
    ):
        """
        Initialize the synthesizer.

        Args:
            soundfont_path: Path to .sf2 soundfont file (optional, can load later)
            config: Synthesis configuration
        """
        # Initialize attributes first to avoid __del__ errors
        self.config = config or SynthConfig()
        self._fs: "fluidsynth.Synth | None" = None
        self._soundfonts: dict[str, SoundfontInfo] = {}
        self._initialized = False

        if not FLUIDSYNTH_AVAILABLE:
            raise ImportError(
                "FluidSynth is not available. Install it with:\n"
                "  brew install fluid-synth && pip install pyfluidsynth  (macOS)\n"
                "  apt install fluidsynth && pip install pyfluidsynth    (Linux)"
            )

        if soundfont_path:
            self._init_synth()
            self.load_soundfont(soundfont_path)

    def _init_synth(self) -> None:
        """Initialize the FluidSynth synthesizer."""
        if self._initialized:
            return

        self._fs = fluidsynth.Synth(
            samplerate=float(self.config.sample_rate),
            gain=self.config.gain,
        )

        # Configure reverb
        if self.config.reverb_enabled:
            self._fs.set_reverb(
                roomsize=self.config.reverb_room_size,
                damping=self.config.reverb_damping,
                width=self.config.reverb_width,
                level=self.config.reverb_level,
            )

        # Configure chorus
        if self.config.chorus_enabled:
            self._fs.set_chorus(
                nr=self.config.chorus_nr,
                level=self.config.chorus_level,
                speed=self.config.chorus_speed,
                depth=self.config.chorus_depth,
                type=0,  # Sine wave
            )

        self._initialized = True

    def load_soundfont(
        self,
        soundfont_path: Path | str,
        name: str | None = None,
    ) -> str:
        """
        Load a soundfont file.

        Args:
            soundfont_path: Path to .sf2 file
            name: Optional name for referencing this soundfont

        Returns:
            Name of the loaded soundfont
        """
        self._init_synth()

        path = Path(soundfont_path)
        if not path.exists():
            raise FileNotFoundError(f"Soundfont not found: {path}")

        if name is None:
            name = path.stem

        sfont_id = self._fs.sfload(str(path))
        if sfont_id == -1:
            raise RuntimeError(f"Failed to load soundfont: {path}")

        self._soundfonts[name] = SoundfontInfo(
            path=path,
            name=name,
            preset_count=0,  # FluidSynth doesn't easily expose this
            sfont_id=sfont_id,
        )

        return name

    def render(
        self,
        midi_data: bytes | None = None,
        midi_path: Path | str | None = None,
        soundfont_name: str | None = None,
        duration_hint_sec: float | None = None,
    ) -> np.ndarray:
        """
        Render MIDI to audio samples.

        Args:
            midi_data: Raw MIDI bytes
            midi_path: Path to MIDI file (alternative to midi_data)
            soundfont_name: Name of soundfont to use (uses first loaded if None)
            duration_hint_sec: Approximate duration for buffer allocation

        Returns:
            Audio samples as numpy array (shape: [samples, channels])
        """
        if midi_data is None and midi_path is None:
            raise ValueError("Must provide either midi_data or midi_path")

        if not self._soundfonts:
            raise RuntimeError("No soundfont loaded. Call load_soundfont() first.")

        self._init_synth()

        # Get soundfont to use
        if soundfont_name is None:
            soundfont_name = next(iter(self._soundfonts.keys()))

        if soundfont_name not in self._soundfonts:
            raise KeyError(f"Soundfont '{soundfont_name}' not loaded")

        sfont_info = self._soundfonts[soundfont_name]

        # Write MIDI data to temp file if needed
        if midi_data is not None:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                f.write(midi_data)
                midi_file = f.name
            cleanup_midi = True
        else:
            midi_file = str(midi_path)
            cleanup_midi = False

        try:
            # Parse MIDI to get events and duration
            import mido

            mid = mido.MidiFile(midi_file)
            total_duration_sec = mid.length

            if duration_hint_sec:
                total_duration_sec = max(total_duration_sec, duration_hint_sec)

            # Add padding for reverb tail
            total_duration_sec += 0.5

            # Calculate buffer size
            num_samples = int(total_duration_sec * self.config.sample_rate)

            # Reset synth state
            self._fs.system_reset()

            # Select the soundfont's first program on channel 9 (drums)
            self._fs.program_select(9, sfont_info.sfont_id, 0, 0)

            # Process MIDI events
            samples_list = []
            current_time = 0.0

            for msg in mid:
                # Advance time
                if msg.time > 0:
                    wait_samples = int(msg.time * self.config.sample_rate)
                    if wait_samples > 0:
                        chunk = self._fs.get_samples(wait_samples)
                        # Reshape to stereo
                        chunk = np.array(chunk).reshape(-1, 2)
                        samples_list.append(chunk)
                    current_time += msg.time

                # Process MIDI message
                if msg.type == "note_on":
                    channel = msg.channel if hasattr(msg, "channel") else 9
                    self._fs.noteon(channel, msg.note, msg.velocity)
                elif msg.type == "note_off":
                    channel = msg.channel if hasattr(msg, "channel") else 9
                    self._fs.noteoff(channel, msg.note)
                elif msg.type == "control_change":
                    channel = msg.channel if hasattr(msg, "channel") else 9
                    self._fs.cc(channel, msg.control, msg.value)

            # Render remaining samples (reverb tail)
            remaining = num_samples - sum(len(s) for s in samples_list)
            if remaining > 0:
                chunk = self._fs.get_samples(remaining)
                chunk = np.array(chunk).reshape(-1, 2)
                samples_list.append(chunk)

            # Concatenate all samples
            if samples_list:
                audio = np.concatenate(samples_list, axis=0)
            else:
                audio = np.zeros((num_samples, 2), dtype=np.float32)

            # Normalize to float32 [-1, 1]
            # FluidSynth returns int16 samples
            audio = audio.astype(np.float32) / 32768.0

            return audio

        finally:
            if cleanup_midi:
                Path(midi_file).unlink(missing_ok=True)

    def render_to_file(
        self,
        output_path: Path | str,
        midi_data: bytes | None = None,
        midi_path: Path | str | None = None,
        soundfont_name: str | None = None,
        format: Literal["wav", "flac"] = "flac",
        subtype: str | None = None,
    ) -> Path:
        """
        Render MIDI to an audio file.

        Args:
            output_path: Output file path
            midi_data: Raw MIDI bytes
            midi_path: Path to MIDI file
            soundfont_name: Soundfont to use
            format: Output format ('wav' or 'flac')
            subtype: Soundfile subtype (e.g., 'PCM_16', 'PCM_24')

        Returns:
            Path to the output file
        """
        audio = self.render(
            midi_data=midi_data,
            midi_path=midi_path,
            soundfont_name=soundfont_name,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Default subtypes
        if subtype is None:
            subtype = "PCM_24" if format == "flac" else "PCM_16"

        sf.write(
            str(output_path),
            audio,
            self.config.sample_rate,
            format=format.upper(),
            subtype=subtype,
        )

        return output_path

    def close(self) -> None:
        """Clean up FluidSynth resources."""
        if self._fs is not None:
            self._fs.delete()
            self._fs = None
            self._initialized = False
            self._soundfonts.clear()

    def __enter__(self) -> "AudioSynthesizer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def apply_velocity_curve(
    audio: np.ndarray,
    strokes: list,
    sample_rate: int,
    veltrack: float = 1.0,
    crossfade_ms: float = 3.0,
) -> np.ndarray:
    """Apply velocity-dependent gain to compensate for flat soundfont mapping.

    FluidSynth's snare soundfonts produce nearly identical amplitude regardless
    of MIDI velocity. This function applies a post-synthesis gain envelope
    based on each stroke's actual_velocity, using the same formula as SFZ
    amp_veltrack.

    Args:
        audio: Synthesized audio (samples,) or (samples, channels)
        strokes: List of StrokeEvent objects with actual_time_ms and actual_velocity
        sample_rate: Audio sample rate in Hz
        veltrack: Velocity sensitivity 0.0-1.0 (1.0 = full linear mapping)
        crossfade_ms: Crossfade duration at stroke boundaries to prevent clicks

    Returns:
        Audio with velocity-dependent gain applied
    """
    if not strokes:
        return audio

    n_samples = audio.shape[0]
    gain_envelope = np.ones(n_samples, dtype=np.float32)

    # Sort strokes by time
    sorted_strokes = sorted(strokes, key=lambda s: s.actual_time_ms)
    crossfade_samples = int(crossfade_ms / 1000 * sample_rate)

    for i, stroke in enumerate(sorted_strokes):
        # Gain formula (SFZ standard): gain = (1 - veltrack) + veltrack * (vel / 127)
        vel_gain = (1.0 - veltrack) + veltrack * (stroke.actual_velocity / 127.0)

        start = int(stroke.actual_time_ms / 1000 * sample_rate)
        start = max(0, min(start, n_samples))

        if i + 1 < len(sorted_strokes):
            end = int(sorted_strokes[i + 1].actual_time_ms / 1000 * sample_rate)
            end = max(0, min(end, n_samples))
        else:
            end = n_samples

        if start < end:
            gain_envelope[start:end] = vel_gain

    # Smooth transitions with crossfade to prevent clicks
    if crossfade_samples > 1:
        from scipy.ndimage import uniform_filter1d

        gain_envelope = uniform_filter1d(gain_envelope, size=crossfade_samples)

    # Apply gain
    if audio.ndim == 2:
        result = audio * gain_envelope[:, np.newaxis]
    else:
        result = audio * gain_envelope

    # Prevent clipping
    max_val = np.max(np.abs(result))
    if max_val > 1.0:
        result = result / max_val

    return result


def render_midi_to_audio(
    midi_data: bytes | None = None,
    midi_path: Path | str | None = None,
    soundfont_path: Path | str | None = None,
    output_path: Path | str | None = None,
    sample_rate: int = 44100,
    format: Literal["wav", "flac"] = "flac",
) -> np.ndarray | Path:
    """
    Convenience function to render MIDI to audio.

    Args:
        midi_data: Raw MIDI bytes
        midi_path: Path to MIDI file
        soundfont_path: Path to soundfont (required)
        output_path: If provided, write to file and return path
        sample_rate: Sample rate in Hz
        format: Output format if writing to file

    Returns:
        Audio samples as numpy array, or Path if output_path provided
    """
    if soundfont_path is None:
        raise ValueError("soundfont_path is required")

    config = SynthConfig(sample_rate=sample_rate)

    with AudioSynthesizer(soundfont_path, config) as synth:
        if output_path:
            return synth.render_to_file(
                output_path,
                midi_data=midi_data,
                midi_path=midi_path,
                format=format,
            )
        else:
            return synth.render(
                midi_data=midi_data,
                midi_path=midi_path,
            )
