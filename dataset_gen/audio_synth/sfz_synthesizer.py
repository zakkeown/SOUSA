"""
Top-level SFZ synthesizer that matches the AudioSynthesizer API.

Combines SfzParser + SfzDrumRenderer behind a render() interface
so it plugs into the dataset generation pipeline alongside FluidSynth.
"""

from __future__ import annotations

from pathlib import Path
import logging

import mido
import numpy as np

from dataset_gen.audio_synth.sfz_parser import SfzParser, SfzInstrument
from dataset_gen.audio_synth.sfz_renderer import SfzDrumRenderer

logger = logging.getLogger(__name__)


class SfzSynthesizer:
    """
    Synthesize audio from MIDI using SFZ sample libraries.

    Drop-in complement to AudioSynthesizer for SFZ-format soundfonts.
    Supports multiple loaded instruments and per-instrument note remapping.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._parser = SfzParser()
        self._renderer = SfzDrumRenderer(sample_rate=sample_rate)
        self._instruments: dict[str, SfzInstrument] = {}
        self._note_remaps: dict[str, dict[int, int]] = {}

    def load_soundfont(
        self,
        path: Path | str,
        name: str | None = None,
        note_remap: dict[int, int] | None = None,
    ) -> str:
        """
        Parse and load an SFZ instrument.

        Args:
            path: Path to .sfz file
            name: Name to reference this instrument (defaults to stem of filename)
            note_remap: Optional MIDI note remapping dict, e.g. {38: 64}

        Returns:
            Name of the loaded instrument
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SFZ file not found: {path}")

        if name is None:
            name = path.stem

        instrument = self._parser.parse(path)
        self._instruments[name] = instrument

        if note_remap:
            self._note_remaps[name] = note_remap

        mapped_notes = sorted(instrument.get_mapped_notes())
        logger.info(
            f"Loaded SFZ: {name} ({len(instrument.regions)} regions, " f"notes: {mapped_notes})"
        )
        if note_remap:
            logger.info(f"  Note remap: {note_remap}")

        return name

    def render(
        self,
        midi_data: bytes | None = None,
        midi_path: Path | str | None = None,
        soundfont_name: str | None = None,
        duration_hint_sec: float | None = None,
    ) -> np.ndarray:
        """
        Render MIDI to audio using the specified SFZ instrument.

        API matches AudioSynthesizer.render() for pipeline compatibility.

        Args:
            midi_data: Raw MIDI bytes
            midi_path: Path to MIDI file (alternative to midi_data)
            soundfont_name: Name of loaded SFZ instrument (uses first if None)
            duration_hint_sec: Approximate duration for buffer allocation

        Returns:
            Audio samples as numpy array (shape: [samples, 2])
        """
        if midi_data is None and midi_path is None:
            raise ValueError("Must provide either midi_data or midi_path")

        if not self._instruments:
            raise RuntimeError("No SFZ instrument loaded. Call load_soundfont() first.")

        # Select instrument
        if soundfont_name is None:
            soundfont_name = next(iter(self._instruments.keys()))

        if soundfont_name not in self._instruments:
            raise KeyError(f"SFZ instrument '{soundfont_name}' not loaded")

        instrument = self._instruments[soundfont_name]
        note_remap = self._note_remaps.get(soundfont_name, {})

        # Parse MIDI to extract note events
        events, midi_duration = self._extract_events(midi_data, midi_path, note_remap)

        total_duration = midi_duration
        if duration_hint_sec:
            total_duration = max(total_duration, duration_hint_sec)

        # Render through the drum renderer
        audio = self._renderer.render_events(
            instrument,
            events,
            total_duration,
            seed=hash((soundfont_name, len(events))) % (2**31),
        )

        return audio

    def _extract_events(
        self,
        midi_data: bytes | None,
        midi_path: Path | str | None,
        note_remap: dict[int, int],
    ) -> tuple[list[tuple[float, int, int]], float]:
        """
        Extract note events from MIDI data.

        Returns:
            Tuple of (events, total_duration_sec) where events are
            (time_sec, note, velocity) tuples.
        """
        import tempfile

        if midi_data is not None:
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                f.write(midi_data)
                tmp_path = f.name
            try:
                mid = mido.MidiFile(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            mid = mido.MidiFile(str(midi_path))

        events: list[tuple[float, int, int]] = []
        current_time = 0.0

        for msg in mid:
            current_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note = msg.note
                # Apply note remapping
                note = note_remap.get(note, note)
                events.append((current_time, note, msg.velocity))

        total_duration = mid.length
        return events, total_duration

    @property
    def soundfont_names(self) -> list[str]:
        """List of loaded instrument names."""
        return list(self._instruments.keys())

    def close(self) -> None:
        """Clean up resources."""
        self._renderer.close()
        self._instruments.clear()
        self._note_remaps.clear()

    def __enter__(self) -> "SfzSynthesizer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
