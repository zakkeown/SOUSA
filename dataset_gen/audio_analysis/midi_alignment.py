"""MIDI onset extraction and alignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

__all__ = ["MidiOnset", "extract_midi_onsets"]


@dataclass
class MidiOnset:
    """A MIDI note-on event with absolute timing.

    Attributes:
        time_sec: Onset time in seconds from the start of the MIDI file.
        velocity: MIDI velocity (0-127).
        note: MIDI note number.
    """

    time_sec: float
    velocity: int
    note: int


def extract_midi_onsets(midi_path: str | Path) -> list[MidiOnset]:
    """Extract note-on events from a MIDI file with absolute timestamps.

    Uses mido's built-in tempo-aware iteration which converts tick deltas
    to seconds.

    Args:
        midi_path: Path to the MIDI file.

    Returns:
        List of MidiOnset objects sorted by time.
    """
    mid = mido.MidiFile(str(midi_path))
    onsets: list[MidiOnset] = []
    current_time = 0.0
    for msg in mid:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            onsets.append(MidiOnset(time_sec=current_time, velocity=msg.velocity, note=msg.note))
    onsets.sort(key=lambda o: o.time_sec)
    return onsets
