"""Audio analysis module for onset detection and MIDI alignment."""

from dataset_gen.audio_analysis.onsets import DetectedOnset, detect_onsets, get_onset_activation
from dataset_gen.audio_analysis.midi_alignment import MidiOnset

__all__ = [
    "DetectedOnset",
    "MidiOnset",
    "detect_onsets",
    "get_onset_activation",
]
