"""Audio analysis module for high-resolution visualization of drum rudiments."""

from dataset_gen.audio_analysis.midi_alignment import MidiOnset, extract_midi_onsets
from dataset_gen.audio_analysis.onsets import (
    DetectedOnset,
    detect_onsets,
    get_onset_activation,
)
from dataset_gen.audio_analysis.views import (
    render_dashboard,
    render_onset_timeline,
    render_waveform,
)

__all__ = [
    "DetectedOnset",
    "MidiOnset",
    "detect_onsets",
    "extract_midi_onsets",
    "get_onset_activation",
    "render_dashboard",
    "render_onset_timeline",
    "render_waveform",
]
