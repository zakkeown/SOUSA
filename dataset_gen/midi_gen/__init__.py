"""MIDI generation engine for synthetic drum performances."""

from dataset_gen.midi_gen.generator import (
    MIDIGenerator,
    GeneratedPerformance,
    StrokeEvent,
)
from dataset_gen.midi_gen.articulations import (
    ArticulationEngine,
    apply_flam_spacing,
    apply_diddle_timing,
    apply_roll_velocity_decay,
)

__all__ = [
    "MIDIGenerator",
    "GeneratedPerformance",
    "StrokeEvent",
    "ArticulationEngine",
    "apply_flam_spacing",
    "apply_diddle_timing",
    "apply_roll_velocity_decay",
]
