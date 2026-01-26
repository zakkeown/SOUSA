# MIDI Generation Module

The MIDI generation module creates synthetic drum performances from rudiment definitions and player profiles. It applies realistic timing deviations, velocity variations, and articulation-specific processing to produce varied but consistent performances.

The generation process:

1. Creates an ideal timing grid from the rudiment pattern
2. Applies player-specific deviations based on profile dimensions
3. Handles articulation-specific timing (flam spacing, diddle ratios)
4. Converts stroke events to MIDI bytes

## Generator

The main MIDI generation engine that transforms rudiments and profiles into performances.

::: dataset_gen.midi_gen.generator
    options:
      show_root_heading: false
      members:
        - StrokeEvent
        - GeneratedPerformance
        - MIDIGenerator
        - generate_performance
        - regenerate_midi

## Articulations

Articulation-specific processing for flams, drags, diddles, and rolls. Refines timing and velocity relationships based on rudiment type and player skill.

::: dataset_gen.midi_gen.articulations
    options:
      show_root_heading: false
      members:
        - ArticulationParams
        - ArticulationEngine
        - apply_flam_spacing
        - apply_diddle_timing
        - apply_roll_velocity_decay
