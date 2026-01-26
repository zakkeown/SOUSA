# Technical Reference

This section provides comprehensive technical documentation for SOUSA's internal systems, data structures, and algorithms.

## Overview

SOUSA (Synthetic Open Unified Snare Assessment) generates synthetic drum rudiment datasets for machine learning training. The system produces over 100,000 labeled samples covering all 40 PAS (Percussive Arts Society) drum rudiments with MIDI, audio, and hierarchical performance labels.

## Reference Documentation

### [Architecture](architecture.md)

System architecture documentation including:

- **Pipeline Flow**: Complete data flow from rudiment definitions through audio output
- **Class Diagrams**: Data structure relationships (Sample, StrokeLabel, MeasureLabel, ExerciseScores)
- **Module Dependencies**: How components interact

### [Rudiment Schema](rudiment-schema.md)

YAML schema specification for rudiment definitions:

- **Schema Fields**: Complete field reference with types and constraints
- **Stroke Types**: tap, accent, grace, diddle, buzz
- **Category Parameters**: Flam, drag, diddle, and roll-specific configurations
- **Examples**: Annotated YAML files for common rudiments

### [Score Computation](score-computation.md)

Mathematical foundations for performance scoring:

- **Timing Metrics**: Accuracy, consistency, tempo stability formulas
- **Dynamics Metrics**: Velocity control, accent differentiation
- **Hand Balance**: Combined velocity and timing balance scoring
- **Overall Score**: Weighted composite calculation
- **Perceptual Scaling**: Sigmoid transformations for human-aligned scores

### [Audio Processing](audio-processing.md)

Audio augmentation pipeline documentation:

- **Preamp Simulation**: CLEAN, WARM, AGGRESSIVE, VINTAGE types
- **Compression**: Threshold, ratio, attack, release, knee parameters
- **Equalization**: Highpass, lowpass, and shelving filters
- **Room Simulation**: Impulse response convolution
- **Degradation**: Sample rate, bit depth, and noise injection

### [Data Format](data-format.md)

Complete data schema documentation:

- **File Structure**: Directory layout for local and HuggingFace datasets
- **Parquet Schemas**: Column definitions for samples, exercises, measures, strokes
- **Audio/MIDI Formats**: Technical specifications
- **Augmentation Presets**: Configuration details for each preset

## Quick Reference

### Key Data Types

| Type | Description | Location |
|------|-------------|----------|
| `Rudiment` | Complete rudiment definition | `dataset_gen/rudiments/schema.py` |
| `PlayerProfile` | Player execution characteristics | `dataset_gen/profiles/archetypes.py` |
| `Sample` | Complete labeled sample | `dataset_gen/labels/schema.py` |
| `StrokeEvent` | Generated stroke with timing/velocity | `dataset_gen/midi_gen/generator.py` |

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Load | `rudiments/loader.py` | Parse YAML definitions |
| 2. Profile | `profiles/archetypes.py` | Generate player profiles |
| 3. MIDI | `midi_gen/generator.py` | Generate stroke events |
| 4. Audio | `audio_synth/synthesizer.py` | Render via FluidSynth |
| 5. Augment | `audio_aug/pipeline.py` | Apply audio augmentations |
| 6. Label | `labels/compute.py` | Compute hierarchical scores |
| 7. Store | `pipeline/storage.py` | Write Parquet files |

### Score Ranges

All scores use a 0-100 scale where higher values indicate better performance:

| Score | Perfect | Professional | Advanced | Intermediate | Beginner |
|-------|---------|--------------|----------|--------------|----------|
| `overall_score` | 100 | 70-85 | 55-70 | 40-55 | 25-40 |
| `timing_accuracy` | 100 | 80-95 | 65-80 | 45-65 | 20-45 |
| `hand_balance` | 100 | 90-98 | 80-90 | 70-80 | 50-70 |

## See Also

- [Getting Started Guide](../getting-started/index.md) - Installation and first steps
- [User Guide](../user-guide/index.md) - Configuration and generation
- [API Reference](../api/index.md) - Python API documentation
