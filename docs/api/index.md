# API Reference

The SOUSA (Synthetic Open Unified Snare Assessment) package is organized into several modules that handle different aspects of synthetic drum rudiment dataset generation.

## Module Structure

The `dataset_gen` package follows this hierarchical structure:

```
dataset_gen/
├── rudiments/        # Rudiment definitions and loading
├── profiles/         # Player profile generation
├── midi_gen/         # MIDI performance generation
├── audio_synth/      # Audio synthesis via FluidSynth
├── audio_aug/        # Audio augmentation chain
├── labels/           # Hierarchical label computation
├── pipeline/         # Dataset generation orchestration
└── validation/       # Dataset validation and analysis
```

## Module Overview

### [Rudiments](rudiments.md)

Handles loading and parsing of the 40 PAS (Percussive Arts Society) drum rudiment definitions from YAML files. Provides schema classes for representing strokes, sticking patterns, and complete rudiment specifications.

### [Profiles](profiles.md)

Generates player profiles with skill-tier-based archetypes. Each profile represents a virtual drummer with correlated execution dimensions covering timing accuracy, dynamics control, and hand balance.

### [MIDI Generation](midi-gen.md)

Generates MIDI performances from rudiment definitions and player profiles. Applies realistic timing deviations, velocity variations, and articulation-specific processing for flams, diddles, and rolls.

### [Audio Synthesis](audio-synth.md)

FluidSynth wrapper for rendering MIDI performances to audio. Supports multiple soundfonts for different drum sounds (practice pad, marching snare, drum kits).

### [Audio Augmentation](audio-aug.md)

Comprehensive audio augmentation pipeline including room simulation (convolution reverb), microphone modeling, recording chain simulation (preamp, compression, EQ), and degradation effects (noise, bit reduction).

### [Labels](labels.md)

Computes hierarchical labels at three levels:

- **Stroke-level**: Individual timing and velocity measurements
- **Measure-level**: Aggregate statistics per measure
- **Exercise-level**: Overall performance scores (0-100 scale)

### [Pipeline](pipeline.md)

Orchestrates the complete dataset generation pipeline from profiles to stored samples. Handles parallel processing, checkpointing, and Parquet/audio file storage.

### [Validation](validation.md)

Dataset validation and statistical analysis tools. Verifies label correctness, data integrity, and skill tier ordering. Generates comprehensive validation reports.

## Quick Start

```python
from dataset_gen.pipeline.generate import generate_dataset

# Generate a small dataset
splits = generate_dataset(
    output_dir="output/dataset",
    num_profiles=10,
    soundfont_path="data/soundfonts/snare.sf2",
    seed=42,
)
```

## Key Types

The most commonly used types across the package:

- `Rudiment` - Complete rudiment definition from YAML
- `StickingPattern` - Stroke sequence for a rudiment
- `PlayerProfile` - Virtual drummer characteristics
- `SkillTier` - Skill level enum (beginner/intermediate/advanced/professional)
- `Sample` - Complete sample with all hierarchical labels
- `GeneratedPerformance` - MIDI performance with stroke events
