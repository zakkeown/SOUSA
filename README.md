# SOUSA: Synthetic Open Unified Snare Assessment

[![CI](https://github.com/zakkeown/SOUSA/actions/workflows/test.yml/badge.svg)](https://github.com/zakkeown/SOUSA/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/zakkeown/SOUSA/branch/main/graph/badge.svg)](https://codecov.io/gh/zakkeown/SOUSA)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/zkeown/sousa)

A synthetic dataset generator for all 40 PAS (Percussive Arts Society) drum rudiments, designed to train machine learning models for drumming performance assessment.

## Load from HuggingFace

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("zkeown/sousa")

# Load specific split
train = load_dataset("zkeown/sousa", split="train")

# Stream for memory efficiency
dataset = load_dataset("zkeown/sousa", streaming=True)

# Access a sample
sample = dataset["train"][0]
print(f"Rudiment: {sample['rudiment_slug']}")
print(f"Overall Score: {sample['overall_score']:.1f}")
```

## Overview

SOUSA generates 100K+ synthetic drum rudiment performances with:

- **MIDI performances** with realistic timing/velocity variations modeled from player skill profiles
- **Multi-soundfont audio synthesis** via FluidSynth (practice pad, marching snare, drum kits)
- **Extensive audio augmentation** (room acoustics, mic simulation, compression, noise)
- **Hierarchical labels** at stroke, measure, and exercise levels
- **Profile-based splits** ensuring train/val/test generalization

## Quick Start

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Download soundfonts for audio generation
python scripts/setup_soundfonts.py

# Generate a small test dataset (~1,200 samples)
python scripts/generate_dataset.py --preset small --with-audio

# Generate the full 100K dataset
python scripts/generate_dataset.py --with-audio
```

## Upload to HuggingFace Hub

```bash
# Install hub dependencies
pip install 'sousa[hub]'

# Login to HuggingFace
huggingface-cli login

# Upload dataset
python scripts/push_to_hub.py zkeown/sousa

# Upload with options
python scripts/push_to_hub.py zkeown/sousa --private          # Private repo
python scripts/push_to_hub.py zkeown/sousa --no-audio         # Skip audio (smaller)
python scripts/push_to_hub.py zkeown/sousa --dry-run          # Test without upload
```

## Dataset Structure

```
output/dataset/
├── midi/              # MIDI files
├── audio/             # FLAC audio files (if --with-audio)
├── labels/            # Parquet files with hierarchical labels
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
└── validation_report.json
```

### Sample Naming Convention

Samples use readable IDs:
```
{skill_tier}{profile_num}_{rudiment}_{tempo}bpm_{soundfont}_{augmentation_preset}
```

Example: `beg042_single_paradiddle_100bpm_marching_practicedry`

## Generation Presets

| Preset | Profiles | Tempos | Augmentations | Samples | Storage |
|--------|----------|--------|---------------|---------|---------|
| small  | 10       | 3      | 1             | ~1,200  | ~1 GB   |
| medium | 50       | 3      | 2             | ~12,000 | ~10 GB  |
| full   | 100      | 5      | 5             | ~100,000| ~97 GB  |

### Full Dataset Storage Breakdown

| Component | Size | Description |
|-----------|------|-------------|
| Audio     | 96 GB | FLAC 44.1kHz 24-bit mono (~138 hours) |
| MIDI      | 79 MB | Type 1 MIDI files |
| Labels    | 41 MB | Parquet files (strokes, measures, exercises) |
| **Total** | **~97 GB** | Full dataset with audio |

## Rudiments Covered

All 40 PAS International Drum Rudiments:

- **Roll Rudiments (15)**: Single/Double/Triple Stroke Rolls, 5-17 Stroke Rolls
- **Diddle Rudiments (5)**: Paradiddles and variants
- **Flam Rudiments (12)**: Flam, Flam Accent, Flam Tap, Flamacue, etc.
- **Drag Rudiments (8)**: Drags, Drag Taps, Ratamacue variants

## Player Skill Modeling

Profiles model realistic skill correlations:

| Dimension | Beginner | Intermediate | Advanced |
|-----------|----------|--------------|----------|
| Timing accuracy | 25ms std | 12ms std | 5ms std |
| L/R balance | 0.75 ratio | 0.88 ratio | 0.95 ratio |
| Velocity consistency | High variance | Medium | Low variance |
| Accent differentiation | Weak | Clear | Precise |

## Audio Augmentation Presets

- **clean_studio**: Dry, close-miked, no processing
- **practice_dry**: Small room, practice pad character
- **studio_warm**: Medium room, light compression
- **live_room**: Large room, dynamic range
- **lo_fi**: Vintage degradation, tape saturation

## Validation

SOUSA includes comprehensive validation comparing generated data against peer-reviewed research:

```python
from dataset_gen.validation.report import generate_report

report = generate_report('output/dataset')
print(report.summary())
```

### Validation Checks

| Category | Checks | Status |
|----------|--------|--------|
| **Data Integrity** | 13 checks (unique IDs, valid references, ranges) | All pass |
| **Literature Benchmarks** | 8 comparisons to published timing/velocity research | All pass |
| **Skill Separation** | ANOVA confirms tier differences (F > 18,000) | All pass |
| **Correlation Structure** | 5 expected score correlations | 4/5 pass |

### Literature References

- Fujii et al. (2011) - Professional drummer timing variability
- Repp (2005) - Sensorimotor synchronization review
- Wing & Kristofferson (1973) - Timing response model
- Schmidt & Lee (2011) - Motor control velocity CV

See [docs/VALIDATION.md](docs/VALIDATION.md) for full validation documentation.

## Project Structure

```
Rudimentary/
├── dataset_gen/           # Core generation modules
│   ├── rudiments/         # Rudiment definitions (YAML)
│   ├── profiles/          # Player skill modeling
│   ├── midi_gen/          # MIDI generation engine
│   ├── audio_synth/       # FluidSynth wrapper
│   ├── audio_aug/         # Augmentation pipeline
│   ├── labels/            # Label computation
│   ├── pipeline/          # Orchestration
│   └── validation/        # Dataset validation
├── data/
│   ├── soundfonts/        # SF2 files
│   ├── impulse_responses/ # Room IRs
│   └── noise_profiles/    # Background noise
├── scripts/
│   ├── generate_dataset.py
│   └── setup_soundfonts.py
└── output/                # Generated datasets
```

## Requirements

- Python 3.10+
- FluidSynth (`brew install fluid-synth` on macOS)
- Dependencies: `pip install -e .`

## Reproducibility

Dataset generation is fully deterministic with seeded random number generators:

| Parameter | Default Value |
|-----------|---------------|
| Global seed | 42 |
| Profile generation | Seeded from global |
| MIDI generation | Seeded from global |
| Audio augmentation | Seeded per-sample from sample_id hash |

To regenerate an identical dataset:
```bash
python scripts/generate_dataset.py --seed 42 --with-audio
```

## Future Work

Potential enhancements for v2:

- **Beat Group Labels**: Add `beat_group_index` and `beat_group_name` to stroke labels to identify which portion of a compound rudiment each stroke belongs to (e.g., "paradiddle" vs "diddle" in a paradiddle-diddle). This enables localized technique assessment—identifying that a player's diddles drag while their paradiddles are clean. Currently derivable post-hoc from `rudiment_slug` + pattern definitions.

- **Per-Group Aggregate Scores**: Compute timing/velocity metrics per beat group, creating a three-tier pedagogical hierarchy: stroke → beat_group → exercise.

- **Additional Rudiment Variations**: Expand to include inverted, reversed, and accent-shifted variations of the 40 PAS rudiments.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues and contributing to SOUSA.

## License

MIT
