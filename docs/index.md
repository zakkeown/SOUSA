# SOUSA: Synthetic Open Unified Snare Assessment

[![CI](https://github.com/zakkeown/SOUSA/actions/workflows/test.yml/badge.svg)](https://github.com/zakkeown/SOUSA/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/zkeown/sousa)

A synthetic dataset generator for all **40 PAS drum rudiments**, designed to train machine learning models for drumming performance assessment.

## Features

- **100K+ Samples** - Comprehensive coverage of all PAS rudiments
- **Multi-Soundfont Audio** - Practice pad, marching snare, drum kits
- **Audio Augmentation** - Room acoustics, mic simulation, compression
- **Hierarchical Labels** - Stroke, measure, and exercise-level scores
- **Profile-Based Splits** - Proper train/val/test generalization

## Quick Start

```python
from datasets import load_dataset

dataset = load_dataset("zkeown/sousa")
sample = dataset["train"][0]
print(f"Rudiment: {sample['rudiment_slug']}")
print(f"Overall Score: {sample['overall_score']:.1f}")
```

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on HuggingFace](https://huggingface.co/datasets/zkeown/sousa){ .md-button }

## What is SOUSA?

SOUSA generates synthetic drum rudiment performances with realistic timing and velocity variations modeled from player skill profiles. Each sample includes:

- **MIDI performance data** with per-stroke timing and velocity
- **Synthesized audio** using multiple soundfonts (practice pad, marching snare, drum kits)
- **Hierarchical labels** at stroke, measure, and exercise levels
- **Player profile metadata** including skill tier and performance characteristics

## Dataset Overview

| Component | Description |
|-----------|-------------|
| **Rudiments** | All 40 PAS International Drum Rudiments |
| **Profiles** | 100 player profiles across 4 skill tiers |
| **Audio** | 44.1kHz FLAC, multiple soundfonts and augmentations |
| **Labels** | 0-100 scores for timing, velocity, balance, and more |
| **Splits** | Profile-based train/val/test (70/15/15) |

## Use Cases

SOUSA is designed for:

- **Performance Assessment** - Train models to score drumming quality
- **Skill Classification** - Classify performances by skill level
- **Rudiment Recognition** - Identify which rudiment is being played
- **Audio Understanding** - General research on percussive audio

## Documentation

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install SOUSA and its dependencies

    [:octicons-arrow-right-24: Installation Guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Generate your first dataset in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-database:{ .lg .middle } **Loading Data**

    ---

    Access the dataset from HuggingFace or locally

    [:octicons-arrow-right-24: Loading Guide](getting-started/loading.md)

-   :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Detailed examples for ML tasks

    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

</div>

## Generation Presets

| Preset | Profiles | Tempos | Augmentations | Samples | Storage |
|--------|----------|--------|---------------|---------|---------|
| small  | 10       | 3      | 1             | ~1,200  | ~1 GB   |
| medium | 50       | 3      | 2             | ~12,000 | ~10 GB  |
| full   | 100      | 5      | 5             | ~100,000| ~97 GB  |

## Rudiments Covered

All 40 PAS International Drum Rudiments:

- **Roll Rudiments (15)**: Single/Double/Triple Stroke Rolls, 5-17 Stroke Rolls
- **Diddle Rudiments (5)**: Paradiddles and variants
- **Flam Rudiments (12)**: Flam, Flam Accent, Flam Tap, Flamacue, etc.
- **Drag Rudiments (8)**: Drags, Drag Taps, Ratamacue variants

## License

SOUSA is released under the [MIT License](https://github.com/zakkeown/SOUSA/blob/main/LICENSE).
