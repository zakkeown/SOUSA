# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SOUSA (Synthetic Open Unified Snare Assessment) generates synthetic drum rudiment datasets for ML training. It produces 100K+ samples of all 40 PAS drum rudiments with MIDI, audio, and hierarchical labels.

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install (editable mode)
pip install -e .
pip install -e '.[dev]'      # With dev dependencies (pytest, black, ruff)
pip install -e '.[hub]'      # With HuggingFace dependencies

# Install pre-commit hooks (required - CI will fail without this)
make setup-hooks             # Or: pip install pre-commit && pre-commit install
```

## Commands

```bash

# Run tests
pytest                       # All tests
pytest tests/test_profiles.py  # Single module
pytest tests/test_profiles.py::TestProfileGeneration::test_generate_profile_beginner  # Single test
pytest -v                    # Verbose output

# Lint/format
ruff check dataset_gen/
black dataset_gen/           # Line length: 100

# Generate dataset
python scripts/generate_dataset.py --preset small   # ~1,200 samples (testing)
python scripts/generate_dataset.py --preset medium  # ~12,000 samples (dev)
python scripts/generate_dataset.py --with-audio     # Include audio (requires soundfonts)

# Setup soundfonts (required for audio)
python scripts/setup_soundfonts.py

# Upload to HuggingFace
python scripts/push_to_hub.py zkeown/sousa
```

## Architecture

### Generation Pipeline

The pipeline flows: **Rudiment YAML → PlayerProfile → MIDI → Audio → Labels → Parquet**

1. **Rudiments** (`dataset_gen/rudiments/`) - 40 YAML definitions in `definitions/` specify stroke patterns, sticking (R/L), articulations (accent, tap, diddle, grace), tempo ranges, and subdivisions

2. **Profiles** (`dataset_gen/profiles/`) - `archetypes.py` generates `PlayerProfile` objects with correlated skill dimensions (timing accuracy, hand balance, velocity control). Profiles are partitioned for train/val/test splits

3. **MIDI Gen** (`dataset_gen/midi_gen/`) - `generator.py` applies profile-based timing/velocity variations to rudiment patterns. `articulations.py` handles flams, drags, diddles

4. **Audio Synth** (`dataset_gen/audio_synth/`) - FluidSynth wrapper renders MIDI to audio using SF2 soundfonts (practice pad, marching snare, drum kits)

5. **Audio Aug** (`dataset_gen/audio_aug/`) - Augmentation chain: `room.py` (IR convolution), `mic.py` (mic modeling), `chain.py` (compression/EQ), `degradation.py` (noise, lo-fi)

6. **Labels** (`dataset_gen/labels/`) - `compute.py` calculates hierarchical scores at stroke/measure/exercise levels. `groove.py` computes feel metrics. Scores are 0-100 scale

7. **Pipeline** (`dataset_gen/pipeline/`) - `generate.py` orchestrates generation, `parallel.py` handles multiprocessing, `checkpoint.py` enables resumable generation, `storage.py` writes Parquet

### Key Types

- `Rudiment` / `StickingPattern` - Rudiment definitions from YAML
- `PlayerProfile` / `SkillTier` - Player skill modeling (beginner/intermediate/advanced/professional)
- `Sample` / `StrokeLabel` / `MeasureLabel` / `ExerciseScores` - Hierarchical label schemas

### Data Locations

- `data/soundfonts/` - SF2 files (downloaded via setup_soundfonts.py)
- `data/impulse_responses/` - Room IRs for reverb
- `data/noise_profiles/` - Background noise samples
- `output/dataset/` - Generated dataset (midi/, audio/, labels/)

## External Dependencies

- **FluidSynth**: Required for audio synthesis. Install via `brew install fluid-synth` (macOS)
- **Ray**: Optional distributed processing. Install with `pip install 'sousa[cloud]'`
