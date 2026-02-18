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

## Evaluation & Validation

After generating a dataset, use the following tools to validate quality. Run them in order from quick checks to deep analysis.

### Post-Generation Validation (run after every regeneration)

```bash
# 1. Quick health check (seconds) — exit code 1 = critical issues
python scripts/check_generation.py output/dataset

# 2. Dataset statistics overview
python scripts/dataset_stats.py output/dataset              # Console output
python scripts/dataset_stats.py output/dataset --markdown    # For dataset card
python scripts/dataset_stats.py output/dataset --json stats.json

# 3. Full test suite (includes label verification, tier ordering, realism checks)
pytest tests/ -v

# 4. Audio quality validation (requires --with-audio generation)
python -m dataset_gen.validation.audio_check output/dataset --sample-size 100
```

### Deep Analysis (run before publishing or after pipeline changes)

```bash
# 5. ML utility proof — trains baseline classifiers/regressors, reports learnability
#    Outputs: output/dataset/proofs/utility_report.json
#    Pass criteria: classification acc > 0.7 AND regression R² > 0.5
python scripts/prove_utility.py output/dataset

# 6. Publication plots — 6 PNGs for dataset card
#    Outputs: score_distribution, timing_accuracy, rudiment_distribution,
#             skill_tier_distribution, tempo_distribution, timing_vs_velocity
python scripts/generate_plots.py output/dataset --output-dir output/dataset/plots
```

### Audio Inspection (for debugging individual samples)

```bash
# Analyze a single audio file — renders waveform, onsets, dashboard, cycle zoom
python scripts/analyze_audio.py output/dataset/audio/SAMPLE.flac \
    --midi output/dataset/midi/SAMPLE.mid --view all

# Views: waveform | onsets | dashboard | cycles | all
# Dashboard panel 4 shows velocity-amplitude correlation per stroke
# Cycle zoom shows per-stroke match quality (green=matched, red=missed)
```

### Validation Modules (programmatic access)

| Module | Purpose | Key checks |
|--------|---------|------------|
| `dataset_gen.validation.verify` | Label integrity | 14 checks: parquet integrity, ID uniqueness, ref validity, score ranges, tier ordering, MIDI alignment |
| `dataset_gen.validation.analysis` | Statistical analysis | Distribution stats by stroke/measure/exercise level, per-tier breakdowns |
| `dataset_gen.validation.realism` | Literature comparison | Timing SD and velocity CV vs. published benchmarks (Fujii 2011, Repp 2005, etc.) |
| `dataset_gen.validation.audio_check` | Audio quality | File integrity, silence, clipping, duration match, RMS consistency |
| `dataset_gen.validation.report` | Integrated report | Combines all validators into single JSON report |

### Expected Pass Criteria

- **Label verification**: 14/14 checks pass
- **Realism - literature**: 8/8 comparisons within published ranges
- **Realism - correlations**: timing_accuracy↔consistency r≥0.5, timing↔overall r≥0.6
- **Skill tier ordering**: timing_accuracy and hand_balance properly ordered across all 4 tiers
- **Audio quality**: <1% clipping, <20% silence, RMS in [-40, -6] dB range
- **ML utility**: classification accuracy > 0.7, regression R² > 0.5
- **Velocity-amplitude correlation**: mean within-sample r > 0.3 (post velocity gain curve)

## External Dependencies

- **FluidSynth**: Required for audio synthesis. Install via `brew install fluid-synth` (macOS)
- **Ray**: Optional distributed processing. Install with `pip install 'sousa[cloud]'`
