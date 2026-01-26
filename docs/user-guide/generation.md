# Dataset Generation

This guide covers generating SOUSA datasets with different configurations for testing, development, and production use.

## Generation Presets

SOUSA provides three presets optimized for different use cases:

### Small Preset (Testing)

```bash
python scripts/generate_dataset.py --preset small
```

| Parameter | Value |
|-----------|-------|
| Profiles | 10 |
| Tempos per rudiment | 3 |
| Augmentations per sample | 1 |
| **Total samples** | ~1,200 |

**Storage estimates:**

| Component | Size |
|-----------|------|
| MIDI files | ~2.4 MB |
| Labels (Parquet) | ~0.6 MB |
| Audio (if enabled) | ~1.8 GB |
| **Total (MIDI only)** | ~3 MB |

!!! tip "Use case"
    Quick testing of pipeline changes, CI/CD workflows, and development iteration.

### Medium Preset (Development)

```bash
python scripts/generate_dataset.py --preset medium
```

| Parameter | Value |
|-----------|-------|
| Profiles | 50 |
| Tempos per rudiment | 3 |
| Augmentations per sample | 2 |
| **Total samples** | ~12,000 |

**Storage estimates:**

| Component | Size |
|-----------|------|
| MIDI files | ~24 MB |
| Labels (Parquet) | ~6 MB |
| Audio (if enabled) | ~18 GB |
| **Total (MIDI only)** | ~30 MB |

!!! tip "Use case"
    Model development, hyperparameter tuning, and ablation studies.

### Full Preset (Production)

```bash
python scripts/generate_dataset.py --preset full
```

| Parameter | Value |
|-----------|-------|
| Profiles | 100 |
| Tempos per rudiment | 5 |
| Augmentations per sample | 5 |
| **Total samples** | ~100,000 |

**Storage estimates:**

| Component | Size |
|-----------|------|
| MIDI files | ~200 MB |
| Labels (Parquet) | ~50 MB |
| Audio (if enabled) | ~150 GB |
| **Total (MIDI only)** | ~250 MB |

!!! tip "Use case"
    Final model training, benchmark evaluation, and dataset releases.

## Command Line Options

### Basic Options

```bash
python scripts/generate_dataset.py [OPTIONS]
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `output/dataset` | Output directory |
| `--preset` | | None | Use preset (small/medium/full) |
| `--profiles` | `-p` | 100 | Number of player profiles |
| `--tempos` | `-t` | 5 | Tempos per rudiment |
| `--augmentations` | `-a` | 5 | Augmented versions per sample |
| `--seed` | | 42 | Random seed for reproducibility |
| `--quiet` | `-q` | False | Reduce output verbosity |

### Audio Options

```bash
python scripts/generate_dataset.py --with-audio --soundfont path/to/soundfonts
```

| Option | Description |
|--------|-------------|
| `--with-audio` | Generate audio files (requires soundfonts) |
| `--soundfont` | Path to .sf2 file or directory |

!!! warning "Soundfont setup"
    Audio generation requires soundfonts. Run `python scripts/setup_soundfonts.py` first.

### Parallel Processing

```bash
python scripts/generate_dataset.py --workers 8
```

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--workers` | `-w` | 1 | Number of parallel workers |

Set `--workers 0` to auto-detect CPU count.

### Validation

```bash
python scripts/generate_dataset.py --skip-validation
```

| Option | Description |
|--------|-------------|
| `--skip-validation` | Skip validation report generation |

## Examples

### Quick Test Generation

```bash
# Minimal dataset for testing
python scripts/generate_dataset.py --preset small -o output/test_dataset
```

### Development with Audio

```bash
# Medium dataset with audio for model development
python scripts/generate_dataset.py \
    --preset medium \
    --with-audio \
    --workers 4 \
    -o output/dev_dataset
```

### Production Generation

```bash
# Full dataset with parallel processing
python scripts/generate_dataset.py \
    --preset full \
    --with-audio \
    --workers 0 \
    --seed 42 \
    -o output/production_dataset
```

### Custom Configuration

```bash
# Override preset values
python scripts/generate_dataset.py \
    --preset medium \
    --profiles 30 \
    --tempos 4 \
    --augmentations 3
```

## Reproducibility and Seeding

SOUSA supports deterministic generation through seeding:

```bash
# Both commands produce identical datasets
python scripts/generate_dataset.py --seed 42 --preset small
python scripts/generate_dataset.py --seed 42 --preset small
```

### Seed Behavior

- **Profile generation**: Same seed produces identical player profiles
- **MIDI generation**: Timing/velocity variations are deterministic per seed
- **Audio synthesis**: FluidSynth rendering is deterministic
- **Augmentation**: Random augmentations use seed-derived values

!!! note "Parallel generation seeds"
    When using multiple workers, each worker uses `seed + worker_id` to ensure reproducibility while avoiding identical samples across workers.

### Recommended Seeds

For benchmarking and reproducibility, document your seed:

```python
# In your experiment config
DATASET_CONFIG = {
    "seed": 42,
    "preset": "full",
    "version": "1.0.0",
}
```

## Generation Pipeline

The generation process follows this flow:

```
1. Load Rudiments (40 YAML definitions)
         ↓
2. Generate Player Profiles
         ↓
3. Assign Train/Val/Test Splits
         ↓
4. For each Profile × Rudiment × Tempo:
   a. Generate base MIDI performance
   b. Apply profile-based timing/velocity variations
   c. Compute hierarchical labels (stroke → measure → exercise)
   d. For each augmentation variant:
      - Synthesize audio (if enabled)
      - Apply augmentation preset
      - Write to storage
         ↓
5. Write Parquet files (samples, exercises, measures, strokes)
         ↓
6. Run validation (unless skipped)
```

## Output Structure

```
output/dataset/
├── audio/                    # FLAC audio files (if --with-audio)
│   └── {sample_id}.flac
├── midi/                     # Standard MIDI files
│   └── {sample_id}.mid
├── labels/                   # Parquet files
│   ├── samples.parquet       # Sample metadata
│   ├── exercises.parquet     # Exercise-level scores
│   ├── measures.parquet      # Measure-level statistics
│   └── strokes.parquet       # Stroke-level events
├── splits.json               # Train/val/test assignments
└── validation_report.json    # Validation results
```

## Resumable Generation

!!! warning "Coming soon"
    Checkpoint-based resumable generation is planned for a future release.

For now, if generation is interrupted:

1. Check the `output/dataset/labels/` directory for partially written parquet files
2. Delete the incomplete output directory
3. Re-run generation with the same seed

## Performance Tips

### Memory Usage

For large datasets, consider:

```bash
# Use more workers with smaller batches
python scripts/generate_dataset.py --preset full --workers 8
```

### Disk I/O

- Use an SSD for the output directory
- Audio generation is I/O intensive; consider generating MIDI-only first

### Parallel Scaling

| Workers | Speedup (approx) | Notes |
|---------|------------------|-------|
| 1 | 1x | Baseline |
| 4 | 3.5x | Good for most systems |
| 8 | 6x | Diminishing returns |
| 16+ | 7-8x | Limited by I/O |

## Validation After Generation

By default, validation runs automatically after generation:

```
=== SOUSA Validation Report ===
Dataset: output/dataset
Generated: 2026-01-25T21:18:35

Samples: 99,770
Profiles: 100
Rudiments: 40

Verification: 13/13 checks passed
Literature validation: 100% pass rate
Correlation checks: 80% pass rate
```

See [Validation Guide](validation.md) for details on interpreting results.
