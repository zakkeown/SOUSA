# Quick Start

Generate and explore a SOUSA dataset in 5 minutes.

!!! info "Prerequisites"
    This guide assumes you have completed the [Installation](installation.md) steps, including:

    - Python 3.10+ installed
    - SOUSA installed (`pip install -e .`)
    - Soundfonts downloaded (`python scripts/setup_soundfonts.py`)
    - FluidSynth installed (for audio generation)

## Generate a Test Dataset

Generate a small test dataset (~1,200 samples) to verify your setup:

```bash
python scripts/generate_dataset.py --preset small --with-audio
```

??? note "Without Audio"
    If you don't have FluidSynth installed or want faster generation:
    ```bash
    python scripts/generate_dataset.py --preset small
    ```
    This generates MIDI files and labels only (no audio).

Expected output:

```
============================================================
DATASET GENERATION PLAN
============================================================
Preset: small - Quick testing (~1,200 samples)

Configuration:
  10 player profiles
  40 rudiments
  3 tempos per rudiment
  1 augmentations per sample

Total samples: 1,200

Estimated storage:
  MIDI files:  ~2.3 MB
  Labels:      ~0.6 MB
  Audio files: ~1.8 GB
  TOTAL:       ~1.8 GB
============================================================

Loaded 40 rudiments
...
Dataset saved to: output/dataset
Total time: X.X minutes
```

## Explore the Dataset

### Dataset Structure

After generation, your dataset is organized as:

```
output/dataset/
├── midi/              # MIDI files (*.mid)
├── audio/             # Audio files (*.flac)
├── labels/            # Parquet files with labels
│   ├── samples.parquet
│   ├── exercises.parquet
│   ├── measures.parquet
│   └── strokes.parquet
├── splits.json        # Train/val/test split assignments
└── validation_report.json
```

### Load Samples in Python

```python
import pandas as pd
from pathlib import Path

# Load the samples table
samples_df = pd.read_parquet("output/dataset/labels/samples.parquet")

# View basic info
print(f"Total samples: {len(samples_df)}")
print(f"Columns: {list(samples_df.columns)}")
print(samples_df.head())
```

Output:

```
Total samples: 1200
Columns: ['sample_id', 'profile_id', 'rudiment_slug', 'tempo_bpm', 'soundfont',
          'augmentation_preset', 'split', 'skill_tier', 'overall_score', ...]
```

### Access a Sample

```python
# Get the first sample
sample = samples_df.iloc[0]

print(f"Sample ID: {sample['sample_id']}")
print(f"Rudiment: {sample['rudiment_slug']}")
print(f"Tempo: {sample['tempo_bpm']} BPM")
print(f"Skill Tier: {sample['skill_tier']}")
print(f"Overall Score: {sample['overall_score']:.1f}")
```

### Load Audio

```python
import soundfile as sf

# Get audio path from sample
audio_path = Path("output/dataset/audio") / f"{sample['sample_id']}.flac"

# Load audio
audio_data, sample_rate = sf.read(audio_path)
print(f"Duration: {len(audio_data) / sample_rate:.2f} seconds")
print(f"Sample rate: {sample_rate} Hz")
```

### Load MIDI

```python
import mido

# Get MIDI path from sample
midi_path = Path("output/dataset/midi") / f"{sample['sample_id']}.mid"

# Load MIDI
mid = mido.MidiFile(midi_path)

# Count notes
note_count = sum(1 for track in mid.tracks for msg in track if msg.type == 'note_on')
print(f"Total notes: {note_count}")
```

## Filter Samples

### By Skill Level

```python
# Filter to beginners only
beginners = samples_df[samples_df['skill_tier'] == 'beginner']
print(f"Beginner samples: {len(beginners)}")

# Filter to advanced and professional
advanced = samples_df[samples_df['skill_tier'].isin(['advanced', 'professional'])]
print(f"Advanced/Professional samples: {len(advanced)}")
```

### By Rudiment

```python
# Get all paradiddle variants
paradiddles = samples_df[samples_df['rudiment_slug'].str.contains('paradiddle')]
print(f"Paradiddle samples: {len(paradiddles)}")

# List unique rudiments
rudiments = samples_df['rudiment_slug'].unique()
print(f"Rudiments in dataset: {len(rudiments)}")
```

### By Score

```python
# High performers (score > 80)
high_scores = samples_df[samples_df['overall_score'] > 80]
print(f"High score samples: {len(high_scores)}")

# Poor timing (timing_accuracy < 50)
poor_timing = samples_df[samples_df['timing_accuracy'] < 50]
print(f"Poor timing samples: {len(poor_timing)}")
```

### By Split

```python
# Get training set
train = samples_df[samples_df['split'] == 'train']
val = samples_df[samples_df['split'] == 'val']
test = samples_df[samples_df['split'] == 'test']

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

## View Validation Report

The generation process automatically validates the dataset:

```python
import json

with open("output/dataset/validation_report.json") as f:
    report = json.load(f)

print(f"All checks passed: {report['verification']['all_passed']}")
print(f"Total checks: {report['verification']['total_checks']}")
```

## Play Audio (Jupyter)

If you're in a Jupyter notebook:

```python
from IPython.display import Audio

# Load audio
audio_path = Path("output/dataset/audio") / f"{sample['sample_id']}.flac"
audio_data, sr = sf.read(audio_path)

# Play in notebook
Audio(audio_data, rate=sr)
```

## Quick Statistics

```python
# Score statistics by skill tier
print(samples_df.groupby('skill_tier')['overall_score'].describe())

# Sample counts by rudiment
print(samples_df['rudiment_slug'].value_counts().head(10))

# Tempo distribution
print(samples_df['tempo_bpm'].value_counts().sort_index())
```

## Generation Presets

SOUSA provides three presets for different use cases:

| Preset | Use Case | Samples | Time | Storage |
|--------|----------|---------|------|---------|
| `small` | Testing/debugging | ~1,200 | ~5 min | ~1 GB |
| `medium` | Development | ~12,000 | ~30 min | ~10 GB |
| `full` | Production | ~100,000 | ~4 hours | ~97 GB |

```bash
# Quick test
python scripts/generate_dataset.py --preset small --with-audio

# Development
python scripts/generate_dataset.py --preset medium --with-audio

# Full production dataset
python scripts/generate_dataset.py --preset full --with-audio
```

## Custom Configuration

Override preset values for custom generation:

```bash
# Custom profile count
python scripts/generate_dataset.py --preset small --profiles 20 --with-audio

# Custom tempo sampling
python scripts/generate_dataset.py --preset small --tempos 5 --with-audio

# Custom output directory
python scripts/generate_dataset.py --preset small --output ./my_dataset --with-audio

# Parallel generation (faster)
python scripts/generate_dataset.py --preset medium --workers 4 --with-audio
```

## Next Steps

Now that you have a dataset:

1. [Loading Data](loading.md) - Load from HuggingFace or local files
2. [User Guide](../user-guide/index.md) - Detailed ML task examples
3. [Data Format](../reference/data-format.md) - Full schema documentation
4. [Validation](../user-guide/validation.md) - Quality assurance details
