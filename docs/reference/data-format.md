# Data Format

This document describes the complete data schema for the SOUSA dataset, including file structure, Parquet schemas, and format specifications.

## Overview

SOUSA uses a hierarchical label structure with three levels of granularity:

```
Exercise (1 per sample)
  |-- Measures (N per exercise)
        |-- Strokes (M per measure)
```

Each sample includes:

- **MIDI file**: Symbolic performance data
- **Audio file**: Rendered and augmented audio (FLAC)
- **Labels**: Hierarchical scores in Parquet format

---

## File Structure

### Local Dataset

```
output/dataset/
|-- audio/                    # FLAC audio files (44.1kHz, 24-bit)
|   |-- {sample_id}.flac
|-- midi/                     # Standard MIDI files
|   |-- {sample_id}.mid
|-- labels/                   # Parquet files with hierarchical labels
|   |-- samples.parquet       # Sample metadata
|   |-- exercises.parquet     # Exercise-level scores
|   |-- measures.parquet      # Measure-level statistics
|   |-- strokes.parquet       # Stroke-level events
|-- index.json                # Dataset index
|-- splits.json               # Train/val/test profile assignments
|-- README.md                 # Dataset card
```

### HuggingFace Format

```
username/sousa/
|-- data/
|   |-- train-00000-of-00001.parquet
|   |-- validation-00000-of-00001.parquet
|   |-- test-00000-of-00001.parquet
|-- audio/
|   |-- {sample_id}.flac
|-- midi/
|   |-- {sample_id}.mid
|-- README.md
```

---

## Sample ID Convention

Sample IDs encode key metadata for easy filtering:

```
{skill_tier}{profile_num}_{rudiment}_{tempo}bpm_{soundfont}_{augmentation}
```

| Component | Description | Examples |
|-----------|-------------|----------|
| `skill_tier` | 3-letter skill code | `beg`, `int`, `adv`, `pro` |
| `profile_num` | 3-digit profile number | `000`-`099` |
| `rudiment` | Snake_case rudiment name | `single_paradiddle`, `flam_tap` |
| `tempo` | BPM value | `60`-`180` |
| `soundfont` | Soundfont identifier | `generalu`, `marching`, `mtpowerd` |
| `augmentation` | Augmentation preset | `cleanstudio`, `practiceroom`, `gym` |

**Example**: `adv042_single_paradiddle_100bpm_marching_practiceroom`

---

## Schema Definitions

### Sample Metadata (`samples.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique sample identifier |
| `profile_id` | string | UUID of player profile |
| `rudiment_slug` | string | Rudiment identifier (snake_case) |
| `tempo_bpm` | int | Performance tempo |
| `duration_sec` | float | Audio duration in seconds |
| `num_cycles` | int | Number of rudiment repetitions |
| `skill_tier` | string | One of: `beginner`, `intermediate`, `advanced`, `professional` |
| `skill_tier_binary` | string | One of: `novice`, `skilled` |
| `dominant_hand` | string | `right` or `left` |
| `midi_path` | string | Relative path to MIDI file |
| `audio_path` | string | Relative path to audio file |
| `num_strokes` | int | Total strokes in performance |
| `num_measures` | int | Total measures in performance |

#### Audio Augmentation Columns

Prefixed with `aug_`:

| Column | Type | Description |
|--------|------|-------------|
| `aug_soundfont` | string | Soundfont used for synthesis |
| `aug_room_type` | string | Room simulation type |
| `aug_room_wet_dry` | float | Room reverb wet/dry ratio (0-1) |
| `aug_mic_distance` | float | Simulated mic distance (meters) |
| `aug_mic_type` | string | Microphone type simulation |
| `aug_compression_ratio` | float | Compression ratio applied |
| `aug_noise_level_db` | float | Added noise level (dB) |
| `aug_bit_depth` | int | Output bit depth |
| `aug_sample_rate` | int | Output sample rate (Hz) |

---

### Exercise Scores (`exercises.parquet`)

All scores are 0-100 (higher = better performance).

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Sample identifier |

#### Timing Scores

| Column | Type | Description |
|--------|------|-------------|
| `timing_accuracy` | float | How close strokes are to intended timing |
| `timing_consistency` | float | Variance in timing errors (lower variance = higher score) |
| `tempo_stability` | float | Consistency of overall tempo throughout |
| `subdivision_evenness` | float | Evenness of note subdivisions |

#### Dynamics Scores

| Column | Type | Description |
|--------|------|-------------|
| `velocity_control` | float | Control over stroke dynamics |
| `accent_differentiation` | float | Clarity between accented/unaccented strokes |
| `accent_accuracy` | float | Correct placement of accents |

#### Balance Scores

| Column | Type | Description |
|--------|------|-------------|
| `hand_balance` | float | Evenness between L/R hand strokes |
| `weak_hand_index` | float | 0=left weak, 100=right weak, 50=balanced |

#### Rudiment-Specific Scores (nullable)

| Column | Type | Description |
|--------|------|-------------|
| `flam_quality` | float? | Grace note spacing quality (flam rudiments only) |
| `diddle_quality` | float? | Diddle stroke evenness (paradiddles only) |
| `roll_sustain` | float? | Roll smoothness (roll rudiments only) |

#### Composite Scores

| Column | Type | Description |
|--------|------|-------------|
| `groove_feel_proxy` | float | Groove/feel metric (0-1 scale) |
| `overall_score` | float | Weighted composite of all metrics |
| `tier_confidence` | float | Confidence that skill_tier label is unambiguous (0-1) |

---

### Measure Labels (`measures.parquet`)

Per-measure aggregate statistics.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Sample identifier |
| `index` | int | Measure index (0-based) |
| `stroke_start` | int | First stroke index in measure |
| `stroke_end` | int | Last stroke index (exclusive) |

#### Timing Statistics

| Column | Type | Description |
|--------|------|-------------|
| `timing_mean_error_ms` | float | Mean timing error (milliseconds) |
| `timing_std_ms` | float | Timing error standard deviation |
| `timing_max_error_ms` | float | Maximum timing error |

#### Velocity Statistics

| Column | Type | Description |
|--------|------|-------------|
| `velocity_mean` | float | Mean velocity (0-127 MIDI scale) |
| `velocity_std` | float | Velocity standard deviation |
| `velocity_consistency` | float | 1 - (std/mean), higher = more consistent |

#### Hand Balance

| Column | Type | Description |
|--------|------|-------------|
| `lr_velocity_ratio` | float? | Left/Right velocity ratio (null if single-hand) |
| `lr_timing_diff_ms` | float? | L/R timing difference (ms) |

---

### Stroke Labels (`strokes.parquet`)

Individual stroke-level events.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Sample identifier |
| `index` | int | Stroke index (0-based) |
| `hand` | string | `L` or `R` |
| `stroke_type` | string | Stroke type identifier |

#### Timing

| Column | Type | Description |
|--------|------|-------------|
| `intended_time_ms` | float | Intended stroke time (ms from start) |
| `actual_time_ms` | float | Actual stroke time |
| `timing_error_ms` | float | actual - intended (positive = late) |

#### Velocity

| Column | Type | Description |
|--------|------|-------------|
| `intended_velocity` | int | Intended velocity (0-127) |
| `actual_velocity` | int | Actual velocity |
| `velocity_error` | int | actual - intended |

#### Articulation Flags

| Column | Type | Description |
|--------|------|-------------|
| `is_grace_note` | bool | True if grace note (flams) |
| `is_accent` | bool | True if accented stroke |
| `diddle_position` | int? | Position in diddle (null if not diddle) |
| `flam_spacing_ms` | float? | Actual spacing to primary stroke (grace notes only) |
| `parent_stroke_index` | int? | Index of primary stroke (grace notes only) |

---

## Audio Format

| Property | Value |
|----------|-------|
| Format | FLAC (lossless) |
| Sample Rate | 44,100 Hz |
| Bit Depth | 24-bit |
| Channels | Mono |
| Duration | 4-12 seconds typical |

---

## MIDI Format

Standard MIDI Type 1 files with:

| Property | Value |
|----------|-------|
| Track 0 | Tempo and time signature |
| Track 1 | Note events (channel 10 for drums) |
| Note Numbers | Standard GM drum mapping (38 = snare) |
| Velocity | 1-127 (dynamics) |
| PPQ | 480 ticks per quarter note |

---

## Augmentation Presets

| Preset | Room | Mic | Compression | Noise |
|--------|------|-----|-------------|-------|
| `cleanstudio` | None | Close | None | None |
| `cleanclosed` | Small | Close | Light | None |
| `practiceroom` | Small practice | Medium | Light | Low |
| `concerthall` | Large hall | Far | None | Low |
| `gym` | Gymnasium | Far | None | Medium |
| `garage` | Garage | Medium | Medium | Medium |
| `vintagetape` | Medium | Medium | Tape saturation | Tape hiss |
| `lofi` | Variable | Variable | Heavy | High |
| `phonerecording` | None | Poor | Heavy limiting | High |

---

## Soundfonts

| Identifier | Description |
|------------|-------------|
| `generalu` | GeneralUser GS - General purpose |
| `marching` | Marching snare focused |
| `mtpowerd` | MT Power Drums - Rock kit |
| `douglasn` | Douglas Drums - Natural kit |
| `fluidr3` | FluidR3 GM - Standard GM |

---

## Train/Val/Test Splits

Default 70/15/15 split by **profile** (not by sample):

| Split | Profiles | Samples (approx.) |
|-------|----------|-------------------|
| train | 70 | ~70,000 |
| validation | 15 | ~15,000 |
| test | 15 | ~15,000 |

!!! note "Profile-Based Splits"
    Splits are assigned at the profile level to prevent data leakage from player-specific timing/velocity patterns. This tests generalization to "new players" not seen during training.

**Stratification**: Splits maintain skill tier proportions:

```
Train skill distribution == Val skill distribution == Test skill distribution
```

---

## Class Balance and Distribution

### Skill Tier Distribution

SOUSA uses a non-uniform distribution approximating realistic player populations:

| Skill Tier | Target Proportion | Full Dataset (~100K) | Medium (~12K) |
|------------|-------------------|---------------------|---------------|
| Beginner | 25% | ~25,000 | ~3,000 |
| Intermediate | 35% | ~35,000 | ~4,200 |
| Advanced | 25% | ~25,000 | ~3,000 |
| Professional | 15% | ~15,000 | ~1,800 |

### Rudiment Distribution

All 40 PAS rudiments are equally represented:

| Preset | Profiles | Tempos/Rudiment | Augmentations | Samples/Rudiment |
|--------|----------|-----------------|---------------|------------------|
| small | 10 | 3 | 1 | ~30 |
| medium | 50 | 3 | 2 | ~300 |
| full | 100 | 5 | 5 | ~2,500 |

### Score-Specific Availability

Some scores are only computed for rudiments containing specific articulations:

| Score Column | Available For | % Non-Null |
|--------------|---------------|------------|
| `flam_quality` | Flam rudiments (20-30) | ~27.5% |
| `diddle_quality` | Diddle/roll rudiments (6-19, 24-26, 35-37) | ~47.5% |
| `roll_sustain` | Roll rudiments (1-15) | ~37.5% |

**Handling in Training:**

- Use masking for null values in loss computation
- Consider separate prediction heads per articulation type
- Filter to non-null subset if focusing on specific rudiments

---

## Rudiments

### Roll Rudiments (15)

| Slug | Name |
|------|------|
| `single_stroke_roll` | Single Stroke Roll |
| `single_stroke_four` | Single Stroke Four |
| `single_stroke_seven` | Single Stroke Seven |
| `multiple_bounce_roll` | Multiple Bounce Roll |
| `triple_stroke_roll` | Triple Stroke Roll |
| `double_stroke_roll` | Double Stroke Open Roll |
| `five_stroke_roll` | Five Stroke Roll |
| `six_stroke_roll` | Six Stroke Roll |
| `seven_stroke_roll` | Seven Stroke Roll |
| `nine_stroke_roll` | Nine Stroke Roll |
| `ten_stroke_roll` | Ten Stroke Roll |
| `eleven_stroke_roll` | Eleven Stroke Roll |
| `thirteen_stroke_roll` | Thirteen Stroke Roll |
| `fifteen_stroke_roll` | Fifteen Stroke Roll |
| `seventeen_stroke_roll` | Seventeen Stroke Roll |

### Diddle Rudiments (4)

| Slug | Name |
|------|------|
| `single_paradiddle` | Single Paradiddle |
| `double_paradiddle` | Double Paradiddle |
| `triple_paradiddle` | Triple Paradiddle |
| `paradiddle_diddle` | Paradiddle-Diddle |

### Flam Rudiments (11)

| Slug | Name |
|------|------|
| `flam` | Flam |
| `flam_accent` | Flam Accent |
| `flam_tap` | Flam Tap |
| `flamacue` | Flamacue |
| `flam_paradiddle` | Flam Paradiddle |
| `single_flammed_mill` | Single Flammed Mill |
| `flam_paradiddle_diddle` | Flam Paradiddle-Diddle |
| `pataflafla` | Pataflafla |
| `swiss_army_triplet` | Swiss Army Triplet |
| `inverted_flam_tap` | Inverted Flam Tap |
| `flam_drag` | Flam Drag |

### Drag Rudiments (10)

| Slug | Name |
|------|------|
| `drag` | Drag |
| `single_drag_tap` | Single Drag Tap |
| `double_drag_tap` | Double Drag Tap |
| `lesson_25` | Lesson 25 |
| `single_dragadiddle` | Single Dragadiddle |
| `drag_paradiddle_1` | Drag Paradiddle #1 |
| `drag_paradiddle_2` | Drag Paradiddle #2 |
| `single_ratamacue` | Single Ratamacue |
| `double_ratamacue` | Double Ratamacue |
| `triple_ratamacue` | Triple Ratamacue |

---

## Score Computation Reference

### Timing Accuracy

Uses sigmoid scaling for perceptual alignment:

$$
\text{timing\_accuracy} = 100 \times \frac{1}{1 + e^{(\bar{e} - 25) / 10}}
$$

Where `max_expected_error_ms` scales with tempo (faster = tighter tolerance).

### Velocity Control

$$
\text{velocity\_control} = 100 \times (1 - \text{velocity\_std} / \text{expected\_std\_for\_tier})
$$

### Overall Score

Weighted combination:

| Component | Weight |
|-----------|--------|
| timing_accuracy | 0.20 |
| timing_consistency | 0.15 |
| velocity_control | 0.10 |
| accent_accuracy | 0.10 |
| hand_balance | 0.15 |
| tempo_stability | 0.10 |
| subdivision_evenness | 0.10 |
| accent_differentiation | 0.10 |

Plus rudiment-specific bonuses/penalties when applicable.

---

## Validation Report Schema

The `validation_report.json` file contains comprehensive dataset validation results.

### Top-Level Structure

| Field | Type | Description |
|-------|------|-------------|
| `dataset_path` | string | Path to validated dataset |
| `generated_at` | string | ISO 8601 timestamp |
| `stats` | object | Statistical analysis results |
| `verification` | object | Data integrity check results |
| `skill_tier_ordering` | object | Skill tier ordering verification |
| `realism` | object | Literature validation results |

### Stats Object

| Field | Type | Description |
|-------|------|-------------|
| `num_samples` | int | Total sample count |
| `num_profiles` | int | Unique player profiles |
| `num_rudiments` | int | Unique rudiments |
| `tempo` | DistributionStats | Tempo BPM distribution |
| `duration` | DistributionStats | Duration in seconds |
| `num_strokes` | DistributionStats | Strokes per sample |
| `timing_error` | DistributionStats | Stroke timing error (ms) |
| `velocity` | DistributionStats | MIDI velocity values |
| `skill_tier_counts` | object | Samples per skill tier |
| `rudiment_counts` | object | Samples per rudiment |
| `split_counts` | object | Train/val/test counts |

### DistributionStats Object

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Metric name |
| `count` | int | Number of observations |
| `mean` | float | Arithmetic mean |
| `std` | float | Standard deviation |
| `min` | float | Minimum value |
| `max` | float | Maximum value |
| `median` | float | 50th percentile |
| `q25` | float | 25th percentile |
| `q75` | float | 75th percentile |
| `skewness` | float | Distribution skewness |
| `kurtosis` | float | Distribution kurtosis |
| `by_group` | object? | Optional breakdown by skill tier |

---

## Loading the Dataset

### With Pandas

```python
import pandas as pd

# Load label files
samples = pd.read_parquet("output/dataset/labels/samples.parquet")
exercises = pd.read_parquet("output/dataset/labels/exercises.parquet")
measures = pd.read_parquet("output/dataset/labels/measures.parquet")
strokes = pd.read_parquet("output/dataset/labels/strokes.parquet")

# Join for analysis
data = samples.merge(exercises, on="sample_id")
```

### With HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("zkeown/sousa")

# Access splits
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

# Filter by rudiment
paradiddles = train.filter(lambda x: "paradiddle" in x["rudiment_slug"])
```

### Loading Audio

```python
import soundfile as sf

# Load audio file
audio, sr = sf.read("output/dataset/audio/sample_id.flac")
print(f"Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s")
```

### Loading MIDI

```python
import mido

midi = mido.MidiFile("output/dataset/midi/sample_id.mid")
for track in midi.tracks:
    for msg in track:
        if msg.type == "note_on":
            print(f"Note: {msg.note}, Velocity: {msg.velocity}, Time: {msg.time}")
```

---

## See Also

- [Architecture](architecture.md) - Pipeline overview
- [Score Computation](score-computation.md) - Detailed scoring algorithms
- [Rudiment Schema](rudiment-schema.md) - YAML format for rudiments
