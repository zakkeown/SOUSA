---
license: mit
task_categories:
  - audio-classification
  - tabular-classification
  - tabular-regression
tags:
  - audio
  - music
  - drum-rudiments
  - synthetic
  - snare-drum
  - midi
  - percussion
  - music-information-retrieval
size_categories:
  - 100K<n<1M
language:
  - en
pretty_name: SOUSA - Synthetic Open Unified Snare Assessment
configs:
  - config_name: audio
    data_files:
      - split: train
        path: audio/train-*.parquet
      - split: validation
        path: audio/validation-*.parquet
      - split: test
        path: audio/test-*.parquet
    default: true
  - config_name: midi_only
    data_files:
      - split: train
        path: midi_only/train-*.parquet
      - split: validation
        path: midi_only/validation-*.parquet
      - split: test
        path: midi_only/test-*.parquet
  - config_name: labels_only
    data_files:
      - split: train
        path: labels_only/train-*.parquet
      - split: validation
        path: labels_only/validation-*.parquet
      - split: test
        path: labels_only/test-*.parquet
---

# SOUSA: Synthetic Open Unified Snare Assessment

SOUSA is a synthetic dataset of 100,000 drum rudiment performances for machine learning. It covers all 40 PAS (Percussive Arts Society) standard rudiments performed by 100 simulated player profiles across 4 skill tiers, with hierarchical quality labels at the stroke, measure, and exercise level.

## Dataset Summary

| Attribute | Value |
|-----------|-------|
| Total samples | 100,000 |
| Audio files (FLAC) | 100,000 |
| MIDI files | 20,000 |
| Rudiments | 40 (all PAS standard) |
| Skill tiers | beginner, intermediate, advanced, professional |
| Player profiles | 100 |
| Tempo range | 60-180 BPM |
| Splits | train (68K) / val (13K) / test (19K) |
| Audio size | ~120 GB |
| License | MIT |

## Usage

```python
from datasets import load_dataset

# Full dataset with audio (~96GB download)
ds = load_dataset("zkeown/sousa")

# MIDI + labels only (~2.5GB download)
ds = load_dataset("zkeown/sousa", "midi_only")

# Pure tabular labels (~50MB download)
ds = load_dataset("zkeown/sousa", "labels_only")

# Access a sample
sample = ds["train"][0]
print(sample["rudiment_slug"], sample["skill_tier"], sample["overall_score"])

# Access audio (decoded automatically)
audio_array = sample["audio"]["array"]          # numpy array
sample_rate = sample["audio"]["sampling_rate"]  # 44100

# Access MIDI bytes
midi_bytes = sample["midi"]  # raw .mid file bytes
```

## Dataset Structure

### Splits

Samples are split by player profile (not by rudiment or tempo) to prevent data leakage. All performances from a given player appear in exactly one split.

- **Train**: 68 profiles (68,000 samples)
- **Validation**: 13 profiles (13,000 samples)
- **Test**: 19 profiles (19,000 samples)

### File Organization

The dataset uses three configurations with different levels of content:

| Config | Contents | Size |
|--------|----------|------|
| `audio` (default) | FLAC audio + MIDI bytes + metadata + scores | ~96GB |
| `midi_only` | MIDI bytes + metadata + scores | ~2.5GB |
| `labels_only` | Metadata + scores only | ~50MB |

Stroke-level and measure-level labels are available as auxiliary files:

```python
import pandas as pd
strokes = pd.read_parquet("hf://datasets/zkeown/sousa/auxiliary/strokes.parquet")
measures = pd.read_parquet("hf://datasets/zkeown/sousa/auxiliary/measures.parquet")
```

### Features

#### Sample Metadata

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique identifier |
| `profile_id` | string | Player profile UUID |
| `rudiment_slug` | string | Rudiment name (e.g., `single_stroke_roll`) |
| `tempo_bpm` | int | Performance tempo (60-180) |
| `duration_sec` | float | Performance duration in seconds |
| `num_cycles` | int | Number of rudiment repetitions |
| `skill_tier` | string | `beginner` / `intermediate` / `advanced` / `professional` |
| `skill_tier_binary` | string | `novice` / `skilled` |
| `dominant_hand` | string | `right` / `left` |
| `num_strokes` | int | Total strokes in the performance |
| `num_measures` | int | Number of measures |
| `soundfont` | string | Soundfont used for synthesis |
| `augmentation_preset` | string | Audio environment preset name |
| `augmentation_group_id` | string | Links augmented variants of the same MIDI |

#### Audio Augmentation Parameters

| Column | Type | Description |
|--------|------|-------------|
| `aug_soundfont` | string | Soundfont used for this variant |
| `aug_room_type` | string | Simulated room (studio, gym, garage, etc.) |
| `aug_room_wet_dry` | float | Reverb wet/dry mix |
| `aug_mic_distance` | float | Simulated mic distance (meters) |
| `aug_mic_type` | string | Mic model (condenser, dynamic) |
| `aug_compression_ratio` | float | Audio compression ratio |
| `aug_noise_level_db` | float | Background noise level |
| `aug_bit_depth` | int | Bit depth (if degraded) |
| `aug_sample_rate` | int | Sample rate (if degraded) |

#### Exercise-Level Scores (0-100)

| Column | Type | Description |
|--------|------|-------------|
| `timing_accuracy` | float | How close strokes are to the ideal beat grid |
| `timing_consistency` | float | Variability of timing deviations |
| `tempo_stability` | float | Drift from target tempo over time |
| `subdivision_evenness` | float | Evenness of subdivisions within beats |
| `velocity_control` | float | Consistency of stroke dynamics |
| `accent_differentiation` | float | Contrast between accented and unaccented strokes |
| `accent_accuracy` | float | Whether accents land on the correct beats |
| `hand_balance` | float | Evenness between left and right hands |
| `weak_hand_index` | float | Relative weakness of the non-dominant hand |
| `flam_quality` | float | Quality of flam rudiment execution |
| `diddle_quality` | float | Quality of diddle/double-stroke execution |
| `roll_sustain` | float | Sustain quality for roll rudiments |
| `groove_feel_proxy` | float | Overall rhythmic feel metric |
| `overall_score` | float | Weighted composite score |

## The 40 PAS Rudiments

Single stroke roll, single stroke four, single stroke seven, multiple bounce roll, double stroke open roll, five stroke roll, six stroke roll, seven stroke roll, nine stroke roll, ten stroke roll, eleven stroke roll, thirteen stroke roll, fifteen stroke roll, seventeen stroke roll, flam, flam accent, flam tap, flamacue, flam paradiddle, single flammed mill, flam paradiddle-diddle, pataflafla, swiss army triplet, inverted flam tap, flam drag, drag, single drag tap, double drag tap, lesson 25, single dragadiddle, drag paradiddle #1, drag paradiddle #2, single ratamacue, double ratamacue, triple ratamacue, single paradiddle, double paradiddle, triple paradiddle, paradiddle-diddle.

## Generation

This dataset was generated using [SOUSA](https://github.com/zkeown/sousa), which synthesizes realistic drum performances by:

1. Defining rudiment patterns from YAML specifications
2. Creating player profiles with correlated skill dimensions
3. Generating MIDI with profile-based timing/velocity variation
4. Rendering audio via FluidSynth with multiple soundfonts
5. Applying augmentation chains (room IR, mic modeling, compression, noise)
6. Computing hierarchical quality labels at stroke, measure, and exercise level

```bash
pip install sousa
python3 scripts/generate_dataset.py --preset full --with-audio
```

## Citation

```bibtex
@dataset{keown2026sousa,
  title={SOUSA: Synthetic Open Unified Snare Assessment},
  author={Keown, Zak},
  year={2026},
  url={https://huggingface.co/datasets/zkeown/sousa},
  publisher={Hugging Face}
}
```
