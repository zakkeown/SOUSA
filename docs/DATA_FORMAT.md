# SOUSA Data Format Documentation

This document describes the complete data schema for the SOUSA dataset.

## Overview

SOUSA uses a hierarchical label structure with three levels of granularity:

```
Exercise (1 per sample)
  └── Measures (N per exercise)
        └── Strokes (M per measure)
```

## File Structure

### Local Dataset

```
output/dataset/
├── audio/                    # FLAC audio files (44.1kHz, 24-bit)
│   └── {sample_id}.flac
├── midi/                     # Standard MIDI files
│   └── {sample_id}.mid
├── labels/                   # Parquet files with hierarchical labels
│   ├── samples.parquet       # Sample metadata
│   ├── exercises.parquet     # Exercise-level scores
│   ├── measures.parquet      # Measure-level statistics
│   └── strokes.parquet       # Stroke-level events
├── index.json                # Dataset index
├── splits.json               # Train/val/test profile assignments
└── README.md                 # Dataset card
```

### HuggingFace Format

```
username/sousa/
├── data/
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── audio/
│   └── {sample_id}.flac
├── midi/
│   └── {sample_id}.mid
└── README.md
```

## Sample ID Convention

Sample IDs encode key metadata:

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
| `dominant_hand` | string | `right` or `left` |
| `midi_path` | string | Relative path to MIDI file |
| `audio_path` | string | Relative path to audio file |
| `num_strokes` | int | Total strokes in performance |
| `num_measures` | int | Total measures in performance |

**Audio Augmentation Columns** (prefixed with `aug_`):

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

### Exercise Scores (`exercises.parquet`)

All scores are 0-100 (higher = better performance).

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Sample identifier |

**Timing Scores**:

| Column | Type | Description |
|--------|------|-------------|
| `timing_accuracy` | float | How close strokes are to intended timing |
| `timing_consistency` | float | Variance in timing errors (lower variance = higher score) |
| `tempo_stability` | float | Consistency of overall tempo throughout |
| `subdivision_evenness` | float | Evenness of note subdivisions |

**Dynamics Scores**:

| Column | Type | Description |
|--------|------|-------------|
| `velocity_control` | float | Control over stroke dynamics |
| `accent_differentiation` | float | Clarity between accented/unaccented strokes |
| `accent_accuracy` | float | Correct placement of accents |

**Balance Scores**:

| Column | Type | Description |
|--------|------|-------------|
| `hand_balance` | float | Evenness between L/R hand strokes |
| `weak_hand_index` | float | 0=left weak, 100=right weak, 50=balanced |

**Rudiment-Specific Scores** (nullable):

| Column | Type | Description |
|--------|------|-------------|
| `flam_quality` | float? | Grace note spacing quality (flam rudiments only) |
| `diddle_quality` | float? | Diddle stroke evenness (paradiddles only) |
| `roll_sustain` | float? | Roll smoothness (roll rudiments only) |

**Composite Scores**:

| Column | Type | Description |
|--------|------|-------------|
| `groove_feel_proxy` | float | Groove/feel metric (0-1 scale) |
| `overall_score` | float | Weighted composite of all metrics |

### Measure Labels (`measures.parquet`)

Per-measure aggregate statistics.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Sample identifier |
| `index` | int | Measure index (0-based) |
| `stroke_start` | int | First stroke index in measure |
| `stroke_end` | int | Last stroke index (exclusive) |

**Timing Statistics**:

| Column | Type | Description |
|--------|------|-------------|
| `timing_mean_error_ms` | float | Mean timing error (milliseconds) |
| `timing_std_ms` | float | Timing error standard deviation |
| `timing_max_error_ms` | float | Maximum timing error |

**Velocity Statistics**:

| Column | Type | Description |
|--------|------|-------------|
| `velocity_mean` | float | Mean velocity (0-127 MIDI scale) |
| `velocity_std` | float | Velocity standard deviation |
| `velocity_consistency` | float | 1 - (std/mean), higher = more consistent |

**Hand Balance**:

| Column | Type | Description |
|--------|------|-------------|
| `lr_velocity_ratio` | float? | Left/Right velocity ratio (null if single-hand) |
| `lr_timing_diff_ms` | float? | L/R timing difference (ms) |

### Stroke Labels (`strokes.parquet`)

Individual stroke-level events.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Sample identifier |
| `index` | int | Stroke index (0-based) |
| `hand` | string | `L` or `R` |
| `stroke_type` | string | Stroke type identifier |

**Timing**:

| Column | Type | Description |
|--------|------|-------------|
| `intended_time_ms` | float | Intended stroke time (ms from start) |
| `actual_time_ms` | float | Actual stroke time |
| `timing_error_ms` | float | actual - intended (positive = late) |

**Velocity**:

| Column | Type | Description |
|--------|------|-------------|
| `intended_velocity` | int | Intended velocity (0-127) |
| `actual_velocity` | int | Actual velocity |
| `velocity_error` | int | actual - intended |

**Articulation Flags**:

| Column | Type | Description |
|--------|------|-------------|
| `is_grace_note` | bool | True if grace note (flams) |
| `is_accent` | bool | True if accented stroke |
| `diddle_position` | int? | Position in diddle (null if not diddle) |

## Audio Format

- **Format**: FLAC (lossless)
- **Sample Rate**: 44,100 Hz
- **Bit Depth**: 24-bit
- **Channels**: Mono
- **Duration**: 4-12 seconds typical

## MIDI Format

Standard MIDI Type 1 files with:

- **Track 0**: Tempo and time signature
- **Track 1**: Note events (channel 10 for drums)
- **Note Numbers**: Standard GM drum mapping (38 = snare)
- **Velocity**: 1-127 (dynamics)

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

## Soundfonts

| Identifier | Description |
|------------|-------------|
| `generalu` | GeneralUser GS - General purpose |
| `marching` | Marching snare focused |
| `mtpowerd` | MT Power Drums - Rock kit |
| `douglasn` | Douglas Drums - Natural kit |
| `fluidr3` | FluidR3 GM - Standard GM |

## Split Statistics

Default 70/15/15 train/validation/test split by profile.

| Split | Profiles | Samples (approx.) |
|-------|----------|-------------------|
| train | 68 | 68,000 |
| validation | 13 | 13,000 |
| test | 19 | 19,000 |

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

### Diddle Rudiments (5)

| Slug | Name |
|------|------|
| `single_paradiddle` | Single Paradiddle |
| `double_paradiddle` | Double Paradiddle |
| `triple_paradiddle` | Triple Paradiddle |
| `single_paradiddle_diddle` | Single Paradiddle-Diddle |
| `paradiddle_diddle` | Paradiddle-Diddle |

### Flam Rudiments (12)

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

### Drag Rudiments (8)

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

## Score Computation

### Timing Accuracy

```
timing_accuracy = 100 * (1 - mean_abs_error_ms / max_expected_error_ms)
```

Where `max_expected_error_ms` scales with tempo (faster = tighter tolerance).

### Velocity Control

```
velocity_control = 100 * (1 - velocity_std / expected_std_for_tier)
```

### Overall Score

Weighted combination:

```
overall_score = (
    0.30 * timing_accuracy +
    0.15 * timing_consistency +
    0.15 * velocity_control +
    0.10 * accent_accuracy +
    0.10 * hand_balance +
    0.10 * tempo_stability +
    0.10 * subdivision_evenness
)
```

Plus rudiment-specific bonuses/penalties when applicable.

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
| `num_samples` | int | Total sample count (99,770) |
| `num_profiles` | int | Unique player profiles (100) |
| `num_rudiments` | int | Unique rudiments (40) |
| `tempo` | DistributionStats | Tempo BPM distribution |
| `duration` | DistributionStats | Duration in seconds |
| `num_strokes` | DistributionStats | Strokes per sample |
| `timing_error` | DistributionStats | Stroke timing error (ms) |
| `velocity` | DistributionStats | MIDI velocity values |
| `measure_timing` | DistributionStats | Measure-level timing |
| `measure_velocity` | DistributionStats | Measure velocity consistency |
| `exercise_timing_accuracy` | DistributionStats | Exercise timing scores |
| `exercise_hand_balance` | DistributionStats | Exercise balance scores |
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

### Verification Object

| Field | Type | Description |
|-------|------|-------------|
| `all_passed` | bool | True if all checks passed |
| `num_passed` | int | Count of passed checks |
| `num_failed` | int | Count of failed checks |
| `checks` | array | Individual check results |

### Check Result Object

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Check identifier |
| `passed` | bool | Whether check passed |
| `message` | string | Human-readable result |
| `details` | object | Check-specific details |

### Realism Object

| Field | Type | Description |
|-------|------|-------------|
| `literature_comparisons` | array | Comparisons to published research |
| `literature_pass_rate` | float | Percentage within expected ranges |
| `correlation_checks` | array | Expected correlation validations |
| `correlation_pass_rate` | float | Percentage meeting expectations |
| `correlation_matrix` | object | Full score correlation matrix |
| `skill_tier_separation` | object | ANOVA results by metric |

### Literature Comparison Object

| Field | Type | Description |
|-------|------|-------------|
| `metric` | string | Metric being compared |
| `skill_tier` | string | Skill tier |
| `dataset_value` | float | Observed value |
| `expected_range` | [float, float] | Expected range from literature |
| `within_range` | bool | Whether value is in range |
| `citation` | string | Academic citation |
| `details` | object | Additional methodology details |

---

## Class Balance and Distribution

This section documents the exact sample distribution across skill tiers, rudiments, and their combinations.

### Skill Tier Distribution

SOUSA uses a non-uniform skill tier distribution that approximates realistic player populations:

| Skill Tier | Target Proportion | Full Dataset (~100K) | Medium (~12K) |
|------------|-------------------|---------------------|---------------|
| Beginner | 25% | ~25,000 | ~3,000 |
| Intermediate | 35% | ~35,000 | ~4,200 |
| Advanced | 25% | ~25,000 | ~3,000 |
| Professional | 15% | ~15,000 | ~1,800 |

**Note:** Actual counts may vary slightly due to:
- Profile generation uses probabilistic sampling
- Samples are generated per-profile × per-rudiment × per-tempo × per-augmentation

### Rudiment Distribution

All 40 PAS rudiments are equally represented. Each profile plays each rudiment at multiple tempos:

| Preset | Profiles | Tempos/Rudiment | Augmentations | Samples/Rudiment |
|--------|----------|-----------------|---------------|------------------|
| small | 10 | 3 | 1 | ~30 |
| medium | 50 | 3 | 2 | ~300 |
| full | 100 | 5 | 5 | ~2,500 |

**Complete Rudiment List (40):**

| # | Rudiment Slug | Category | Contains |
|---|---------------|----------|----------|
| 1 | `single_stroke_roll` | Roll | - |
| 2 | `single_stroke_four` | Roll | - |
| 3 | `single_stroke_seven` | Roll | - |
| 4 | `multiple_bounce_roll` | Roll | Buzz |
| 5 | `triple_stroke_roll` | Roll | - |
| 6 | `double_stroke_open_roll` | Roll | Diddle |
| 7 | `five_stroke_roll` | Roll | Diddle |
| 8 | `six_stroke_roll` | Roll | Diddle |
| 9 | `seven_stroke_roll` | Roll | Diddle |
| 10 | `nine_stroke_roll` | Roll | Diddle |
| 11 | `ten_stroke_roll` | Roll | Diddle |
| 12 | `eleven_stroke_roll` | Roll | Diddle |
| 13 | `thirteen_stroke_roll` | Roll | Diddle |
| 14 | `fifteen_stroke_roll` | Roll | Diddle |
| 15 | `seventeen_stroke_roll` | Roll | Diddle |
| 16 | `single_paradiddle` | Paradiddle | Diddle |
| 17 | `double_paradiddle` | Paradiddle | Diddle |
| 18 | `triple_paradiddle` | Paradiddle | Diddle |
| 19 | `paradiddle_diddle` | Paradiddle | Diddle |
| 20 | `flam` | Flam | Flam |
| 21 | `flam_accent` | Flam | Flam |
| 22 | `flam_tap` | Flam | Flam |
| 23 | `flamacue` | Flam | Flam |
| 24 | `flam_paradiddle` | Flam | Flam, Diddle |
| 25 | `single_flammed_mill` | Flam | Flam, Diddle |
| 26 | `flam_paradiddle_diddle` | Flam | Flam, Diddle |
| 27 | `pataflafla` | Flam | Flam |
| 28 | `swiss_army_triplet` | Flam | Flam |
| 29 | `inverted_flam_tap` | Flam | Flam |
| 30 | `flam_drag` | Flam | Flam, Drag |
| 31 | `drag` | Drag | Drag |
| 32 | `single_drag_tap` | Drag | Drag |
| 33 | `double_drag_tap` | Drag | Drag |
| 34 | `lesson_25` | Drag | Drag |
| 35 | `single_dragadiddle` | Drag | Drag, Diddle |
| 36 | `drag_paradiddle_1` | Drag | Drag, Diddle |
| 37 | `drag_paradiddle_2` | Drag | Drag, Diddle |
| 38 | `single_ratamacue` | Drag | Drag |
| 39 | `double_ratamacue` | Drag | Drag |
| 40 | `triple_ratamacue` | Drag | Drag |

### Cross-Tabulation: Skill Tier × Rudiment

All skill tier × rudiment combinations are populated. The formula for samples per cell:

```
samples_per_cell = (profiles_in_tier × tempos_per_rudiment × augmentations_per_sample)
```

For the full preset with 100 profiles (25 beginner, 35 intermediate, 25 advanced, 15 professional):

| Tier | Profiles | × Tempos | × Augs | = Samples/Rudiment |
|------|----------|----------|--------|-------------------|
| Beginner | 25 | 5 | 5 | 625 |
| Intermediate | 35 | 5 | 5 | 875 |
| Advanced | 25 | 5 | 5 | 625 |
| Professional | 15 | 5 | 5 | 375 |

**Imbalance Ratio:** Professional samples are ~2.3× less frequent than Intermediate.

### Score-Specific Availability

Some scores are only computed for rudiments containing specific articulations:

| Score Column | Available For | Null For | % of Samples |
|--------------|---------------|----------|--------------|
| `flam_quality` | Rudiments 20-30 | All others | ~27.5% non-null |
| `diddle_quality` | Rudiments 6-19, 24-26, 35-37 | All others | ~47.5% non-null |
| `roll_sustain` | Rudiments 1-15 | All others | ~37.5% non-null |

**Handling in Training:**
- Use masking for null values in loss computation
- Consider separate prediction heads per articulation type
- Filter to non-null subset if focusing on specific rudiments

### Train/Val/Test Split Distribution

Splits are assigned at the **profile level**, not sample level:

| Split | Profiles | Target % | Samples (~100K) |
|-------|----------|----------|-----------------|
| Train | 70 | 70% | ~70,000 |
| Validation | 15 | 15% | ~15,000 |
| Test | 15 | 15% | ~15,000 |

**Stratification:** Splits maintain skill tier proportions:

```
Train skill distribution ≈ Val skill distribution ≈ Test skill distribution
```

**Why profile-based splits?**
- Prevents data leakage from player-specific timing/velocity patterns
- Tests generalization to "new players" not seen during training
- More realistic evaluation for deployment scenarios

See [Split Methodology Experiment](../experiments/split_validation.py) for empirical analysis.

---

## Score Correlation Analysis

Understanding correlations between scores is critical for multi-task learning and avoiding multicollinearity.

### High Correlation Pairs (r > 0.7)

| Score 1 | Score 2 | Expected r | Reason |
|---------|---------|------------|--------|
| `timing_accuracy` | `timing_consistency` | ~0.85-0.90 | Both derived from timing errors |
| `timing_accuracy` | `subdivision_evenness` | ~0.75-0.85 | Both measure rhythmic precision |
| `timing_consistency` | `tempo_stability` | ~0.70-0.80 | Consistent players don't drift |
| `velocity_control` | `accent_differentiation` | ~0.60-0.75 | Both require dynamic range mastery |

### Score Clusters

Based on correlation structure, scores cluster into conceptual groups:

**Cluster 1: Timing Quality**
- `timing_accuracy`
- `timing_consistency`
- `tempo_stability`
- `subdivision_evenness`

**Cluster 2: Dynamics Quality**
- `velocity_control`
- `accent_differentiation`
- `accent_accuracy`

**Cluster 3: Hand Balance**
- `hand_balance`
- `weak_hand_index`

**Cluster 4: Rudiment-Specific** (independent)
- `flam_quality`
- `diddle_quality`
- `roll_sustain`

### Recommendations for Training

**For score regression:**
- Predicting `overall_score` alone is sufficient for most use cases
- If predicting individual scores, consider:
  - PCA on the 8 core scores → 3-4 components capture ~95% variance
  - Predict cluster representatives instead of all scores
  - Use multi-task learning with appropriate loss weighting

**For multi-task learning:**
- Weight timing cluster lower (high redundancy)
- Give rudiment-specific scores separate heads with masking
- Consider predicting `overall_score` + `hand_balance` + rudiment-specific (low correlation set)

**Minimal orthogonal score set:**
1. `overall_score` (composite, always available)
2. `hand_balance` (independent axis)
3. `flam_quality` (when available)
4. `diddle_quality` (when available)

See [Score Analysis Experiment](../experiments/score_analysis.py) for detailed correlation matrix and PCA results.
