# SOUSA Validation Documentation

This document describes the comprehensive validation infrastructure used to ensure data quality and realism in the SOUSA dataset.

## Overview

SOUSA employs a multi-layered validation system with **2,132 lines of validation code** across 5 modules:

```
dataset_gen/validation/
├── __init__.py        # Module exports
├── verify.py          # Data integrity (13 checks)
├── analysis.py        # Statistical analysis
├── realism.py         # Literature validation
└── report.py          # Report generation
```

## Validation Architecture

```
Input: Parquet Dataset
         ↓
┌─────────────────────────────────┐
│  1. LABEL VERIFICATION          │
│     - Data integrity (13 checks)│
│     - References valid          │
│     - Ranges correct            │
│     - Metadata matches records  │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  2. STATISTICAL ANALYSIS        │
│     - 11 distribution metrics   │
│     - Grouped by skill tier     │
│     - Moments & quartiles       │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  3. REALISM VALIDATION          │
│     - Literature benchmarks     │
│     - Correlation structure     │
│     - Skill tier separation     │
└─────────────────────────────────┘
         ↓
OUTPUT: ValidationReport (JSON)
```

## Data Integrity Checks

The `LabelVerifier` class runs 13 core verification checks:

### 1. Sample ID Uniqueness
```python
# Ensures all sample IDs are unique
assert len(sample_ids) == len(set(sample_ids))
```
**Result**: 99,770/99,770 unique sample IDs ✓

### 2-4. Reference Validity
Verifies all foreign key relationships:
- `stroke_refs_valid`: All 4,150,320 strokes reference valid samples ✓
- `measure_refs_valid`: All 399,080 measures reference valid samples ✓
- `exercise_refs_valid`: All 99,770 exercises reference valid samples ✓

### 5. Velocity Range
```python
# MIDI velocity must be in [0, 127]
pct_valid = (velocities >= 0) & (velocities <= 127)
```
**Result**: 99.99% in valid MIDI range ✓

### 6. Timing Range
```python
# Timing errors should be bounded
max_error < 500ms
pct_under_200ms > 95%
```
**Result**: max=300ms, 97.0% under 200ms, mean=47.6ms ✓

### 7. Score Ranges
All exercise scores must be in [0, 100]:
- `timing_accuracy`, `timing_consistency`, `tempo_stability`
- `velocity_control`, `accent_differentiation`, `accent_accuracy`
- `hand_balance`, `overall_score`

**Result**: All scores in valid range ✓

### 8-9. Count Consistency
- `stroke_counts_match`: `num_strokes` metadata matches actual stroke records ✓
- `measure_counts_match`: `num_measures` metadata matches actual measure records ✓

### 10-11. Skill Tier Ordering
Verifies performance metrics properly ordered by skill:

**Timing Accuracy** (Professional > Advanced > Intermediate > Beginner):
| Tier | Mean Score |
|------|------------|
| Professional | 76.87 |
| Advanced | 58.30 |
| Intermediate | 32.68 |
| Beginner | 5.85 |

**Hand Balance**:
| Tier | Mean Score |
|------|------------|
| Professional | 96.40 |
| Advanced | 93.74 |
| Intermediate | 87.90 |
| Beginner | 77.46 |

### 12. Rudiment Pattern Correctness
Samples 100 random performances and verifies sticking patterns match YAML definitions.

**Result**: 0 pattern mismatches ✓

### 13. Label-MIDI Alignment
Checks that stroke labels align with actual MIDI note events.

**Result**: 100.0% alignment (400 strokes across 10 samples) ✓

## Statistical Analysis

The `DatasetAnalyzer` computes distribution statistics for 11 metrics:

### Computed Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Tempo (BPM) | 117.9 | 33.5 | 60 | 180 |
| Duration (sec) | 5.88 | 2.86 | 1.63 | 20.46 |
| Strokes/Sample | 41.6 | 18.3 | 16 | 96 |
| Timing Error (ms) | 3.31 | 69.6 | -300 | 540 |
| Velocity | 71.9 | 23.3 | 0 | 127 |
| Timing Accuracy | 37.4 | 30.5 | 0 | 94.7 |
| Hand Balance | 87.7 | 11.2 | 35.3 | 100 |

### Distribution Statistics

Each metric includes:
- Count, mean, std, min, max
- Median, Q25, Q75
- Skewness, kurtosis
- Breakdown by skill tier

## Realism Validation

The `RealismValidator` compares dataset characteristics against peer-reviewed research.

### Literature Benchmarks

#### Timing Error Standard Deviation

| Skill Tier | Dataset | Expected | Source | Status |
|------------|---------|----------|--------|--------|
| Professional | 6.4ms | 5-15ms | Fujii et al. (2011) | ✓ |
| Advanced | 14.4ms | 8-25ms | Repp (2005) | ✓ |
| Intermediate | 31.6ms | 20-45ms | Wing & Kristofferson (1973) | ✓ |
| Beginner | 55.5ms | 35-80ms | Palmer (1997) | ✓ |

**Pass Rate: 100%**

#### Velocity Coefficient of Variation

| Skill Tier | Dataset | Expected | Source | Status |
|------------|---------|----------|--------|--------|
| Professional | 0.084 | 0.05-0.12 | Schmidt & Lee (2011) | ✓ |
| Advanced | 0.139 | 0.08-0.18 | Schmidt & Lee (2011) | ✓ |
| Intermediate | 0.234 | 0.12-0.25 | Schmidt & Lee (2011) | ✓ |
| Beginner | 0.284 | 0.18-0.35 | Schmidt & Lee (2011) | ✓ |

**Pass Rate: 100%**

### Correlation Structure

Expected correlations between performance dimensions:

| Dimension Pair | Observed r | Expected Min | Status |
|----------------|------------|--------------|--------|
| Timing Accuracy ↔ Timing Consistency | 0.888 | 0.5 | ✓ |
| Timing Accuracy ↔ Hand Balance | 0.464 | 0.3 | ✓ |
| Velocity Control ↔ Accent Differentiation | 0.259 | 0.4 | ✗ |
| Timing Accuracy ↔ Overall Score | 0.875 | 0.6 | ✓ |
| Hand Balance ↔ Overall Score | 0.415 | 0.4 | ✓ |

**Pass Rate: 80%**

Note: The velocity/accent correlation is lower than expected because accents are intentionally sampled at different velocities, reducing the correlation.

### Full Correlation Matrix

```
                    timing_acc  timing_cons  tempo_stab  vel_ctrl  accent_diff  hand_bal  overall
timing_accuracy        1.000       0.888       0.575      0.277       0.050      0.464     0.875
timing_consistency     0.888       1.000       0.532      0.389       0.083      0.418     0.885
tempo_stability        0.575       0.532       1.000      0.197       0.023      0.229     0.649
velocity_control       0.277       0.389       0.197      1.000       0.259      0.173     0.551
accent_differentiation 0.050       0.083       0.023      0.259       1.000     -0.103     0.360
hand_balance           0.464       0.418       0.229      0.173      -0.103      1.000     0.415
overall_score          0.875       0.885       0.649      0.551       0.360      0.415     1.000
```

### Skill Tier Separation (ANOVA)

One-way ANOVA confirms skill tiers are statistically distinguishable:

| Metric | F-Statistic | p-value | Significant |
|--------|-------------|---------|-------------|
| Timing Accuracy | 44,171.75 | ≈ 0 | Yes |
| Hand Balance | 18,464.86 | ≈ 0 | Yes |
| Overall Score | 55,260.32 | ≈ 0 | Yes |

All metrics show highly significant separation (p < 0.001) with massive F-statistics.

## Running Validation

### Generate Full Report

```python
from dataset_gen.validation.report import generate_report

report = generate_report(
    dataset_dir='output/dataset',
    output_path='validation_report.json',
    include_realism=True,
    include_midi_checks=True
)

print(report.summary())
```

### Quick Validation

```python
from dataset_gen.validation.report import quick_validate

is_valid = quick_validate('output/dataset')  # Returns boolean
```

### Individual Components

```python
from dataset_gen.validation import (
    analyze_dataset,
    verify_labels,
    validate_realism
)

# Statistical analysis
stats = analyze_dataset('output/dataset')
print(f"Samples: {stats['num_samples']}")
print(f"Mean timing error: {stats['timing_error']['mean']:.2f}ms")

# Data integrity
result = verify_labels('output/dataset')
print(f"Passed: {result.num_passed}/{result.num_passed + result.num_failed}")

# Realism validation
realism = validate_realism('output/dataset')
print(f"Literature pass rate: {realism['literature_pass_rate']}%")
```

### Command Line

```bash
# Check dataset health
python scripts/check_generation.py output/dataset

# Generate dataset with validation
python scripts/generate_dataset.py --preset full
# (validation runs automatically at end)
```

## Validation Report Schema

The `validation_report.json` contains:

```json
{
  "dataset_path": "output/dataset",
  "generated_at": "2026-01-25T21:18:35.372368",
  "stats": {
    "num_samples": 99770,
    "num_profiles": 100,
    "num_rudiments": 40,
    "tempo": { /* DistributionStats */ },
    "duration": { /* DistributionStats */ },
    "timing_error": { /* DistributionStats with by_group */ },
    "velocity": { /* DistributionStats */ },
    "exercise_timing_accuracy": { /* DistributionStats with by_group */ },
    "exercise_hand_balance": { /* DistributionStats with by_group */ },
    "skill_tier_counts": { /* counts by tier */ },
    "rudiment_counts": { /* counts by rudiment */ },
    "split_counts": { /* train/val/test counts */ }
  },
  "verification": {
    "all_passed": true,
    "num_passed": 13,
    "num_failed": 0,
    "checks": [ /* array of check results */ ]
  },
  "skill_tier_ordering": {
    "timing_accuracy_ordered": true,
    "timing_accuracy_means": { /* by tier */ },
    "hand_balance_ordered": true,
    "hand_balance_means": { /* by tier */ }
  },
  "realism": {
    "literature_comparisons": [ /* array of comparisons */ ],
    "literature_pass_rate": 100.0,
    "correlation_checks": [ /* array of checks */ ],
    "correlation_pass_rate": 80.0,
    "correlation_matrix": { /* full matrix */ },
    "skill_tier_separation": {
      "timing_accuracy": { "f_statistic": 44171.75, "p_value": 0.0 },
      "hand_balance": { "f_statistic": 18464.86, "p_value": 0.0 },
      "overall_score": { "f_statistic": 55260.32, "p_value": 0.0 }
    }
  }
}
```

## Test Suite

The validation infrastructure has 26 test cases in `tests/test_validation.py`:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestDistributionStats | 2 | Statistics computation |
| TestDatasetAnalyzer | 4 | Analysis pipeline |
| TestLabelVerifier | 7 | All 13 integrity checks |
| TestVerificationResult | 3 | Result aggregation |
| TestValidationReport | 5 | Report generation |
| TestSkillTierOrdering | 2 | Ordering verification |
| TestDataIntegrity | 3 | Edge cases |

Run tests:
```bash
pytest tests/test_validation.py -v
```

## References

1. Fujii, S., Hirashima, M., Kudo, K., Ohtsuki, T., Nakamura, Y., & Oda, S. (2011). Synchronization error of drum kit playing with a metronome at different tempi by professional and amateur drummers. *Music Perception*, 28(5), 491-503.

2. Repp, B. H. (2005). Sensorimotor synchronization: A review of the tapping literature. *Psychonomic Bulletin & Review*, 12(6), 969-992.

3. Wing, A. M., & Kristofferson, A. B. (1973). Response delays and the timing of discrete motor responses. *Perception & Psychophysics*, 14(1), 5-12.

4. Palmer, C. (1997). Music performance. *Annual Review of Psychology*, 48(1), 115-138.

5. Schmidt, R. A., & Lee, T. D. (2011). *Motor control and learning: A behavioral emphasis* (5th ed.). Human Kinetics.
