# Validation Guide

SOUSA includes a comprehensive validation infrastructure with 2,132 lines of validation code across 5 modules. This guide covers running validation, interpreting results, and understanding the quality benchmarks.

## Validation Architecture

```
dataset_gen/validation/
├── __init__.py        # Module exports
├── verify.py          # Data integrity (13 checks)
├── analysis.py        # Statistical analysis
├── realism.py         # Literature validation
└── report.py          # Report generation
```

The validation pipeline processes datasets through three stages:

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

## Running Validation

### Automatic Validation (Default)

Validation runs automatically after dataset generation:

```bash
python scripts/generate_dataset.py --preset small
# Validation runs at the end
```

To skip validation:

```bash
python scripts/generate_dataset.py --preset small --skip-validation
```

### Generate Validation Report

```python
from dataset_gen.validation.report import generate_report

# Generate comprehensive report
report = generate_report(
    dataset_dir='output/dataset',
    output_path='validation_report.json',
    include_realism=True,
    include_midi_checks=True
)

# Print summary
print(report.summary())
```

### Quick Validation Check

```python
from dataset_gen.validation.report import quick_validate

# Returns True if all critical checks pass
is_valid = quick_validate('output/dataset')
print(f"Dataset valid: {is_valid}")
```

### Individual Validation Components

```python
from dataset_gen.validation import (
    analyze_dataset,
    verify_labels,
    validate_realism
)

# Statistical analysis
stats = analyze_dataset('output/dataset')
print(f"Total samples: {stats['num_samples']}")
print(f"Mean timing error: {stats['timing_error']['mean']:.2f}ms")
print(f"Skill tier distribution: {stats['skill_tier_counts']}")

# Data integrity verification
result = verify_labels('output/dataset')
print(f"Checks passed: {result.num_passed}/{result.num_passed + result.num_failed}")
for check in result.checks:
    status = "PASS" if check.passed else "FAIL"
    print(f"  [{status}] {check.name}: {check.message}")

# Realism validation
realism = validate_realism('output/dataset')
print(f"Literature pass rate: {realism['literature_pass_rate']}%")
print(f"Correlation pass rate: {realism['correlation_pass_rate']}%")
```

### Command Line Validation

```bash
# Check dataset health
python scripts/check_generation.py output/dataset
```

## Interpreting Results

### Validation Report Summary

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

### Understanding Check Results

| Result | Meaning | Action |
|--------|---------|--------|
| 13/13 passed | All integrity checks pass | Dataset is valid |
| Literature 100% | Timing matches research | Realistic profiles |
| Correlation 80% | Most correlations expected | Minor deviations OK |

!!! warning "Failed checks"
    If verification checks fail, the dataset may have integrity issues. Review the specific check messages and regenerate if necessary.

## Data Integrity Checks

The `LabelVerifier` runs 13 verification checks:

### 1. Sample ID Uniqueness

Ensures all sample IDs are unique across the dataset.

```
Result: 99,770/99,770 unique sample IDs
```

### 2-4. Reference Validity

Verifies foreign key relationships between tables:

| Check | Description | Expected |
|-------|-------------|----------|
| `stroke_refs_valid` | All strokes reference valid samples | 100% valid |
| `measure_refs_valid` | All measures reference valid samples | 100% valid |
| `exercise_refs_valid` | All exercises reference valid samples | 100% valid |

### 5. Velocity Range

MIDI velocity must be in [0, 127]:

```
Result: 99.99% in valid MIDI range
```

### 6. Timing Range

Timing errors should be bounded:

| Metric | Expected | Typical Result |
|--------|----------|----------------|
| Max error | < 500ms | ~300ms |
| % under 200ms | > 95% | ~97% |
| Mean error | < 100ms | ~48ms |

### 7. Score Ranges

All exercise scores must be in [0, 100]:

- `timing_accuracy`, `timing_consistency`, `tempo_stability`
- `velocity_control`, `accent_differentiation`, `accent_accuracy`
- `hand_balance`, `overall_score`

### 8-9. Count Consistency

Metadata counts must match actual records:

| Check | Description |
|-------|-------------|
| `stroke_counts_match` | `num_strokes` matches stroke records |
| `measure_counts_match` | `num_measures` matches measure records |

### 10-11. Skill Tier Ordering

Performance metrics must be properly ordered by skill:

**Timing Accuracy** (Professional > Advanced > Intermediate > Beginner):

| Tier | Expected Mean |
|------|---------------|
| Professional | ~77 |
| Advanced | ~58 |
| Intermediate | ~33 |
| Beginner | ~6 |

**Hand Balance** (same ordering):

| Tier | Expected Mean |
|------|---------------|
| Professional | ~96 |
| Advanced | ~94 |
| Intermediate | ~88 |
| Beginner | ~77 |

### 12. Rudiment Pattern Correctness

Samples sticking patterns and verifies they match YAML definitions.

```
Result: 0 pattern mismatches (100 samples checked)
```

### 13. Label-MIDI Alignment

Checks that stroke labels align with actual MIDI note events.

```
Result: 100.0% alignment (400 strokes across 10 samples)
```

## Literature Benchmarks

### Timing Error Standard Deviation

Compared against peer-reviewed motor learning research:

| Skill Tier | Expected Range | Source |
|------------|----------------|--------|
| Professional | 5-15ms | Fujii et al. (2011) |
| Advanced | 8-25ms | Repp (2005) |
| Intermediate | 20-45ms | Wing & Kristofferson (1973) |
| Beginner | 35-80ms | Palmer (1997) |

### Velocity Coefficient of Variation

| Skill Tier | Expected Range | Source |
|------------|----------------|--------|
| Professional | 0.05-0.12 | Schmidt & Lee (2011) |
| Advanced | 0.08-0.18 | Schmidt & Lee (2011) |
| Intermediate | 0.12-0.25 | Schmidt & Lee (2011) |
| Beginner | 0.18-0.35 | Schmidt & Lee (2011) |

### Expected Pass Rates

| Validation Type | Expected | Notes |
|-----------------|----------|-------|
| Literature comparisons | 100% | All tiers within ranges |
| Correlation checks | 80%+ | Some expected deviations |

## Correlation Structure

### Expected Correlations

| Dimension Pair | Expected r | Reason |
|----------------|------------|--------|
| Timing Accuracy - Timing Consistency | > 0.5 | Both from timing errors |
| Timing Accuracy - Hand Balance | > 0.3 | Correlated skills |
| Velocity Control - Accent Differentiation | > 0.4 | Dynamic range mastery |
| Timing Accuracy - Overall Score | > 0.6 | Major component |
| Hand Balance - Overall Score | > 0.4 | Component score |

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

!!! note "Velocity/Accent correlation"
    The velocity_control to accent_differentiation correlation is lower than expected because accents are intentionally sampled at different velocities, reducing the correlation.

### Skill Tier Separation (ANOVA)

One-way ANOVA confirms skill tiers are statistically distinguishable:

| Metric | F-Statistic | p-value | Significant |
|--------|-------------|---------|-------------|
| Timing Accuracy | 44,172 | < 0.001 | Yes |
| Hand Balance | 18,465 | < 0.001 | Yes |
| Overall Score | 55,260 | < 0.001 | Yes |

All metrics show highly significant separation with massive F-statistics.

## Validation Report Schema

The `validation_report.json` file contains:

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
    "exercise_timing_accuracy": { /* DistributionStats with by_group */ },
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

### Reading the Report

```python
import json

with open('output/dataset/validation_report.json') as f:
    report = json.load(f)

# Check if all verifications passed
print(f"All checks passed: {report['verification']['all_passed']}")

# Get timing accuracy by skill tier
timing = report['stats']['exercise_timing_accuracy']['by_group']
for tier, stats in timing.items():
    print(f"{tier}: {stats['mean']:.1f} +/- {stats['std']:.1f}")

# Check realism validation
print(f"Literature: {report['realism']['literature_pass_rate']}% pass")
print(f"Correlations: {report['realism']['correlation_pass_rate']}% pass")
```

## Test Suite Overview

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

### Running Tests

```bash
# Run all validation tests
pytest tests/test_validation.py -v

# Run specific test class
pytest tests/test_validation.py::TestLabelVerifier -v

# Run with coverage
pytest tests/test_validation.py --cov=dataset_gen.validation
```

## Troubleshooting

### Failed Integrity Checks

| Check | Common Cause | Solution |
|-------|--------------|----------|
| Sample ID uniqueness | Duplicate generation | Regenerate with different seed |
| Reference validity | Incomplete writes | Check disk space, regenerate |
| Score ranges | Bug in score computation | Update dataset_gen, regenerate |
| Skill tier ordering | Insufficient samples | Use larger preset |

### Low Literature Pass Rate

If literature comparisons fail:

1. Check skill tier distribution matches expected proportions
2. Verify timing parameters in profile generation
3. Review `dataset_gen/profiles/archetypes.py` for parameter ranges

### Unexpected Correlation Structure

Some correlation deviations are expected:

- **Low velocity/accent correlation**: By design (accents vary velocity)
- **High timing correlations**: Expected (scores derived from same errors)

If correlations are significantly off:

1. Check profile generation parameters
2. Verify score computation in `dataset_gen/labels/compute.py`

## References

The realism validation compares against these peer-reviewed sources:

1. Fujii, S., et al. (2011). Synchronization error of drum kit playing. *Music Perception*, 28(5), 491-503.

2. Repp, B. H. (2005). Sensorimotor synchronization: A review. *Psychonomic Bulletin & Review*, 12(6), 969-992.

3. Wing, A. M., & Kristofferson, A. B. (1973). Response delays and timing. *Perception & Psychophysics*, 14(1), 5-12.

4. Palmer, C. (1997). Music performance. *Annual Review of Psychology*, 48(1), 115-138.

5. Schmidt, R. A., & Lee, T. D. (2011). *Motor control and learning* (5th ed.). Human Kinetics.
