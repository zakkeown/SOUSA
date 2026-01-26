# Score Computation

This document describes the mathematical foundations for computing performance scores in SOUSA. All scores use a 0-100 scale where higher values indicate better performance.

## Overview

SOUSA computes scores at three hierarchical levels:

1. **Stroke Level**: Individual timing and velocity measurements
2. **Measure Level**: Aggregate statistics per measure
3. **Exercise Level**: Composite performance scores

The exercise-level scores are the primary targets for machine learning, while stroke and measure labels provide fine-grained supervision.

---

## Perceptual Scaling

SOUSA uses **sigmoid (logistic) scaling** to transform raw error measurements into perceptual scores. This approach aligns with human perception of timing errors:

- Errors < 10ms: Nearly imperceptible (high score)
- Errors 20-30ms: Noticeable but acceptable (medium score)
- Errors > 50ms: Clearly audible (low score)

The general sigmoid transformation:

$$
\text{score} = 100 \times \frac{1}{1 + e^{(x - c) / k}}
$$

Where:

- $x$ is the raw measurement (e.g., mean absolute timing error)
- $c$ is the center point (50% score threshold)
- $k$ controls the steepness of the transition

---

## Timing Scores

### Timing Accuracy

Measures how close strokes are to their intended timing.

**Formula:**

$$
\text{timing\_accuracy} = 100 \times \frac{1}{1 + e^{(\bar{e} - 25) / 10}}
$$

Where:

- $\bar{e}$ = mean absolute timing error in milliseconds

**Parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Center ($c$) | 25 ms | 50% score at 25ms mean error |
| Steepness ($k$) | 10 | Transition width |

**Score Distribution:**

| Mean Error | Score |
|------------|-------|
| 5 ms | 88 |
| 10 ms | 82 |
| 25 ms | 50 |
| 40 ms | 18 |
| 60 ms | 3 |

### Timing Consistency

Measures the variance in timing errors (lower variance = higher score).

**Formula:**

$$
\text{timing\_consistency} = 100 \times \frac{1}{1 + e^{(\sigma_t - 15) / 8}}
$$

Where:

- $\sigma_t$ = standard deviation of timing errors in milliseconds

**Parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Center ($c$) | 15 ms | 50% score at 15ms std dev |
| Steepness ($k$) | 8 | Transition width |

### Tempo Stability

Measures drift in timing over the exercise duration.

**Computation:**

1. Fit a linear regression to timing errors over time:
   $$\text{error}_i = \beta_0 + \beta_1 \cdot t_i$$

2. Calculate slope in ms/second:
   $$\text{slope} = \beta_1 \times 1000$$

3. Convert to score:
   $$\text{tempo\_stability} = \max(0, 100 - |\text{slope}| \times 10)$$

**Interpretation:**

| Slope (ms/s) | Score | Meaning |
|--------------|-------|---------|
| 0 | 100 | No drift |
| 5 | 50 | Moderate drift |
| 10 | 0 | Severe drift |

### Subdivision Evenness

Measures consistency of inter-onset intervals (IOI).

**Formula:**

1. Calculate IOI ratios:
   $$r_i = \frac{\text{IOI}_{\text{actual},i}}{\text{IOI}_{\text{intended},i}}$$

2. Calculate variance:
   $$\sigma_r = \text{std}(r_1, r_2, \ldots, r_n)$$

3. Convert to score:
   $$\text{subdivision\_evenness} = \max(0, 100 - \sigma_r \times 100)$$

---

## Dynamics Scores

### Velocity Control

Measures consistency of stroke dynamics.

**Formula:**

$$
\text{velocity\_control} = \max(0, 100 - \sigma_v \times 2)
$$

Where:

- $\sigma_v$ = standard deviation of MIDI velocities (0-127 scale)

**Interpretation:**

| Velocity Std Dev | Score |
|------------------|-------|
| 0 | 100 |
| 10 | 80 |
| 25 | 50 |
| 50 | 0 |

### Accent Differentiation

Measures the dynamic contrast between accented and unaccented strokes.

**Formula:**

1. Calculate velocity difference:
   $$\Delta v = \bar{v}_{\text{accent}} - \bar{v}_{\text{tap}}$$

2. Convert to approximate dB:
   $$\Delta_{\text{dB}} = \frac{\Delta v}{127} \times 20$$

3. Calculate score:
   $$\text{accent\_differentiation} = \min(100, \max(0, \Delta_{\text{dB}} \times 8))$$

**Target Range:**

| dB Difference | Score | Quality |
|---------------|-------|---------|
| 0 dB | 0 | No differentiation |
| 6 dB | 48 | Minimal |
| 12 dB | 96 | Good |
| 15+ dB | 100 | Excellent |

### Accent Accuracy

Measures whether accents are placed on the correct beats.

**Formula:**

$$
\text{accent\_accuracy} = \frac{n_{\text{correct}}}{n_{\text{total}}} \times 100
$$

Where:

- $n_{\text{correct}}$ = accented strokes with velocity above average
- $n_{\text{total}}$ = total accented strokes in the pattern

---

## Hand Balance Scores

### Hand Balance (Composite)

Measures evenness between left and right hand performance, combining **velocity balance** and **timing balance** equally.

**Formula:**

$$
\text{hand\_balance} = 0.5 \times \text{velocity\_balance} + 0.5 \times \text{timing\_balance}
$$

**Velocity Balance:**

$$
\text{velocity\_balance} = \frac{\min(\bar{v}_L, \bar{v}_R)}{\max(\bar{v}_L, \bar{v}_R)} \times 100
$$

**Timing Balance:**

$$
\text{timing\_balance} = \frac{\min(\bar{|e|}_L, \bar{|e|}_R)}{\max(\bar{|e|}_L, \bar{|e|}_R)} \times 100
$$

Where:

- $\bar{v}_L$, $\bar{v}_R$ = mean velocity for left/right hand
- $\bar{|e|}_L$, $\bar{|e|}_R$ = mean absolute timing error for left/right hand

!!! note "Composite Score Rationale"
    Earlier versions used velocity-only balance, which exhibited a ceiling effect (nearly always high). Adding timing balance provides better discrimination between skill levels.

### Weak Hand Index

Identifies which hand is weaker for diagnostic purposes.

**Formula:**

$$
\text{weak\_hand\_index} = \frac{\bar{|e|}_L}{\bar{|e|}_L + \bar{|e|}_R} \times 100
$$

**Interpretation:**

| Value | Meaning |
|-------|---------|
| 0 | Left hand much weaker |
| 50 | Balanced |
| 100 | Right hand much weaker |

---

## Rudiment-Specific Scores

These scores are only computed for rudiments containing the relevant articulations.

### Flam Quality

Measures grace note spacing consistency for flam rudiments.

**Ideal Range:** 20-40ms (center: 30ms)

**Computation:**

For each grace note with spacing $s$ (ms from grace to primary):

$$
\text{score}_i = \begin{cases}
100 & \text{if } 20 \leq s \leq 40 \\
100 - (20 - s) \times 5 & \text{if } s < 20 \text{ (too tight)} \\
100 - (s - 40) \times 3 & \text{if } s > 40 \text{ (too wide)}
\end{cases}
$$

**Final Score:**

$$
\text{flam\_quality} = \frac{1}{n} \sum_{i=1}^{n} \text{score}_i
$$

### Diddle Quality

Measures evenness of double strokes.

**Computation:**

For each diddle pair with actual gap $g_a$ and intended gap $g_i$:

$$
r = \frac{g_a}{g_i}
$$

**Score:**

$$
\text{diddle\_quality} = \max(0, 100 - \bar{|r - 1|} \times 200)
$$

Where perfect evenness yields $r = 1$.

### Roll Sustain

Measures velocity consistency across sustained rolls (penalizes decay).

**Computation:**

1. Calculate mean velocity for first quarter: $\bar{v}_{\text{start}}$
2. Calculate mean velocity for last quarter: $\bar{v}_{\text{end}}$
3. Calculate decay ratio:
   $$\text{decay} = \frac{\bar{v}_{\text{end}}}{\bar{v}_{\text{start}}}$$

**Score:**

$$
\text{roll\_sustain} = \min(100, \text{decay} \times 100)
$$

| Decay Ratio | Score | Interpretation |
|-------------|-------|----------------|
| 1.0 | 100 | Perfect sustain |
| 0.8 | 80 | Slight decay |
| 0.5 | 50 | Significant decay |

---

## Composite Scores

### Overall Score

Weighted average of all component scores.

**Formula:**

$$
\text{overall\_score} = \sum_{i} w_i \times s_i
$$

**Weight Table:**

| Component | Weight | Rationale |
|-----------|--------|-----------|
| `timing_accuracy` | 0.20 | Most critical for rhythmic precision |
| `timing_consistency` | 0.15 | Important for professional sound |
| `tempo_stability` | 0.10 | Shows control over time |
| `subdivision_evenness` | 0.10 | Essential for clean rudiments |
| `velocity_control` | 0.10 | Dynamics consistency |
| `accent_differentiation` | 0.10 | Musical expression |
| `accent_accuracy` | 0.10 | Pattern correctness |
| `hand_balance` | 0.15 | Technical evenness |
| **Total** | **1.00** | |

!!! note "Rudiment-Specific Bonuses"
    Rudiment-specific scores (flam_quality, diddle_quality, roll_sustain) are computed separately and not included in the overall score to maintain consistency across rudiment types.

### Groove Feel Proxy

A supplementary metric capturing the "feel" of the performance.

**Formula:**

$$
\text{groove\_feel} = f(\text{micro-timing patterns, velocity contours})
$$

This is a 0-1 scale metric computed in `labels/groove.py`. It captures intentional micro-timing deviations that contribute to musical groove rather than penalizing all deviations equally.

---

## Tier Confidence

Measures how clearly a sample belongs to its assigned skill tier.

**Formula:**

$$
\text{tier\_confidence} = e^{-0.5 \cdot z^2}
$$

Where:

$$
z = \frac{|\text{overall\_score} - \mu_{\text{tier}}|}{\sigma_{\text{tier}}}
$$

**Tier Distribution Parameters:**

| Tier | Mean ($\mu$) | Std Dev ($\sigma$) |
|------|--------------|---------------------|
| Beginner | 34.3 | 6.6 |
| Intermediate | 45.7 | 10.1 |
| Advanced | 61.0 | 10.8 |
| Professional | 73.5 | 10.7 |

**Interpretation:**

| Confidence | Meaning |
|------------|---------|
| > 0.9 | Clearly within tier |
| 0.5-0.9 | Typical for tier |
| < 0.5 | Near tier boundary (label noise) |

!!! tip "Using Tier Confidence"
    When training classification models, consider:

    - Filtering samples with `tier_confidence < 0.3` to reduce label noise
    - Using `tier_confidence` as sample weights
    - Using the binary `skill_tier_binary` (novice/skilled) for cleaner separation

---

## Score Correlations

Understanding score correlations is important for multi-task learning.

### High Correlation Pairs ($r > 0.7$)

| Score 1 | Score 2 | Expected $r$ | Reason |
|---------|---------|--------------|--------|
| `timing_accuracy` | `timing_consistency` | 0.85-0.90 | Both from timing errors |
| `timing_accuracy` | `subdivision_evenness` | 0.75-0.85 | Both measure rhythm |
| `timing_consistency` | `tempo_stability` | 0.70-0.80 | Consistent players don't drift |
| `velocity_control` | `accent_differentiation` | 0.60-0.75 | Both require dynamic control |

### Score Clusters

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

### Recommendations

For **score regression**:

- Predicting `overall_score` alone is sufficient for most use cases
- PCA on the 8 core scores captures ~95% variance in 3-4 components

For **multi-task learning**:

- Weight timing cluster lower (high redundancy)
- Give rudiment-specific scores separate heads with masking
- Consider predicting: `overall_score` + `hand_balance` + rudiment-specific

**Minimal orthogonal score set:**

1. `overall_score` (composite, always available)
2. `hand_balance` (independent axis)
3. `flam_quality` (when available)
4. `diddle_quality` (when available)

---

## Implementation Reference

Score computation is implemented in `dataset_gen/labels/compute.py`:

```python
def compute_exercise_scores(
    stroke_labels: list[StrokeLabel],
    events: list[StrokeEvent],
    rudiment: Rudiment,
) -> ExerciseScores:
    """Compute all exercise-level scores."""

    # Timing scores (use corrected values from labels)
    timing_errors = np.array([s.timing_error_ms for s in stroke_labels])
    timing_errors_abs = np.abs(timing_errors)

    # Timing accuracy: sigmoid scaling
    mean_abs_error = np.mean(timing_errors_abs)
    timing_accuracy = 100 * (1 / (1 + np.exp((mean_abs_error - 25) / 10)))

    # ... additional scores computed similarly
```

## See Also

- [Architecture](architecture.md) - Pipeline overview showing where scores are computed
- [Data Format](data-format.md) - Schema for score columns in Parquet files
- [Rudiment Schema](rudiment-schema.md) - How rudiment definitions affect scoring
