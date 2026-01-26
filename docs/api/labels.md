# Labels Module

The labels module computes hierarchical labels from generated performances at three levels:

1. **Stroke-level**: Individual timing and velocity measurements for each stroke
2. **Measure-level**: Aggregate statistics within each measure (timing consistency, hand balance)
3. **Exercise-level**: Overall performance scores on a 0-100 scale

Labels are designed for ML training with features like:

- Corrected timing errors for grace notes (relative to ideal flam spacing, not grid)
- Perceptual (sigmoid) scaling for timing accuracy scores
- Tier confidence scores indicating label ambiguity near skill tier boundaries

## Schema

Pydantic models defining the hierarchical label structure.

::: dataset_gen.labels.schema
    options:
      show_root_heading: false
      members:
        - StrokeLabel
        - MeasureLabel
        - ExerciseScores
        - AudioAugmentation
        - Sample

## Label Computation

Functions for computing labels from stroke events and performances.

::: dataset_gen.labels.compute
    options:
      show_root_heading: false
      members:
        - compute_stroke_labels
        - compute_measure_labels
        - compute_exercise_scores
        - compute_tier_confidence
        - compute_skill_tier_binary
        - compute_sample_labels

## Groove Analysis

Groove feel heuristic proxy computation that distinguishes intentional microtiming from sloppy playing.

::: dataset_gen.labels.groove
    options:
      show_root_heading: false
      members:
        - GrooveMetrics
        - compute_groove_feel_proxy
        - compute_groove_metrics
        - analyze_groove_quality
