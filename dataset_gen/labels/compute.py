"""
Compute hierarchical labels from stroke events.

This module transforms raw stroke events into the three-level
label hierarchy used for ML training.
"""

from __future__ import annotations

import numpy as np

from dataset_gen.midi_gen.generator import StrokeEvent, GeneratedPerformance
from dataset_gen.rudiments.schema import Rudiment, Hand, StrokeType, RudimentCategory
from dataset_gen.profiles.archetypes import PlayerProfile
from dataset_gen.labels.schema import (
    StrokeLabel,
    MeasureLabel,
    ExerciseScores,
    Sample,
)
from dataset_gen.labels.groove import compute_groove_feel_proxy


def compute_stroke_labels(events: list[StrokeEvent]) -> list[StrokeLabel]:
    """
    Convert stroke events to stroke labels.

    For grace notes, timing_error_ms is recomputed as deviation from ideal
    flam spacing (25-35ms), not grid deviation. This provides a meaningful
    timing quality metric for ornamental strokes.

    Args:
        events: List of stroke events from MIDI generation

    Returns:
        List of StrokeLabel objects
    """
    # Build index lookup for finding primary strokes
    events_by_index = {e.index: e for e in events}

    # Ideal flam spacing (center of acceptable range)
    IDEAL_FLAM_SPACING_MS = 30.0

    labels = []
    for event in events:
        # Compute grace-note-specific fields
        flam_spacing_ms = None
        timing_error = event.timing_error_ms
        parent_index = event.parent_stroke_index

        if event.is_grace_note and parent_index is not None:
            primary = events_by_index.get(parent_index)
            if primary is not None:
                # Compute actual flam spacing (primary - grace)
                flam_spacing_ms = primary.actual_time_ms - event.actual_time_ms

                # Recompute timing_error as deviation from ideal spacing
                # Positive = too wide (grace too early), Negative = too tight
                timing_error = flam_spacing_ms - IDEAL_FLAM_SPACING_MS

        label = StrokeLabel(
            index=event.index,
            hand="R" if event.hand == Hand.RIGHT else "L",
            stroke_type=event.stroke_type.value,
            intended_time_ms=event.intended_time_ms,
            actual_time_ms=event.actual_time_ms,
            timing_error_ms=timing_error,
            intended_velocity=event.intended_velocity,
            actual_velocity=event.actual_velocity,
            velocity_error=event.velocity_error,
            is_grace_note=event.is_grace_note,
            is_accent=event.stroke_type == StrokeType.ACCENT,
            diddle_position=event.diddle_position,
            flam_spacing_ms=flam_spacing_ms,
            parent_stroke_index=parent_index,
        )
        labels.append(label)

    # Compute buzz_count for primary buzz strokes
    for label in labels:
        if label.stroke_type == "buzz" and not label.is_grace_note:
            sub_count = sum(1 for l in labels if l.parent_stroke_index == label.index)
            label.buzz_count = 1 + sub_count  # primary + subs

    return labels


def compute_measure_labels(
    stroke_labels: list[StrokeLabel],
    strokes_per_measure: int,
) -> list[MeasureLabel]:
    """
    Compute per-measure aggregate statistics.

    Args:
        stroke_labels: List of stroke labels (with corrected timing for grace notes)
        strokes_per_measure: Number of strokes in each measure

    Returns:
        List of MeasureLabel objects
    """
    if not stroke_labels:
        return []

    measures = []
    n_measures = (len(stroke_labels) + strokes_per_measure - 1) // strokes_per_measure

    for m in range(n_measures):
        start_idx = m * strokes_per_measure
        end_idx = min((m + 1) * strokes_per_measure, len(stroke_labels))
        measure_strokes = stroke_labels[start_idx:end_idx]

        if not measure_strokes:
            continue

        # Timing statistics (using corrected timing_error_ms from labels)
        timing_errors = [s.timing_error_ms for s in measure_strokes]
        timing_mean = np.mean(timing_errors)
        timing_std = np.std(timing_errors)
        timing_max = np.max(np.abs(timing_errors))

        # Velocity statistics
        velocities = [s.actual_velocity for s in measure_strokes]
        vel_mean = np.mean(velocities)
        vel_std = np.std(velocities)
        vel_consistency = 1 - (vel_std / vel_mean) if vel_mean > 0 else 0

        # Hand balance within measure
        left_strokes = [s for s in measure_strokes if s.hand == "L"]
        right_strokes = [s for s in measure_strokes if s.hand == "R"]

        lr_ratio = None
        lr_timing_diff = None
        if left_strokes and right_strokes:
            left_vel = np.mean([s.actual_velocity for s in left_strokes])
            right_vel = np.mean([s.actual_velocity for s in right_strokes])
            lr_ratio = left_vel / right_vel if right_vel > 0 else 1.0

            left_timing = np.mean([s.timing_error_ms for s in left_strokes])
            right_timing = np.mean([s.timing_error_ms for s in right_strokes])
            lr_timing_diff = left_timing - right_timing

        measure = MeasureLabel(
            index=m,
            stroke_start=start_idx,
            stroke_end=end_idx,
            timing_mean_error_ms=timing_mean,
            timing_std_ms=timing_std,
            timing_max_error_ms=timing_max,
            velocity_mean=vel_mean,
            velocity_std=vel_std,
            velocity_consistency=vel_consistency,
            lr_velocity_ratio=lr_ratio,
            lr_timing_diff_ms=lr_timing_diff,
        )
        measures.append(measure)

    return measures


def compute_exercise_scores(
    stroke_labels: list[StrokeLabel],
    events: list[StrokeEvent],
    rudiment: Rudiment,
) -> ExerciseScores:
    """
    Compute overall exercise scores from stroke labels and events.

    Uses stroke labels for timing metrics (corrected for grace notes)
    and events for rudiment-specific calculations that need full event data.

    Scores are normalized to 0-100 where 100 is perfect performance.

    Args:
        stroke_labels: List of stroke labels (with corrected timing)
        events: List of stroke events (for rudiment-specific calculations)
        rudiment: The rudiment being performed

    Returns:
        ExerciseScores object
    """
    if not stroke_labels:
        return _empty_scores()

    # === TIMING SCORES (use corrected values from labels) ===
    # Uses perceptual (sigmoid) scaling to match human perception:
    # - Errors <10ms are nearly imperceptible (high score)
    # - Errors 20-30ms are noticeable but acceptable (medium score)
    # - Errors >50ms are clearly audible (low score)

    timing_errors = np.array([s.timing_error_ms for s in stroke_labels])
    timing_errors_abs = np.abs(timing_errors)

    # Timing accuracy: sigmoid scaling (perceptual)
    # Center at 25ms (50% score), steepness controlled by /10
    mean_abs_error = np.mean(timing_errors_abs)
    timing_accuracy = 100 * (1 / (1 + np.exp((mean_abs_error - 25) / 10)))

    # Timing consistency: sigmoid scaling
    # Center at 15ms std (50% score)
    timing_std = np.std(timing_errors)
    timing_consistency = 100 * (1 / (1 + np.exp((timing_std - 15) / 8)))

    # Tempo stability: based on drift over time (for non-grace strokes only)
    non_grace_labels = [s for s in stroke_labels if not s.is_grace_note]
    if len(non_grace_labels) > 1:
        times = np.array([s.intended_time_ms for s in non_grace_labels])
        errors = np.array([s.timing_error_ms for s in non_grace_labels])
        slope = np.polyfit(times, errors, 1)[0] * 1000  # ms/second
        tempo_stability = max(0, 100 - abs(slope) * 10)
    else:
        tempo_stability = 100

    # Subdivision evenness: check inter-onset intervals
    if len(stroke_labels) > 1:
        actual_times = np.array([s.actual_time_ms for s in stroke_labels])
        intended_times = np.array([s.intended_time_ms for s in stroke_labels])
        actual_ioi = np.diff(actual_times)
        intended_ioi = np.diff(intended_times)
        ioi_ratios = actual_ioi / np.maximum(intended_ioi, 1)
        ioi_variance = np.std(ioi_ratios)
        subdivision_evenness = max(0, 100 - ioi_variance * 100)
    else:
        subdivision_evenness = 100

    # === DYNAMICS SCORES ===

    velocities = np.array([s.actual_velocity for s in stroke_labels])

    # Velocity control: consistency of dynamics
    vel_std = np.std(velocities)
    velocity_control = max(0, 100 - vel_std * 2)

    # Accent differentiation
    accent_strokes = [s for s in stroke_labels if s.is_accent]
    tap_strokes = [s for s in stroke_labels if s.stroke_type in ("tap", "buzz")]

    if accent_strokes and tap_strokes:
        accent_vel = np.mean([s.actual_velocity for s in accent_strokes])
        tap_vel = np.mean([s.actual_velocity for s in tap_strokes])
        diff_db = (accent_vel - tap_vel) / 127 * 20  # Rough dB approximation
        accent_differentiation = min(100, max(0, diff_db * 8))
    else:
        accent_differentiation = 80

    # Accent accuracy: are accents on correct beats?
    if accent_strokes:
        correct_accents = sum(1 for s in accent_strokes if s.actual_velocity > np.mean(velocities))
        accent_accuracy = (correct_accents / len(accent_strokes)) * 100
    else:
        accent_accuracy = 100

    # === HAND BALANCE SCORES ===
    # Now incorporates BOTH velocity AND timing balance (50% each)
    # This addresses the ceiling effect where velocity-only was nearly always high

    left_strokes = [s for s in stroke_labels if s.hand == "L"]
    right_strokes = [s for s in stroke_labels if s.hand == "R"]

    if left_strokes and right_strokes:
        # Velocity balance (0-100)
        left_vel = np.mean([s.actual_velocity for s in left_strokes])
        right_vel = np.mean([s.actual_velocity for s in right_strokes])
        vel_ratio = min(left_vel, right_vel) / max(left_vel, right_vel)
        velocity_balance = vel_ratio * 100

        # Timing balance (0-100): combines symmetry AND absolute quality
        left_timing_abs = np.mean([np.abs(s.timing_error_ms) for s in left_strokes])
        right_timing_abs = np.mean([np.abs(s.timing_error_ms) for s in right_strokes])

        # Symmetry factor: how similar are the hands?
        max_timing = max(left_timing_abs, right_timing_abs, 1)
        min_timing = min(left_timing_abs, right_timing_abs)
        symmetry_factor = min_timing / max_timing

        # Quality factor: sigmoid penalty for high absolute errors
        avg_abs_error = (left_timing_abs + right_timing_abs) / 2
        quality_factor = 100 / (1 + np.exp((avg_abs_error - 25) / 10))

        # Combined: both symmetry and quality matter
        timing_balance = 0.4 * (symmetry_factor * 100) + 0.6 * quality_factor

        # Combined hand balance: 50% velocity, 50% timing
        hand_balance = 0.5 * velocity_balance + 0.5 * timing_balance

        # Weak hand index: which hand is weaker overall?
        # 0 = left weaker, 100 = right weaker, 50 = balanced
        total_timing = left_timing_abs + right_timing_abs
        weak_hand_index = (left_timing_abs / total_timing * 100) if total_timing > 0 else 50
    else:
        hand_balance = 100
        weak_hand_index = 50

    # === RUDIMENT-SPECIFIC SCORES (use events for these) ===

    flam_quality = None
    diddle_quality = None
    roll_sustain = None

    if rudiment.category == RudimentCategory.FLAM:
        flam_quality = _compute_flam_quality(events)

    if rudiment.category in (RudimentCategory.DIDDLE, RudimentCategory.ROLL):
        diddle_quality = _compute_diddle_quality(events)

    if rudiment.category == RudimentCategory.ROLL:
        roll_sustain = _compute_roll_sustain(events)

    # === DERIVED SCORES ===

    groove_feel = compute_groove_feel_proxy(events)

    # Overall score: weighted average of components
    weights = {
        "timing_accuracy": 0.2,
        "timing_consistency": 0.15,
        "tempo_stability": 0.1,
        "subdivision_evenness": 0.1,
        "velocity_control": 0.1,
        "accent_differentiation": 0.1,
        "accent_accuracy": 0.1,
        "hand_balance": 0.15,
    }
    overall = sum(
        weights[k] * v
        for k, v in {
            "timing_accuracy": timing_accuracy,
            "timing_consistency": timing_consistency,
            "tempo_stability": tempo_stability,
            "subdivision_evenness": subdivision_evenness,
            "velocity_control": velocity_control,
            "accent_differentiation": accent_differentiation,
            "accent_accuracy": accent_accuracy,
            "hand_balance": hand_balance,
        }.items()
    )

    return ExerciseScores(
        timing_accuracy=timing_accuracy,
        timing_consistency=timing_consistency,
        tempo_stability=tempo_stability,
        subdivision_evenness=subdivision_evenness,
        velocity_control=velocity_control,
        accent_differentiation=accent_differentiation,
        accent_accuracy=accent_accuracy,
        hand_balance=hand_balance,
        weak_hand_index=weak_hand_index,
        flam_quality=flam_quality,
        diddle_quality=diddle_quality,
        roll_sustain=roll_sustain,
        groove_feel_proxy=groove_feel,
        overall_score=overall,
    )


def _compute_flam_quality(events: list[StrokeEvent]) -> float:
    """Compute flam quality score based on grace note spacing."""
    grace_events = [e for e in events if e.is_grace_note]
    if not grace_events:
        return 80

    # Ideal flam spacing is 25-35ms
    ideal_min, ideal_max = 20, 40
    spacings = []

    for grace in grace_events:
        if grace.parent_stroke_index is not None:
            primary = next((e for e in events if e.index == grace.parent_stroke_index), None)
            if primary:
                spacing = primary.actual_time_ms - grace.actual_time_ms
                spacings.append(spacing)

    if not spacings:
        return 80

    # Score based on how close to ideal range
    scores = []
    for s in spacings:
        if ideal_min <= s <= ideal_max:
            scores.append(100)
        elif s < ideal_min:
            scores.append(max(0, 100 - (ideal_min - s) * 5))  # Too tight
        else:
            scores.append(max(0, 100 - (s - ideal_max) * 3))  # Too wide

    return np.mean(scores)


def _compute_diddle_quality(events: list[StrokeEvent]) -> float:
    """Compute diddle quality based on evenness of double strokes."""
    # Find diddle pairs
    ratios = []
    i = 0
    while i < len(events) - 1:
        if (
            events[i].stroke_type == StrokeType.DIDDLE
            and events[i].diddle_position == 1
            and events[i + 1].stroke_type == StrokeType.DIDDLE
            and events[i + 1].diddle_position == 2
        ):
            actual_gap = events[i + 1].actual_time_ms - events[i].actual_time_ms
            intended_gap = events[i + 1].intended_time_ms - events[i].intended_time_ms
            if intended_gap > 0:
                ratio = actual_gap / intended_gap
                ratios.append(ratio)
            i += 2
        else:
            i += 1

    if not ratios:
        return 80

    # Score based on how close to 1.0 (even)
    deviations = [abs(r - 1.0) for r in ratios]
    mean_deviation = np.mean(deviations)
    return max(0, 100 - mean_deviation * 200)


def _compute_roll_sustain(events: list[StrokeEvent]) -> float:
    """Compute roll sustain quality based on velocity decay."""
    velocities = [e.actual_velocity for e in events]
    if len(velocities) < 4:
        return 80

    # Check if velocity drops over time (bad) or stays consistent (good)
    first_quarter = velocities[: len(velocities) // 4]
    last_quarter = velocities[-len(velocities) // 4 :]

    first_mean = np.mean(first_quarter)
    last_mean = np.mean(last_quarter)

    if first_mean == 0:
        return 80

    decay_ratio = last_mean / first_mean
    # 1.0 = perfect sustain, 0.5 = 50% decay
    return min(100, decay_ratio * 100)


def _empty_scores() -> ExerciseScores:
    """Return empty scores for edge cases."""
    return ExerciseScores(
        timing_accuracy=0,
        timing_consistency=0,
        tempo_stability=0,
        subdivision_evenness=0,
        velocity_control=0,
        accent_differentiation=0,
        accent_accuracy=0,
        hand_balance=0,
        weak_hand_index=50,
        groove_feel_proxy=0,
        overall_score=0,
        tier_confidence=0.5,
    )


# Tier score distribution parameters (mean, std) derived from dataset analysis
# Used to compute tier_confidence
TIER_SCORE_DISTRIBUTIONS = {
    "beginner": (34.3, 6.6),
    "intermediate": (45.7, 10.1),
    "advanced": (61.0, 10.8),
    "professional": (73.5, 10.7),
}


def compute_tier_confidence(overall_score: float, skill_tier: str) -> float:
    """
    Compute confidence that the skill_tier label is unambiguous.

    Uses Gaussian likelihood based on expected score distributions per tier.
    Returns 0-1 where:
    - 1.0 = score is at the center of the tier's expected distribution
    - 0.0 = score is far from the tier's expected range (likely mislabeled)

    A sample near the boundary between tiers will have lower confidence,
    indicating potential label noise for classification tasks.
    """
    if skill_tier not in TIER_SCORE_DISTRIBUTIONS:
        return 0.5  # Unknown tier

    mean, std = TIER_SCORE_DISTRIBUTIONS[skill_tier]

    # Gaussian likelihood (normalized to 0-1)
    # At mean: confidence = 1.0
    # At 2 std from mean: confidence ≈ 0.14
    z_score = abs(overall_score - mean) / std
    confidence = np.exp(-0.5 * z_score**2)

    return float(confidence)


def compute_skill_tier_binary(skill_tier: str) -> str:
    """
    Convert 4-class skill tier to 2-class binary label.

    This alternative labeling reduces class overlap issues:
    - novice: beginner, intermediate
    - skilled: advanced, professional

    For use when 4-class classification ceiling is too low due to label noise.
    """
    if skill_tier in ("beginner", "intermediate"):
        return "novice"
    else:
        return "skilled"


def compute_sample_labels(
    performance: GeneratedPerformance,
    rudiment: Rudiment,
    profile: PlayerProfile,
) -> Sample:
    """
    Compute all hierarchical labels for a generated performance.

    Args:
        performance: The generated MIDI performance
        rudiment: The rudiment definition
        profile: The player profile

    Returns:
        Complete Sample object with all labels
    """
    stroke_labels = compute_stroke_labels(performance.strokes)

    # Calculate strokes per measure from rudiment pattern
    strokes_per_measure = len(rudiment.pattern.strokes)

    # Use stroke_labels for aggregations (has corrected timing for grace notes)
    measure_labels = compute_measure_labels(stroke_labels, strokes_per_measure)
    exercise_scores = compute_exercise_scores(stroke_labels, performance.strokes, rudiment)

    # Compute tier confidence based on how central the score is to the tier's distribution
    skill_tier_str = profile.skill_tier.value
    tier_confidence = compute_tier_confidence(exercise_scores.overall_score, skill_tier_str)
    exercise_scores.tier_confidence = tier_confidence

    # Compute binary skill tier (novice vs skilled)
    skill_tier_binary = compute_skill_tier_binary(skill_tier_str)

    return Sample(
        sample_id=performance.id,
        profile_id=profile.id,
        rudiment_slug=rudiment.slug,
        tempo_bpm=performance.tempo_bpm,
        duration_sec=performance.duration_sec,
        num_cycles=performance.num_cycles,
        skill_tier=skill_tier_str,
        skill_tier_binary=skill_tier_binary,
        dominant_hand=profile.dominant_hand,
        strokes=stroke_labels,
        measures=measure_labels,
        exercise_scores=exercise_scores,
    )
