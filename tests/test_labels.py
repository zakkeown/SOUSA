"""Tests for label score computation formulas.

This module documents and validates the scoring formulas used in
dataset_gen/labels/compute.py. Each test case verifies that the
formula produces expected results for known inputs.
"""

import pytest

from dataset_gen.labels.schema import StrokeLabel
from dataset_gen.labels.compute import (
    compute_measure_labels,
    compute_exercise_scores,
    _compute_flam_quality,
    _compute_diddle_quality,
    _compute_roll_sustain,
    _empty_scores,
)
from dataset_gen.midi_gen.generator import StrokeEvent
from dataset_gen.rudiments.schema import (
    Rudiment,
    RudimentCategory,
    StickingPattern,
    Stroke,
    Hand,
    StrokeType,
)

# ============================================================================
# Test Fixtures
# ============================================================================


def make_stroke_label(
    index: int = 0,
    hand: str = "R",
    timing_error_ms: float = 0,
    actual_velocity: int = 80,
    is_accent: bool = False,
    is_grace_note: bool = False,
    stroke_type: str = "tap",
) -> StrokeLabel:
    """Create a StrokeLabel with sensible defaults."""
    return StrokeLabel(
        index=index,
        hand=hand,
        stroke_type=stroke_type,
        intended_time_ms=index * 100,
        actual_time_ms=index * 100 + timing_error_ms,
        timing_error_ms=timing_error_ms,
        intended_velocity=80,
        actual_velocity=actual_velocity,
        velocity_error=actual_velocity - 80,
        is_grace_note=is_grace_note,
        is_accent=is_accent,
    )


def make_stroke_event(
    index: int = 0,
    hand: Hand = Hand.RIGHT,
    timing_error_ms: float = 0,
    velocity: int = 80,
    stroke_type: StrokeType = StrokeType.TAP,
    is_grace_note: bool = False,
    parent_stroke_index: int | None = None,
    diddle_position: int | None = None,
) -> StrokeEvent:
    """Create a StrokeEvent with sensible defaults.

    Note: timing_error_ms and velocity_error are computed properties,
    so we set actual_time_ms = intended_time_ms + timing_error_ms.
    """
    intended_time = index * 100.0
    return StrokeEvent(
        index=index,
        hand=hand,
        stroke_type=stroke_type,
        intended_time_ms=intended_time,
        actual_time_ms=intended_time + timing_error_ms,
        intended_velocity=80,
        actual_velocity=velocity,
        is_grace_note=is_grace_note,
        parent_stroke_index=parent_stroke_index,
        diddle_position=diddle_position,
    )


def make_simple_rudiment(category: RudimentCategory = RudimentCategory.ROLL) -> Rudiment:
    """Create a simple rudiment for testing."""
    return Rudiment(
        name="Test Rudiment",
        slug="test_rudiment",
        category=category,
        pattern=StickingPattern(
            strokes=[
                Stroke(hand=Hand.RIGHT, stroke_type=StrokeType.TAP),
                Stroke(hand=Hand.LEFT, stroke_type=StrokeType.TAP),
            ],
            beats_per_cycle=1,
        ),
    )


# ============================================================================
# Timing Accuracy Tests
# ============================================================================


class TestTimingAccuracyFormula:
    """
    Test: timing_accuracy = 100 * (1 / (1 + exp((mean_abs_error - 25) / 10)))

    This formula uses sigmoid/perceptual scaling to match human perception:
    - Errors <10ms are nearly imperceptible (high score ~92+)
    - Errors ~25ms give ~50 (center of sigmoid)
    - Errors >50ms are clearly audible (low score ~7.5 or less)
    """

    def test_perfect_timing_gives_high_score(self):
        """0ms timing error should give score >90 (sigmoid asymptote)."""
        strokes = [make_stroke_label(i, timing_error_ms=0) for i in range(8)]
        events = [make_stroke_event(i, timing_error_ms=0) for i in range(8)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        # sigmoid(0ms) = 100 / (1 + exp(-2.5)) ≈ 92.4
        assert scores.timing_accuracy == pytest.approx(92.4, abs=0.5)

    def test_25ms_error_gives_50(self):
        """25ms mean timing error should give score of ~50 (sigmoid center)."""
        strokes = [make_stroke_label(i, timing_error_ms=25) for i in range(8)]
        events = [make_stroke_event(i, timing_error_ms=25) for i in range(8)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        assert scores.timing_accuracy == pytest.approx(50, abs=0.1)

    def test_50ms_error_gives_low_score(self):
        """50ms mean timing error should give low score (~7.5)."""
        strokes = [make_stroke_label(i, timing_error_ms=50) for i in range(8)]
        events = [make_stroke_event(i, timing_error_ms=50) for i in range(8)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        # sigmoid(50ms) = 100 / (1 + exp(2.5)) ≈ 7.6
        assert scores.timing_accuracy == pytest.approx(7.6, abs=0.5)

    def test_large_error_approaches_zero(self):
        """Errors >50ms should approach 0 (sigmoid tail)."""
        strokes = [make_stroke_label(i, timing_error_ms=100) for i in range(8)]
        events = [make_stroke_event(i, timing_error_ms=100) for i in range(8)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        # sigmoid(100ms) ≈ 0.055
        assert scores.timing_accuracy < 1

    def test_negative_errors_use_absolute_value(self):
        """Negative timing errors (early strokes) should use absolute value."""
        strokes = [make_stroke_label(i, timing_error_ms=-25) for i in range(8)]
        events = [make_stroke_event(i, timing_error_ms=-25) for i in range(8)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        assert scores.timing_accuracy == pytest.approx(50, abs=0.1)


# ============================================================================
# Timing Consistency Tests
# ============================================================================


class TestTimingConsistencyFormula:
    """
    Test: timing_consistency = 100 * (1 / (1 + exp((timing_std - 15) / 8)))

    This formula uses sigmoid scaling for timing consistency.
    - 0ms std gives ~87 (sigmoid asymptote)
    - 15ms std gives 50 (center)
    - High std approaches 0
    """

    def test_zero_variance_gives_high_score(self):
        """All strokes with same error should give high score (~87)."""
        strokes = [make_stroke_label(i, timing_error_ms=10) for i in range(8)]
        events = [make_stroke_event(i, timing_error_ms=10) for i in range(8)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        # sigmoid(0ms std) = 100 / (1 + exp(-15/8)) ≈ 86.7
        assert scores.timing_consistency == pytest.approx(86.7, abs=1.0)

    def test_high_variance_gives_low_score(self):
        """High variance in timing errors should give low score."""
        # Alternating +40 and -40 gives std ≈ 42.5
        strokes = []
        events = []
        for i in range(8):
            error = 40 if i % 2 == 0 else -40
            strokes.append(make_stroke_label(i, timing_error_ms=error))
            events.append(make_stroke_event(i, timing_error_ms=error))

        rudiment = make_simple_rudiment()
        scores = compute_exercise_scores(strokes, events, rudiment)
        # sigmoid(~42ms std) = 100 / (1 + exp((42-15)/8)) ≈ 3.4
        assert scores.timing_consistency < 10


# ============================================================================
# Hand Balance Tests
# ============================================================================


class TestHandBalanceFormula:
    """
    Test: hand_balance = 0.5 * velocity_balance + 0.5 * timing_balance

    Where:
    - velocity_balance = min(left_vel, right_vel) / max(left_vel, right_vel) * 100
    - timing_balance = min(left_timing_error, right_timing_error) / max(...) * 100

    This combined formula considers BOTH velocity AND timing balance.
    """

    def test_equal_hands_same_timing_gives_high_score(self):
        """Equal velocity and timing between hands should give high score."""
        strokes = [
            make_stroke_label(0, hand="R", actual_velocity=80, timing_error_ms=5),
            make_stroke_label(1, hand="L", actual_velocity=80, timing_error_ms=5),
            make_stroke_label(2, hand="R", actual_velocity=80, timing_error_ms=5),
            make_stroke_label(3, hand="L", actual_velocity=80, timing_error_ms=5),
        ]
        events = [
            make_stroke_event(0, hand=Hand.RIGHT, velocity=80, timing_error_ms=5),
            make_stroke_event(1, hand=Hand.LEFT, velocity=80, timing_error_ms=5),
            make_stroke_event(2, hand=Hand.RIGHT, velocity=80, timing_error_ms=5),
            make_stroke_event(3, hand=Hand.LEFT, velocity=80, timing_error_ms=5),
        ]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        # velocity_balance = 100, timing_balance uses quality-aware formula
        # symmetry = 1.0, quality = 100/(1+exp((5-25)/10)) ≈ 88.08
        # timing_balance = 0.4*100 + 0.6*88.08 ≈ 92.85
        # combined = 0.5*100 + 0.5*92.85 ≈ 96.4
        assert scores.hand_balance >= 95

    def test_imbalanced_velocity_same_timing(self):
        """Imbalanced velocity with same timing gives combined score."""
        strokes = [
            make_stroke_label(0, hand="R", actual_velocity=100, timing_error_ms=5),
            make_stroke_label(1, hand="L", actual_velocity=60, timing_error_ms=5),
            make_stroke_label(2, hand="R", actual_velocity=100, timing_error_ms=5),
            make_stroke_label(3, hand="L", actual_velocity=60, timing_error_ms=5),
        ]
        events = [
            make_stroke_event(0, hand=Hand.RIGHT, velocity=100, timing_error_ms=5),
            make_stroke_event(1, hand=Hand.LEFT, velocity=60, timing_error_ms=5),
            make_stroke_event(2, hand=Hand.RIGHT, velocity=100, timing_error_ms=5),
            make_stroke_event(3, hand=Hand.LEFT, velocity=60, timing_error_ms=5),
        ]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        # velocity_balance = 60/100 * 100 = 60
        # timing_balance: symmetry=1.0, quality≈88, tb≈0.4*100+0.6*88≈92.8
        # combined = 0.5*60 + 0.5*92.8 ≈ 76.4
        assert scores.hand_balance == pytest.approx(76.4, abs=2)

    def test_single_hand_gives_100(self):
        """Single-hand exercise should give 100 (no imbalance possible)."""
        strokes = [make_stroke_label(i, hand="R", actual_velocity=80) for i in range(4)]
        events = [make_stroke_event(i, hand=Hand.RIGHT, velocity=80) for i in range(4)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        assert scores.hand_balance == 100


# ============================================================================
# Overall Score Tests
# ============================================================================


class TestOverallScoreWeights:
    """
    Test: overall_score is a weighted average with these weights:
        timing_accuracy: 0.2
        timing_consistency: 0.15
        tempo_stability: 0.1
        subdivision_evenness: 0.1
        velocity_control: 0.1
        accent_differentiation: 0.1
        accent_accuracy: 0.1
        hand_balance: 0.15
    Total: 1.0
    """

    def test_weights_sum_to_one(self):
        """Verify documented weights sum to 1.0."""
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
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_perfect_performance_gives_high_overall(self):
        """Perfect strokes should give high overall score."""
        strokes = [
            make_stroke_label(i, hand="R" if i % 2 == 0 else "L", timing_error_ms=0)
            for i in range(8)
        ]
        events = [
            make_stroke_event(i, hand=Hand.RIGHT if i % 2 == 0 else Hand.LEFT, timing_error_ms=0)
            for i in range(8)
        ]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        # With perfect timing, scores are high but not 100 due to sigmoid asymptotes
        # timing_accuracy ~92.4, timing_consistency ~86.7, etc.
        # Overall should still be solidly high
        assert scores.overall_score >= 85


# ============================================================================
# Velocity Control Tests
# ============================================================================


class TestVelocityControlFormula:
    """
    Test: velocity_control = max(0, 100 - vel_std * 2)

    Lower velocity variance = higher control score.
    """

    def test_consistent_velocity_gives_high_score(self):
        """Consistent velocity should give high score."""
        strokes = [make_stroke_label(i, actual_velocity=80) for i in range(8)]
        events = [make_stroke_event(i, velocity=80) for i in range(8)]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)
        assert scores.velocity_control == 100

    def test_variable_velocity_gives_low_score(self):
        """High velocity variance should give low score."""
        strokes = []
        events = []
        for i in range(8):
            vel = 40 if i % 2 == 0 else 120
            strokes.append(make_stroke_label(i, actual_velocity=vel))
            events.append(make_stroke_event(i, velocity=vel))

        rudiment = make_simple_rudiment()
        scores = compute_exercise_scores(strokes, events, rudiment)
        # std of [40,120,40,120,...] = 40
        # 100 - 40*2 = 20
        assert scores.velocity_control == pytest.approx(20, abs=1)


# ============================================================================
# Flam Quality Tests
# ============================================================================


class TestFlamQualityFormula:
    """
    Test: flam_quality based on grace note spacing.
    Ideal range: 20-40ms
    - Inside range: 100
    - Outside range: penalty based on distance
    """

    def test_ideal_spacing_gives_100(self):
        """Grace note at 30ms before primary should give 100."""
        events = [
            make_stroke_event(
                0,
                is_grace_note=True,
                parent_stroke_index=1,
                timing_error_ms=0,
            ),
            make_stroke_event(1, timing_error_ms=30),  # Primary 30ms after grace
        ]
        # Set actual times so spacing is 30ms
        events[0].actual_time_ms = 0
        events[1].actual_time_ms = 30

        score = _compute_flam_quality(events)
        assert score == 100

    def test_too_tight_spacing_reduces_score(self):
        """Grace note <20ms before primary should reduce score."""
        events = [
            make_stroke_event(0, is_grace_note=True, parent_stroke_index=1),
            make_stroke_event(1),
        ]
        events[0].actual_time_ms = 0
        events[1].actual_time_ms = 10  # Only 10ms spacing

        score = _compute_flam_quality(events)
        # 10ms is 10ms below ideal_min (20ms)
        # Penalty: (20-10) * 5 = 50
        # Score: 100 - 50 = 50
        assert score == pytest.approx(50, abs=1)

    def test_too_wide_spacing_reduces_score(self):
        """Grace note >40ms before primary should reduce score."""
        events = [
            make_stroke_event(0, is_grace_note=True, parent_stroke_index=1),
            make_stroke_event(1),
        ]
        events[0].actual_time_ms = 0
        events[1].actual_time_ms = 60  # 60ms spacing

        score = _compute_flam_quality(events)
        # 60ms is 20ms above ideal_max (40ms)
        # Penalty: (60-40) * 3 = 60
        # Score: 100 - 60 = 40
        assert score == pytest.approx(40, abs=1)


# ============================================================================
# Measure Label Tests
# ============================================================================


class TestMeasureLabelComputation:
    """Test aggregation of strokes into measure-level statistics."""

    def test_empty_strokes_returns_empty_list(self):
        """Empty stroke list should return empty measure list."""
        measures = compute_measure_labels([], strokes_per_measure=4)
        assert measures == []

    def test_single_measure_aggregation(self):
        """Four strokes should produce one measure."""
        strokes = [make_stroke_label(i, timing_error_ms=10) for i in range(4)]
        measures = compute_measure_labels(strokes, strokes_per_measure=4)

        assert len(measures) == 1
        assert measures[0].timing_mean_error_ms == 10
        assert measures[0].timing_std_ms == 0  # All same error
        assert measures[0].stroke_start == 0
        assert measures[0].stroke_end == 4

    def test_multiple_measures(self):
        """Eight strokes with 4 per measure should produce two measures."""
        strokes = [make_stroke_label(i, timing_error_ms=i * 5) for i in range(8)]
        measures = compute_measure_labels(strokes, strokes_per_measure=4)

        assert len(measures) == 2
        assert measures[0].stroke_start == 0
        assert measures[0].stroke_end == 4
        assert measures[1].stroke_start == 4
        assert measures[1].stroke_end == 8


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_strokes_returns_zero_scores(self):
        """Empty stroke list should return all-zero scores."""
        scores = _empty_scores()

        assert scores.timing_accuracy == 0
        assert scores.overall_score == 0
        assert scores.weak_hand_index == 50  # Default balanced

    def test_single_stroke_handles_gracefully(self):
        """Single stroke should not cause division errors."""
        strokes = [make_stroke_label(0, timing_error_ms=10)]
        events = [make_stroke_event(0, timing_error_ms=10)]
        rudiment = make_simple_rudiment()

        # Should not raise
        scores = compute_exercise_scores(strokes, events, rudiment)
        assert 0 <= scores.overall_score <= 100

    def test_all_scores_in_valid_range(self):
        """All score fields should be in [0, 100] range."""
        strokes = [
            make_stroke_label(i, hand="R" if i % 2 == 0 else "L", timing_error_ms=i * 5)
            for i in range(16)
        ]
        events = [
            make_stroke_event(
                i, hand=Hand.RIGHT if i % 2 == 0 else Hand.LEFT, timing_error_ms=i * 5
            )
            for i in range(16)
        ]
        rudiment = make_simple_rudiment()

        scores = compute_exercise_scores(strokes, events, rudiment)

        assert 0 <= scores.timing_accuracy <= 100
        assert 0 <= scores.timing_consistency <= 100
        assert 0 <= scores.tempo_stability <= 100
        assert 0 <= scores.subdivision_evenness <= 100
        assert 0 <= scores.velocity_control <= 100
        assert 0 <= scores.accent_differentiation <= 100
        assert 0 <= scores.accent_accuracy <= 100
        assert 0 <= scores.hand_balance <= 100
        assert 0 <= scores.overall_score <= 100


# ============================================================================
# Diddle Quality Tests
# ============================================================================


class TestDiddleQualityFormula:
    """
    Test: diddle_quality based on evenness of double strokes.
    Score based on how close actual/intended gap ratio is to 1.0.
    """

    def test_even_diddles_gives_high_score(self):
        """Perfectly even diddles should give high score."""
        events = [
            make_stroke_event(0, stroke_type=StrokeType.DIDDLE, diddle_position=1),
            make_stroke_event(1, stroke_type=StrokeType.DIDDLE, diddle_position=2),
        ]
        events[0].intended_time_ms = 0
        events[0].actual_time_ms = 0
        events[1].intended_time_ms = 50
        events[1].actual_time_ms = 50  # Ratio = 1.0

        score = _compute_diddle_quality(events)
        assert score == 100

    def test_uneven_diddles_reduces_score(self):
        """Uneven diddles should reduce score."""
        events = [
            make_stroke_event(0, stroke_type=StrokeType.DIDDLE, diddle_position=1),
            make_stroke_event(1, stroke_type=StrokeType.DIDDLE, diddle_position=2),
        ]
        events[0].intended_time_ms = 0
        events[0].actual_time_ms = 0
        events[1].intended_time_ms = 50
        events[1].actual_time_ms = 75  # 50% longer than intended

        score = _compute_diddle_quality(events)
        # Ratio = 75/50 = 1.5, deviation = 0.5
        # Score = 100 - 0.5 * 200 = 0
        assert score == 0


# ============================================================================
# Roll Sustain Tests
# ============================================================================


class TestRollSustainFormula:
    """
    Test: roll_sustain based on velocity decay over time.
    Score = min(100, last_quarter_mean / first_quarter_mean * 100)
    """

    def test_consistent_velocity_gives_high_score(self):
        """Consistent velocity throughout should give high score."""
        events = [make_stroke_event(i, velocity=80) for i in range(16)]
        score = _compute_roll_sustain(events)
        assert score == 100

    def test_decaying_velocity_reduces_score(self):
        """Velocity decay should reduce score."""
        events = []
        for i in range(16):
            # Velocity decreases from 100 to 50
            vel = 100 - (i * 50 // 15)
            events.append(make_stroke_event(i, velocity=vel))

        score = _compute_roll_sustain(events)
        # First quarter avg ~100, last quarter avg ~50
        # Score = 50/100 * 100 = 50
        assert score < 80  # Approximate check

    def test_few_strokes_returns_default(self):
        """Fewer than 4 strokes should return default score."""
        events = [make_stroke_event(i, velocity=80) for i in range(3)]
        score = _compute_roll_sustain(events)
        assert score == 80  # Default
