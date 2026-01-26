"""
Groove feel heuristic proxy computation.

Since "groove" is subjective and hard to define, we use computed proxies
that capture aspects of intentional vs. sloppy microtiming.
"""

from dataclasses import dataclass
import numpy as np

from dataset_gen.midi_gen.generator import StrokeEvent
from dataset_gen.rudiments.schema import StrokeType


@dataclass
class GrooveMetrics:
    """Detailed groove analysis metrics."""

    # Pattern consistency: high = systematic deviations, low = random errors
    pattern_consistency: float

    # Swing detection: how consistently timing deviates in a swing pattern
    swing_ratio: float | None

    # Accent timing bias: do accents lead (driving) or lag (laid back)
    accent_timing_bias: float

    # Overall groove feel proxy (0-1)
    groove_feel: float


def compute_groove_feel_proxy(events: list[StrokeEvent]) -> float:
    """
    Compute a groove feel proxy score.

    The proxy distinguishes between:
    - Intentional microtiming (groove): Consistent patterns of deviation
    - Sloppy playing: Random, inconsistent errors

    Args:
        events: List of stroke events

    Returns:
        Score from 0 (sloppy) to 1 (good groove or perfectly accurate)
    """
    if len(events) < 4:
        return 0.5  # Not enough data

    metrics = compute_groove_metrics(events)
    return metrics.groove_feel


def compute_groove_metrics(events: list[StrokeEvent]) -> GrooveMetrics:
    """
    Compute detailed groove metrics.

    Args:
        events: List of stroke events

    Returns:
        GrooveMetrics object with all computed values
    """
    timing_errors = np.array([e.timing_error_ms for e in events])

    # 1. PATTERN CONSISTENCY
    # Use autocorrelation to detect systematic timing patterns
    # High autocorrelation = consistent patterns (good groove or good timing)
    # Low autocorrelation = random errors (sloppy)
    pattern_consistency = _compute_autocorrelation(timing_errors)

    # 2. SWING DETECTION
    # Check for consistent push/pull on alternate notes
    swing_ratio = _detect_swing(events)

    # 3. ACCENT TIMING BIAS
    # Check if accented notes consistently lead or lag
    accent_bias = _compute_accent_timing_bias(events)

    # 4. COMBINE INTO GROOVE FEEL
    # High groove = either accurate OR consistently deviant
    # Low groove = random, inconsistent errors

    # Base score on timing accuracy
    mean_abs_error = np.mean(np.abs(timing_errors))
    accuracy_score = max(0, 1 - mean_abs_error / 50)  # 0-50ms -> 1-0

    # Boost for pattern consistency (intentional deviations)
    consistency_boost = pattern_consistency * 0.3

    # Small boost for detected swing
    swing_boost = 0.1 if swing_ratio is not None and 0.55 < swing_ratio < 0.75 else 0

    groove_feel = min(1.0, accuracy_score + consistency_boost + swing_boost)

    return GrooveMetrics(
        pattern_consistency=pattern_consistency,
        swing_ratio=swing_ratio,
        accent_timing_bias=accent_bias,
        groove_feel=groove_feel,
    )


def _compute_autocorrelation(errors: np.ndarray, max_lag: int = 4) -> float:
    """
    Compute mean autocorrelation of timing errors.

    High autocorrelation indicates systematic (non-random) deviations.
    """
    if len(errors) < max_lag + 2:
        return 0.5

    # Normalize
    errors = errors - np.mean(errors)
    std = np.std(errors)
    if std == 0:
        return 1.0  # Perfect timing = perfect "consistency"

    errors = errors / std

    # Compute autocorrelation for lags 1 to max_lag
    autocorrs = []
    n = len(errors)
    for lag in range(1, min(max_lag + 1, n)):
        autocorr = np.sum(errors[:-lag] * errors[lag:]) / (n - lag)
        autocorrs.append(autocorr)

    # Take mean absolute autocorrelation
    mean_autocorr = np.mean(np.abs(autocorrs))
    return float(mean_autocorr)


def _detect_swing(events: list[StrokeEvent]) -> float | None:
    """
    Detect swing feel by looking at alternating note timing.

    Returns swing ratio (proportion of time on downbeats) or None if not detected.
    """
    if len(events) < 4:
        return None

    # Look at pairs of notes and compute timing ratio
    ratios = []
    for i in range(0, len(events) - 1, 2):
        e1, e2 = events[i], events[i + 1]
        gap1 = e1.actual_time_ms - e1.intended_time_ms
        gap2 = e2.actual_time_ms - e2.intended_time_ms

        # Compute the relative position of note 2 within the beat
        intended_gap = e2.intended_time_ms - e1.intended_time_ms
        actual_gap = e2.actual_time_ms - e1.actual_time_ms

        if intended_gap > 0:
            ratio = actual_gap / intended_gap
            if 0.3 < ratio < 1.7:  # Sanity check
                ratios.append(ratio)

    if len(ratios) < 2:
        return None

    # Check if ratios are consistent (swing) vs random
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)

    # Swing typically has ratio > 0.55 (triplet feel ~0.67)
    # with low variance (consistent application)
    if std_ratio < 0.15:  # Consistent timing
        return mean_ratio

    return None


def _compute_accent_timing_bias(events: list[StrokeEvent]) -> float:
    """
    Compute timing bias of accented notes.

    Positive = accents lead (driving feel)
    Negative = accents lag (laid back)
    """
    accent_events = [e for e in events if e.stroke_type == StrokeType.ACCENT]

    if not accent_events:
        return 0.0

    accent_errors = [e.timing_error_ms for e in accent_events]
    return float(np.mean(accent_errors))


def analyze_groove_quality(events: list[StrokeEvent]) -> dict:
    """
    Produce a detailed groove analysis for debugging/inspection.

    Args:
        events: List of stroke events

    Returns:
        Dictionary with detailed groove analysis
    """
    metrics = compute_groove_metrics(events)
    timing_errors = np.array([e.timing_error_ms for e in events])

    return {
        "groove_feel_proxy": metrics.groove_feel,
        "pattern_consistency": metrics.pattern_consistency,
        "swing_ratio": metrics.swing_ratio,
        "accent_timing_bias_ms": metrics.accent_timing_bias,
        "timing_stats": {
            "mean_error_ms": float(np.mean(timing_errors)),
            "std_error_ms": float(np.std(timing_errors)),
            "max_error_ms": float(np.max(np.abs(timing_errors))),
        },
        "interpretation": _interpret_groove(metrics),
    }


def _interpret_groove(metrics: GrooveMetrics) -> str:
    """Generate human-readable interpretation of groove metrics."""
    interpretations = []

    if metrics.groove_feel > 0.8:
        interpretations.append("Excellent groove or timing accuracy")
    elif metrics.groove_feel > 0.6:
        interpretations.append("Good groove feel")
    elif metrics.groove_feel > 0.4:
        interpretations.append("Moderate groove, some inconsistency")
    else:
        interpretations.append("Inconsistent timing, lacking groove")

    if metrics.pattern_consistency > 0.5:
        interpretations.append("Systematic timing patterns detected")
    elif metrics.pattern_consistency < 0.2:
        interpretations.append("Random/erratic timing deviations")

    if metrics.swing_ratio is not None:
        if metrics.swing_ratio > 0.6:
            interpretations.append(f"Swing feel detected (ratio: {metrics.swing_ratio:.2f})")
        else:
            interpretations.append("Straight timing")

    if abs(metrics.accent_timing_bias) > 5:
        direction = "ahead" if metrics.accent_timing_bias > 0 else "behind"
        interpretations.append(f"Accents tend to be {direction} the beat")

    return "; ".join(interpretations)
