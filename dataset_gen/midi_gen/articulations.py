"""
Articulation-specific processing for different rudiment types.

This module handles the nuances of flams, diddles, rolls, and buzzes,
applying appropriate timing and velocity relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from dataset_gen.rudiments.schema import Rudiment, RudimentCategory, StrokeType
from dataset_gen.profiles.archetypes import PlayerProfile, RudimentSpecificDimensions
from dataset_gen.midi_gen.generator import StrokeEvent


@dataclass
class ArticulationParams:
    """Parameters for articulation processing."""

    flam_spacing_ms: float = 30.0
    flam_spacing_variance: float = 0.1
    flam_grace_velocity_ratio: float = 0.65

    diddle_ratio: float = 1.0  # Second/first stroke duration ratio
    diddle_ratio_variance: float = 0.05
    diddle_velocity_decay: float = 0.95  # Second stroke velocity multiplier

    roll_velocity_decay_per_stroke: float = 0.02
    roll_acceleration_factor: float = 0.0  # Tendency to speed up in rolls

    buzz_strokes_per_primary: int = 5
    buzz_velocity_decay: float = 0.85


class ArticulationEngine:
    """
    Process stroke events with rudiment-specific articulation rules.

    This engine refines the base timing and velocity deviations
    with articulation-specific relationships.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def process(
        self,
        events: list[StrokeEvent],
        rudiment: Rudiment,
        profile: PlayerProfile,
    ) -> list[StrokeEvent]:
        """
        Apply articulation-specific processing to stroke events.

        Args:
            events: List of stroke events to process
            rudiment: The rudiment being performed
            profile: Player profile with articulation parameters

        Returns:
            Processed stroke events with refined timing/velocity
        """
        dims = profile.dimensions.rudiment_specific

        # Process based on rudiment category
        if rudiment.category == RudimentCategory.FLAM:
            events = self._process_flams(events, dims)
        elif rudiment.category == RudimentCategory.DRAG:
            events = self._process_drags(events, dims)

        # Process diddles in all relevant rudiments
        events = self._process_diddles(events, dims)

        # Process rolls
        if rudiment.category == RudimentCategory.ROLL:
            events = self._process_rolls(events, dims, rudiment)

        return events

    def _process_flams(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
    ) -> list[StrokeEvent]:
        """
        Refine flam articulations.

        Ensures grace notes have proper spacing relative to primary strokes.
        """
        # Group grace notes with their primary strokes
        grace_to_primary: dict[int, int] = {}
        for event in events:
            if event.is_grace_note and event.parent_stroke_index is not None:
                grace_to_primary[event.index] = event.parent_stroke_index

        # Process each grace note
        for event in events:
            if event.index not in grace_to_primary:
                continue

            primary_idx = grace_to_primary[event.index]
            primary_event = next((e for e in events if e.index == primary_idx), None)
            if primary_event is None:
                continue

            # Calculate actual flam spacing
            base_spacing = dims.flam_spacing
            variance = dims.flam_spacing_variance * base_spacing
            actual_spacing = max(5, min(100, self.rng.normal(base_spacing, variance)))

            # Grace note should be before primary
            # Cap the timing error to prevent extreme deviations
            new_time = primary_event.actual_time_ms - actual_spacing
            max_error = 300.0
            timing_error = new_time - event.intended_time_ms
            timing_error = np.clip(timing_error, -max_error, max_error)
            event.actual_time_ms = event.intended_time_ms + timing_error

            # Grace note velocity is lower than primary
            grace_velocity_ratio = 0.6 + self.rng.normal(0, 0.05)
            event.actual_velocity = int(
                np.clip(primary_event.actual_velocity * grace_velocity_ratio, 1, 127)
            )

        return events

    def _process_drags(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
    ) -> list[StrokeEvent]:
        """
        Refine drag articulations.

        Drags have two grace notes before the primary, spaced evenly.
        """
        # Find drag patterns (two consecutive grace notes followed by accent)
        i = 0
        while i < len(events) - 2:
            if (
                events[i].is_grace_note
                and events[i + 1].is_grace_note
                and events[i + 2].stroke_type in (StrokeType.ACCENT, StrokeType.TAP)
            ):
                # This is a drag pattern
                grace1 = events[i]
                grace2 = events[i + 1]
                primary = events[i + 2]

                # Calculate drag spacing
                drag_spacing = dims.flam_spacing * 0.6  # Drags are tighter than flams
                variance = dims.flam_spacing_variance * drag_spacing

                spacing1 = max(5, min(80, self.rng.normal(drag_spacing * 1.5, variance)))
                spacing2 = max(3, min(50, self.rng.normal(drag_spacing * 0.8, variance)))

                # Calculate new times with error capping
                max_error = 300.0
                new_time1 = primary.actual_time_ms - spacing1 - spacing2
                new_time2 = primary.actual_time_ms - spacing2

                error1 = np.clip(new_time1 - grace1.intended_time_ms, -max_error, max_error)
                error2 = np.clip(new_time2 - grace2.intended_time_ms, -max_error, max_error)

                grace1.actual_time_ms = grace1.intended_time_ms + error1
                grace2.actual_time_ms = grace2.intended_time_ms + error2

                # Drag grace notes are typically played as a "buzz" diddle
                grace_velocity = int(primary.actual_velocity * 0.55)
                grace1.actual_velocity = grace_velocity
                grace2.actual_velocity = int(grace_velocity * 0.9)  # Second slightly softer

                i += 3
            else:
                i += 1

        return events

    def _process_diddles(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
    ) -> list[StrokeEvent]:
        """
        Refine diddle (double stroke) articulations.

        Adjusts timing ratio between first and second stroke of each diddle.
        """
        # Find diddle pairs
        i = 0
        while i < len(events) - 1:
            if (
                events[i].stroke_type == StrokeType.DIDDLE
                and events[i].diddle_position == 1
                and events[i + 1].stroke_type == StrokeType.DIDDLE
                and events[i + 1].diddle_position == 2
            ):
                first = events[i]
                second = events[i + 1]

                # Calculate the intended gap between strokes
                intended_gap = second.intended_time_ms - first.intended_time_ms

                # Apply evenness ratio with variance
                evenness = dims.diddle_evenness
                variance = dims.diddle_variance
                actual_ratio = max(0.5, min(1.5, self.rng.normal(evenness, variance)))

                # Adjust second stroke timing based on ratio
                # ratio > 1 means second stroke is late (lazy doubles)
                # ratio < 1 means second stroke is early (rushed doubles)
                actual_gap = intended_gap * actual_ratio
                new_time = first.actual_time_ms + actual_gap

                # Cap timing error
                max_error = 300.0
                error = np.clip(new_time - second.intended_time_ms, -max_error, max_error)
                second.actual_time_ms = second.intended_time_ms + error

                # Second stroke of diddle is typically slightly softer
                velocity_decay = 0.92 + self.rng.normal(0, 0.03)
                second.actual_velocity = int(
                    np.clip(first.actual_velocity * velocity_decay, 1, 127)
                )

                i += 2
            else:
                i += 1

        return events

    def _process_rolls(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
        rudiment: Rudiment,
    ) -> list[StrokeEvent]:
        """
        Apply roll-specific processing.

        Handles velocity decay across sustained rolls and buzz rolls.
        """
        params = rudiment.params

        if params.roll_type == "buzz":
            return self._process_buzz_roll(events, dims, params)

        # For open rolls, apply gradual velocity decay
        decay_rate = dims.roll_sustain
        for i, event in enumerate(events):
            if event.stroke_type in (StrokeType.DIDDLE, StrokeType.TAP):
                # Apply cumulative decay
                decay_factor = 1.0 - (i * decay_rate * 0.05)
                decay_factor = max(0.5, decay_factor)  # Don't decay below 50%
                event.actual_velocity = int(np.clip(event.actual_velocity * decay_factor, 1, 127))

        return events

    def _process_buzz_roll(
        self,
        events: list[StrokeEvent],
        dims: RudimentSpecificDimensions,
        params,
    ) -> list[StrokeEvent]:
        """
        Process buzz roll events.

        For buzz rolls, we'd typically generate multiple bounce strokes
        per primary stroke. This is a simplified version.
        """
        # Buzz density affects consistency of roll sound
        density_consistency = dims.buzz_density_consistency

        for event in events:
            # Add slight randomness to velocity based on buzz consistency
            jitter = (1 - density_consistency) * 20
            velocity_jitter = self.rng.normal(0, jitter)
            event.actual_velocity = int(np.clip(event.actual_velocity + velocity_jitter, 1, 127))

        return events


def apply_flam_spacing(
    events: list[StrokeEvent],
    spacing_ms: float,
    variance: float = 0.1,
    rng: np.random.Generator | None = None,
) -> list[StrokeEvent]:
    """
    Convenience function to apply flam spacing to events.

    Args:
        events: List of stroke events
        spacing_ms: Target spacing between grace and primary
        variance: Variance as proportion of spacing
        rng: Random number generator

    Returns:
        Modified events
    """
    if rng is None:
        rng = np.random.default_rng()

    for event in events:
        if event.is_grace_note and event.parent_stroke_index is not None:
            primary = next((e for e in events if e.index == event.parent_stroke_index), None)
            if primary:
                actual_spacing = max(5, rng.normal(spacing_ms, spacing_ms * variance))
                event.actual_time_ms = primary.actual_time_ms - actual_spacing

    return events


def apply_diddle_timing(
    events: list[StrokeEvent],
    evenness: float = 1.0,
    variance: float = 0.05,
    rng: np.random.Generator | None = None,
) -> list[StrokeEvent]:
    """
    Convenience function to apply diddle timing adjustments.

    Args:
        events: List of stroke events
        evenness: Target ratio between first and second stroke (1.0 = even)
        variance: Variance in evenness
        rng: Random number generator

    Returns:
        Modified events
    """
    if rng is None:
        rng = np.random.default_rng()

    i = 0
    while i < len(events) - 1:
        if (
            events[i].stroke_type == StrokeType.DIDDLE
            and events[i].diddle_position == 1
            and events[i + 1].stroke_type == StrokeType.DIDDLE
            and events[i + 1].diddle_position == 2
        ):
            first, second = events[i], events[i + 1]
            gap = second.intended_time_ms - first.intended_time_ms
            actual_ratio = max(0.5, min(1.5, rng.normal(evenness, variance)))
            second.actual_time_ms = first.actual_time_ms + gap * actual_ratio
            i += 2
        else:
            i += 1

    return events


def apply_roll_velocity_decay(
    events: list[StrokeEvent],
    decay_rate: float = 0.02,
) -> list[StrokeEvent]:
    """
    Apply velocity decay across a roll.

    Args:
        events: List of stroke events
        decay_rate: Decay per stroke (0.02 = 2% per stroke)

    Returns:
        Modified events
    """
    for i, event in enumerate(events):
        decay = 1.0 - (i * decay_rate)
        decay = max(0.5, decay)
        event.actual_velocity = int(np.clip(event.actual_velocity * decay, 1, 127))

    return events
