"""
MIDI generation engine for synthetic drum performances.

This module generates MIDI sequences from rudiment definitions and player profiles,
applying realistic timing deviations, velocity variations, and articulations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
import numpy as np
import mido

from dataset_gen.rudiments.schema import (
    Rudiment,
    StrokeType,
    Hand,
)
from dataset_gen.profiles.archetypes import PlayerProfile

# MIDI drum note mappings (General MIDI)
SNARE_NOTE = 38  # Acoustic snare
LEFT_HAND_NOTE = 38  # Same pitch, different track/channel for analysis
RIGHT_HAND_NOTE = 38


@dataclass
class StrokeEvent:
    """A single stroke event with timing and velocity information."""

    # Stroke identity
    index: int
    hand: Hand
    stroke_type: StrokeType

    # Timing (in milliseconds from start)
    intended_time_ms: float
    actual_time_ms: float

    # Velocity (MIDI 0-127)
    intended_velocity: int
    actual_velocity: int

    # Articulation-specific
    is_grace_note: bool = False
    parent_stroke_index: int | None = None  # For grace notes, which primary they belong to
    diddle_position: int | None = None  # 1 or 2 for diddle strokes

    @property
    def timing_error_ms(self) -> float:
        """Deviation from intended timing."""
        return self.actual_time_ms - self.intended_time_ms

    @property
    def velocity_error(self) -> int:
        """Deviation from intended velocity."""
        return self.actual_velocity - self.intended_velocity


@dataclass
class GeneratedPerformance:
    """Complete generated performance with all stroke events and metadata."""

    id: str = field(default_factory=lambda: str(uuid4()))

    # Source information
    rudiment_slug: str = ""
    profile_id: str = ""
    tempo_bpm: int = 120
    duration_sec: float = 0.0
    num_cycles: int = 1

    # Stroke events
    strokes: list[StrokeEvent] = field(default_factory=list)

    # The generated MIDI data
    midi_data: bytes | None = None

    def to_midi_file(self, path: Path | str) -> None:
        """Write MIDI data to file."""
        if self.midi_data is None:
            raise ValueError("No MIDI data to write")
        Path(path).write_bytes(self.midi_data)


class MIDIGenerator:
    """
    Generate MIDI performances from rudiments and player profiles.

    The generator applies the player's execution dimensions to create
    realistic deviations in timing and velocity.
    """

    def __init__(
        self,
        seed: int | None = None,
        ticks_per_beat: int = 480,
    ):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
            ticks_per_beat: MIDI resolution (PPQ)
        """
        self.rng = np.random.default_rng(seed)
        self.ticks_per_beat = ticks_per_beat

    def generate(
        self,
        rudiment: Rudiment,
        profile: PlayerProfile,
        tempo_bpm: int = 120,
        num_cycles: int | None = None,
        target_duration_sec: float | None = None,
        include_midi: bool = True,
    ) -> GeneratedPerformance:
        """
        Generate a performance of a rudiment.

        Args:
            rudiment: The rudiment to perform
            profile: The player profile defining execution characteristics
            tempo_bpm: Tempo in BPM
            num_cycles: How many times to repeat the rudiment pattern (if target_duration_sec not set)
            target_duration_sec: Target duration in seconds (overrides num_cycles if set)
            include_midi: Whether to generate MIDI bytes

        Returns:
            GeneratedPerformance with stroke events and optional MIDI data
        """
        # Calculate num_cycles from target duration if specified
        if target_duration_sec is not None:
            # Calculate how long one cycle takes at this tempo
            cycle_duration = rudiment.duration_at_tempo(tempo_bpm, num_cycles=1)
            # Calculate cycles needed (at least 1, round to nearest int)
            num_cycles = max(1, round(target_duration_sec / cycle_duration))
        elif num_cycles is None:
            num_cycles = 4  # Default fallback

        # Generate ideal timing grid
        ideal_events = self._generate_ideal_events(rudiment, tempo_bpm, num_cycles)

        # Apply player deviations
        actual_events = self._apply_deviations(ideal_events, profile, rudiment, tempo_bpm)

        # Calculate duration
        if actual_events:
            duration_sec = max(e.actual_time_ms for e in actual_events) / 1000.0 + 0.5
        else:
            duration_sec = 0.0

        performance = GeneratedPerformance(
            rudiment_slug=rudiment.slug,
            profile_id=profile.id,
            tempo_bpm=tempo_bpm,
            duration_sec=duration_sec,
            num_cycles=num_cycles,
            strokes=actual_events,
        )

        if include_midi:
            performance.midi_data = self._events_to_midi(actual_events, tempo_bpm)

        return performance

    def _generate_ideal_events(
        self,
        rudiment: Rudiment,
        tempo_bpm: int,
        num_cycles: int,
    ) -> list[StrokeEvent]:
        """Generate ideal (perfect) stroke events from rudiment pattern."""
        events = []
        ms_per_beat = 60000.0 / tempo_bpm

        # Calculate timing per stroke based on subdivision
        strokes_in_pattern = len(rudiment.pattern.strokes)
        beats_per_cycle = rudiment.pattern.beats_per_cycle
        ms_per_stroke = (beats_per_cycle * ms_per_beat) / strokes_in_pattern

        stroke_index = 0
        for cycle in range(num_cycles):
            cycle_start_ms = cycle * beats_per_cycle * ms_per_beat

            for i, stroke in enumerate(rudiment.pattern.strokes):
                # Calculate ideal timing
                stroke_offset_ms = i * ms_per_stroke
                ideal_time = cycle_start_ms + stroke_offset_ms

                # Adjust for grace notes (they come before the beat)
                if stroke.stroke_type == StrokeType.GRACE and stroke.grace_offset:
                    # grace_offset is in beats, convert to ms
                    ideal_time += stroke.grace_offset * ms_per_beat

                # Calculate ideal velocity
                if stroke.stroke_type == StrokeType.ACCENT:
                    ideal_velocity = 120
                elif stroke.stroke_type == StrokeType.GRACE:
                    ideal_velocity = 40
                elif stroke.stroke_type == StrokeType.DIDDLE:
                    ideal_velocity = 75
                elif stroke.stroke_type == StrokeType.BUZZ:
                    ideal_velocity = 70
                else:  # TAP
                    ideal_velocity = 65

                # Find parent stroke for grace notes
                parent_index = None
                if stroke.stroke_type == StrokeType.GRACE:
                    # Parent is the next non-grace stroke
                    for j in range(i + 1, len(rudiment.pattern.strokes)):
                        if rudiment.pattern.strokes[j].stroke_type != StrokeType.GRACE:
                            parent_index = stroke_index + (j - i)
                            break

                event = StrokeEvent(
                    index=stroke_index,
                    hand=stroke.hand,
                    stroke_type=stroke.stroke_type,
                    intended_time_ms=ideal_time,
                    actual_time_ms=ideal_time,  # Will be modified by deviations
                    intended_velocity=ideal_velocity,
                    actual_velocity=ideal_velocity,  # Will be modified by deviations
                    is_grace_note=stroke.stroke_type == StrokeType.GRACE,
                    parent_stroke_index=parent_index,
                    diddle_position=stroke.diddle_position,
                )
                events.append(event)
                stroke_index += 1

        return events

    def _apply_deviations(
        self,
        events: list[StrokeEvent],
        profile: PlayerProfile,
        rudiment: Rudiment,
        tempo_bpm: int,
    ) -> list[StrokeEvent]:
        """Apply player-specific deviations to ideal events."""
        dims = profile.dimensions
        modified_events = []

        # Calculate tempo penalty (worse outside comfort zone)
        tempo_penalty = profile.get_tempo_penalty(tempo_bpm)

        # Pre-calculate base timing noise for the exercise
        n_events = len(events)
        base_timing_noise = self.rng.normal(
            0,
            dims.timing.timing_accuracy * tempo_penalty,
            size=n_events,
        )

        # Add timing consistency variation (how random vs systematic errors are)
        consistency_noise = (
            self.rng.normal(0, 1, size=n_events) * dims.timing.timing_consistency * 10
        )

        # Calculate tempo drift parameters
        # tempo_drift represents ms of drift per beat (not fraction of total duration)
        total_duration_ms = events[-1].intended_time_ms if events else 0
        ms_per_beat = 60000.0 / tempo_bpm
        total_beats = total_duration_ms / ms_per_beat if ms_per_beat > 0 else 0

        for i, event in enumerate(events):
            modified = StrokeEvent(
                index=event.index,
                hand=event.hand,
                stroke_type=event.stroke_type,
                intended_time_ms=event.intended_time_ms,
                actual_time_ms=event.intended_time_ms,
                intended_velocity=event.intended_velocity,
                actual_velocity=event.intended_velocity,
                is_grace_note=event.is_grace_note,
                parent_stroke_index=event.parent_stroke_index,
                diddle_position=event.diddle_position,
            )

            # === TIMING DEVIATIONS ===

            # Base timing error
            timing_error = base_timing_noise[i] + consistency_noise[i]

            # Hand-specific bias
            is_non_dominant = (profile.dominant_hand == "right" and event.hand == Hand.LEFT) or (
                profile.dominant_hand == "left" and event.hand == Hand.RIGHT
            )
            if is_non_dominant:
                timing_error += dims.hand_balance.lr_timing_bias

            # Tempo drift (rushing or dragging over time)
            # tempo_drift is fraction of beat duration that accumulates per beat
            # e.g., tempo_drift=0.02 at 120BPM (500ms/beat) = 10ms drift per beat
            current_beat = event.intended_time_ms / ms_per_beat if ms_per_beat > 0 else 0
            # Drift direction: positive = rushing (early), negative = dragging (late)
            # Use a consistent direction for this performance (set once per performance)
            if i == 0:
                self._drift_direction = 1.0 if self.rng.random() > 0.5 else -1.0
            drift_amount = (
                current_beat * dims.timing.tempo_drift * ms_per_beat * self._drift_direction
            )
            timing_error += drift_amount

            # Fatigue effect (performance degrades over time)
            progress = event.intended_time_ms / max(total_duration_ms, 1)
            fatigue_factor = 1.0 + (progress * profile.fatigue_coefficient * 0.5)
            timing_error *= fatigue_factor

            # Cap timing error to prevent runaway accumulation
            # Even beginners shouldn't be more than ~300ms off
            max_timing_error = 300.0
            timing_error = np.clip(timing_error, -max_timing_error, max_timing_error)

            modified.actual_time_ms = event.intended_time_ms + timing_error

            # Ensure non-negative time (MIDI can't represent negative times)
            if modified.actual_time_ms < 0:
                modified.actual_time_ms = 0.0

            # === VELOCITY DEVIATIONS ===

            # Base velocity with variance
            velocity_error = self.rng.normal(
                0, event.intended_velocity * dims.dynamics.velocity_variance
            )

            # Hand balance: non-dominant hand hits softer
            if is_non_dominant:
                velocity_reduction = event.intended_velocity * (
                    1 - dims.hand_balance.lr_velocity_ratio
                )
                velocity_error -= velocity_reduction

            # Add consistency delta for non-dominant hand
            if is_non_dominant:
                velocity_error += self.rng.normal(
                    0, event.intended_velocity * dims.hand_balance.lr_consistency_delta
                )

            # Accent accuracy: sometimes miss accents or add unwanted accents
            if event.stroke_type == StrokeType.ACCENT:
                if self.rng.random() > dims.dynamics.accent_accuracy:
                    # Missed accent - reduce velocity
                    velocity_error -= dims.dynamics.accent_differentiation * 3
            else:
                # Non-accent strokes: occasionally add unwanted emphasis
                if self.rng.random() > dims.dynamics.accent_accuracy:
                    # False accent
                    velocity_error += dims.dynamics.accent_differentiation * 2

            # Apply fatigue to dynamics
            velocity_error -= progress * profile.fatigue_coefficient * event.intended_velocity * 0.5

            modified.actual_velocity = int(
                np.clip(event.intended_velocity + velocity_error, 1, 127)
            )

            # === RUDIMENT-SPECIFIC ARTICULATIONS ===

            # Flam spacing variation
            if event.is_grace_note and event.parent_stroke_index is not None:
                # Add variance to the grace note timing relative to parent
                base_spacing = dims.rudiment_specific.flam_spacing
                spacing_variance = dims.rudiment_specific.flam_spacing_variance * base_spacing
                actual_spacing = self.rng.normal(base_spacing, spacing_variance)
                # Grace notes should be before the beat
                modified.actual_time_ms = (
                    modified.actual_time_ms
                )  # Keep base, articulation engine refines

            # Diddle evenness
            if event.diddle_position == 2:
                # Second diddle stroke: adjust timing based on evenness
                evenness = dims.rudiment_specific.diddle_evenness
                evenness_variance = dims.rudiment_specific.diddle_variance
                actual_ratio = self.rng.normal(evenness, evenness_variance)
                # The second stroke timing deviation is relative to the first
                # (This is a simplification - real implementation would look at stroke pairs)

            modified_events.append(modified)

        return modified_events

    def _events_to_midi(self, events: list[StrokeEvent], tempo_bpm: int) -> bytes:
        """Convert stroke events to MIDI bytes."""
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)

        # Create a single track for simplicity
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        tempo_us = mido.bpm2tempo(tempo_bpm)
        track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

        # Sort events by actual time
        sorted_events = sorted(events, key=lambda e: e.actual_time_ms)

        # Convert to MIDI messages
        current_time_ticks = 0
        ticks_per_ms = self.ticks_per_beat * tempo_bpm / 60000.0

        for event in sorted_events:
            # MIDI can't represent negative times - clamp to 0
            event_time_ms = max(0.0, event.actual_time_ms)
            event_time_ticks = int(event_time_ms * ticks_per_ms)
            delta_ticks = max(0, event_time_ticks - current_time_ticks)

            # Note on
            track.append(
                mido.Message(
                    "note_on",
                    note=SNARE_NOTE,
                    velocity=event.actual_velocity,
                    time=delta_ticks,
                )
            )
            current_time_ticks = event_time_ticks

            # Note off (minimal duration for percussion - timing is based on attacks only)
            # Use 1 tick to avoid affecting subsequent note timing calculations
            track.append(
                mido.Message(
                    "note_off",
                    note=SNARE_NOTE,
                    velocity=0,
                    time=1,
                )
            )
            current_time_ticks += 1

        # End of track
        track.append(mido.MetaMessage("end_of_track", time=0))

        # Convert to bytes
        from io import BytesIO

        buffer = BytesIO()
        mid.save(file=buffer)
        return buffer.getvalue()


def generate_performance(
    rudiment: Rudiment,
    profile: PlayerProfile,
    tempo_bpm: int = 120,
    num_cycles: int | None = None,
    target_duration_sec: float | None = None,
    seed: int | None = None,
) -> GeneratedPerformance:
    """
    Convenience function to generate a single performance.

    Args:
        rudiment: The rudiment to perform
        profile: Player profile
        tempo_bpm: Tempo
        num_cycles: Number of pattern repetitions (if target_duration_sec not set)
        target_duration_sec: Target duration in seconds (overrides num_cycles)
        seed: Random seed

    Returns:
        GeneratedPerformance
    """
    generator = MIDIGenerator(seed=seed)
    return generator.generate(
        rudiment,
        profile,
        tempo_bpm,
        num_cycles=num_cycles,
        target_duration_sec=target_duration_sec,
    )


def regenerate_midi(
    performance: GeneratedPerformance,
    ticks_per_beat: int = 480,
) -> bytes:
    """
    Regenerate MIDI data from a performance's stroke events.

    Use this after modifying stroke timings/velocities (e.g., via ArticulationEngine)
    to ensure MIDI matches the labels.

    Args:
        performance: Performance with (possibly modified) stroke events
        ticks_per_beat: MIDI resolution

    Returns:
        New MIDI data as bytes
    """
    generator = MIDIGenerator(ticks_per_beat=ticks_per_beat)
    return generator._events_to_midi(performance.strokes, performance.tempo_bpm)
