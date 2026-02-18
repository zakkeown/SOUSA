"""
Pydantic models for rudiment definitions.

The schema captures all 40 PAS rudiments with their sticking patterns,
accent patterns, and articulation-specific parameters.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class Hand(str, Enum):
    """Which hand plays the stroke."""

    LEFT = "L"
    RIGHT = "R"


class StrokeType(str, Enum):
    """Type of drum stroke."""

    TAP = "tap"  # Unaccented single stroke
    ACCENT = "accent"  # Accented single stroke
    GRACE = "grace"  # Grace note (flams, drags)
    DIDDLE = "diddle"  # Double stroke (part of a diddle)
    BUZZ = "buzz"  # Buzz/press roll stroke


class Stroke(BaseModel):
    """A single stroke in a rudiment pattern."""

    hand: Hand
    stroke_type: StrokeType = StrokeType.TAP

    # For grace notes: relative timing offset from primary (negative = before)
    grace_offset: float | None = Field(
        default=None, description="Timing offset in beats for grace notes (e.g., -0.05)"
    )

    # For diddles: which stroke in the diddle (1 or 2)
    diddle_position: Literal[1, 2] | None = Field(
        default=None, description="Position within a diddle (1=first, 2=second)"
    )

    def is_accented(self) -> bool:
        """Returns True if this is an accented stroke."""
        return self.stroke_type == StrokeType.ACCENT

    def is_grace_note(self) -> bool:
        """Returns True if this is a grace note."""
        return self.stroke_type == StrokeType.GRACE


class StickingPattern(BaseModel):
    """
    A complete sticking pattern for a rudiment.

    Patterns are defined per beat grouping (e.g., one paradiddle cycle).
    The pattern repeats for the duration of the exercise.
    """

    strokes: list[Stroke]
    beats_per_cycle: float = Field(
        default=1.0, description="How many beats one full cycle of the pattern takes"
    )

    def __len__(self) -> int:
        return len(self.strokes)

    def stroke_count(self) -> int:
        """Total number of strokes in one cycle."""
        return len(self.strokes)

    def accent_positions(self) -> list[int]:
        """Indices of accented strokes."""
        return [i for i, s in enumerate(self.strokes) if s.is_accented()]

    def grace_note_positions(self) -> list[int]:
        """Indices of grace notes."""
        return [i for i, s in enumerate(self.strokes) if s.is_grace_note()]


class RudimentCategory(str, Enum):
    """PAS rudiment categories."""

    ROLL = "roll"
    DIDDLE = "diddle"
    FLAM = "flam"
    DRAG = "drag"


class RudimentParams(BaseModel):
    """
    Rudiment-specific parameters that affect generation.

    Different rudiment types have different articulation requirements
    that affect how we generate timing and velocity.
    """

    # Flam parameters
    flam_spacing_range: tuple[float, float] | None = Field(
        default=None, description="Min/max grace note spacing in ms (e.g., (15, 40))"
    )

    # Diddle/double stroke parameters
    diddle_ratio_range: tuple[float, float] | None = Field(
        default=None, description="Min/max ratio between first and second diddle stroke duration"
    )

    # Roll parameters
    roll_type: Literal["open", "closed", "buzz"] | None = Field(
        default=None, description="Type of roll for roll rudiments"
    )
    roll_strokes_per_beat: int | None = Field(
        default=None, description="Number of strokes per beat for rolls"
    )

    # Buzz roll parameters
    buzz_strokes_range: tuple[int, int] | None = Field(
        default=None, description="Min/max bounce strokes per primary stroke"
    )
    buzz_detail: Literal["sub_strokes", "marking"] | None = Field(
        default=None,
        description="Buzz generation mode: sub_strokes expands in MIDI, marking only tags stroke type",
    )

    # Drag parameters (drags have 2 grace notes)
    drag_spacing_range: tuple[float, float] | None = Field(
        default=None, description="Min/max spacing between drag grace notes in ms"
    )


class Subdivision(str, Enum):
    """Base subdivision for the rudiment."""

    QUARTER = "quarter"
    EIGHTH = "eighth"
    TRIPLET = "triplet"
    SIXTEENTH = "sixteenth"
    SEXTUPLET = "sextuplet"
    THIRTYSECOND = "thirtysecond"


class Rudiment(BaseModel):
    """
    Complete definition of a PAS drum rudiment.

    This captures everything needed to generate MIDI for the rudiment
    with appropriate timing, velocity, and articulation.
    """

    name: str = Field(description="Official PAS name")
    slug: str = Field(description="URL-safe identifier (e.g., 'single_paradiddle')")
    category: RudimentCategory

    # The sticking pattern
    pattern: StickingPattern

    # Timing
    subdivision: Subdivision = Field(
        default=Subdivision.SIXTEENTH, description="Base note subdivision"
    )
    tempo_range: tuple[int, int] = Field(
        default=(60, 180), description="Recommended tempo range (BPM)"
    )

    # Rudiment-specific parameters
    params: RudimentParams = Field(default_factory=RudimentParams)

    # Metadata
    pas_number: int | None = Field(default=None, description="Official PAS rudiment number (1-40)")
    description: str | None = Field(default=None, description="Brief description of the rudiment")

    # Variations
    starts_on_left: bool = Field(
        default=False, description="If True, the 'alternate' version starts on left hand"
    )

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        """Ensure slug is URL-safe."""
        import re

        if not re.match(r"^[a-z0-9_]+$", v):
            raise ValueError("Slug must be lowercase alphanumeric with underscores only")
        return v

    def strokes_per_beat(self) -> float:
        """Calculate how many strokes occur per beat."""
        return self.pattern.stroke_count() / self.pattern.beats_per_cycle

    def duration_at_tempo(self, tempo_bpm: int, num_cycles: int = 1) -> float:
        """Calculate duration in seconds for given number of cycles at tempo."""
        beats = self.pattern.beats_per_cycle * num_cycles
        return beats * (60.0 / tempo_bpm)

    def get_alternating_pattern(self) -> StickingPattern:
        """
        Return the pattern with hands swapped (for practicing alternating starts).
        """
        swapped_strokes = []
        for stroke in self.pattern.strokes:
            new_hand = Hand.LEFT if stroke.hand == Hand.RIGHT else Hand.RIGHT
            swapped_strokes.append(stroke.model_copy(update={"hand": new_hand}))

        return StickingPattern(
            strokes=swapped_strokes, beats_per_cycle=self.pattern.beats_per_cycle
        )


# Convenience functions for building patterns


def parse_sticking_string(sticking: str, accents: str | None = None) -> StickingPattern:
    """
    Parse a simple sticking string into a StickingPattern.

    Args:
        sticking: String of R and L characters (e.g., "RLRR LRLL")
        accents: Optional string marking accent positions with > (e.g., ">... >...")

    Returns:
        StickingPattern with the parsed strokes

    Example:
        >>> parse_sticking_string("RLRR LRLL", ">... >...")
        # Creates single paradiddle with accents on first stroke of each group
    """
    # Remove spaces for alignment
    sticking = sticking.replace(" ", "")
    if accents:
        accents = accents.replace(" ", "")
        if len(accents) != len(sticking):
            raise ValueError(
                f"Accent string length ({len(accents)}) must match "
                f"sticking length ({len(sticking)})"
            )

    strokes = []
    for i, char in enumerate(sticking.upper()):
        if char not in ("R", "L"):
            raise ValueError(f"Invalid sticking character: {char}")

        hand = Hand.RIGHT if char == "R" else Hand.LEFT
        is_accent = accents and accents[i] == ">"
        stroke_type = StrokeType.ACCENT if is_accent else StrokeType.TAP

        strokes.append(Stroke(hand=hand, stroke_type=stroke_type))

    return StickingPattern(strokes=strokes, beats_per_cycle=len(strokes) / 4)  # Assume 16ths


def create_flam_pattern(primary_hand: Hand, accent: bool = True) -> list[Stroke]:
    """
    Create a two-stroke flam pattern (grace + primary).

    Args:
        primary_hand: Which hand plays the primary (main) stroke
        accent: Whether the primary stroke is accented

    Returns:
        List of two strokes: [grace_note, primary]
    """
    grace_hand = Hand.LEFT if primary_hand == Hand.RIGHT else Hand.RIGHT
    return [
        Stroke(hand=grace_hand, stroke_type=StrokeType.GRACE, grace_offset=-0.05),
        Stroke(hand=primary_hand, stroke_type=StrokeType.ACCENT if accent else StrokeType.TAP),
    ]


def create_diddle(hand: Hand) -> list[Stroke]:
    """
    Create a two-stroke diddle (double stroke).

    Args:
        hand: Which hand plays both strokes

    Returns:
        List of two strokes representing the diddle
    """
    return [
        Stroke(hand=hand, stroke_type=StrokeType.DIDDLE, diddle_position=1),
        Stroke(hand=hand, stroke_type=StrokeType.DIDDLE, diddle_position=2),
    ]


def create_drag(primary_hand: Hand) -> list[Stroke]:
    """
    Create a three-stroke drag pattern (2 grace notes + primary).

    Args:
        primary_hand: Which hand plays the primary stroke

    Returns:
        List of three strokes: [grace1, grace2, primary]
    """
    grace_hand = Hand.LEFT if primary_hand == Hand.RIGHT else Hand.RIGHT
    return [
        Stroke(hand=grace_hand, stroke_type=StrokeType.GRACE, grace_offset=-0.08),
        Stroke(hand=grace_hand, stroke_type=StrokeType.GRACE, grace_offset=-0.04),
        Stroke(hand=primary_hand, stroke_type=StrokeType.ACCENT),
    ]
