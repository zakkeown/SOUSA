"""
Pydantic models for hierarchical label schema.

Labels are computed at three levels:
1. Per-stroke: Individual timing and velocity measurements
2. Per-measure: Aggregate statistics within each measure
3. Per-exercise: Overall performance scores
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class StrokeLabel(BaseModel):
    """Labels for a single stroke event."""

    index: int
    hand: Literal["L", "R"]
    stroke_type: str

    # Timing
    intended_time_ms: float
    actual_time_ms: float
    timing_error_ms: float  # For grace notes: deviation from ideal flam spacing

    # Velocity
    intended_velocity: int
    actual_velocity: int
    velocity_error: int

    # Articulation-specific
    is_grace_note: bool = False
    is_accent: bool = False
    diddle_position: int | None = None

    # Grace note specific (None for non-grace strokes)
    flam_spacing_ms: float | None = None  # Actual spacing to primary stroke
    parent_stroke_index: int | None = None  # Index of the primary stroke

    # Buzz roll specific
    buzz_count: int | None = None  # Total strokes in buzz group (primary + subs)


class MeasureLabel(BaseModel):
    """Aggregate labels for a single measure."""

    index: int
    stroke_start: int  # First stroke index in this measure
    stroke_end: int  # Last stroke index (exclusive)

    # Timing statistics
    timing_mean_error_ms: float
    timing_std_ms: float
    timing_max_error_ms: float

    # Velocity statistics
    velocity_mean: float
    velocity_std: float
    velocity_consistency: float  # 1 - (std / mean), higher is more consistent

    # Hand balance for this measure
    lr_velocity_ratio: float | None = None  # None if single-hand measure
    lr_timing_diff_ms: float | None = None


class ExerciseScores(BaseModel):
    """Composite scores for the entire exercise/performance."""

    # Timing scores (0-100, higher is better)
    timing_accuracy: float = Field(ge=0, le=100)
    timing_consistency: float = Field(ge=0, le=100)
    tempo_stability: float = Field(ge=0, le=100)
    subdivision_evenness: float = Field(ge=0, le=100)

    # Dynamics scores
    velocity_control: float = Field(ge=0, le=100)
    accent_differentiation: float = Field(ge=0, le=100)
    accent_accuracy: float = Field(ge=0, le=100)

    # Hand balance scores
    hand_balance: float = Field(ge=0, le=100)
    weak_hand_index: float = Field(
        ge=0, le=100, description="0 = left hand weak, 100 = right hand weak, 50 = balanced"
    )

    # Rudiment-specific scores (if applicable)
    flam_quality: float | None = Field(default=None, ge=0, le=100)
    diddle_quality: float | None = Field(default=None, ge=0, le=100)
    roll_sustain: float | None = Field(default=None, ge=0, le=100)

    # Derived/composite scores
    groove_feel_proxy: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=100)

    # Tier confidence: how central this score is to the assigned tier (0-1)
    # Higher values indicate the sample is clearly within its tier's expected range
    # Lower values indicate the sample is near tier boundaries (label noise)
    tier_confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Confidence that skill_tier label is unambiguous (1=central, 0=boundary)",
    )


class AudioAugmentation(BaseModel):
    """Parameters describing audio augmentation applied to the sample."""

    soundfont: str = "default"
    room_type: str | None = None
    room_wet_dry: float | None = None
    mic_distance: float | None = None
    mic_type: str | None = None
    compression_ratio: float | None = None
    noise_level_db: float | None = None
    bit_depth: int | None = None
    sample_rate: int | None = None


class Sample(BaseModel):
    """
    Complete sample with all hierarchical labels.

    This is the primary data structure for the dataset, containing
    everything needed for ML training.
    """

    # Identifiers
    sample_id: str
    profile_id: str
    rudiment_slug: str

    # Generation parameters
    tempo_bpm: int
    duration_sec: float
    num_cycles: int

    # Profile snapshot (for reference)
    skill_tier: str
    skill_tier_binary: Literal["novice", "skilled"] = "novice"  # 2-class alternative
    dominant_hand: Literal["right", "left"]

    # Hierarchical labels
    strokes: list[StrokeLabel]
    measures: list[MeasureLabel]
    exercise_scores: ExerciseScores

    # Audio augmentation (if applicable)
    audio_augmentation: AudioAugmentation | None = None

    # Augmentation tracking (always present for ML filtering)
    soundfont: str | None = None  # Name of soundfont used for audio synthesis
    augmentation_preset: str | None = None  # "none" for clean, preset name otherwise
    augmentation_group_id: str | None = None  # Links clean sample to its augmented variants

    # File paths (populated during dataset generation)
    midi_path: str | None = None
    audio_path: str | None = None
