"""
Player profile definitions and archetype generators.

Defines the execution dimensions that characterize drum performance quality
and provides archetypes for different skill levels with realistic correlations.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4
import numpy as np
from pydantic import BaseModel, Field


class SkillTier(str, Enum):
    """Broad skill categories for player profiles."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"


class TimingDimensions(BaseModel):
    """Timing-related execution dimensions."""

    timing_accuracy: float = Field(
        description="Mean deviation from grid in milliseconds (lower is better)",
        ge=0,
        le=100,
    )
    timing_consistency: float = Field(
        description="Variance in timing errors, 0-1 where 0 is perfectly consistent",
        ge=0,
        le=1,
    )
    tempo_drift: float = Field(
        description="Slope of tempo over exercise (positive=rushing, negative=dragging)",
        ge=-0.5,
        le=0.5,
    )
    subdivision_evenness: float = Field(
        description="Ratio consistency between subdivisions, 0-1 where 1 is perfectly even",
        ge=0,
        le=1,
    )


class DynamicsDimensions(BaseModel):
    """Dynamics-related execution dimensions."""

    velocity_mean: float = Field(
        description="Average MIDI velocity (0-127)",
        ge=30,
        le=127,
    )
    velocity_variance: float = Field(
        description="Variance in base stroke velocity, 0-1 where 0 is robotic consistency",
        ge=0,
        le=0.5,
    )
    accent_differentiation: float = Field(
        description="dB gap between accented and unaccented strokes (higher is clearer)",
        ge=0,
        le=20,
    )
    accent_accuracy: float = Field(
        description="Proportion of accents landing correctly, 0-1",
        ge=0,
        le=1,
    )


class HandBalanceDimensions(BaseModel):
    """Hand balance execution dimensions."""

    lr_velocity_ratio: float = Field(
        description="Left/Right hand velocity ratio (1.0 = balanced, <1 = weak left)",
        ge=0.5,
        le=1.0,
    )
    lr_timing_bias: float = Field(
        description="Timing bias in ms (positive = left rushes, negative = left drags)",
        ge=-20,
        le=20,
    )
    lr_consistency_delta: float = Field(
        description="Difference in consistency between hands (0 = both equally consistent)",
        ge=0,
        le=0.3,
    )


class RudimentSpecificDimensions(BaseModel):
    """Rudiment-specific execution dimensions."""

    flam_spacing: float = Field(
        description="Grace note distance from primary in ms (ideal ~25-35ms)",
        ge=10,
        le=80,
    )
    flam_spacing_variance: float = Field(
        description="Variance in flam spacing, 0-1",
        ge=0,
        le=0.5,
    )
    diddle_evenness: float = Field(
        description="Ratio between first and second diddle stroke, 1.0 = even",
        ge=0.6,
        le=1.4,
    )
    diddle_variance: float = Field(
        description="Variance in diddle evenness across the exercise",
        ge=0,
        le=0.3,
    )
    roll_sustain: float = Field(
        description="Velocity decay coefficient across sustained rolls (0 = no decay)",
        ge=0,
        le=0.5,
    )
    buzz_density_consistency: float = Field(
        description="Consistency of buzz strokes per beat, 0-1",
        ge=0,
        le=1,
    )


class ExecutionDimensions(BaseModel):
    """
    Complete execution dimensions for a player profile.

    These capture all the measurable aspects of how a drummer executes rudiments.
    """

    timing: TimingDimensions
    dynamics: DynamicsDimensions
    hand_balance: HandBalanceDimensions
    rudiment_specific: RudimentSpecificDimensions


class PlayerProfile(BaseModel):
    """
    Complete player profile for generating synthetic performances.

    Each profile represents a unique "virtual drummer" with consistent
    execution characteristics across multiple rudiments.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    skill_tier: SkillTier
    dimensions: ExecutionDimensions

    # Metadata
    dominant_hand: Literal["right", "left"] = "right"
    fatigue_coefficient: float = Field(
        default=0.0,
        description="How much performance degrades over time (0 = no fatigue)",
        ge=0,
        le=0.3,
    )
    tempo_comfort_range: tuple[int, int] = Field(
        default=(80, 140),
        description="BPM range where player performs best",
    )

    def get_tempo_penalty(self, tempo: int) -> float:
        """
        Calculate performance penalty for playing outside comfort range.

        Returns a multiplier (1.0 = no penalty, >1.0 = worse performance).
        """
        low, high = self.tempo_comfort_range
        if low <= tempo <= high:
            return 1.0

        # Linear penalty outside comfort zone
        if tempo < low:
            return 1.0 + (low - tempo) * 0.01
        else:
            return 1.0 + (tempo - high) * 0.01

    def to_dict(self) -> dict:
        """Serialize profile to dictionary for storage/transfer."""
        data = self.model_dump()
        # Convert enum to string for JSON compatibility
        data["skill_tier"] = self.skill_tier.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerProfile":
        """Deserialize profile from dictionary."""
        # Convert string back to enum if needed
        if isinstance(data.get("skill_tier"), str):
            data = data.copy()
            data["skill_tier"] = SkillTier(data["skill_tier"])
        return cls.model_validate(data)


# Archetype definitions with typical parameter ranges
# Format: (mean, std) for Gaussian sampling

ARCHETYPE_PARAMS: dict[SkillTier, dict] = {
    SkillTier.BEGINNER: {
        "timing": {
            "timing_accuracy": (50, 10),  # Literature: 35-80ms for untrained
            "timing_consistency": (0.6, 0.15),  # Highly inconsistent
            "tempo_drift": (0.025, 0.015),  # ~100ms total drift over 8 beats at 120bpm
            "subdivision_evenness": (0.6, 0.15),  # Uneven subdivisions
        },
        "dynamics": {
            "velocity_mean": (85, 15),
            "velocity_variance": (0.12, 0.04),  # Literature: CV 0.18-0.35 (after hand balance)
            "accent_differentiation": (6, 2),  # Weak accent contrast
            "accent_accuracy": (0.7, 0.1),  # Misses some accents
        },
        "hand_balance": {
            "lr_velocity_ratio": (0.72, 0.08),  # Weak non-dominant hand
            "lr_timing_bias": (8, 5),  # Non-dominant hand drags
            "lr_consistency_delta": (0.15, 0.05),  # Non-dominant less consistent
        },
        "rudiment_specific": {
            "flam_spacing": (50, 15),  # Flams too wide
            "flam_spacing_variance": (0.3, 0.1),
            "diddle_evenness": (0.75, 0.15),  # Uneven diddles
            "diddle_variance": (0.2, 0.05),
            "roll_sustain": (0.3, 0.1),  # Velocity drops in rolls
            "buzz_density_consistency": (0.5, 0.15),
        },
        "meta": {
            "fatigue_coefficient": (0.15, 0.05),
            "tempo_comfort_range": ((60, 100), None),  # Narrow comfort zone
        },
    },
    SkillTier.INTERMEDIATE: {
        "timing": {
            "timing_accuracy": (28, 8),  # Literature: 20-45ms for moderate training
            "timing_consistency": (0.35, 0.1),
            "tempo_drift": (0.012, 0.008),  # ~50ms total drift over 8 beats at 120bpm
            "subdivision_evenness": (0.8, 0.1),
        },
        "dynamics": {
            "velocity_mean": (90, 10),
            "velocity_variance": (0.15, 0.04),  # Literature: CV 0.12-0.25 (after hand balance)
            "accent_differentiation": (10, 2),
            "accent_accuracy": (0.88, 0.05),
        },
        "hand_balance": {
            "lr_velocity_ratio": (0.85, 0.05),
            "lr_timing_bias": (4, 3),
            "lr_consistency_delta": (0.08, 0.03),
        },
        "rudiment_specific": {
            "flam_spacing": (35, 8),
            "flam_spacing_variance": (0.15, 0.05),
            "diddle_evenness": (0.9, 0.08),
            "diddle_variance": (0.1, 0.03),
            "roll_sustain": (0.15, 0.05),
            "buzz_density_consistency": (0.75, 0.1),
        },
        "meta": {
            "fatigue_coefficient": (0.08, 0.03),
            "tempo_comfort_range": ((70, 150), None),
        },
    },
    SkillTier.ADVANCED: {
        "timing": {
            "timing_accuracy": (14, 4),  # Literature: 8-25ms for trained musicians
            "timing_consistency": (0.15, 0.05),
            "tempo_drift": (0.006, 0.004),  # ~25ms total drift over 8 beats at 120bpm
            "subdivision_evenness": (0.92, 0.05),
        },
        "dynamics": {
            "velocity_mean": (95, 8),
            "velocity_variance": (0.10, 0.03),  # Literature: CV 0.08-0.18
            "accent_differentiation": (14, 2),
            "accent_accuracy": (0.96, 0.02),
        },
        "hand_balance": {
            "lr_velocity_ratio": (0.94, 0.03),
            "lr_timing_bias": (1.5, 1.5),
            "lr_consistency_delta": (0.03, 0.015),
        },
        "rudiment_specific": {
            "flam_spacing": (28, 4),
            "flam_spacing_variance": (0.08, 0.03),
            "diddle_evenness": (0.97, 0.03),
            "diddle_variance": (0.05, 0.02),
            "roll_sustain": (0.05, 0.03),
            "buzz_density_consistency": (0.9, 0.05),
        },
        "meta": {
            "fatigue_coefficient": (0.03, 0.02),
            "tempo_comfort_range": ((60, 180), None),
        },
    },
    SkillTier.PROFESSIONAL: {
        "timing": {
            "timing_accuracy": (7, 2),  # Literature: 5-15ms for professionals
            "timing_consistency": (0.08, 0.03),
            "tempo_drift": (0.002, 0.002),  # ~8ms total drift over 8 beats at 120bpm
            "subdivision_evenness": (0.97, 0.02),
        },
        "dynamics": {
            "velocity_mean": (100, 5),
            "velocity_variance": (0.07, 0.02),  # Literature: CV 0.05-0.12
            "accent_differentiation": (16, 1.5),
            "accent_accuracy": (0.99, 0.005),
        },
        "hand_balance": {
            "lr_velocity_ratio": (0.98, 0.015),
            "lr_timing_bias": (0.5, 0.5),
            "lr_consistency_delta": (0.01, 0.005),
        },
        "rudiment_specific": {
            "flam_spacing": (25, 2),
            "flam_spacing_variance": (0.04, 0.015),
            "diddle_evenness": (0.99, 0.01),
            "diddle_variance": (0.02, 0.01),
            "roll_sustain": (0.02, 0.01),
            "buzz_density_consistency": (0.96, 0.02),
        },
        "meta": {
            "fatigue_coefficient": (0.01, 0.005),
            "tempo_comfort_range": ((50, 220), None),
        },
    },
}


def _sample_param(spec: tuple[float, float], rng: np.random.Generator) -> float:
    """Sample a parameter from a (mean, std) specification."""
    mean, std = spec
    return rng.normal(mean, std)


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def generate_profile(
    skill_tier: SkillTier,
    rng: np.random.Generator | None = None,
    profile_id: str | None = None,
) -> PlayerProfile:
    """
    Generate a player profile for a given skill tier.

    Args:
        skill_tier: The skill level archetype to base the profile on
        rng: Random number generator for reproducibility
        profile_id: Optional specific ID for the profile

    Returns:
        A PlayerProfile with sampled execution dimensions
    """
    if rng is None:
        rng = np.random.default_rng()

    params = ARCHETYPE_PARAMS[skill_tier]

    # Sample timing dimensions
    timing = TimingDimensions(
        timing_accuracy=_clamp(_sample_param(params["timing"]["timing_accuracy"], rng), 0, 100),
        timing_consistency=_clamp(_sample_param(params["timing"]["timing_consistency"], rng), 0, 1),
        tempo_drift=_clamp(_sample_param(params["timing"]["tempo_drift"], rng), -0.5, 0.5),
        subdivision_evenness=_clamp(
            _sample_param(params["timing"]["subdivision_evenness"], rng), 0, 1
        ),
    )

    # Sample dynamics dimensions
    dynamics = DynamicsDimensions(
        velocity_mean=_clamp(_sample_param(params["dynamics"]["velocity_mean"], rng), 30, 127),
        velocity_variance=_clamp(
            _sample_param(params["dynamics"]["velocity_variance"], rng), 0, 0.5
        ),
        accent_differentiation=_clamp(
            _sample_param(params["dynamics"]["accent_differentiation"], rng), 0, 20
        ),
        accent_accuracy=_clamp(_sample_param(params["dynamics"]["accent_accuracy"], rng), 0, 1),
    )

    # Sample hand balance dimensions
    hand_balance = HandBalanceDimensions(
        lr_velocity_ratio=_clamp(
            _sample_param(params["hand_balance"]["lr_velocity_ratio"], rng), 0.5, 1.0
        ),
        lr_timing_bias=_clamp(
            _sample_param(params["hand_balance"]["lr_timing_bias"], rng), -20, 20
        ),
        lr_consistency_delta=_clamp(
            _sample_param(params["hand_balance"]["lr_consistency_delta"], rng), 0, 0.3
        ),
    )

    # Sample rudiment-specific dimensions
    rudiment_specific = RudimentSpecificDimensions(
        flam_spacing=_clamp(
            _sample_param(params["rudiment_specific"]["flam_spacing"], rng), 10, 80
        ),
        flam_spacing_variance=_clamp(
            _sample_param(params["rudiment_specific"]["flam_spacing_variance"], rng), 0, 0.5
        ),
        diddle_evenness=_clamp(
            _sample_param(params["rudiment_specific"]["diddle_evenness"], rng), 0.6, 1.4
        ),
        diddle_variance=_clamp(
            _sample_param(params["rudiment_specific"]["diddle_variance"], rng), 0, 0.3
        ),
        roll_sustain=_clamp(
            _sample_param(params["rudiment_specific"]["roll_sustain"], rng), 0, 0.5
        ),
        buzz_density_consistency=_clamp(
            _sample_param(params["rudiment_specific"]["buzz_density_consistency"], rng), 0, 1
        ),
    )

    # Sample meta parameters
    fatigue = _clamp(_sample_param(params["meta"]["fatigue_coefficient"], rng), 0, 0.3)
    tempo_range = params["meta"]["tempo_comfort_range"][0]

    # Determine dominant hand (90% right-handed)
    dominant_hand = "left" if rng.random() < 0.1 else "right"

    return PlayerProfile(
        id=profile_id or str(uuid4()),
        skill_tier=skill_tier,
        dimensions=ExecutionDimensions(
            timing=timing,
            dynamics=dynamics,
            hand_balance=hand_balance,
            rudiment_specific=rudiment_specific,
        ),
        dominant_hand=dominant_hand,
        fatigue_coefficient=fatigue,
        tempo_comfort_range=tempo_range,
    )


def generate_profiles_batch(
    n_profiles: int,
    skill_distribution: dict[SkillTier, float] | None = None,
    seed: int | None = None,
) -> list[PlayerProfile]:
    """
    Generate a batch of player profiles with specified skill distribution.

    Args:
        n_profiles: Number of profiles to generate
        skill_distribution: Dict mapping skill tiers to proportions (must sum to 1).
                           Defaults to uniform distribution.
        seed: Random seed for reproducibility

    Returns:
        List of PlayerProfile objects
    """
    rng = np.random.default_rng(seed)

    if skill_distribution is None:
        skill_distribution = {tier: 0.25 for tier in SkillTier}

    # Normalize distribution
    total = sum(skill_distribution.values())
    skill_distribution = {k: v / total for k, v in skill_distribution.items()}

    # Calculate counts per tier
    tiers = list(skill_distribution.keys())
    probs = [skill_distribution[t] for t in tiers]

    # Use indices to avoid numpy string conversion of enum values
    tier_indices = rng.choice(len(tiers), size=n_profiles, p=probs)

    profiles = []
    for idx in tier_indices:
        profiles.append(generate_profile(tiers[idx], rng))

    return profiles
