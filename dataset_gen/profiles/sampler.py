"""
Correlated profile sampling using multivariate distributions.

This module enables realistic correlations between execution dimensions,
such as beginners being weak across multiple areas simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from dataset_gen.profiles.archetypes import (
    PlayerProfile,
    SkillTier,
    ExecutionDimensions,
    TimingDimensions,
    DynamicsDimensions,
    HandBalanceDimensions,
    RudimentSpecificDimensions,
    ARCHETYPE_PARAMS,
)

# Dimension names in order (for correlation matrix indexing)
DIMENSION_ORDER = [
    # Timing (0-3)
    "timing_accuracy",
    "timing_consistency",
    "tempo_drift",
    "subdivision_evenness",
    # Dynamics (4-7)
    "velocity_mean",
    "velocity_variance",
    "accent_differentiation",
    "accent_accuracy",
    # Hand balance (8-10)
    "lr_velocity_ratio",
    "lr_timing_bias",
    "lr_consistency_delta",
    # Rudiment-specific (11-16)
    "flam_spacing",
    "flam_spacing_variance",
    "diddle_evenness",
    "diddle_variance",
    "roll_sustain",
    "buzz_density_consistency",
]


@dataclass
class CorrelationStructure:
    """
    Defines correlation patterns between execution dimensions.

    The correlation matrix captures how dimensions co-vary. For example:
    - High timing_accuracy (poor) correlates with high timing_consistency (inconsistent)
    - Good subdivision_evenness correlates with good diddle_evenness

    Sign convention: Positive correlation means dimensions move together.
    Note that some dimensions are "good when low" (accuracy error) and
    others are "good when high" (accent_accuracy), so interpret carefully.
    """

    # Correlation coefficient presets for different relationships
    # Note: These are transformed to match actual value directions
    STRONG_POSITIVE = 0.7
    MODERATE_POSITIVE = 0.4
    WEAK_POSITIVE = 0.2
    WEAK_NEGATIVE = -0.2
    MODERATE_NEGATIVE = -0.4
    STRONG_NEGATIVE = -0.7

    @classmethod
    def get_default_correlation_matrix(cls) -> np.ndarray:
        """
        Build the default correlation matrix based on pedagogical knowledge.

        This encodes how execution dimensions typically co-vary in real drummers.
        """
        n = len(DIMENSION_ORDER)
        corr = np.eye(n)

        def set_corr(dim1: str, dim2: str, value: float):
            """Helper to set symmetric correlation."""
            i = DIMENSION_ORDER.index(dim1)
            j = DIMENSION_ORDER.index(dim2)
            corr[i, j] = value
            corr[j, i] = value

        # WITHIN-CATEGORY CORRELATIONS

        # Timing correlations
        # Poor timing accuracy → poor consistency
        set_corr("timing_accuracy", "timing_consistency", cls.STRONG_POSITIVE)
        # Poor timing → uneven subdivisions
        set_corr("timing_accuracy", "subdivision_evenness", cls.MODERATE_NEGATIVE)
        set_corr("timing_consistency", "subdivision_evenness", cls.MODERATE_NEGATIVE)
        # Rushing (tempo drift) correlates with timing issues
        set_corr("tempo_drift", "timing_accuracy", cls.WEAK_POSITIVE)

        # Dynamics correlations
        # Good accent accuracy → good differentiation
        set_corr("accent_accuracy", "accent_differentiation", cls.MODERATE_POSITIVE)
        # High velocity variance (bad) → lower accent accuracy
        set_corr("velocity_variance", "accent_accuracy", cls.MODERATE_NEGATIVE)

        # Hand balance correlations
        # Weak hand velocity → weak hand timing
        set_corr(
            "lr_velocity_ratio", "lr_timing_bias", cls.MODERATE_NEGATIVE
        )  # Low ratio, high bias
        # Weak hand overall → less consistent
        set_corr("lr_velocity_ratio", "lr_consistency_delta", cls.MODERATE_NEGATIVE)

        # Rudiment-specific internal correlations
        # Flam spacing variance correlates with diddle variance
        set_corr("flam_spacing_variance", "diddle_variance", cls.MODERATE_POSITIVE)
        # Good diddle evenness → good buzz consistency
        set_corr("diddle_evenness", "buzz_density_consistency", cls.MODERATE_POSITIVE)

        # CROSS-CATEGORY CORRELATIONS

        # Timing → Dynamics
        # Poor timing → poor accent placement
        set_corr("timing_accuracy", "accent_accuracy", cls.MODERATE_NEGATIVE)
        set_corr("timing_consistency", "accent_accuracy", cls.WEAK_NEGATIVE)

        # Timing → Rudiment-specific
        # Poor timing → poor diddle evenness
        set_corr("timing_accuracy", "diddle_evenness", cls.MODERATE_NEGATIVE)
        set_corr("timing_consistency", "diddle_evenness", cls.MODERATE_NEGATIVE)
        # Poor subdivision evenness → poor diddle evenness
        set_corr("subdivision_evenness", "diddle_evenness", cls.STRONG_POSITIVE)

        # Hand balance → Dynamics
        # Poor hand balance → higher velocity variance
        set_corr("lr_velocity_ratio", "velocity_variance", cls.WEAK_NEGATIVE)

        # Hand balance → Timing
        # Poor hand balance → worse timing (from weak hand)
        set_corr("lr_velocity_ratio", "timing_accuracy", cls.WEAK_NEGATIVE)
        set_corr("lr_consistency_delta", "timing_consistency", cls.MODERATE_POSITIVE)

        # Hand balance → Rudiment-specific
        # Poor hand balance → worse flams/diddles
        set_corr("lr_velocity_ratio", "flam_spacing", cls.WEAK_POSITIVE)  # Wide flams
        set_corr("lr_velocity_ratio", "diddle_evenness", cls.WEAK_POSITIVE)

        # Dynamics → Rudiment-specific
        # Poor dynamics control → worse roll sustain
        set_corr("velocity_variance", "roll_sustain", cls.MODERATE_POSITIVE)

        # Ensure matrix is positive semi-definite (required for multivariate normal)
        # Find nearest positive semi-definite matrix
        corr = cls._nearest_positive_definite(corr)

        return corr

    @staticmethod
    def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive-definite matrix to A.

        Uses the Higham (2002) algorithm.
        """
        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)
        H = V.T @ np.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if cls_is_positive_definite(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not cls_is_positive_definite(A3):
            min_eig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-min_eig * k**2 + spacing)
            k += 1

        return A3


def cls_is_positive_definite(A: np.ndarray) -> bool:
    """Check if matrix is positive definite."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


class ProfileSampler:
    """
    Sample player profiles with correlated execution dimensions.

    Uses multivariate normal distribution with correlation structure
    to generate profiles where dimensions co-vary realistically.
    """

    def __init__(
        self,
        correlation_matrix: np.ndarray | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the sampler.

        Args:
            correlation_matrix: Custom correlation matrix. If None, uses default.
            seed: Random seed for reproducibility.
        """
        if correlation_matrix is None:
            self.correlation = CorrelationStructure.get_default_correlation_matrix()
        else:
            self.correlation = correlation_matrix

        self.rng = np.random.default_rng(seed)

    def _get_means_and_stds(self, skill_tier: SkillTier) -> tuple[np.ndarray, np.ndarray]:
        """Get means and standard deviations for all dimensions given skill tier."""
        params = ARCHETYPE_PARAMS[skill_tier]

        means = []
        stds = []

        # Timing
        for dim in ["timing_accuracy", "timing_consistency", "tempo_drift", "subdivision_evenness"]:
            mean, std = params["timing"][dim]
            means.append(mean)
            stds.append(std)

        # Dynamics
        for dim in [
            "velocity_mean",
            "velocity_variance",
            "accent_differentiation",
            "accent_accuracy",
        ]:
            mean, std = params["dynamics"][dim]
            means.append(mean)
            stds.append(std)

        # Hand balance
        for dim in ["lr_velocity_ratio", "lr_timing_bias", "lr_consistency_delta"]:
            mean, std = params["hand_balance"][dim]
            means.append(mean)
            stds.append(std)

        # Rudiment-specific
        for dim in [
            "flam_spacing",
            "flam_spacing_variance",
            "diddle_evenness",
            "diddle_variance",
            "roll_sustain",
            "buzz_density_consistency",
        ]:
            mean, std = params["rudiment_specific"][dim]
            means.append(mean)
            stds.append(std)

        return np.array(means), np.array(stds)

    def _build_covariance_matrix(self, stds: np.ndarray) -> np.ndarray:
        """Build covariance matrix from correlation and standard deviations."""
        # Cov[i,j] = Corr[i,j] * std[i] * std[j]
        outer_stds = np.outer(stds, stds)
        return self.correlation * outer_stds

    def sample(self, skill_tier: SkillTier, profile_id: str | None = None) -> PlayerProfile:
        """
        Sample a single profile with correlated dimensions.

        Args:
            skill_tier: The skill level to sample from
            profile_id: Optional ID for the profile

        Returns:
            PlayerProfile with correlated execution dimensions
        """
        means, stds = self._get_means_and_stds(skill_tier)
        cov = self._build_covariance_matrix(stds)

        # Sample from multivariate normal
        samples = self.rng.multivariate_normal(means, cov)

        # Clamp to valid ranges
        samples = self._clamp_samples(samples)

        # Build profile from samples
        return self._samples_to_profile(samples, skill_tier, profile_id)

    def sample_batch(
        self,
        n_profiles: int,
        skill_distribution: dict[SkillTier, float] | None = None,
    ) -> list[PlayerProfile]:
        """
        Sample multiple profiles with specified skill distribution.

        Args:
            n_profiles: Number of profiles to generate
            skill_distribution: Proportion per skill tier (defaults to uniform)

        Returns:
            List of PlayerProfile objects
        """
        if skill_distribution is None:
            skill_distribution = {tier: 0.25 for tier in SkillTier}

        # Normalize
        total = sum(skill_distribution.values())
        skill_distribution = {k: v / total for k, v in skill_distribution.items()}

        # Sample tier assignments using indices to avoid numpy string conversion
        tiers = list(skill_distribution.keys())
        probs = [skill_distribution[t] for t in tiers]
        tier_indices = self.rng.choice(len(tiers), size=n_profiles, p=probs)

        return [self.sample(tiers[idx]) for idx in tier_indices]

    def _clamp_samples(self, samples: np.ndarray) -> np.ndarray:
        """Clamp samples to valid ranges."""
        # Define (min, max) for each dimension
        bounds = [
            (0, 100),  # timing_accuracy
            (0, 1),  # timing_consistency
            (-0.5, 0.5),  # tempo_drift
            (0, 1),  # subdivision_evenness
            (30, 127),  # velocity_mean
            (0, 0.5),  # velocity_variance
            (0, 20),  # accent_differentiation
            (0, 1),  # accent_accuracy
            (0.5, 1.0),  # lr_velocity_ratio
            (-20, 20),  # lr_timing_bias
            (0, 0.3),  # lr_consistency_delta
            (10, 80),  # flam_spacing
            (0, 0.5),  # flam_spacing_variance
            (0.6, 1.4),  # diddle_evenness
            (0, 0.3),  # diddle_variance
            (0, 0.5),  # roll_sustain
            (0, 1),  # buzz_density_consistency
        ]

        clamped = np.copy(samples)
        for i, (low, high) in enumerate(bounds):
            clamped[i] = np.clip(samples[i], low, high)

        return clamped

    def _samples_to_profile(
        self,
        samples: np.ndarray,
        skill_tier: SkillTier,
        profile_id: str | None,
    ) -> PlayerProfile:
        """Convert sample array to PlayerProfile."""
        from uuid import uuid4

        params = ARCHETYPE_PARAMS[skill_tier]

        timing = TimingDimensions(
            timing_accuracy=samples[0],
            timing_consistency=samples[1],
            tempo_drift=samples[2],
            subdivision_evenness=samples[3],
        )

        dynamics = DynamicsDimensions(
            velocity_mean=samples[4],
            velocity_variance=samples[5],
            accent_differentiation=samples[6],
            accent_accuracy=samples[7],
        )

        hand_balance = HandBalanceDimensions(
            lr_velocity_ratio=samples[8],
            lr_timing_bias=samples[9],
            lr_consistency_delta=samples[10],
        )

        rudiment_specific = RudimentSpecificDimensions(
            flam_spacing=samples[11],
            flam_spacing_variance=samples[12],
            diddle_evenness=samples[13],
            diddle_variance=samples[14],
            roll_sustain=samples[15],
            buzz_density_consistency=samples[16],
        )

        # Sample meta parameters (independent)
        fatigue_mean, fatigue_std = params["meta"]["fatigue_coefficient"]
        fatigue = np.clip(self.rng.normal(fatigue_mean, fatigue_std), 0, 0.3)
        tempo_range = params["meta"]["tempo_comfort_range"][0]
        dominant_hand = "left" if self.rng.random() < 0.1 else "right"

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


def visualize_correlation_matrix():
    """Print correlation matrix for inspection (useful for debugging)."""
    corr = CorrelationStructure.get_default_correlation_matrix()

    print("Correlation Matrix:")
    print("-" * 80)

    # Header
    header = "".ljust(25) + "".join(f"{d[:8]:>9}" for d in DIMENSION_ORDER)
    print(header)

    # Rows
    for i, dim in enumerate(DIMENSION_ORDER):
        row = f"{dim[:24]:24} "
        for j in range(len(DIMENSION_ORDER)):
            val = corr[i, j]
            if abs(val) < 0.1:
                row += "    .    "
            else:
                row += f"{val:8.2f} "
        print(row)


if __name__ == "__main__":
    visualize_correlation_matrix()
