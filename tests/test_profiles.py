"""Tests for player profile generation."""

import numpy as np

from dataset_gen.profiles.archetypes import (
    SkillTier,
    generate_profile,
    generate_profiles_batch,
    ARCHETYPE_PARAMS,
)
from dataset_gen.profiles.sampler import (
    ProfileSampler,
    CorrelationStructure,
    DIMENSION_ORDER,
)


class TestProfileGeneration:
    """Tests for independent profile generation."""

    def test_generate_profile_beginner(self):
        """Test generating a beginner profile."""
        profile = generate_profile(SkillTier.BEGINNER, rng=np.random.default_rng(42))

        assert profile.skill_tier == SkillTier.BEGINNER
        assert profile.id is not None

        # Beginners should have higher timing errors (worse)
        dims = profile.dimensions
        assert dims.timing.timing_accuracy > 10  # Should be relatively high

    def test_generate_profile_professional(self):
        """Test generating a professional profile."""
        profile = generate_profile(SkillTier.PROFESSIONAL, rng=np.random.default_rng(42))

        assert profile.skill_tier == SkillTier.PROFESSIONAL

        # Professionals should have lower timing errors (better)
        dims = profile.dimensions
        assert dims.timing.timing_accuracy < 15

        # Should have good hand balance
        assert dims.hand_balance.lr_velocity_ratio > 0.9

    def test_profile_dimensions_within_bounds(self):
        """Test that all dimensions are within valid bounds."""
        for tier in SkillTier:
            for _ in range(10):
                profile = generate_profile(tier)
                dims = profile.dimensions

                # Timing bounds
                assert 0 <= dims.timing.timing_accuracy <= 100
                assert 0 <= dims.timing.timing_consistency <= 1
                assert -0.5 <= dims.timing.tempo_drift <= 0.5

                # Velocity bounds
                assert 30 <= dims.dynamics.velocity_mean <= 127
                assert 0 <= dims.dynamics.velocity_variance <= 0.5

                # Hand balance bounds
                assert 0.5 <= dims.hand_balance.lr_velocity_ratio <= 1.0

    def test_generate_batch(self):
        """Test batch profile generation."""
        profiles = generate_profiles_batch(100, seed=42)
        assert len(profiles) == 100

        # Check that we have a mix of skill levels (uniform by default)
        tiers = [p.skill_tier for p in profiles]
        for tier in SkillTier:
            assert tier in tiers

    def test_generate_batch_with_distribution(self):
        """Test batch generation with custom skill distribution."""
        distribution = {
            SkillTier.BEGINNER: 0.5,
            SkillTier.INTERMEDIATE: 0.3,
            SkillTier.ADVANCED: 0.15,
            SkillTier.PROFESSIONAL: 0.05,
        }
        profiles = generate_profiles_batch(1000, skill_distribution=distribution, seed=42)

        # Check approximate distribution
        tiers = [p.skill_tier for p in profiles]
        beginner_ratio = tiers.count(SkillTier.BEGINNER) / len(profiles)
        assert 0.4 < beginner_ratio < 0.6  # Should be around 50%

    def test_profile_tempo_penalty(self):
        """Test tempo penalty calculation."""
        profile = generate_profile(SkillTier.BEGINNER, rng=np.random.default_rng(42))

        # Within comfort zone: no penalty
        in_range_penalty = profile.get_tempo_penalty(90)
        assert abs(in_range_penalty - 1.0) < 0.1

        # Outside comfort zone: should have penalty
        out_of_range_penalty = profile.get_tempo_penalty(200)
        assert out_of_range_penalty > 1.0


class TestCorrelatedSampling:
    """Tests for correlated profile sampling."""

    def test_correlation_matrix_positive_definite(self):
        """Test that default correlation matrix is valid."""
        corr = CorrelationStructure.get_default_correlation_matrix()

        # Check it's symmetric
        assert np.allclose(corr, corr.T)

        # Check diagonal is 1
        assert np.allclose(np.diag(corr), 1.0)

        # Check positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvals(corr)
        assert all(v >= -1e-10 for v in eigenvalues)

    def test_sampler_generates_valid_profiles(self):
        """Test that correlated sampler produces valid profiles."""
        sampler = ProfileSampler(seed=42)

        for tier in SkillTier:
            for _ in range(10):
                profile = sampler.sample(tier)
                dims = profile.dimensions

                # Check bounds (same as independent test)
                assert 0 <= dims.timing.timing_accuracy <= 100
                assert 0 <= dims.timing.timing_consistency <= 1
                assert 0.5 <= dims.hand_balance.lr_velocity_ratio <= 1.0

    def test_sampler_produces_correlated_dimensions(self):
        """Test that sampled dimensions show expected correlations."""
        sampler = ProfileSampler(seed=42)

        # Generate many profiles
        profiles = sampler.sample_batch(500, skill_distribution={SkillTier.INTERMEDIATE: 1.0})

        # Extract correlated dimensions
        timing_accuracy = [p.dimensions.timing.timing_accuracy for p in profiles]
        timing_consistency = [p.dimensions.timing.timing_consistency for p in profiles]

        # These should be positively correlated (poor accuracy -> poor consistency)
        correlation = np.corrcoef(timing_accuracy, timing_consistency)[0, 1]
        assert correlation > 0.3  # Should show positive correlation

    def test_dimension_order_matches_archetype_params(self):
        """Test that dimension order is consistent with archetype params."""
        # Verify we have the right number of dimensions
        assert len(DIMENSION_ORDER) == 17

        # Verify all dimensions are in archetype params
        params = ARCHETYPE_PARAMS[SkillTier.BEGINNER]
        for dim in DIMENSION_ORDER[:4]:
            assert dim in params["timing"]
        for dim in DIMENSION_ORDER[4:8]:
            assert dim in params["dynamics"]
        for dim in DIMENSION_ORDER[8:11]:
            assert dim in params["hand_balance"]
        for dim in DIMENSION_ORDER[11:]:
            assert dim in params["rudiment_specific"]


class TestArchetypeParams:
    """Tests for archetype parameter definitions."""

    def test_skill_tiers_have_decreasing_errors(self):
        """Test that better skill tiers have lower error values."""
        tiers = [
            SkillTier.BEGINNER,
            SkillTier.INTERMEDIATE,
            SkillTier.ADVANCED,
            SkillTier.PROFESSIONAL,
        ]

        timing_accuracies = [ARCHETYPE_PARAMS[t]["timing"]["timing_accuracy"][0] for t in tiers]

        # Should decrease with skill level
        for i in range(len(timing_accuracies) - 1):
            assert timing_accuracies[i] > timing_accuracies[i + 1]

    def test_skill_tiers_have_increasing_hand_balance(self):
        """Test that better skill tiers have better hand balance."""
        tiers = [
            SkillTier.BEGINNER,
            SkillTier.INTERMEDIATE,
            SkillTier.ADVANCED,
            SkillTier.PROFESSIONAL,
        ]

        ratios = [ARCHETYPE_PARAMS[t]["hand_balance"]["lr_velocity_ratio"][0] for t in tiers]

        # Should increase with skill level (closer to 1.0)
        for i in range(len(ratios) - 1):
            assert ratios[i] < ratios[i + 1]
