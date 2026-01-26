"""Player profile generation with correlated skill dimensions."""

from dataset_gen.profiles.archetypes import (
    PlayerProfile,
    SkillTier,
    ExecutionDimensions,
    generate_profile,
    generate_profiles_batch,
)
from dataset_gen.profiles.sampler import (
    ProfileSampler,
    CorrelationStructure,
)

__all__ = [
    "PlayerProfile",
    "SkillTier",
    "ExecutionDimensions",
    "generate_profile",
    "generate_profiles_batch",
    "ProfileSampler",
    "CorrelationStructure",
]
