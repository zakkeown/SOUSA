"""
Dataset split generation by player profile.

Splits are done by profile to ensure the model generalizes to
unseen players rather than memorizing specific profile characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

import numpy as np

from dataset_gen.profiles.archetypes import PlayerProfile, SkillTier


class DatasetSplit(str, Enum):
    """Dataset split types."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Stratify by skill tier to maintain distribution
    stratify_by_skill: bool = True

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class SplitAssignment:
    """Assignment of profiles to splits."""

    train_profile_ids: list[str] = field(default_factory=list)
    val_profile_ids: list[str] = field(default_factory=list)
    test_profile_ids: list[str] = field(default_factory=list)

    # Mapping from profile_id to split
    profile_to_split: dict[str, DatasetSplit] = field(default_factory=dict)

    # Statistics
    stats: dict = field(default_factory=dict)

    def get_split(self, profile_id: str) -> DatasetSplit:
        """Get the split for a given profile."""
        return self.profile_to_split.get(profile_id, DatasetSplit.TRAIN)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "train_profile_ids": self.train_profile_ids,
            "val_profile_ids": self.val_profile_ids,
            "test_profile_ids": self.test_profile_ids,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SplitAssignment":
        """Create from dictionary."""
        assignment = cls(
            train_profile_ids=data["train_profile_ids"],
            val_profile_ids=data["val_profile_ids"],
            test_profile_ids=data["test_profile_ids"],
            stats=data.get("stats", {}),
        )

        # Rebuild profile_to_split mapping
        for pid in assignment.train_profile_ids:
            assignment.profile_to_split[pid] = DatasetSplit.TRAIN
        for pid in assignment.val_profile_ids:
            assignment.profile_to_split[pid] = DatasetSplit.VALIDATION
        for pid in assignment.test_profile_ids:
            assignment.profile_to_split[pid] = DatasetSplit.TEST

        return assignment


class SplitGenerator:
    """
    Generate train/val/test splits by player profile.

    Ensures that all samples from a single profile end up in the same split,
    preventing data leakage and testing generalization to new players.
    """

    def __init__(self, config: SplitConfig | None = None):
        """
        Initialize split generator.

        Args:
            config: Split configuration
        """
        self.config = config or SplitConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate_splits(
        self,
        profiles: list[PlayerProfile],
    ) -> SplitAssignment:
        """
        Assign profiles to train/val/test splits.

        Args:
            profiles: List of player profiles to split

        Returns:
            SplitAssignment with profile assignments
        """
        if self.config.stratify_by_skill:
            return self._stratified_split(profiles)
        else:
            return self._random_split(profiles)

    def _random_split(self, profiles: list[PlayerProfile]) -> SplitAssignment:
        """Simple random split without stratification."""
        profile_ids = [p.id for p in profiles]
        self.rng.shuffle(profile_ids)

        n = len(profile_ids)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        train_ids = profile_ids[:n_train]
        val_ids = profile_ids[n_train : n_train + n_val]
        test_ids = profile_ids[n_train + n_val :]

        return self._create_assignment(train_ids, val_ids, test_ids, profiles)

    def _stratified_split(self, profiles: list[PlayerProfile]) -> SplitAssignment:
        """Split with stratification by skill tier."""
        # Group profiles by skill tier
        tier_groups: dict[SkillTier, list[str]] = {tier: [] for tier in SkillTier}
        for profile in profiles:
            tier_groups[profile.skill_tier].append(profile.id)

        train_ids = []
        val_ids = []
        test_ids = []

        # Split each tier proportionally
        for tier, ids in tier_groups.items():
            if not ids:
                continue

            # Shuffle within tier
            ids = list(ids)
            self.rng.shuffle(ids)

            n = len(ids)
            n_train = max(1, int(n * self.config.train_ratio))
            n_val = max(0, int(n * self.config.val_ratio))

            # Ensure at least 1 in test if we have enough profiles
            if n >= 3:
                n_test = n - n_train - n_val
                if n_test == 0 and n_val > 1:
                    n_val -= 1
                    n_test = 1

            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train : n_train + n_val])
            test_ids.extend(ids[n_train + n_val :])

        return self._create_assignment(train_ids, val_ids, test_ids, profiles)

    def _create_assignment(
        self,
        train_ids: list[str],
        val_ids: list[str],
        test_ids: list[str],
        profiles: list[PlayerProfile],
    ) -> SplitAssignment:
        """Create SplitAssignment with statistics."""
        assignment = SplitAssignment(
            train_profile_ids=train_ids,
            val_profile_ids=val_ids,
            test_profile_ids=test_ids,
        )

        # Build mapping
        for pid in train_ids:
            assignment.profile_to_split[pid] = DatasetSplit.TRAIN
        for pid in val_ids:
            assignment.profile_to_split[pid] = DatasetSplit.VALIDATION
        for pid in test_ids:
            assignment.profile_to_split[pid] = DatasetSplit.TEST

        # Compute statistics
        profile_map = {p.id: p for p in profiles}
        assignment.stats = self._compute_stats(assignment, profile_map)

        return assignment

    def _compute_stats(
        self,
        assignment: SplitAssignment,
        profile_map: dict[str, PlayerProfile],
    ) -> dict:
        """Compute statistics for the split."""
        stats = {
            "total_profiles": len(profile_map),
            "train_profiles": len(assignment.train_profile_ids),
            "val_profiles": len(assignment.val_profile_ids),
            "test_profiles": len(assignment.test_profile_ids),
        }

        # Skill tier distribution per split
        for split_name, ids in [
            ("train", assignment.train_profile_ids),
            ("val", assignment.val_profile_ids),
            ("test", assignment.test_profile_ids),
        ]:
            tier_counts = {tier.value: 0 for tier in SkillTier}
            for pid in ids:
                if pid in profile_map:
                    tier_counts[profile_map[pid].skill_tier.value] += 1
            stats[f"{split_name}_skill_distribution"] = tier_counts

        return stats

    def save_splits(
        self,
        assignment: SplitAssignment,
        output_path: Path | str,
    ) -> None:
        """
        Save split assignment to JSON file.

        Args:
            assignment: Split assignment to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(assignment.to_dict(), f, indent=2)

    def load_splits(self, input_path: Path | str) -> SplitAssignment:
        """
        Load split assignment from JSON file.

        Args:
            input_path: Input file path

        Returns:
            Loaded SplitAssignment
        """
        with open(input_path) as f:
            data = json.load(f)
        return SplitAssignment.from_dict(data)


class SampleSplitter:
    """
    Assign samples to splits based on their profile.

    Works with generated samples to determine which split they belong to.
    """

    def __init__(self, assignment: SplitAssignment):
        """
        Initialize sample splitter.

        Args:
            assignment: Profile split assignment
        """
        self.assignment = assignment

    def get_sample_split(self, profile_id: str) -> DatasetSplit:
        """
        Get the split for a sample based on its profile.

        Args:
            profile_id: The profile ID of the sample

        Returns:
            The dataset split
        """
        return self.assignment.get_split(profile_id)

    def split_samples(
        self,
        samples: list[dict],
        profile_id_key: str = "profile_id",
    ) -> dict[DatasetSplit, list[dict]]:
        """
        Split a list of samples by their profile.

        Args:
            samples: List of sample dictionaries
            profile_id_key: Key for profile ID in sample dict

        Returns:
            Dictionary mapping splits to sample lists
        """
        result = {
            DatasetSplit.TRAIN: [],
            DatasetSplit.VALIDATION: [],
            DatasetSplit.TEST: [],
        }

        for sample in samples:
            profile_id = sample.get(profile_id_key)
            split = self.get_sample_split(profile_id)
            result[split].append(sample)

        return result


def generate_splits(
    profiles: list[PlayerProfile],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> SplitAssignment:
    """
    Convenience function to generate dataset splits.

    Args:
        profiles: List of player profiles
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        stratify: Whether to stratify by skill tier
        seed: Random seed

    Returns:
        SplitAssignment object
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by_skill=stratify,
        seed=seed,
    )

    generator = SplitGenerator(config)
    return generator.generate_splits(profiles)
