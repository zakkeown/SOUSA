"""
Statistical analysis of generated datasets.

This module provides tools for analyzing the distribution of
generated samples across various dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy import stats

from dataset_gen.pipeline.storage import ParquetReader

logger = logging.getLogger(__name__)


@dataclass
class DistributionStats:
    """Statistics for a single distribution."""

    name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float

    # Optional grouping
    by_group: dict[str, "DistributionStats"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }
        if self.by_group:
            result["by_group"] = {k: v.to_dict() for k, v in self.by_group.items()}
        return result

    @classmethod
    def from_array(cls, name: str, data: np.ndarray) -> "DistributionStats":
        """Compute statistics from numpy array."""
        if len(data) == 0:
            return cls(
                name=name,
                count=0,
                mean=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                median=0.0,
                q25=0.0,
                q75=0.0,
                skewness=0.0,
                kurtosis=0.0,
            )

        return cls(
            name=name,
            count=len(data),
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            median=float(np.median(data)),
            q25=float(np.percentile(data, 25)),
            q75=float(np.percentile(data, 75)),
            skewness=float(stats.skew(data)) if len(data) > 2 else 0.0,
            kurtosis=float(stats.kurtosis(data)) if len(data) > 3 else 0.0,
        )


@dataclass
class DatasetStats:
    """Complete statistics for a dataset."""

    num_samples: int
    num_profiles: int
    num_rudiments: int

    # Sample-level distributions
    tempo_stats: DistributionStats
    duration_stats: DistributionStats
    num_strokes_stats: DistributionStats

    # Stroke-level distributions
    timing_error_stats: DistributionStats
    velocity_stats: DistributionStats

    # Measure-level distributions
    measure_timing_stats: DistributionStats
    measure_velocity_stats: DistributionStats

    # Exercise-level distributions
    exercise_timing_accuracy: DistributionStats
    exercise_hand_balance: DistributionStats

    # Categorical distributions
    skill_tier_counts: dict[str, int]
    rudiment_counts: dict[str, int]
    split_counts: dict[str, int]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "num_samples": self.num_samples,
            "num_profiles": self.num_profiles,
            "num_rudiments": self.num_rudiments,
            "tempo": self.tempo_stats.to_dict(),
            "duration": self.duration_stats.to_dict(),
            "num_strokes": self.num_strokes_stats.to_dict(),
            "timing_error": self.timing_error_stats.to_dict(),
            "velocity": self.velocity_stats.to_dict(),
            "measure_timing": self.measure_timing_stats.to_dict(),
            "measure_velocity": self.measure_velocity_stats.to_dict(),
            "exercise_timing_accuracy": self.exercise_timing_accuracy.to_dict(),
            "exercise_hand_balance": self.exercise_hand_balance.to_dict(),
            "skill_tier_counts": self.skill_tier_counts,
            "rudiment_counts": self.rudiment_counts,
            "split_counts": self.split_counts,
        }


class DatasetAnalyzer:
    """
    Analyze generated dataset statistics.

    Provides comprehensive analysis of distributions across
    all hierarchical levels (stroke, measure, exercise, sample).
    """

    def __init__(self, dataset_dir: Path | str):
        """
        Initialize analyzer.

        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.reader = ParquetReader(dataset_dir)

        # Cached data
        self._samples_df: pd.DataFrame | None = None
        self._strokes_df: pd.DataFrame | None = None
        self._measures_df: pd.DataFrame | None = None
        self._exercises_df: pd.DataFrame | None = None

    def load_data(self) -> None:
        """Load all parquet data into memory."""
        logger.info("Loading dataset...")
        self._samples_df = self.reader.load_samples()
        self._strokes_df = self.reader.load_strokes()
        self._measures_df = self.reader.load_measures()
        self._exercises_df = self.reader.load_exercises()
        logger.info(f"Loaded {len(self._samples_df)} samples")

    @property
    def samples(self) -> pd.DataFrame:
        if self._samples_df is None:
            self.load_data()
        return self._samples_df

    @property
    def strokes(self) -> pd.DataFrame:
        if self._strokes_df is None:
            self.load_data()
        return self._strokes_df

    @property
    def measures(self) -> pd.DataFrame:
        if self._measures_df is None:
            self.load_data()
        return self._measures_df

    @property
    def exercises(self) -> pd.DataFrame:
        if self._exercises_df is None:
            self.load_data()
        return self._exercises_df

    def compute_stats(self) -> DatasetStats:
        """Compute comprehensive dataset statistics."""
        samples = self.samples
        strokes = self.strokes
        measures = self.measures
        exercises = self.exercises

        # Basic counts
        num_samples = len(samples)
        num_profiles = samples["profile_id"].nunique()
        num_rudiments = samples["rudiment_slug"].nunique()

        # Sample-level stats
        tempo_stats = DistributionStats.from_array(
            "tempo_bpm",
            samples["tempo_bpm"].values,
        )
        duration_stats = DistributionStats.from_array(
            "duration_sec",
            samples["duration_sec"].values,
        )
        num_strokes_stats = DistributionStats.from_array(
            "num_strokes",
            samples["num_strokes"].values,
        )

        # Stroke-level stats
        timing_error_stats = DistributionStats.from_array(
            "timing_error_ms",
            strokes["timing_error_ms"].values,
        )
        velocity_stats = DistributionStats.from_array(
            "actual_velocity",
            strokes["actual_velocity"].values,
        )

        # Add by-skill-tier breakdown for timing errors
        timing_by_tier = self._compute_stroke_stats_by_tier(strokes, samples, "timing_error_ms")
        timing_error_stats.by_group = timing_by_tier

        # Measure-level stats
        measure_timing_stats = DistributionStats.from_array(
            "timing_mean_error_ms",
            measures["timing_mean_error_ms"].values,
        )
        measure_velocity_stats = DistributionStats.from_array(
            "velocity_consistency",
            measures["velocity_consistency"].values,
        )

        # Exercise-level stats
        exercise_timing_accuracy = DistributionStats.from_array(
            "timing_accuracy",
            exercises["timing_accuracy"].values,
        )
        exercise_hand_balance = DistributionStats.from_array(
            "hand_balance",
            exercises["hand_balance"].values,
        )

        # Add by-skill-tier breakdown
        exercise_timing_by_tier = self._compute_exercise_stats_by_tier(
            exercises, samples, "timing_accuracy"
        )
        exercise_timing_accuracy.by_group = exercise_timing_by_tier

        exercise_balance_by_tier = self._compute_exercise_stats_by_tier(
            exercises, samples, "hand_balance"
        )
        exercise_hand_balance.by_group = exercise_balance_by_tier

        # Categorical counts
        skill_tier_counts = samples["skill_tier"].value_counts().to_dict()
        rudiment_counts = samples["rudiment_slug"].value_counts().to_dict()

        # Split counts (if available from splits.json)
        split_counts = self._get_split_counts()

        return DatasetStats(
            num_samples=num_samples,
            num_profiles=num_profiles,
            num_rudiments=num_rudiments,
            tempo_stats=tempo_stats,
            duration_stats=duration_stats,
            num_strokes_stats=num_strokes_stats,
            timing_error_stats=timing_error_stats,
            velocity_stats=velocity_stats,
            measure_timing_stats=measure_timing_stats,
            measure_velocity_stats=measure_velocity_stats,
            exercise_timing_accuracy=exercise_timing_accuracy,
            exercise_hand_balance=exercise_hand_balance,
            skill_tier_counts=skill_tier_counts,
            rudiment_counts=rudiment_counts,
            split_counts=split_counts,
        )

    def _compute_stroke_stats_by_tier(
        self,
        strokes: pd.DataFrame,
        samples: pd.DataFrame,
        column: str,
    ) -> dict[str, DistributionStats]:
        """Compute stroke statistics grouped by skill tier."""
        # Merge to get skill tier for each stroke
        merged = strokes.merge(
            samples[["sample_id", "skill_tier"]],
            on="sample_id",
        )

        result = {}
        for tier in merged["skill_tier"].unique():
            tier_data = merged[merged["skill_tier"] == tier][column].values
            result[tier] = DistributionStats.from_array(f"{column}_{tier}", tier_data)

        return result

    def _compute_exercise_stats_by_tier(
        self,
        exercises: pd.DataFrame,
        samples: pd.DataFrame,
        column: str,
    ) -> dict[str, DistributionStats]:
        """Compute exercise statistics grouped by skill tier."""
        merged = exercises.merge(
            samples[["sample_id", "skill_tier"]],
            on="sample_id",
        )

        result = {}
        for tier in merged["skill_tier"].unique():
            tier_data = merged[merged["skill_tier"] == tier][column].values
            result[tier] = DistributionStats.from_array(f"{column}_{tier}", tier_data)

        return result

    def _get_split_counts(self) -> dict[str, int]:
        """Get sample counts per split."""
        splits_path = self.dataset_dir / "splits.json"
        if not splits_path.exists():
            return {}

        import json

        with open(splits_path) as f:
            splits = json.load(f)

        # Count samples per split
        samples = self.samples
        train_ids = set(splits.get("train_profile_ids", []))
        val_ids = set(splits.get("val_profile_ids", []))
        test_ids = set(splits.get("test_profile_ids", []))

        train_count = len(samples[samples["profile_id"].isin(train_ids)])
        val_count = len(samples[samples["profile_id"].isin(val_ids)])
        test_count = len(samples[samples["profile_id"].isin(test_ids)])

        return {
            "train": train_count,
            "validation": val_count,
            "test": test_count,
        }

    def check_skill_tier_ordering(self) -> dict[str, bool]:
        """
        Verify that skill tiers show expected ordering.

        Higher skill should correlate with:
        - Lower timing errors
        - Higher timing accuracy scores
        - Better hand balance
        """
        exercises = self.exercises
        samples = self.samples

        merged = exercises.merge(
            samples[["sample_id", "skill_tier"]],
            on="sample_id",
        )

        # Expected order (best to worst for timing accuracy)
        tier_order = ["professional", "advanced", "intermediate", "beginner"]

        results = {}

        # Check timing accuracy ordering
        tier_means = {}
        for tier in tier_order:
            tier_data = merged[merged["skill_tier"] == tier]["timing_accuracy"]
            if len(tier_data) > 0:
                tier_means[tier] = tier_data.mean()

        if len(tier_means) >= 2:
            # Check if professional > advanced > intermediate > beginner
            available_tiers = [t for t in tier_order if t in tier_means]
            is_ordered = all(
                tier_means[available_tiers[i]] >= tier_means[available_tiers[i + 1]]
                for i in range(len(available_tiers) - 1)
            )
            results["timing_accuracy_ordered"] = is_ordered
            results["timing_accuracy_means"] = tier_means

        # Check hand balance ordering
        tier_means = {}
        for tier in tier_order:
            tier_data = merged[merged["skill_tier"] == tier]["hand_balance"]
            if len(tier_data) > 0:
                tier_means[tier] = tier_data.mean()

        if len(tier_means) >= 2:
            available_tiers = [t for t in tier_order if t in tier_means]
            is_ordered = all(
                tier_means[available_tiers[i]] >= tier_means[available_tiers[i + 1]]
                for i in range(len(available_tiers) - 1)
            )
            results["hand_balance_ordered"] = is_ordered
            results["hand_balance_means"] = tier_means

        return results

    def compute_correlation_matrix(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for exercise scores.

        Args:
            columns: Columns to include (default: all numeric)

        Returns:
            Correlation matrix as DataFrame
        """
        exercises = self.exercises

        if columns is None:
            # Use all numeric columns except sample_id
            numeric_cols = exercises.select_dtypes(include=[np.number]).columns
            columns = [c for c in numeric_cols if c != "sample_id"]

        return exercises[columns].corr()


def analyze_dataset(dataset_dir: Path | str) -> DatasetStats:
    """
    Convenience function to analyze a dataset.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        DatasetStats with comprehensive statistics
    """
    analyzer = DatasetAnalyzer(dataset_dir)
    return analyzer.compute_stats()
