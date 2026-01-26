#!/usr/bin/env python3
"""
Dataset Filtering Examples
==========================

Examples for filtering SOUSA dataset by various criteria:
- Clean samples only (no augmentation)
- Specific augmentation presets
- Specific soundfonts
- Grouping by augmentation_group_id

Usage:
    python -m examples.filtering output/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.pytorch_dataloader import SOUSADataset


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Load and merge sample metadata."""
    samples_path = data_dir / "labels" / "samples.parquet"
    exercises_path = data_dir / "labels" / "exercises.parquet"

    samples = pd.read_parquet(samples_path)
    exercises = pd.read_parquet(exercises_path)

    return samples.merge(exercises, on="sample_id", how="left")


def filter_clean_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to clean (non-augmented) samples only.

    Clean samples are identified by:
    - augmentation_preset is null/NaN
    - augmentation_preset == "none"
    """
    if "augmentation_preset" not in df.columns:
        print("Warning: 'augmentation_preset' column not found.")
        print("Dataset may not have augmentation metadata.")
        return df

    mask = df["augmentation_preset"].isna() | (df["augmentation_preset"] == "none")
    return df[mask].reset_index(drop=True)


def filter_augmented_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to augmented samples only.
    """
    if "augmentation_preset" not in df.columns:
        return df

    mask = df["augmentation_preset"].notna() & (df["augmentation_preset"] != "none")
    return df[mask].reset_index(drop=True)


def filter_by_augmentation_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """
    Filter to specific augmentation preset.

    Common presets: "subtle", "moderate", "heavy", "none"
    """
    if "augmentation_preset" not in df.columns:
        return df

    return df[df["augmentation_preset"] == preset].reset_index(drop=True)


def filter_by_soundfont(df: pd.DataFrame, soundfont: str) -> pd.DataFrame:
    """
    Filter to specific soundfont.

    Common soundfonts: "practice_pad", "marching_snare", "acoustic_kit", etc.
    """
    if "soundfont" not in df.columns:
        print("Warning: 'soundfont' column not found.")
        return df

    return df[df["soundfont"] == soundfont].reset_index(drop=True)


def filter_by_soundfonts(df: pd.DataFrame, soundfonts: list[str]) -> pd.DataFrame:
    """Filter to multiple soundfonts."""
    if "soundfont" not in df.columns:
        return df

    return df[df["soundfont"].isin(soundfonts)].reset_index(drop=True)


def filter_by_skill_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """
    Filter to specific skill tier.

    Tiers: "beginner", "intermediate", "advanced", "professional"
    """
    return df[df["skill_tier"] == tier].reset_index(drop=True)


def filter_by_rudiment(df: pd.DataFrame, rudiment_slug: str) -> pd.DataFrame:
    """Filter to specific rudiment."""
    return df[df["rudiment_slug"] == rudiment_slug].reset_index(drop=True)


def filter_by_score_range(
    df: pd.DataFrame,
    min_score: float = 0,
    max_score: float = 100,
    score_column: str = "overall_score",
) -> pd.DataFrame:
    """Filter to samples within score range."""
    mask = (df[score_column] >= min_score) & (df[score_column] <= max_score)
    return df[mask].reset_index(drop=True)


def group_by_augmentation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Group samples by augmentation_group_id.

    This links clean samples with their augmented variants.
    Returns dict mapping group_id to dataframe of variants.
    """
    if "augmentation_group_id" not in df.columns:
        print("Warning: 'augmentation_group_id' column not found.")
        return {}

    groups = {}
    for group_id, group_df in df.groupby("augmentation_group_id"):
        groups[group_id] = group_df.reset_index(drop=True)

    return groups


def get_clean_augmented_pairs(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """
    Get pairs of (clean_sample_id, [augmented_sample_ids]).

    Useful for studying augmentation effects on the same underlying performance.
    """
    if "augmentation_group_id" not in df.columns or "augmentation_preset" not in df.columns:
        return []

    pairs = []
    for group_id, group_df in df.groupby("augmentation_group_id"):
        clean_mask = group_df["augmentation_preset"].isna() | (group_df["augmentation_preset"] == "none")
        clean_samples = group_df[clean_mask]["sample_id"].tolist()
        augmented_samples = group_df[~clean_mask]["sample_id"].tolist()

        if clean_samples and augmented_samples:
            for clean_id in clean_samples:
                pairs.append((clean_id, augmented_samples))

    return pairs


class FilteredSOUSADataset(SOUSADataset):
    """
    SOUSA Dataset with built-in filtering support.

    Example:
        # Get only clean practice pad samples
        dataset = FilteredSOUSADataset(
            data_dir="output/dataset",
            split="train",
            target="skill_tier",
            filters={
                "augmentation_preset": "none",  # or None for clean
                "soundfont": "practice_pad",
            }
        )
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str | None = None,
        target: str = "overall_score",
        filters: dict | None = None,
        **kwargs,
    ):
        super().__init__(data_dir, split=split, target=target, **kwargs)

        if filters:
            self._apply_filters(filters)

    def _apply_filters(self, filters: dict):
        """Apply filter criteria to data."""
        for column, value in filters.items():
            if column not in self.data.columns:
                print(f"Warning: Filter column '{column}' not found in data.")
                continue

            if value is None:
                # Filter for null values
                self.data = self.data[self.data[column].isna()]
            elif isinstance(value, list):
                # Filter for multiple values
                self.data = self.data[self.data[column].isin(value)]
            else:
                # Filter for exact value
                self.data = self.data[self.data[column] == value]

        self.data = self.data.reset_index(drop=True)


def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset"):
    """Print summary statistics for a filtered dataset."""
    print(f"\n{name}")
    print("=" * 50)
    print(f"Total samples: {len(df)}")

    if "skill_tier" in df.columns:
        print(f"\nBy skill tier:")
        for tier, count in df["skill_tier"].value_counts().items():
            print(f"  {tier}: {count}")

    if "soundfont" in df.columns:
        print(f"\nBy soundfont:")
        for sf, count in df["soundfont"].value_counts().items():
            print(f"  {sf}: {count}")

    if "augmentation_preset" in df.columns:
        print(f"\nBy augmentation preset:")
        for preset, count in df["augmentation_preset"].value_counts(dropna=False).items():
            label = preset if pd.notna(preset) else "(clean/none)"
            print(f"  {label}: {count}")

    if "overall_score" in df.columns:
        print(f"\nScore statistics:")
        print(f"  Mean: {df['overall_score'].mean():.1f}")
        print(f"  Std:  {df['overall_score'].std():.1f}")
        print(f"  Min:  {df['overall_score'].min():.1f}")
        print(f"  Max:  {df['overall_score'].max():.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset filtering examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        default="output/dataset",
        help="Path to SOUSA dataset directory",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("SOUSA Dataset Filtering Examples")
    print("=" * 60)
    print(f"Data directory: {data_dir}")

    # Load metadata
    print("\nLoading metadata...")
    df = load_metadata(data_dir)
    print(f"Total samples in dataset: {len(df)}")

    # Show available columns
    print(f"\nAvailable columns: {list(df.columns)}")

    # Example 1: All data
    print_dataset_summary(df, "Full Dataset")

    # Example 2: Clean samples only
    clean_df = filter_clean_samples(df)
    print_dataset_summary(clean_df, "Clean Samples Only")

    # Example 3: Augmented samples only
    aug_df = filter_augmented_samples(df)
    print_dataset_summary(aug_df, "Augmented Samples Only")

    # Example 4: By soundfont (if available)
    if "soundfont" in df.columns:
        soundfonts = df["soundfont"].unique()
        print(f"\n\nAvailable soundfonts: {list(soundfonts)}")

        for sf in soundfonts[:2]:  # Show first 2
            sf_df = filter_by_soundfont(df, sf)
            print_dataset_summary(sf_df, f"Soundfont: {sf}")

    # Example 5: Combined filters
    print("\n\n" + "=" * 60)
    print("Combined Filter Example")
    print("=" * 60)
    print("""
    # Using FilteredSOUSADataset class:

    from examples.filtering import FilteredSOUSADataset

    # Get only clean samples from practice pad, advanced skill level
    dataset = FilteredSOUSADataset(
        data_dir="output/dataset",
        split="train",
        target="skill_tier",
        filters={
            "augmentation_preset": None,  # None = clean/null
            "soundfont": "practice_pad",
            "skill_tier": "advanced",
        }
    )
    """)

    # Example 6: Augmentation groups
    if "augmentation_group_id" in df.columns:
        print("\n" + "=" * 60)
        print("Augmentation Groups")
        print("=" * 60)

        pairs = get_clean_augmented_pairs(df)
        print(f"Found {len(pairs)} clean-augmented pairs")

        if pairs:
            clean_id, aug_ids = pairs[0]
            print(f"\nExample pair:")
            print(f"  Clean sample: {clean_id}")
            print(f"  Augmented variants: {aug_ids[:3]}...")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
