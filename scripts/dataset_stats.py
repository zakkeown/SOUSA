#!/usr/bin/env python3
"""
Generate comprehensive statistics summary for SOUSA dataset.

This script provides detailed statistics about a generated dataset,
including completeness verification, distribution analysis, and
data quality checks.

Usage:
    python scripts/dataset_stats.py                      # Print to console
    python scripts/dataset_stats.py output/dataset       # Specify directory
    python scripts/dataset_stats.py --json stats.json    # Save as JSON
    python scripts/dataset_stats.py --markdown           # Output markdown tables

Examples:
    # Quick overview before publishing
    python scripts/dataset_stats.py

    # Generate stats for dataset card
    python scripts/dataset_stats.py --markdown > STATS.md

    # Machine-readable output
    python scripts/dataset_stats.py --json dataset_stats.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from dataset_gen.pipeline.storage import ParquetReader
from dataset_gen.rudiments.loader import load_all_rudiments
from dataset_gen.profiles.archetypes import SkillTier


@dataclass
class CompletenessReport:
    """Report on dataset completeness."""

    expected_rudiments: int
    found_rudiments: int
    missing_rudiments: list[str]
    expected_skill_tiers: int
    found_skill_tiers: int
    missing_skill_tiers: list[str]
    all_complete: bool

    def to_dict(self) -> dict:
        return {
            "expected_rudiments": self.expected_rudiments,
            "found_rudiments": self.found_rudiments,
            "missing_rudiments": self.missing_rudiments,
            "expected_skill_tiers": self.expected_skill_tiers,
            "found_skill_tiers": self.found_skill_tiers,
            "missing_skill_tiers": self.missing_skill_tiers,
            "all_complete": self.all_complete,
        }


@dataclass
class RudimentStats:
    """Statistics for a single rudiment."""

    slug: str
    name: str
    sample_count: int
    mean_score: float
    score_std: float
    tier_counts: dict[str, int]


@dataclass
class DatasetSummary:
    """Complete dataset summary."""

    # Overview
    total_samples: int
    total_profiles: int
    total_rudiments: int
    total_strokes: int
    total_measures: int

    # Audio
    total_audio_duration_sec: float
    total_audio_hours: float
    audio_file_count: int

    # MIDI
    midi_file_count: int

    # Splits
    train_count: int
    validation_count: int
    test_count: int

    # Skill tier distribution
    skill_tier_counts: dict[str, int]

    # Score summaries
    overall_score_mean: float
    overall_score_std: float
    timing_accuracy_mean: float
    hand_balance_mean: float

    # Per-rudiment stats
    rudiment_stats: list[RudimentStats]

    # Completeness
    completeness: CompletenessReport

    def to_dict(self) -> dict:
        return {
            "overview": {
                "total_samples": self.total_samples,
                "total_profiles": self.total_profiles,
                "total_rudiments": self.total_rudiments,
                "total_strokes": self.total_strokes,
                "total_measures": self.total_measures,
            },
            "audio": {
                "total_duration_sec": self.total_audio_duration_sec,
                "total_hours": round(self.total_audio_hours, 2),
                "file_count": self.audio_file_count,
            },
            "midi": {
                "file_count": self.midi_file_count,
            },
            "splits": {
                "train": self.train_count,
                "validation": self.validation_count,
                "test": self.test_count,
            },
            "skill_tiers": self.skill_tier_counts,
            "scores": {
                "overall_score_mean": round(self.overall_score_mean, 2),
                "overall_score_std": round(self.overall_score_std, 2),
                "timing_accuracy_mean": round(self.timing_accuracy_mean, 2),
                "hand_balance_mean": round(self.hand_balance_mean, 2),
            },
            "completeness": self.completeness.to_dict(),
            "per_rudiment": [
                {
                    "slug": r.slug,
                    "name": r.name,
                    "count": r.sample_count,
                    "mean_score": round(r.mean_score, 2),
                    "score_std": round(r.score_std, 2),
                    "tier_counts": r.tier_counts,
                }
                for r in self.rudiment_stats
            ],
        }


class DatasetStatsGenerator:
    """Generate comprehensive dataset statistics."""

    def __init__(self, dataset_dir: Path | str):
        self.dataset_dir = Path(dataset_dir)
        self.reader = ParquetReader(dataset_dir)

        # Load expected rudiments
        self.expected_rudiments = load_all_rudiments()
        self.expected_skill_tiers = [tier.value for tier in SkillTier]

    def generate_summary(self) -> DatasetSummary:
        """Generate complete dataset summary."""
        # Load data
        samples = self.reader.load_samples()
        strokes = self.reader.load_strokes()
        measures = self.reader.load_measures()
        exercises = self.reader.load_exercises()

        # Basic counts
        total_samples = len(samples)
        total_profiles = samples["profile_id"].nunique()
        total_rudiments = samples["rudiment_slug"].nunique()
        total_strokes = len(strokes)
        total_measures = len(measures)

        # Audio statistics
        audio_dir = self.dataset_dir / "audio"
        audio_files = list(audio_dir.glob("*.flac")) if audio_dir.exists() else []
        total_audio_duration_sec = samples["duration_sec"].sum()
        total_audio_hours = total_audio_duration_sec / 3600

        # MIDI statistics
        midi_dir = self.dataset_dir / "midi"
        midi_files = list(midi_dir.glob("*.mid")) if midi_dir.exists() else []

        # Splits
        splits = self._load_splits()
        train_ids = set(splits.get("train_profile_ids", []))
        val_ids = set(splits.get("val_profile_ids", []))
        test_ids = set(splits.get("test_profile_ids", []))

        train_count = len(samples[samples["profile_id"].isin(train_ids)])
        val_count = len(samples[samples["profile_id"].isin(val_ids)])
        test_count = len(samples[samples["profile_id"].isin(test_ids)])

        # Skill tier distribution
        skill_tier_counts = samples["skill_tier"].value_counts().to_dict()

        # Score summaries
        overall_score_mean = exercises["overall_score"].mean()
        overall_score_std = exercises["overall_score"].std()
        timing_accuracy_mean = exercises["timing_accuracy"].mean()
        hand_balance_mean = exercises["hand_balance"].mean()

        # Per-rudiment statistics
        rudiment_stats = self._compute_per_rudiment_stats(samples, exercises)

        # Completeness check
        completeness = self._check_completeness(samples)

        return DatasetSummary(
            total_samples=total_samples,
            total_profiles=total_profiles,
            total_rudiments=total_rudiments,
            total_strokes=total_strokes,
            total_measures=total_measures,
            total_audio_duration_sec=total_audio_duration_sec,
            total_audio_hours=total_audio_hours,
            audio_file_count=len(audio_files),
            midi_file_count=len(midi_files),
            train_count=train_count,
            validation_count=val_count,
            test_count=test_count,
            skill_tier_counts=skill_tier_counts,
            overall_score_mean=overall_score_mean,
            overall_score_std=overall_score_std,
            timing_accuracy_mean=timing_accuracy_mean,
            hand_balance_mean=hand_balance_mean,
            rudiment_stats=rudiment_stats,
            completeness=completeness,
        )

    def _load_splits(self) -> dict:
        """Load splits.json if it exists."""
        splits_path = self.dataset_dir / "splits.json"
        if not splits_path.exists():
            return {}
        with open(splits_path) as f:
            return json.load(f)

    def _compute_per_rudiment_stats(
        self, samples: pd.DataFrame, exercises: pd.DataFrame
    ) -> list[RudimentStats]:
        """Compute statistics for each rudiment."""
        # Merge exercises with samples for skill tier info
        merged = exercises.merge(
            samples[["sample_id", "rudiment_slug", "skill_tier"]],
            on="sample_id",
        )

        stats = []
        for slug in sorted(samples["rudiment_slug"].unique()):
            rudiment_data = merged[merged["rudiment_slug"] == slug]

            # Get rudiment name
            name = slug
            if slug in self.expected_rudiments:
                name = self.expected_rudiments[slug].name

            # Compute tier counts
            tier_counts = rudiment_data["skill_tier"].value_counts().to_dict()

            stats.append(
                RudimentStats(
                    slug=slug,
                    name=name,
                    sample_count=len(rudiment_data),
                    mean_score=rudiment_data["overall_score"].mean(),
                    score_std=rudiment_data["overall_score"].std(),
                    tier_counts=tier_counts,
                )
            )

        return stats

    def _check_completeness(self, samples: pd.DataFrame) -> CompletenessReport:
        """Check if all expected rudiments and skill tiers are present."""
        found_rudiments = set(samples["rudiment_slug"].unique())
        expected_slugs = set(self.expected_rudiments.keys())
        missing_rudiments = sorted(expected_slugs - found_rudiments)

        found_tiers = set(samples["skill_tier"].unique())
        expected_tiers = set(self.expected_skill_tiers)
        missing_tiers = sorted(expected_tiers - found_tiers)

        return CompletenessReport(
            expected_rudiments=len(expected_slugs),
            found_rudiments=len(found_rudiments),
            missing_rudiments=missing_rudiments,
            expected_skill_tiers=len(expected_tiers),
            found_skill_tiers=len(found_tiers),
            missing_skill_tiers=missing_tiers,
            all_complete=len(missing_rudiments) == 0 and len(missing_tiers) == 0,
        )


def format_console_output(summary: DatasetSummary) -> str:
    """Format summary for console output."""
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("SOUSA DATASET STATISTICS")
    lines.append("=" * 60)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total Samples:     {summary.total_samples:,}")
    lines.append(f"  Total Profiles:    {summary.total_profiles:,}")
    lines.append(f"  Total Rudiments:   {summary.total_rudiments}")
    lines.append(f"  Total Strokes:     {summary.total_strokes:,}")
    lines.append(f"  Total Measures:    {summary.total_measures:,}")
    lines.append("")

    # Audio/MIDI
    lines.append("MEDIA FILES")
    lines.append("-" * 40)
    lines.append(f"  Audio Files:       {summary.audio_file_count:,}")
    lines.append(f"  Audio Duration:    {summary.total_audio_hours:.1f} hours")
    lines.append(f"  MIDI Files:        {summary.midi_file_count:,}")
    lines.append("")

    # Splits
    lines.append("DATA SPLITS")
    lines.append("-" * 40)
    total = summary.train_count + summary.validation_count + summary.test_count
    if total > 0:
        lines.append(
            f"  Train:       {summary.train_count:,} ({100*summary.train_count/total:.1f}%)"
        )
        lines.append(
            f"  Validation:  {summary.validation_count:,} ({100*summary.validation_count/total:.1f}%)"
        )
        lines.append(f"  Test:        {summary.test_count:,} ({100*summary.test_count/total:.1f}%)")
    lines.append("")

    # Skill tiers
    lines.append("SKILL TIER DISTRIBUTION")
    lines.append("-" * 40)
    tier_order = ["beginner", "intermediate", "advanced", "professional"]
    for tier in tier_order:
        count = summary.skill_tier_counts.get(tier, 0)
        pct = 100 * count / summary.total_samples if summary.total_samples > 0 else 0
        lines.append(f"  {tier.capitalize():15s} {count:,} ({pct:.1f}%)")
    lines.append("")

    # Scores
    lines.append("SCORE SUMMARY")
    lines.append("-" * 40)
    lines.append(
        f"  Overall Score:     {summary.overall_score_mean:.1f} +/- {summary.overall_score_std:.1f}"
    )
    lines.append(f"  Timing Accuracy:   {summary.timing_accuracy_mean:.1f}")
    lines.append(f"  Hand Balance:      {summary.hand_balance_mean:.1f}")
    lines.append("")

    # Completeness
    lines.append("COMPLETENESS CHECK")
    lines.append("-" * 40)
    comp = summary.completeness
    status = "PASS" if comp.all_complete else "FAIL"
    lines.append(f"  Status:            {status}")
    lines.append(f"  Rudiments:         {comp.found_rudiments}/{comp.expected_rudiments}")
    lines.append(f"  Skill Tiers:       {comp.found_skill_tiers}/{comp.expected_skill_tiers}")
    if comp.missing_rudiments:
        lines.append(f"  Missing Rudiments: {', '.join(comp.missing_rudiments)}")
    if comp.missing_skill_tiers:
        lines.append(f"  Missing Tiers:     {', '.join(comp.missing_skill_tiers)}")
    lines.append("")

    # Per-rudiment table (abbreviated)
    lines.append("PER-RUDIMENT STATISTICS (Top 10 by count)")
    lines.append("-" * 40)
    sorted_rudiments = sorted(summary.rudiment_stats, key=lambda r: r.sample_count, reverse=True)
    lines.append(f"  {'Rudiment':<30} {'Count':>8} {'Score':>10}")
    lines.append("  " + "-" * 50)
    for r in sorted_rudiments[:10]:
        lines.append(
            f"  {r.name:<30} {r.sample_count:>8} {r.mean_score:>7.1f} +/- {r.score_std:>4.1f}"
        )
    if len(sorted_rudiments) > 10:
        lines.append(f"  ... and {len(sorted_rudiments) - 10} more rudiments")
    lines.append("")

    return "\n".join(lines)


def format_markdown_output(summary: DatasetSummary) -> str:
    """Format summary as markdown tables."""
    lines = []

    # Quick stats table
    lines.append("## Quick Stats")
    lines.append("")
    lines.append("| Statistic | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Total Samples | {summary.total_samples:,} |")
    lines.append(f"| Audio Hours | {summary.total_audio_hours:.1f} |")
    lines.append(f"| Stroke Labels | {summary.total_strokes:,} |")
    lines.append(f"| Rudiments | {summary.total_rudiments}/40 PAS |")
    lines.append(f"| Skill Tiers | {summary.completeness.found_skill_tiers} |")
    lines.append("")

    # Splits
    lines.append("## Data Splits")
    lines.append("")
    lines.append("| Split | Samples | Percentage |")
    lines.append("|-------|---------|------------|")
    total = summary.train_count + summary.validation_count + summary.test_count
    if total > 0:
        lines.append(f"| Train | {summary.train_count:,} | {100*summary.train_count/total:.1f}% |")
        lines.append(
            f"| Validation | {summary.validation_count:,} | {100*summary.validation_count/total:.1f}% |"
        )
        lines.append(f"| Test | {summary.test_count:,} | {100*summary.test_count/total:.1f}% |")
    lines.append("")

    # Skill tier distribution
    lines.append("## Skill Tier Distribution")
    lines.append("")
    lines.append("| Tier | Samples | Percentage |")
    lines.append("|------|---------|------------|")
    tier_order = ["beginner", "intermediate", "advanced", "professional"]
    for tier in tier_order:
        count = summary.skill_tier_counts.get(tier, 0)
        pct = 100 * count / summary.total_samples if summary.total_samples > 0 else 0
        lines.append(f"| {tier.capitalize()} | {count:,} | {pct:.1f}% |")
    lines.append("")

    # Per-rudiment statistics
    lines.append("## Per-Rudiment Statistics")
    lines.append("")
    lines.append("| Rudiment | Samples | Mean Score | Std |")
    lines.append("|----------|---------|------------|-----|")
    sorted_rudiments = sorted(summary.rudiment_stats, key=lambda r: r.slug)
    for r in sorted_rudiments:
        lines.append(f"| {r.name} | {r.sample_count:,} | {r.mean_score:.1f} | {r.score_std:.1f} |")
    lines.append("")

    # Completeness
    lines.append("## Completeness")
    lines.append("")
    comp = summary.completeness
    status = "PASS" if comp.all_complete else "FAIL"
    lines.append(f"- **Status**: {status}")
    lines.append(f"- **Rudiments**: {comp.found_rudiments}/{comp.expected_rudiments}")
    lines.append(f"- **Skill Tiers**: {comp.found_skill_tiers}/{comp.expected_skill_tiers}")
    if comp.missing_rudiments:
        lines.append(f"- **Missing Rudiments**: {', '.join(comp.missing_rudiments)}")
    if comp.missing_skill_tiers:
        lines.append(f"- **Missing Tiers**: {', '.join(comp.missing_skill_tiers)}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive statistics for SOUSA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="output/dataset",
        help="Path to dataset directory (default: output/dataset)",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Save output as JSON to specified file",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as markdown tables",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    labels_dir = dataset_dir / "labels"
    if not labels_dir.exists() or not any(labels_dir.glob("*.parquet")):
        print(f"Error: No parquet files found in {labels_dir}", file=sys.stderr)
        sys.exit(1)

    # Generate statistics
    generator = DatasetStatsGenerator(dataset_dir)
    summary = generator.generate_summary()

    # Output
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        print(f"Statistics saved to {args.json}")
    elif args.markdown:
        print(format_markdown_output(summary))
    else:
        print(format_console_output(summary))

    # Exit code based on completeness
    if not summary.completeness.all_complete:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
