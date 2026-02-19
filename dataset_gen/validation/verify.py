"""
Label verification for generated datasets.

This module verifies that computed labels correctly reflect
the generation parameters and that data integrity is maintained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging

import pandas as pd

from dataset_gen.pipeline.storage import ParquetReader
from dataset_gen.rudiments.loader import load_all_rudiments

logger = logging.getLogger(__name__)


@dataclass
class VerificationCheck:
    """Result of a single verification check."""

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Complete verification results."""

    checks: list[VerificationCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def num_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def num_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Verification: {self.num_passed}/{len(self.checks)} checks passed",
            "",
        ]
        for check in self.checks:
            status = "✓" if check.passed else "✗"
            lines.append(f"  {status} {check.name}: {check.message}")
        return "\n".join(lines)


class LabelVerifier:
    """
    Verify label correctness and data integrity.

    Runs a series of checks to ensure:
    - Labels are within expected ranges
    - Skill tier differences are reflected in metrics
    - Data relationships are consistent
    - No data corruption occurred
    """

    def __init__(self, dataset_dir: Path | str, midi_dir: Path | str | None = None):
        """
        Initialize verifier.

        Args:
            dataset_dir: Path to dataset directory
            midi_dir: Path to MIDI files (defaults to dataset_dir/midi)
        """
        self.dataset_dir = Path(dataset_dir)
        self.reader = ParquetReader(dataset_dir)
        self.midi_dir = Path(midi_dir) if midi_dir else self.dataset_dir / "midi"

        self._samples_df: pd.DataFrame | None = None
        self._strokes_df: pd.DataFrame | None = None
        self._measures_df: pd.DataFrame | None = None
        self._exercises_df: pd.DataFrame | None = None
        self._rudiments: dict | None = None

    @property
    def rudiments(self) -> dict:
        """Lazily load rudiment definitions."""
        if self._rudiments is None:
            self._rudiments = load_all_rudiments()
        return self._rudiments

    def load_data(self) -> None:
        """Load all parquet data."""
        self._samples_df = self.reader.load_samples()
        self._strokes_df = self.reader.load_strokes()
        self._measures_df = self.reader.load_measures()
        self._exercises_df = self.reader.load_exercises()

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

    def verify_all(self, include_midi_checks: bool = True) -> VerificationResult:
        """
        Run all verification checks.

        Args:
            include_midi_checks: Whether to run MIDI alignment checks (slower)
        """
        result = VerificationResult()

        # Parquet file integrity (run first - if this fails, other checks can't run)
        result.checks.append(self._check_parquet_integrity())

        # Data integrity checks
        result.checks.append(self._check_sample_ids_unique())
        result.checks.append(self._check_stroke_sample_refs())
        result.checks.append(self._check_measure_sample_refs())
        result.checks.append(self._check_exercise_sample_refs())

        # Range checks
        result.checks.append(self._check_velocity_range())
        result.checks.append(self._check_timing_range())
        result.checks.append(self._check_score_ranges())

        # Consistency checks
        result.checks.append(self._check_stroke_counts_match())
        result.checks.append(self._check_measure_counts_match())

        # Skill tier checks
        result.checks.append(self._check_skill_tier_timing())
        result.checks.append(self._check_skill_tier_hand_balance())

        # Rudiment pattern correctness
        result.checks.append(self._check_rudiment_pattern_correctness())

        # Label-to-MIDI alignment (optional, slower)
        if include_midi_checks and self.midi_dir.exists():
            result.checks.append(self._check_label_midi_alignment())

        return result

    def _check_parquet_integrity(self) -> VerificationCheck:
        """
        Verify all parquet files can be fully read and are internally consistent.

        Checks:
        - All 4 parquet files exist and are readable
        - Row counts match between samples and exercises
        - All sample_ids in strokes/measures/exercises exist in samples
        - No unexpected null values in required columns
        - Data types are correct
        """
        labels_dir = self.dataset_dir / "labels"
        issues = []
        stats = {}

        # Required columns for each file
        required_columns = {
            "samples": ["sample_id", "profile_id", "rudiment_slug", "skill_tier", "tempo_bpm"],
            "exercises": ["sample_id", "timing_accuracy", "hand_balance", "overall_score"],
            "measures": ["sample_id", "index", "timing_mean_error_ms"],
            "strokes": ["sample_id", "index", "hand", "actual_time_ms", "actual_velocity"],
        }

        # Try to load each parquet file
        dataframes = {}
        for name in ["samples", "strokes", "measures", "exercises"]:
            path = labels_dir / f"{name}.parquet"
            if not path.exists():
                issues.append(f"Missing parquet file: {name}.parquet")
                continue

            try:
                df = pd.read_parquet(path)
                dataframes[name] = df
                stats[f"{name}_rows"] = len(df)

                # Check required columns exist
                missing_cols = set(required_columns[name]) - set(df.columns)
                if missing_cols:
                    issues.append(f"{name}.parquet missing columns: {missing_cols}")

                # Check for nulls in required columns
                for col in required_columns[name]:
                    if col in df.columns:
                        null_count = df[col].isnull().sum()
                        if null_count > 0:
                            issues.append(f"{name}.{col} has {null_count} null values")

            except Exception as e:
                issues.append(f"Failed to read {name}.parquet: {str(e)}")

        # Cross-file consistency checks
        if "samples" in dataframes and "exercises" in dataframes:
            samples_df = dataframes["samples"]
            exercises_df = dataframes["exercises"]

            # Row counts should match (1:1 relationship)
            if len(samples_df) != len(exercises_df):
                issues.append(
                    f"Row count mismatch: samples={len(samples_df)}, exercises={len(exercises_df)}"
                )

            # All exercise sample_ids should exist in samples
            sample_ids = set(samples_df["sample_id"])
            exercise_ids = set(exercises_df["sample_id"])
            orphan_exercises = exercise_ids - sample_ids
            if orphan_exercises:
                issues.append(f"{len(orphan_exercises)} exercises reference non-existent samples")

        if "samples" in dataframes and "strokes" in dataframes:
            sample_ids = set(dataframes["samples"]["sample_id"])
            stroke_ids = set(dataframes["strokes"]["sample_id"].unique())
            orphan_strokes = stroke_ids - sample_ids
            if orphan_strokes:
                issues.append(
                    f"{len(orphan_strokes)} stroke records reference non-existent samples"
                )

        if "samples" in dataframes and "measures" in dataframes:
            sample_ids = set(dataframes["samples"]["sample_id"])
            measure_ids = set(dataframes["measures"]["sample_id"].unique())
            orphan_measures = measure_ids - sample_ids
            if orphan_measures:
                issues.append(
                    f"{len(orphan_measures)} measure records reference non-existent samples"
                )

        # Check for duplicate sample_ids in samples table
        if "samples" in dataframes:
            samples_df = dataframes["samples"]
            duplicate_ids = samples_df["sample_id"].duplicated().sum()
            if duplicate_ids > 0:
                issues.append(f"{duplicate_ids} duplicate sample_ids in samples.parquet")

        passed = len(issues) == 0

        return VerificationCheck(
            name="parquet_integrity",
            passed=passed,
            message=(
                "All parquet files valid and consistent"
                if passed
                else f"{len(issues)} integrity issues found"
            ),
            details={
                "stats": stats,
                "issues": issues,
            },
        )

    def _check_sample_ids_unique(self) -> VerificationCheck:
        """Verify all sample IDs are unique."""
        samples = self.samples
        n_total = len(samples)
        n_unique = samples["sample_id"].nunique()

        passed = n_total == n_unique
        return VerificationCheck(
            name="sample_ids_unique",
            passed=passed,
            message=f"{n_unique}/{n_total} unique sample IDs",
            details={"total": n_total, "unique": n_unique},
        )

    def _check_stroke_sample_refs(self) -> VerificationCheck:
        """Verify all stroke sample_ids reference valid samples."""
        stroke_ids = set(self.strokes["sample_id"].unique())
        sample_ids = set(self.samples["sample_id"].unique())

        orphans = stroke_ids - sample_ids
        passed = len(orphans) == 0

        return VerificationCheck(
            name="stroke_refs_valid",
            passed=passed,
            message=(
                f"{len(orphans)} orphaned stroke records"
                if orphans
                else "All strokes reference valid samples"
            ),
            details={"orphan_count": len(orphans)},
        )

    def _check_measure_sample_refs(self) -> VerificationCheck:
        """Verify all measure sample_ids reference valid samples."""
        measure_ids = set(self.measures["sample_id"].unique())
        sample_ids = set(self.samples["sample_id"].unique())

        orphans = measure_ids - sample_ids
        passed = len(orphans) == 0

        return VerificationCheck(
            name="measure_refs_valid",
            passed=passed,
            message=(
                f"{len(orphans)} orphaned measure records"
                if orphans
                else "All measures reference valid samples"
            ),
            details={"orphan_count": len(orphans)},
        )

    def _check_exercise_sample_refs(self) -> VerificationCheck:
        """Verify all exercise sample_ids reference valid samples."""
        exercise_ids = set(self.exercises["sample_id"].unique())
        sample_ids = set(self.samples["sample_id"].unique())

        orphans = exercise_ids - sample_ids
        passed = len(orphans) == 0

        return VerificationCheck(
            name="exercise_refs_valid",
            passed=passed,
            message=(
                f"{len(orphans)} orphaned exercise records"
                if orphans
                else "All exercises reference valid samples"
            ),
            details={"orphan_count": len(orphans)},
        )

    def _check_velocity_range(self) -> VerificationCheck:
        """Verify velocities are in valid MIDI range [1, 127]."""
        strokes = self.strokes
        velocities = strokes["actual_velocity"]

        in_range = (velocities >= 1) & (velocities <= 127)
        n_valid = in_range.sum()
        n_total = len(velocities)
        pct_valid = 100 * n_valid / max(n_total, 1)

        passed = pct_valid >= 99.9  # Allow tiny tolerance

        return VerificationCheck(
            name="velocity_range",
            passed=passed,
            message=f"{pct_valid:.2f}% in valid MIDI range [1, 127]",
            details={
                "min": float(velocities.min()),
                "max": float(velocities.max()),
                "pct_valid": pct_valid,
            },
        )

    def _check_timing_range(self) -> VerificationCheck:
        """Verify timing errors are within reasonable bounds."""
        strokes = self.strokes
        # Exclude grace notes - their timing_error_ms measures flam spacing deviation,
        # which is a different metric than grid timing error
        primary_strokes = strokes[strokes["stroke_type"] != "grace"]
        timing_errors = primary_strokes["timing_error_ms"].abs()

        # Timing errors above 500ms are almost certainly bugs
        # Allow up to 20% of strokes to exceed 200ms (beginners can have larger errors)
        max_error_threshold = 500  # Hard limit
        soft_threshold = 200  # Most should be under this

        max_error = float(timing_errors.max())
        mean_error = float(timing_errors.mean())

        in_soft_range = timing_errors <= soft_threshold
        pct_in_soft_range = 100 * in_soft_range.sum() / max(len(timing_errors), 1)

        # Pass if: max error < 500ms AND (mean < 100ms OR 80%+ under 200ms)
        passed = max_error < max_error_threshold and (mean_error < 100 or pct_in_soft_range >= 80)

        return VerificationCheck(
            name="timing_range",
            passed=passed,
            message=f"mean={mean_error:.1f}ms, {pct_in_soft_range:.1f}% under {soft_threshold}ms",
            details={
                "max_error": max_error,
                "mean_error": mean_error,
                "pct_under_200ms": pct_in_soft_range,
            },
        )

    def _check_score_ranges(self) -> VerificationCheck:
        """Verify exercise scores are in valid range [0, 100]."""
        exercises = self.exercises

        score_columns = [
            "timing_accuracy",
            "timing_consistency",
            "accent_differentiation",
            "hand_balance",
            "overall_score",
        ]

        issues = []
        for col in score_columns:
            if col not in exercises.columns:
                continue
            values = exercises[col]
            if values.min() < 0 or values.max() > 100:
                issues.append(f"{col}: [{values.min():.1f}, {values.max():.1f}]")

        passed = len(issues) == 0

        return VerificationCheck(
            name="score_ranges",
            passed=passed,
            message="All scores in [0, 100]" if passed else f"Out of range: {', '.join(issues)}",
            details={"issues": issues},
        )

    def _check_stroke_counts_match(self) -> VerificationCheck:
        """Verify stroke counts in samples match actual stroke records."""
        samples = self.samples
        strokes = self.strokes

        # Count strokes per sample
        stroke_counts = strokes.groupby("sample_id").size()

        # Compare to recorded counts
        mismatches = []
        for _, row in samples.iterrows():
            sid = row["sample_id"]
            recorded = row["num_strokes"]
            actual = stroke_counts.get(sid, 0)
            if recorded != actual:
                mismatches.append((sid, recorded, actual))

        passed = len(mismatches) == 0

        return VerificationCheck(
            name="stroke_counts_match",
            passed=passed,
            message=(
                f"{len(mismatches)} samples with mismatched stroke counts"
                if mismatches
                else "All stroke counts match"
            ),
            details={"mismatch_count": len(mismatches)},
        )

    def _check_measure_counts_match(self) -> VerificationCheck:
        """Verify measure counts in samples match actual measure records."""
        samples = self.samples
        measures = self.measures

        # Count measures per sample
        measure_counts = measures.groupby("sample_id").size()

        # Compare to recorded counts
        mismatches = []
        for _, row in samples.iterrows():
            sid = row["sample_id"]
            recorded = row["num_measures"]
            actual = measure_counts.get(sid, 0)
            if recorded != actual:
                mismatches.append((sid, recorded, actual))

        passed = len(mismatches) == 0

        return VerificationCheck(
            name="measure_counts_match",
            passed=passed,
            message=(
                f"{len(mismatches)} samples with mismatched measure counts"
                if mismatches
                else "All measure counts match"
            ),
            details={"mismatch_count": len(mismatches)},
        )

    def _check_skill_tier_timing(self) -> VerificationCheck:
        """
        Verify skill tiers show expected timing accuracy ordering.

        Professional > Advanced > Intermediate > Beginner
        """
        exercises = self.exercises
        samples = self.samples

        merged = exercises.merge(
            samples[["sample_id", "skill_tier"]],
            on="sample_id",
        )

        # Get mean timing accuracy per tier
        tier_means = merged.groupby("skill_tier")["timing_accuracy"].mean()

        # Check ordering
        tier_order = ["professional", "advanced", "intermediate", "beginner"]
        available = [t for t in tier_order if t in tier_means.index]

        if len(available) < 2:
            return VerificationCheck(
                name="skill_tier_timing",
                passed=True,
                message="Not enough skill tiers to verify ordering",
                details={"available_tiers": available},
            )

        # Check if properly ordered (allowing small tolerance)
        ordered = True
        for i in range(len(available) - 1):
            higher = tier_means[available[i]]
            lower = tier_means[available[i + 1]]
            if higher < lower - 2:  # Allow 2-point tolerance
                ordered = False
                break

        return VerificationCheck(
            name="skill_tier_timing",
            passed=ordered,
            message=(
                "Timing accuracy properly ordered by skill tier"
                if ordered
                else "Timing accuracy not ordered by skill tier"
            ),
            details={"tier_means": tier_means.to_dict()},
        )

    def _check_skill_tier_hand_balance(self) -> VerificationCheck:
        """
        Verify skill tiers show expected hand balance ordering.

        Professional > Advanced > Intermediate > Beginner
        """
        exercises = self.exercises
        samples = self.samples

        merged = exercises.merge(
            samples[["sample_id", "skill_tier"]],
            on="sample_id",
        )

        # Get mean hand balance per tier
        tier_means = merged.groupby("skill_tier")["hand_balance"].mean()

        # Check ordering
        tier_order = ["professional", "advanced", "intermediate", "beginner"]
        available = [t for t in tier_order if t in tier_means.index]

        if len(available) < 2:
            return VerificationCheck(
                name="skill_tier_hand_balance",
                passed=True,
                message="Not enough skill tiers to verify ordering",
                details={"available_tiers": available},
            )

        # Check if properly ordered
        ordered = True
        for i in range(len(available) - 1):
            higher = tier_means[available[i]]
            lower = tier_means[available[i + 1]]
            if higher < lower - 2:  # Allow 2-point tolerance
                ordered = False
                break

        return VerificationCheck(
            name="skill_tier_hand_balance",
            passed=ordered,
            message=(
                "Hand balance properly ordered by skill tier"
                if ordered
                else "Hand balance not ordered by skill tier"
            ),
            details={"tier_means": tier_means.to_dict()},
        )

    def _check_rudiment_pattern_correctness(self) -> VerificationCheck:
        """
        Verify generated sticking patterns match rudiment definitions.

        Checks that the sequence of L/R hands in stroke labels matches
        what the rudiment YAML files define.
        """
        samples = self.samples
        strokes = self.strokes
        rudiments = self.rudiments

        mismatches = []
        samples_checked = 0
        max_samples_to_check = 100  # Sample a subset for efficiency

        # Sample some samples from each rudiment
        for rudiment_slug in samples["rudiment_slug"].unique():
            if rudiment_slug not in rudiments:
                continue

            rudiment = rudiments[rudiment_slug]
            expected_pattern = [s.hand.value for s in rudiment.pattern.strokes]
            pattern_len = len(expected_pattern)

            # Get a few samples of this rudiment
            rudiment_samples = samples[samples["rudiment_slug"] == rudiment_slug].head(5)

            for _, sample_row in rudiment_samples.iterrows():
                sample_id = sample_row["sample_id"]
                sample_strokes = strokes[strokes["sample_id"] == sample_id].sort_values("index")

                if len(sample_strokes) == 0:
                    continue

                samples_checked += 1

                # Get the actual hand sequence
                actual_hands = sample_strokes["hand"].tolist()

                # Check first cycle matches pattern
                first_cycle = actual_hands[:pattern_len]
                if first_cycle != expected_pattern:
                    mismatches.append(
                        {
                            "sample_id": sample_id,
                            "rudiment": rudiment_slug,
                            "expected_first_cycle": expected_pattern,
                            "actual_first_cycle": first_cycle,
                        }
                    )

                if samples_checked >= max_samples_to_check:
                    break

            if samples_checked >= max_samples_to_check:
                break

        passed = len(mismatches) == 0

        return VerificationCheck(
            name="rudiment_pattern_correctness",
            passed=passed,
            message=(
                f"Checked {samples_checked} samples, {len(mismatches)} pattern mismatches"
                if passed
                else f"{len(mismatches)} samples have incorrect sticking patterns"
            ),
            details={
                "samples_checked": samples_checked,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches[:5],  # Only include first 5
            },
        )

    def _check_label_midi_alignment(self, tolerance_ms: float = 5.0) -> VerificationCheck:
        """
        Verify stroke labels match actual MIDI events.

        Compares timing and velocity from labels to what's in the MIDI files.

        Args:
            tolerance_ms: Maximum allowed timing deviation between label and MIDI
        """
        import mido

        samples = self.samples
        strokes = self.strokes

        misalignments = []
        samples_checked = 0
        total_strokes_checked = 0
        strokes_aligned = 0
        max_samples_to_check = 50  # Sample a subset for efficiency

        for _, sample_row in samples.head(max_samples_to_check).iterrows():
            sample_id = sample_row["sample_id"]

            # Resolve MIDI path from metadata (supports subdirectory layout)
            midi_path_str = sample_row.get("midi_path")
            if pd.isna(midi_path_str) or not midi_path_str:
                # Fallback to flat layout for backward compatibility
                midi_path = self.midi_dir / f"{sample_id}.mid"
            else:
                midi_path = self.dataset_dir / midi_path_str
            if not midi_path.exists():
                continue

            samples_checked += 1

            try:
                mid = mido.MidiFile(str(midi_path))
            except Exception as e:
                logger.warning(f"Failed to read MIDI file {midi_path}: {e}")
                continue

            # Get tempo from MIDI
            tempo_us = 500000  # Default 120 BPM
            for track in mid.tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        tempo_us = msg.tempo
                        break

            tempo_bpm = mido.tempo2bpm(tempo_us)
            ticks_per_ms = mid.ticks_per_beat * tempo_bpm / 60000.0

            # Extract note events from MIDI
            midi_events = []
            current_time_ticks = 0
            for track in mid.tracks:
                current_time_ticks = 0
                for msg in track:
                    current_time_ticks += msg.time
                    if msg.type == "note_on" and msg.velocity > 0:
                        time_ms = current_time_ticks / ticks_per_ms
                        midi_events.append(
                            {
                                "time_ms": time_ms,
                                "velocity": msg.velocity,
                            }
                        )

            # Sort by time
            midi_events.sort(key=lambda x: x["time_ms"])

            # Get label strokes for this sample
            sample_strokes = strokes[strokes["sample_id"] == sample_id].sort_values(
                "actual_time_ms"
            )

            if len(sample_strokes) != len(midi_events):
                misalignments.append(
                    {
                        "sample_id": sample_id,
                        "issue": "stroke_count_mismatch",
                        "label_count": len(sample_strokes),
                        "midi_count": len(midi_events),
                    }
                )
                continue

            # Compare each stroke
            for (_, stroke_row), midi_event in zip(sample_strokes.iterrows(), midi_events):
                total_strokes_checked += 1

                label_time = stroke_row["actual_time_ms"]
                midi_time = midi_event["time_ms"]
                time_diff = abs(label_time - midi_time)

                label_vel = stroke_row["actual_velocity"]
                midi_vel = midi_event["velocity"]
                vel_diff = abs(label_vel - midi_vel)

                if time_diff <= tolerance_ms and vel_diff == 0:
                    strokes_aligned += 1
                elif time_diff > tolerance_ms:
                    misalignments.append(
                        {
                            "sample_id": sample_id,
                            "issue": "timing_mismatch",
                            "label_time_ms": label_time,
                            "midi_time_ms": midi_time,
                            "diff_ms": time_diff,
                        }
                    )
                elif vel_diff > 0:
                    misalignments.append(
                        {
                            "sample_id": sample_id,
                            "issue": "velocity_mismatch",
                            "label_velocity": label_vel,
                            "midi_velocity": midi_vel,
                        }
                    )

        if total_strokes_checked == 0:
            return VerificationCheck(
                name="label_midi_alignment",
                passed=True,
                message="No MIDI files available for alignment check",
                details={"samples_checked": 0},
            )

        alignment_rate = 100 * strokes_aligned / total_strokes_checked
        passed = alignment_rate >= 99.0  # Allow 1% tolerance

        return VerificationCheck(
            name="label_midi_alignment",
            passed=passed,
            message=f"{alignment_rate:.1f}% of strokes align with MIDI (checked {samples_checked} samples)",
            details={
                "samples_checked": samples_checked,
                "total_strokes": total_strokes_checked,
                "strokes_aligned": strokes_aligned,
                "alignment_rate": alignment_rate,
                "misalignment_examples": misalignments[:5],
            },
        )


def verify_labels(dataset_dir: Path | str) -> VerificationResult:
    """
    Convenience function to verify a dataset.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        VerificationResult with all checks
    """
    verifier = LabelVerifier(dataset_dir)
    return verifier.verify_all()
