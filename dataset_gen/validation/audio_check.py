"""
Audio quality validation for generated dataset.

This module provides checks to verify audio files are valid and
meet quality standards before publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import random

import numpy as np
import soundfile as sf
import pandas as pd

from dataset_gen.pipeline.storage import ParquetReader

logger = logging.getLogger(__name__)


@dataclass
class AudioCheckResult:
    """Result of a single audio quality check."""

    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AudioValidationReport:
    """Complete audio validation report."""

    checks: list[AudioCheckResult] = field(default_factory=list)
    samples_checked: int = 0
    samples_passed: int = 0
    samples_failed: int = 0
    failed_sample_ids: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Audio Validation: {self.samples_passed}/{self.samples_checked} samples passed",
            "",
        ]
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{status}] {check.name}: {check.message}")
        if self.failed_sample_ids:
            lines.append("")
            lines.append(f"  Failed samples: {', '.join(self.failed_sample_ids[:5])}")
            if len(self.failed_sample_ids) > 5:
                lines.append(f"  ... and {len(self.failed_sample_ids) - 5} more")
        return "\n".join(lines)


class AudioQualityChecker:
    """
    Check audio quality for a generated dataset.

    Performs the following checks:
    - File integrity (can be decoded without errors)
    - Silence detection (not mostly silent)
    - Clipping detection (not excessively clipped)
    - Duration verification (matches metadata)
    - RMS level consistency
    """

    # Default thresholds
    SILENCE_THRESHOLD_DB = -60  # dB below which audio is considered silent
    SILENCE_RATIO_MAX = 0.8  # Max ratio of silent frames allowed
    CLIPPING_THRESHOLD = 0.99  # Sample value considered clipping
    CLIPPING_RATIO_MAX = 0.001  # Max ratio of clipped samples (0.1%)
    DURATION_TOLERANCE_SEC = 0.5  # Max deviation from metadata
    RMS_MIN_DB = -40  # Minimum acceptable RMS level
    RMS_MAX_DB = -6  # Maximum acceptable RMS level (before clipping likely)

    def __init__(
        self,
        dataset_dir: Path | str,
        sample_size: int = 100,
        seed: int = 42,
    ):
        """
        Initialize audio checker.

        Args:
            dataset_dir: Path to dataset directory
            sample_size: Number of audio files to check (0 = all)
            seed: Random seed for sampling
        """
        self.dataset_dir = Path(dataset_dir)
        self.audio_dir = self.dataset_dir / "audio"
        self.sample_size = sample_size
        self.seed = seed
        self.reader = ParquetReader(dataset_dir)

    def check_all(self) -> AudioValidationReport:
        """Run all audio quality checks."""
        report = AudioValidationReport()

        # Get list of audio files to check
        samples_df = self.reader.load_samples()
        audio_files = self._get_audio_files(samples_df)

        if not audio_files:
            report.checks.append(
                AudioCheckResult(
                    name="audio_files_exist",
                    passed=False,
                    message="No audio files found",
                )
            )
            return report

        # Sample files if needed
        if self.sample_size > 0 and len(audio_files) > self.sample_size:
            random.seed(self.seed)
            audio_files = random.sample(audio_files, self.sample_size)

        report.samples_checked = len(audio_files)

        # Check each file
        silent_files = []
        clipped_files = []
        corrupted_files = []
        duration_mismatches = []
        rms_outliers = []
        rms_values = []

        for sample_id, audio_path in audio_files:
            # Get expected duration from metadata
            sample_row = samples_df[samples_df["sample_id"] == sample_id]
            expected_duration = (
                sample_row["duration_sec"].values[0] if len(sample_row) > 0 else None
            )

            result = self._check_single_file(audio_path, expected_duration)

            if result["corrupted"]:
                corrupted_files.append(sample_id)
                continue

            if result["is_silent"]:
                silent_files.append(sample_id)
            if result["is_clipped"]:
                clipped_files.append(sample_id)
            if result["duration_mismatch"]:
                duration_mismatches.append(sample_id)
            if result["rms_outlier"]:
                rms_outliers.append(sample_id)
            if result["rms_db"] is not None:
                rms_values.append(result["rms_db"])

        # Compute statistics
        all_failed = set(silent_files + clipped_files + corrupted_files + duration_mismatches)
        report.samples_passed = report.samples_checked - len(all_failed)
        report.samples_failed = len(all_failed)
        report.failed_sample_ids = list(all_failed)

        # Add check results
        report.checks.append(
            AudioCheckResult(
                name="file_integrity",
                passed=len(corrupted_files) == 0,
                message=(
                    f"{len(corrupted_files)} corrupted files"
                    if corrupted_files
                    else "All files readable"
                ),
                details={"corrupted_files": corrupted_files[:10]},
            )
        )

        report.checks.append(
            AudioCheckResult(
                name="silence_detection",
                passed=len(silent_files) == 0,
                message=(
                    f"{len(silent_files)} files >80% silent" if silent_files else "No silent files"
                ),
                details={"silent_files": silent_files[:10]},
            )
        )

        report.checks.append(
            AudioCheckResult(
                name="clipping_detection",
                passed=len(clipped_files) == 0,
                message=(
                    f"{len(clipped_files)} files with excessive clipping"
                    if clipped_files
                    else "No clipped files"
                ),
                details={"clipped_files": clipped_files[:10]},
            )
        )

        report.checks.append(
            AudioCheckResult(
                name="duration_verification",
                passed=len(duration_mismatches) == 0,
                message=(
                    f"{len(duration_mismatches)} files with duration mismatch"
                    if duration_mismatches
                    else "All durations match metadata"
                ),
                details={"duration_mismatches": duration_mismatches[:10]},
            )
        )

        # RMS consistency check
        if rms_values:
            rms_mean = np.mean(rms_values)
            rms_std = np.std(rms_values)
            report.checks.append(
                AudioCheckResult(
                    name="rms_consistency",
                    passed=len(rms_outliers) / len(audio_files) < 0.05,  # <5% outliers
                    message=f"RMS: {rms_mean:.1f} +/- {rms_std:.1f} dB, {len(rms_outliers)} outliers",
                    details={
                        "rms_mean_db": rms_mean,
                        "rms_std_db": rms_std,
                        "outlier_files": rms_outliers[:10],
                    },
                )
            )

        return report

    def _get_audio_files(self, samples_df: pd.DataFrame) -> list[tuple[str, Path]]:
        """Get list of (sample_id, audio_path) tuples."""
        files = []
        for _, row in samples_df.iterrows():
            sample_id = row["sample_id"]
            audio_path = self.audio_dir / f"{sample_id}.flac"
            if audio_path.exists():
                files.append((sample_id, audio_path))
        return files

    def _check_single_file(
        self,
        audio_path: Path,
        expected_duration: float | None = None,
    ) -> dict:
        """
        Check a single audio file.

        Returns dict with check results.
        """
        result = {
            "corrupted": False,
            "is_silent": False,
            "is_clipped": False,
            "duration_mismatch": False,
            "rms_outlier": False,
            "rms_db": None,
        }

        try:
            audio, sr = sf.read(str(audio_path))
        except Exception as e:
            logger.warning(f"Failed to read {audio_path}: {e}")
            result["corrupted"] = True
            return result

        # Handle stereo by taking first channel
        if len(audio.shape) > 1:
            audio = audio[:, 0]

        # Silence check
        silence_ratio = self._compute_silence_ratio(audio)
        if silence_ratio > self.SILENCE_RATIO_MAX:
            result["is_silent"] = True

        # Clipping check
        clip_ratio = self._compute_clip_ratio(audio)
        if clip_ratio > self.CLIPPING_RATIO_MAX:
            result["is_clipped"] = True

        # Duration check
        actual_duration = len(audio) / sr
        if expected_duration is not None:
            if abs(actual_duration - expected_duration) > self.DURATION_TOLERANCE_SEC:
                result["duration_mismatch"] = True

        # RMS level check
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        result["rms_db"] = rms_db

        if rms_db < self.RMS_MIN_DB or rms_db > self.RMS_MAX_DB:
            result["rms_outlier"] = True

        return result

    def _compute_silence_ratio(self, audio: np.ndarray) -> float:
        """Compute ratio of silent frames."""
        # Convert threshold to linear amplitude
        threshold_linear = 10 ** (self.SILENCE_THRESHOLD_DB / 20)
        silent_frames = np.abs(audio) < threshold_linear
        return np.mean(silent_frames)

    def _compute_clip_ratio(self, audio: np.ndarray) -> float:
        """Compute ratio of clipped samples."""
        clipped = np.abs(audio) >= self.CLIPPING_THRESHOLD
        return np.mean(clipped)


def check_audio_quality(
    dataset_dir: Path | str,
    sample_size: int = 100,
    seed: int = 42,
) -> AudioValidationReport:
    """
    Convenience function to check audio quality.

    Args:
        dataset_dir: Path to dataset directory
        sample_size: Number of files to check (0 = all)
        seed: Random seed

    Returns:
        AudioValidationReport with check results
    """
    checker = AudioQualityChecker(dataset_dir, sample_size, seed)
    return checker.check_all()


def main():
    """CLI entry point for audio quality checks."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Check audio quality for SOUSA dataset")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="output/dataset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of files to check (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    audio_dir = dataset_dir / "audio"
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Checking audio quality in {dataset_dir}...")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'all'}")
    print()

    report = check_audio_quality(dataset_dir, args.sample_size, args.seed)
    print(report.summary())

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
