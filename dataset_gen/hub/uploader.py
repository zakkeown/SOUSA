"""
HuggingFace Hub upload utilities for SOUSA dataset.

This module handles:
- Restructuring the local dataset for HuggingFace Hub format
- Creating consolidated parquet files with audio/midi paths
- Uploading to HuggingFace Hub
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HubConfig:
    """Configuration for HuggingFace Hub upload."""

    # Source dataset directory
    dataset_dir: Path

    # Hub settings
    repo_id: str
    private: bool = False
    token: str | None = None

    # Content options
    include_audio: bool = True
    include_midi: bool = True

    # Output staging directory (where HF-format files are prepared)
    staging_dir: Path | None = None

    # Shard settings for large datasets
    max_shard_size: str = "500MB"

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        if self.staging_dir is None:
            self.staging_dir = self.dataset_dir / "hf_staging"
        else:
            self.staging_dir = Path(self.staging_dir)


@dataclass
class UploadStats:
    """Statistics from the upload process."""

    total_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    audio_files: int = 0
    midi_files: int = 0
    total_size_bytes: int = 0

    def summary(self) -> str:
        """Return human-readable summary."""
        size_gb = self.total_size_bytes / (1024**3)
        return (
            f"Upload Statistics:\n"
            f"  Total samples: {self.total_samples:,}\n"
            f"  Train: {self.train_samples:,}\n"
            f"  Validation: {self.val_samples:,}\n"
            f"  Test: {self.test_samples:,}\n"
            f"  Audio files: {self.audio_files:,}\n"
            f"  MIDI files: {self.midi_files:,}\n"
            f"  Total size: {size_gb:.2f} GB"
        )


class DatasetUploader:
    """Upload SOUSA dataset to HuggingFace Hub."""

    def __init__(self, config: HubConfig):
        """Initialize uploader with configuration."""
        self.config = config
        self.stats = UploadStats()

    def prepare(self) -> Path:
        """
        Prepare dataset in HuggingFace Hub format.

        Creates staging directory with:
        - data/ containing split parquet files
        - audio/ with audio files (if include_audio)
        - midi/ with MIDI files (if include_midi)
        - README.md dataset card

        Returns:
            Path to staging directory ready for upload
        """
        logger.info(f"Preparing HuggingFace format in {self.config.staging_dir}")

        # Create staging directory
        staging = self.config.staging_dir
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        (staging / "data").mkdir()

        # Load source data
        samples_df = pd.read_parquet(self.config.dataset_dir / "labels" / "samples.parquet")
        exercises_df = pd.read_parquet(self.config.dataset_dir / "labels" / "exercises.parquet")

        # Load splits
        with open(self.config.dataset_dir / "splits.json") as f:
            splits = json.load(f)

        # Merge samples with exercise scores
        merged_df = self._merge_dataframes(samples_df, exercises_df)

        # Add audio/midi paths for HuggingFace
        if self.config.include_audio:
            merged_df["audio"] = merged_df["audio_path"].apply(
                lambda p: f"audio/{Path(p).name}" if p else None
            )
        if self.config.include_midi:
            merged_df["midi"] = merged_df["midi_path"].apply(
                lambda p: f"midi/{Path(p).name}" if p else None
            )

        # Create split assignment column
        merged_df["split"] = merged_df["profile_id"].apply(lambda pid: self._get_split(pid, splits))

        # Write split parquet files
        for split_name in ["train", "validation", "test"]:
            split_key = split_name if split_name != "validation" else "val"
            split_df = merged_df[merged_df["split"] == split_key].drop(columns=["split"])

            output_path = staging / "data" / f"{split_name}-00000-of-00001.parquet"
            split_df.to_parquet(output_path, index=False)

            if split_name == "train":
                self.stats.train_samples = len(split_df)
            elif split_name == "validation":
                self.stats.val_samples = len(split_df)
            else:
                self.stats.test_samples = len(split_df)

            logger.info(f"Wrote {len(split_df)} samples to {output_path.name}")

        self.stats.total_samples = (
            self.stats.train_samples + self.stats.val_samples + self.stats.test_samples
        )

        # Copy audio files
        if self.config.include_audio:
            self._copy_media_files("audio", "flac")

        # Copy MIDI files
        if self.config.include_midi:
            self._copy_media_files("midi", "mid")

        # Copy README.md (dataset card)
        readme_src = self.config.dataset_dir / "README.md"
        if readme_src.exists():
            shutil.copy(readme_src, staging / "README.md")
        else:
            logger.warning("No README.md found in dataset directory")

        # Calculate total size
        self.stats.total_size_bytes = sum(
            f.stat().st_size for f in staging.rglob("*") if f.is_file()
        )

        logger.info(f"Staging complete: {self.stats.summary()}")
        return staging

    def _merge_dataframes(
        self, samples_df: pd.DataFrame, exercises_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge samples and exercises dataframes."""
        # Exercises has the scores we want to include
        score_columns = [
            "sample_id",
            "timing_accuracy",
            "timing_consistency",
            "tempo_stability",
            "subdivision_evenness",
            "velocity_control",
            "accent_differentiation",
            "accent_accuracy",
            "hand_balance",
            "weak_hand_index",
            "flam_quality",
            "diddle_quality",
            "roll_sustain",
            "groove_feel_proxy",
            "overall_score",
        ]

        exercises_subset = exercises_df[[c for c in score_columns if c in exercises_df.columns]]

        merged = samples_df.merge(exercises_subset, on="sample_id", how="left")
        return merged

    def _get_split(self, profile_id: str, splits: dict) -> str:
        """Determine which split a profile belongs to."""
        if profile_id in splits.get("train_profile_ids", []):
            return "train"
        elif profile_id in splits.get("val_profile_ids", []):
            return "val"
        elif profile_id in splits.get("test_profile_ids", []):
            return "test"
        return "train"  # Default to train

    def _copy_media_files(self, subdir: str, extension: str) -> None:
        """Copy media files to staging directory."""
        src_dir = self.config.dataset_dir / subdir
        dst_dir = self.config.staging_dir / subdir

        if not src_dir.exists():
            logger.warning(f"Source directory {src_dir} does not exist")
            return

        dst_dir.mkdir(exist_ok=True)
        count = 0

        for src_file in src_dir.glob(f"*.{extension}"):
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            count += 1

        if subdir == "audio":
            self.stats.audio_files = count
        elif subdir == "midi":
            self.stats.midi_files = count

        logger.info(f"Copied {count} {extension} files to {dst_dir}")

    def upload(self, dry_run: bool = False) -> str | None:
        """
        Upload prepared dataset to HuggingFace Hub.

        Args:
            dry_run: If True, prepare but don't actually upload

        Returns:
            URL to the uploaded dataset, or None if dry_run
        """
        # Ensure staging is prepared
        staging_dir = self.config.staging_dir
        if not staging_dir.exists() or not (staging_dir / "data").exists():
            staging_dir = self.prepare()

        if dry_run:
            logger.info(f"DRY RUN: Would upload {staging_dir} to {self.config.repo_id}")
            return None

        try:
            from huggingface_hub import HfApi, upload_folder
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for upload. "
                "Install with: pip install 'rudimentary[hub]'"
            )

        api = HfApi(token=self.config.token)

        # Create repository if it doesn't exist
        logger.info(f"Creating/verifying repository: {self.config.repo_id}")
        api.create_repo(
            repo_id=self.config.repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=self.config.private,
        )

        # Upload the entire staging directory
        logger.info(f"Uploading dataset to {self.config.repo_id}...")
        upload_folder(
            folder_path=str(staging_dir),
            repo_id=self.config.repo_id,
            repo_type="dataset",
            commit_message="Upload SOUSA dataset",
            token=self.config.token,
        )

        url = f"https://huggingface.co/datasets/{self.config.repo_id}"
        logger.info(f"Upload complete: {url}")
        return url


def prepare_hf_structure(
    dataset_dir: Path | str,
    staging_dir: Path | str | None = None,
    include_audio: bool = True,
    include_midi: bool = True,
) -> Path:
    """
    Convenience function to prepare HuggingFace format.

    Args:
        dataset_dir: Path to generated dataset
        staging_dir: Output directory for HF format (default: dataset_dir/hf_staging)
        include_audio: Include audio files
        include_midi: Include MIDI files

    Returns:
        Path to staging directory
    """
    config = HubConfig(
        dataset_dir=Path(dataset_dir),
        repo_id="",  # Not needed for prepare-only
        staging_dir=Path(staging_dir) if staging_dir else None,
        include_audio=include_audio,
        include_midi=include_midi,
    )
    uploader = DatasetUploader(config)
    return uploader.prepare()


def push_to_hub(
    dataset_dir: Path | str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    include_audio: bool = True,
    include_midi: bool = True,
    dry_run: bool = False,
) -> str | None:
    """
    Convenience function to upload dataset to HuggingFace Hub.

    Args:
        dataset_dir: Path to generated dataset
        repo_id: HuggingFace repository ID (e.g., "username/sousa")
        token: HuggingFace API token (uses cached token if None)
        private: Make repository private
        include_audio: Include audio files
        include_midi: Include MIDI files
        dry_run: Prepare but don't upload

    Returns:
        URL to uploaded dataset, or None if dry_run
    """
    config = HubConfig(
        dataset_dir=Path(dataset_dir),
        repo_id=repo_id,
        token=token,
        private=private,
        include_audio=include_audio,
        include_midi=include_midi,
    )
    uploader = DatasetUploader(config)
    return uploader.upload(dry_run=dry_run)
