"""
HuggingFace Hub upload utilities for SOUSA dataset.

This module handles:
- Restructuring the local dataset for HuggingFace Hub format
- Creating consolidated parquet files with audio/midi paths
- Organizing media files into rudiment subdirectories
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

    # Which configs to upload: "audio", "midi_only", "labels_only"
    configs: list[str] | None = None

    # Shard settings for large datasets
    max_shard_size: str = "1GB"

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        if self.configs is None:
            self.configs = ["audio", "midi_only", "labels_only"]


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

    def prepare(self, skip_media_copy: bool = False, use_symlinks: bool = True) -> Path:
        """
        Prepare dataset in HuggingFace Hub format.

        Creates staging directory with:
        - data/ containing split parquet files
        - audio/{rudiment_slug}/ with audio files organized by rudiment
        - midi/{rudiment_slug}/ with MIDI files organized by rudiment
        - README.md dataset card

        Args:
            skip_media_copy: If True, skip copying audio/MIDI files (for dry-run)
            use_symlinks: If True, use symlinks instead of copies (saves disk space)

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

        # Add rudiment-organized path columns
        if self.config.include_audio:
            merged_df["audio"] = merged_df.apply(
                lambda row: (
                    f"audio/{row['rudiment_slug']}/{Path(row['audio_path']).name}"
                    if pd.notna(row.get("audio_path")) and row["audio_path"]
                    else None
                ),
                axis=1,
            )
        if self.config.include_midi:
            merged_df["midi"] = merged_df.apply(
                lambda row: (
                    f"midi/{row['rudiment_slug']}/{Path(row['midi_path']).name}"
                    if pd.notna(row.get("midi_path")) and row["midi_path"]
                    else None
                ),
                axis=1,
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

        # Copy/link media files organized by rudiment
        if self.config.include_audio:
            if skip_media_copy:
                self._count_media_files("audio", "flac")
            else:
                self._copy_media_by_rudiment(merged_df, "audio", "flac", use_symlinks=use_symlinks)

        if self.config.include_midi:
            if skip_media_copy:
                self._count_media_files("midi", "mid")
            else:
                self._copy_media_by_rudiment(merged_df, "midi", "mid", use_symlinks=use_symlinks)

        # Copy README.md (dataset card)
        readme_src = self.config.dataset_dir / "README.md"
        if readme_src.exists():
            shutil.copy(readme_src, staging / "README.md")
        else:
            logger.warning("No README.md found in dataset directory")

        # Calculate total size
        if skip_media_copy:
            # Calculate from source directories when media wasn't copied
            self.stats.total_size_bytes = sum(
                f.stat().st_size for f in staging.rglob("*") if f.is_file()
            )
            if self.config.include_audio:
                audio_dir = self.config.dataset_dir / "audio"
                if audio_dir.exists():
                    self.stats.total_size_bytes += sum(
                        f.stat().st_size for f in audio_dir.glob("*.flac")
                    )
            if self.config.include_midi:
                midi_dir = self.config.dataset_dir / "midi"
                if midi_dir.exists():
                    self.stats.total_size_bytes += sum(
                        f.stat().st_size for f in midi_dir.glob("*.mid")
                    )
        else:
            # Calculate from staging directory (symlinks resolve to real sizes)
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

    def _count_media_files(self, subdir: str, extension: str) -> None:
        """Count media files without copying (for dry-run)."""
        src_dir = self.config.dataset_dir / subdir

        if not src_dir.exists():
            logger.warning(f"Source directory {src_dir} does not exist")
            return

        count = sum(1 for _ in src_dir.rglob(f"*.{extension}"))

        if subdir == "audio":
            self.stats.audio_files = count
        elif subdir == "midi":
            self.stats.midi_files = count

        # Estimate size without copying
        total_size = sum(f.stat().st_size for f in src_dir.rglob(f"*.{extension}"))
        logger.info(f"Found {count} {extension} files ({total_size / (1024**3):.2f} GB)")

    def _copy_media_by_rudiment(
        self,
        merged_df: pd.DataFrame,
        subdir: str,
        extension: str,
        use_symlinks: bool = True,
    ) -> None:
        """Copy or symlink media files into rudiment subdirectories.

        Args:
            merged_df: DataFrame with rudiment_slug and path columns
            subdir: Media subdirectory ("audio" or "midi")
            extension: File extension (e.g., "flac", "mid")
            use_symlinks: If True, use symlinks instead of copies
        """
        src_dir = self.config.dataset_dir / subdir
        dst_base = self.config.staging_dir / subdir

        if not src_dir.exists():
            logger.warning(f"Source directory {src_dir} does not exist")
            return

        path_col = f"{subdir}_path"
        count = 0

        for _, row in merged_df.iterrows():
            path_val = row.get(path_col)
            if pd.isna(path_val) or not path_val:
                continue

            slug = row["rudiment_slug"]
            filename = Path(path_val).name
            src_file = src_dir / slug / filename

            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
                continue

            rudiment_dir = dst_base / slug
            rudiment_dir.mkdir(parents=True, exist_ok=True)

            dst_file = rudiment_dir / filename
            if use_symlinks:
                if dst_file.exists() or dst_file.is_symlink():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
            else:
                shutil.copy2(src_file, dst_file)
            count += 1

        if subdir == "audio":
            self.stats.audio_files = count
        elif subdir == "midi":
            self.stats.midi_files = count

        action = "Linked" if use_symlinks else "Copied"
        logger.info(f"{action} {count} {extension} files into {dst_base} by rudiment")

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
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for upload. Install with: pip install 'sousa[hub]'"
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

        # Upload the entire staging directory using upload_large_folder
        # which is optimized for repos with many files (120k+ in our case)
        logger.info(f"Uploading dataset to {self.config.repo_id}...")
        logger.info("Using upload_large_folder for better performance with many files")
        api.upload_large_folder(
            folder_path=str(staging_dir),
            repo_id=self.config.repo_id,
            repo_type="dataset",
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
