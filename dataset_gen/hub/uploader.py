"""
HuggingFace Hub upload utilities for SOUSA dataset.

This module handles:
- Building HuggingFace DatasetDict objects with embedded media
- Uploading Parquet-native datasets to HuggingFace Hub
- Managing multi-config datasets (audio, midi_only, labels_only)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover
    HfApi = None  # type: ignore[assignment,misc]

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


class DatasetUploader:
    """Upload SOUSA dataset to HuggingFace Hub."""

    def __init__(self, config: HubConfig):
        self.config = config

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

    def purge_repo(self, api=None) -> None:
        """Delete all files from the hub repo except .gitattributes.

        This is used before uploading a new dataset structure (e.g., Parquet
        migration) to remove stale files without deleting the entire repo.

        Args:
            api: Optional HfApi instance.  One is created if not provided.
        """
        if api is None:
            try:
                from huggingface_hub import HfApi
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for purge. "
                    "Install with: pip install 'sousa[hub]'"
                )
            api = HfApi(token=self.config.token)

        from huggingface_hub import CommitOperationDelete

        logger.info(f"Listing files in {self.config.repo_id} for purge...")
        items = api.list_repo_tree(
            repo_id=self.config.repo_id,
            repo_type="dataset",
            recursive=True,
        )

        paths_to_delete = [
            item.rfilename
            for item in items
            if hasattr(item, "rfilename") and item.rfilename != ".gitattributes"
        ]

        if not paths_to_delete:
            logger.info("No files to purge (repo is empty or only .gitattributes)")
            return

        logger.info(f"Purging {len(paths_to_delete)} files from {self.config.repo_id}")
        operations = [CommitOperationDelete(path_in_repo=p) for p in paths_to_delete]
        api.create_commit(
            repo_id=self.config.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="chore: purge for Parquet migration",
        )
        logger.info("Purge complete")

    def build_dataset_dict(self, config_name: str):
        """Build a HuggingFace DatasetDict with embedded binary media data.

        Loads samples and exercises parquet files, merges them, assigns splits,
        and constructs a DatasetDict with the appropriate columns for the given
        config type.

        Args:
            config_name: One of "labels_only", "midi_only", or "audio".
                - "labels_only": metadata + scores only, no media columns.
                - "midi_only": metadata + scores + midi column (raw bytes).
                - "audio": metadata + scores + audio (Audio feature) + midi (raw bytes).

        Returns:
            DatasetDict with "train", "validation", and "test" splits.
        """
        from datasets import Audio, Dataset, DatasetDict

        # Load source data
        samples_df = pd.read_parquet(self.config.dataset_dir / "labels" / "samples.parquet")
        exercises_df = pd.read_parquet(self.config.dataset_dir / "labels" / "exercises.parquet")

        with open(self.config.dataset_dir / "splits.json") as f:
            splits = json.load(f)

        # Merge samples with exercise scores
        merged_df = self._merge_dataframes(samples_df, exercises_df)

        # Assign splits
        merged_df["split"] = merged_df["profile_id"].apply(lambda pid: self._get_split(pid, splits))

        # Drop internal filesystem path columns
        internal_cols = [c for c in ["audio_path", "midi_path"] if c in merged_df.columns]

        # For media configs, we need the paths before dropping them
        if config_name in ("midi_only", "audio"):
            midi_paths = merged_df["midi_path"].copy() if "midi_path" in merged_df.columns else None
        else:
            midi_paths = None

        if config_name == "audio":
            audio_paths = (
                merged_df["audio_path"].copy() if "audio_path" in merged_df.columns else None
            )
        else:
            audio_paths = None

        merged_df = merged_df.drop(columns=internal_cols, errors="ignore")

        # Build per-split datasets
        split_datasets = {}
        split_mapping = {"train": "train", "val": "validation", "test": "test"}

        for split_key, split_name in split_mapping.items():
            mask = merged_df["split"] == split_key
            split_df = merged_df[mask].drop(columns=["split"]).reset_index(drop=True)

            # Add media columns based on config_name
            if config_name in ("midi_only", "audio") and midi_paths is not None:
                split_midi_paths = midi_paths[mask].reset_index(drop=True)
                split_df["midi"] = split_midi_paths.apply(
                    lambda p: (
                        (self.config.dataset_dir / p).read_bytes() if pd.notna(p) and p else None
                    )
                )

            if config_name == "audio" and audio_paths is not None:
                split_audio_paths = audio_paths[mask].reset_index(drop=True)
                split_df["audio"] = split_audio_paths.apply(
                    lambda p: str(self.config.dataset_dir / p) if pd.notna(p) and p else None
                )

            # Convert to HF Dataset using from_dict to avoid large_string
            # arrow type that pandas produces (Audio cast requires string, not large_string)
            ds = Dataset.from_dict(split_df.to_dict(orient="list"))

            # Cast audio column to Audio feature type
            if config_name == "audio" and "audio" in ds.column_names:
                ds = ds.cast_column("audio", Audio())

            split_datasets[split_name] = ds

        return DatasetDict(split_datasets)

    def upload(self, dry_run: bool = False, purge: bool = False) -> str | None:
        """
        Upload dataset to HuggingFace Hub using Parquet-native approach.

        For each configured config, builds a DatasetDict via build_dataset_dict()
        and pushes it with push_to_hub(). Also uploads auxiliary label tables
        (strokes.parquet, measures.parquet) and the README.

        Args:
            dry_run: If True, build DatasetDicts but don't push to Hub
            purge: If True, purge existing repo files before uploading

        Returns:
            URL to the uploaded dataset, or None if dry_run
        """
        if HfApi is None:
            raise ImportError(
                "huggingface_hub is required for upload. " "Install with: pip install 'sousa[hub]'"
            )

        api = HfApi(token=self.config.token)

        if not dry_run:
            api.create_repo(
                repo_id=self.config.repo_id,
                repo_type="dataset",
                exist_ok=True,
                private=self.config.private,
            )
            if purge:
                self.purge_repo(api=api)

        for config_name in self.config.configs:
            logger.info(f"Building DatasetDict for config '{config_name}'...")
            dd = self.build_dataset_dict(config_name)

            for split_name, ds in dd.items():
                logger.info(f"  {config_name}/{split_name}: {ds.num_rows} rows")

            if dry_run:
                logger.info(f"DRY RUN: would push config '{config_name}'")
                continue

            dd.push_to_hub(
                self.config.repo_id,
                config_name=config_name,
                max_shard_size=self.config.max_shard_size,
                token=self.config.token,
                private=self.config.private,
            )

        if not dry_run:
            self._upload_auxiliary(api)
            # Upload README
            readme = self.config.dataset_dir / "README.md"
            if readme.exists():
                api.upload_file(
                    path_or_fileobj=str(readme),
                    path_in_repo="README.md",
                    repo_id=self.config.repo_id,
                    repo_type="dataset",
                )

        url = f"https://huggingface.co/datasets/{self.config.repo_id}"
        return None if dry_run else url

    def _upload_auxiliary(self, api) -> None:
        """Upload auxiliary label tables to the hub.

        Uploads strokes.parquet and measures.parquet from the labels directory
        to the ``auxiliary/`` directory on the hub.

        Args:
            api: HfApi instance to use for uploading.
        """
        labels_dir = self.config.dataset_dir / "labels"
        for filename in ["strokes.parquet", "measures.parquet"]:
            src = labels_dir / filename
            if src.exists():
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=f"auxiliary/{filename}",
                    repo_id=self.config.repo_id,
                    repo_type="dataset",
                )


def push_to_hub(
    dataset_dir: Path | str,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    configs: list[str] | None = None,
    max_shard_size: str = "1GB",
    purge: bool = False,
    dry_run: bool = False,
) -> str | None:
    """
    Upload dataset to HuggingFace Hub as Parquet shards.

    Args:
        dataset_dir: Path to generated dataset
        repo_id: HuggingFace repository ID (e.g., "username/sousa")
        token: HuggingFace API token (uses cached token if None)
        private: Make repository private
        configs: Which configs to upload (default: all three)
        max_shard_size: Max Parquet shard size (default: "1GB")
        purge: Delete existing repo files before upload
        dry_run: Prepare but don't upload

    Returns:
        URL to uploaded dataset, or None if dry_run
    """
    config = HubConfig(
        dataset_dir=Path(dataset_dir),
        repo_id=repo_id,
        token=token,
        private=private,
        configs=configs,
        max_shard_size=max_shard_size,
    )
    uploader = DatasetUploader(config)
    return uploader.upload(dry_run=dry_run, purge=purge)
