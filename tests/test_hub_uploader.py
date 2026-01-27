"""Tests for HuggingFace Hub uploader."""

import json
import pytest
import pandas as pd
from pathlib import Path

from dataset_gen.hub.uploader import HubConfig, DatasetUploader


class TestHubConfig:
    """Tests for HubConfig dataclass."""

    def test_use_tar_shards_default_true(self, tmp_path):
        """use_tar_shards defaults to True."""
        config = HubConfig(
            dataset_dir=tmp_path,
            repo_id="test/repo",
        )
        assert config.use_tar_shards is True

    def test_use_tar_shards_can_be_disabled(self, tmp_path):
        """use_tar_shards can be set to False."""
        config = HubConfig(
            dataset_dir=tmp_path,
            repo_id="test/repo",
            use_tar_shards=False,
        )
        assert config.use_tar_shards is False

    def test_tar_shard_size_default(self, tmp_path):
        """tar_shard_size_bytes has sensible default."""
        config = HubConfig(
            dataset_dir=tmp_path,
            repo_id="test/repo",
        )
        assert config.tar_shard_size_bytes == 1_000_000_000  # 1GB


class TestDatasetUploaderHelpers:
    """Tests for DatasetUploader helper methods."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal dataset structure for testing."""
        # Create labels directory
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Create samples parquet
        samples_df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3", "s4"],
            "profile_id": ["p1", "p1", "p2", "p3"],
            "audio_path": ["audio/a1.flac", "audio/a2.flac", "audio/a3.flac", "audio/a4.flac"],
            "midi_path": ["midi/m1.mid", "midi/m2.mid", "midi/m3.mid", "midi/m4.mid"],
        })
        samples_df.to_parquet(labels_dir / "samples.parquet")

        # Create exercises parquet (minimal)
        exercises_df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3", "s4"],
            "overall_score": [80.0, 85.0, 75.0, 90.0],
        })
        exercises_df.to_parquet(labels_dir / "exercises.parquet")

        # Create splits
        splits = {
            "train_profile_ids": ["p1"],
            "val_profile_ids": ["p2"],
            "test_profile_ids": ["p3"],
        }
        with open(tmp_path / "splits.json", "w") as f:
            json.dump(splits, f)

        return tmp_path

    def test_get_filenames_by_split_audio(self, sample_dataset):
        """_get_filenames_by_split returns audio files grouped by split."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        result = uploader._get_filenames_by_split("audio")

        assert "train" in result
        assert "validation" in result
        assert "test" in result

        # p1 has 2 samples -> train
        assert set(result["train"]) == {"a1.flac", "a2.flac"}
        # p2 has 1 sample -> validation
        assert set(result["validation"]) == {"a3.flac"}
        # p3 has 1 sample -> test
        assert set(result["test"]) == {"a4.flac"}
