"""Tests for HuggingFace Hub uploader."""

import pytest
from pathlib import Path

from dataset_gen.hub.uploader import HubConfig


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
