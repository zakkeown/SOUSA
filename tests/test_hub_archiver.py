"""Tests for TAR sharding archiver."""

import pytest
from dataset_gen.hub.archiver import ShardInfo


class TestShardInfo:
    """Tests for ShardInfo dataclass."""

    def test_shard_info_creation(self):
        """ShardInfo stores shard name and filename."""
        info = ShardInfo(shard_name="train-00001.tar", filename="sample_001.flac")
        assert info.shard_name == "train-00001.tar"
        assert info.filename == "sample_001.flac"
