"""Tests for TAR sharding archiver."""

from dataset_gen.hub.archiver import ShardInfo, create_sharded_archives


class TestShardInfo:
    """Tests for ShardInfo dataclass."""

    def test_shard_info_creation(self):
        """ShardInfo stores shard name and filename."""
        info = ShardInfo(shard_name="train-00001.tar", filename="sample_001.flac")
        assert info.shard_name == "train-00001.tar"
        assert info.filename == "sample_001.flac"


class TestCreateShardedArchives:
    """Tests for create_sharded_archives function."""

    def test_creates_single_shard_for_small_files(self, tmp_path):
        """Small files all go into one shard."""
        # Create source files
        src_dir = tmp_path / "audio"
        src_dir.mkdir()
        for i in range(3):
            (src_dir / f"sample_{i:03d}.flac").write_bytes(b"x" * 100)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        filenames_by_split = {"train": [f"sample_{i:03d}.flac" for i in range(3)]}

        result = create_sharded_archives(
            source_dir=src_dir,
            output_dir=out_dir,
            filenames_by_split=filenames_by_split,
            target_shard_size_bytes=10000,  # 10KB - larger than our test files
            extension="flac",
        )

        # Should create one shard
        assert (out_dir / "train-00000.tar").exists()
        assert not (out_dir / "train-00001.tar").exists()

        # All files mapped to that shard
        for i in range(3):
            filename = f"sample_{i:03d}.flac"
            assert filename in result
            assert result[filename].shard_name == "train-00000.tar"
            assert result[filename].filename == filename
