"""Tests for TAR sharding archiver."""

from dataset_gen.hub.archiver import ShardInfo, create_sharded_archives


def test_archiver_exported_from_hub_module():
    """Archiver functions are exported from hub module."""
    from dataset_gen.hub import ShardInfo, create_sharded_archives

    assert ShardInfo is not None
    assert create_sharded_archives is not None


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

    def test_creates_multiple_shards_when_size_exceeded(self, tmp_path):
        """Files are split across shards when target size exceeded."""
        src_dir = tmp_path / "audio"
        src_dir.mkdir()

        # Create 5 files of 100 bytes each
        for i in range(5):
            (src_dir / f"sample_{i:03d}.flac").write_bytes(b"x" * 100)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        filenames_by_split = {"train": [f"sample_{i:03d}.flac" for i in range(5)]}

        # Target 250 bytes per shard - should fit 2 files each (with TAR overhead, may vary)
        result = create_sharded_archives(
            source_dir=src_dir,
            output_dir=out_dir,
            filenames_by_split=filenames_by_split,
            target_shard_size_bytes=250,
            extension="flac",
        )

        # Should have multiple shards
        shards = list(out_dir.glob("train-*.tar"))
        assert len(shards) >= 2, f"Expected multiple shards, got {len(shards)}"

        # All files should be mapped
        assert len(result) == 5

    def test_separate_shards_per_split(self, tmp_path):
        """Each split gets its own shard files."""
        src_dir = tmp_path / "audio"
        src_dir.mkdir()

        # Create files for train and validation
        for i in range(3):
            (src_dir / f"train_{i:03d}.flac").write_bytes(b"x" * 100)
            (src_dir / f"val_{i:03d}.flac").write_bytes(b"x" * 100)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        filenames_by_split = {
            "train": [f"train_{i:03d}.flac" for i in range(3)],
            "validation": [f"val_{i:03d}.flac" for i in range(3)],
        }

        result = create_sharded_archives(
            source_dir=src_dir,
            output_dir=out_dir,
            filenames_by_split=filenames_by_split,
            target_shard_size_bytes=10000,
            extension="flac",
        )

        # Should have separate shards for each split
        assert (out_dir / "train-00000.tar").exists()
        assert (out_dir / "validation-00000.tar").exists()

        # Verify files are in correct shards
        assert result["train_000.flac"].shard_name == "train-00000.tar"
        assert result["val_000.flac"].shard_name == "validation-00000.tar"

    def test_tar_contains_correct_files(self, tmp_path):
        """TAR archives contain the source files with correct content."""
        src_dir = tmp_path / "audio"
        src_dir.mkdir()

        # Create files with distinct content
        (src_dir / "a.flac").write_bytes(b"content_a")
        (src_dir / "b.flac").write_bytes(b"content_b")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        filenames_by_split = {"train": ["a.flac", "b.flac"]}

        create_sharded_archives(
            source_dir=src_dir,
            output_dir=out_dir,
            filenames_by_split=filenames_by_split,
            target_shard_size_bytes=10000,
            extension="flac",
        )

        # Verify TAR contents
        import tarfile

        with tarfile.open(out_dir / "train-00000.tar") as tar:
            names = tar.getnames()
            assert "a.flac" in names
            assert "b.flac" in names

            # Check content
            a_content = tar.extractfile("a.flac").read()
            assert a_content == b"content_a"
