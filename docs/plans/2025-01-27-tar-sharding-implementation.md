# TAR Sharding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bundle audio/MIDI files into sharded TAR archives to reduce file count from 240k to ~100 for HuggingFace upload.

**Architecture:** New `archiver.py` module creates TAR shards grouped by split. Modified `uploader.py` calls archiver instead of copying files, then updates parquet with shard references.

**Tech Stack:** Python tarfile (stdlib), pandas, existing hub module

---

## Task 1: Create archiver module with ShardInfo dataclass

**Files:**
- Create: `dataset_gen/hub/archiver.py`
- Test: `tests/test_hub_archiver.py`

**Step 1: Write the failing test for ShardInfo**

```python
# tests/test_hub_archiver.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_archiver.py::TestShardInfo::test_shard_info_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'dataset_gen.hub.archiver'"

**Step 3: Write minimal implementation**

```python
# dataset_gen/hub/archiver.py
"""
TAR archive creation for HuggingFace Hub uploads.

Creates sharded TAR archives from audio/MIDI files, grouped by split,
to reduce total file count for HuggingFace's 100k file limit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShardInfo:
    """Information about which shard contains a file."""

    shard_name: str  # e.g., "train-00001.tar"
    filename: str  # e.g., "sample_001.flac"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_archiver.py::TestShardInfo::test_shard_info_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/archiver.py tests/test_hub_archiver.py
git commit -m "feat(hub): add ShardInfo dataclass for TAR sharding"
```

---

## Task 2: Add create_sharded_archives function signature and basic test

**Files:**
- Modify: `dataset_gen/hub/archiver.py`
- Modify: `tests/test_hub_archiver.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_hub_archiver.py
import tempfile
from pathlib import Path

from dataset_gen.hub.archiver import ShardInfo, create_sharded_archives


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

        filenames_by_split = {
            "train": [f"sample_{i:03d}.flac" for i in range(3)]
        }

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_archiver.py::TestCreateShardedArchives::test_creates_single_shard_for_small_files -v`
Expected: FAIL with "cannot import name 'create_sharded_archives'"

**Step 3: Write minimal implementation**

```python
# Add to dataset_gen/hub/archiver.py after ShardInfo class

import logging
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)


def create_sharded_archives(
    source_dir: Path,
    output_dir: Path,
    filenames_by_split: dict[str, list[str]],
    target_shard_size_bytes: int = 1_000_000_000,  # 1GB
    extension: str = "flac",
) -> dict[str, ShardInfo]:
    """
    Create TAR archives from source files, sharded by size.

    Args:
        source_dir: Directory containing source files
        output_dir: Directory to write TAR archives
        filenames_by_split: Mapping of split name to list of filenames
        target_shard_size_bytes: Target size per shard (default 1GB)
        extension: File extension being processed (for logging)

    Returns:
        Mapping of original filename to ShardInfo(shard_name, filename)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, ShardInfo] = {}

    for split_name, filenames in filenames_by_split.items():
        shard_index = 0
        current_shard_size = 0
        current_shard_path = output_dir / f"{split_name}-{shard_index:05d}.tar"
        current_tar = tarfile.open(current_shard_path, "w")

        for filename in sorted(filenames):  # Sort for determinism
            src_file = source_dir / filename
            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
                continue

            file_size = src_file.stat().st_size

            # Check if we need a new shard (but always put at least one file per shard)
            if current_shard_size > 0 and current_shard_size + file_size > target_shard_size_bytes:
                current_tar.close()
                logger.info(
                    f"Closed {current_shard_path.name} with {current_shard_size / (1024**2):.1f} MB"
                )
                shard_index += 1
                current_shard_size = 0
                current_shard_path = output_dir / f"{split_name}-{shard_index:05d}.tar"
                current_tar = tarfile.open(current_shard_path, "w")

            # Add file to current shard
            current_tar.add(src_file, arcname=filename)
            current_shard_size += file_size

            shard_name = f"{split_name}-{shard_index:05d}.tar"
            result[filename] = ShardInfo(shard_name=shard_name, filename=filename)

        # Close final shard
        current_tar.close()
        if current_shard_size > 0:
            logger.info(
                f"Closed {current_shard_path.name} with {current_shard_size / (1024**2):.1f} MB"
            )

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_archiver.py::TestCreateShardedArchives::test_creates_single_shard_for_small_files -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/archiver.py tests/test_hub_archiver.py
git commit -m "feat(hub): add create_sharded_archives function"
```

---

## Task 3: Test multiple shards are created when size exceeded

**Files:**
- Modify: `tests/test_hub_archiver.py`

**Step 1: Write the test**

```python
# Add to TestCreateShardedArchives class in tests/test_hub_archiver.py

    def test_creates_multiple_shards_when_size_exceeded(self, tmp_path):
        """Files are split across shards when target size exceeded."""
        src_dir = tmp_path / "audio"
        src_dir.mkdir()

        # Create 5 files of 100 bytes each
        for i in range(5):
            (src_dir / f"sample_{i:03d}.flac").write_bytes(b"x" * 100)

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        filenames_by_split = {
            "train": [f"sample_{i:03d}.flac" for i in range(5)]
        }

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
```

**Step 2: Run test**

Run: `pytest tests/test_hub_archiver.py::TestCreateShardedArchives::test_creates_multiple_shards_when_size_exceeded -v`
Expected: PASS (implementation already handles this)

**Step 3: Commit if needed**

```bash
git add tests/test_hub_archiver.py
git commit -m "test(hub): add test for multiple shard creation"
```

---

## Task 4: Test separate shards per split

**Files:**
- Modify: `tests/test_hub_archiver.py`

**Step 1: Write the test**

```python
# Add to TestCreateShardedArchives class

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
```

**Step 2: Run test**

Run: `pytest tests/test_hub_archiver.py::TestCreateShardedArchives::test_separate_shards_per_split -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_hub_archiver.py
git commit -m "test(hub): add test for split separation in shards"
```

---

## Task 5: Test TAR contents are correct

**Files:**
- Modify: `tests/test_hub_archiver.py`

**Step 1: Write the test**

```python
# Add to TestCreateShardedArchives class

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
```

**Step 2: Run test**

Run: `pytest tests/test_hub_archiver.py::TestCreateShardedArchives::test_tar_contains_correct_files -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_hub_archiver.py
git commit -m "test(hub): add test verifying TAR contents"
```

---

## Task 6: Export archiver from hub module

**Files:**
- Modify: `dataset_gen/hub/__init__.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_hub_archiver.py at top level

def test_archiver_exported_from_hub_module():
    """Archiver functions are exported from hub module."""
    from dataset_gen.hub import ShardInfo, create_sharded_archives

    assert ShardInfo is not None
    assert create_sharded_archives is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_archiver.py::test_archiver_exported_from_hub_module -v`
Expected: FAIL with "cannot import name 'ShardInfo' from 'dataset_gen.hub'"

**Step 3: Update __init__.py**

```python
# dataset_gen/hub/__init__.py
"""
HuggingFace Hub integration for SOUSA dataset.

This module provides utilities to:
- Prepare the dataset for HuggingFace Hub format
- Upload the dataset to HuggingFace Hub
- Generate consolidated parquet files for efficient loading
- Create sharded TAR archives for large media files
"""

from dataset_gen.hub.archiver import (
    ShardInfo,
    create_sharded_archives,
)
from dataset_gen.hub.uploader import (
    HubConfig,
    DatasetUploader,
    prepare_hf_structure,
    push_to_hub,
)

__all__ = [
    # Archiver
    "ShardInfo",
    "create_sharded_archives",
    # Uploader
    "HubConfig",
    "DatasetUploader",
    "prepare_hf_structure",
    "push_to_hub",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_archiver.py::test_archiver_exported_from_hub_module -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/__init__.py
git commit -m "feat(hub): export archiver from hub module"
```

---

## Task 7: Add use_tar_shards option to HubConfig

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Create: `tests/test_hub_uploader.py`

**Step 1: Write the failing test**

```python
# tests/test_hub_uploader.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_uploader.py::TestHubConfig -v`
Expected: FAIL with "unexpected keyword argument 'use_tar_shards'"

**Step 3: Update HubConfig**

```python
# In dataset_gen/hub/uploader.py, modify HubConfig dataclass:

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

    # TAR sharding options (for staying under HuggingFace 100k file limit)
    use_tar_shards: bool = True
    tar_shard_size_bytes: int = 1_000_000_000  # 1GB per shard

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        if self.staging_dir is None:
            self.staging_dir = self.dataset_dir / "hf_staging"
        else:
            self.staging_dir = Path(self.staging_dir)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_uploader.py::TestHubConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "feat(hub): add use_tar_shards option to HubConfig"
```

---

## Task 8: Add _get_filenames_by_split helper method

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Modify: `tests/test_hub_uploader.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_hub_uploader.py

import json
import pandas as pd
from dataset_gen.hub.uploader import HubConfig, DatasetUploader


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_uploader.py::TestDatasetUploaderHelpers::test_get_filenames_by_split_audio -v`
Expected: FAIL with "'DatasetUploader' object has no attribute '_get_filenames_by_split'"

**Step 3: Add the helper method to DatasetUploader**

```python
# Add to DatasetUploader class in dataset_gen/hub/uploader.py

    def _get_filenames_by_split(self, media_type: str) -> dict[str, list[str]]:
        """
        Get filenames grouped by split for a media type.

        Args:
            media_type: Either "audio" or "midi"

        Returns:
            Dict mapping split name to list of filenames
        """
        # Load source data
        samples_df = pd.read_parquet(self.config.dataset_dir / "labels" / "samples.parquet")

        # Load splits
        with open(self.config.dataset_dir / "splits.json") as f:
            splits = json.load(f)

        # Determine path column
        path_col = f"{media_type}_path"
        if path_col not in samples_df.columns:
            return {"train": [], "validation": [], "test": []}

        result: dict[str, list[str]] = {"train": [], "validation": [], "test": []}

        for _, row in samples_df.iterrows():
            path = row[path_col]
            if pd.isna(path) or not path:
                continue

            filename = Path(path).name
            profile_id = row["profile_id"]

            if profile_id in splits.get("train_profile_ids", []):
                result["train"].append(filename)
            elif profile_id in splits.get("val_profile_ids", []):
                result["validation"].append(filename)
            elif profile_id in splits.get("test_profile_ids", []):
                result["test"].append(filename)
            else:
                result["train"].append(filename)  # Default to train

        return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_uploader.py::TestDatasetUploaderHelpers::test_get_filenames_by_split_audio -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "feat(hub): add _get_filenames_by_split helper"
```

---

## Task 9: Add _create_media_archives method

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Modify: `tests/test_hub_uploader.py`

**Step 1: Write the failing test**

```python
# Add to TestDatasetUploaderHelpers class

    def test_create_media_archives_creates_tar_files(self, sample_dataset):
        """_create_media_archives creates TAR files in staging directory."""
        # Create actual audio files
        audio_dir = sample_dataset / "audio"
        audio_dir.mkdir()
        for name in ["a1.flac", "a2.flac", "a3.flac", "a4.flac"]:
            (audio_dir / name).write_bytes(b"fake audio content")

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        # Create staging dir
        staging = config.staging_dir
        staging.mkdir(parents=True)

        shard_map = uploader._create_media_archives("audio", "flac")

        # Should have created TAR files
        audio_staging = staging / "audio"
        assert audio_staging.exists()
        assert (audio_staging / "train-00000.tar").exists()
        assert (audio_staging / "validation-00000.tar").exists()
        assert (audio_staging / "test-00000.tar").exists()

        # Shard map should have all files
        assert len(shard_map) == 4
        assert "a1.flac" in shard_map
        assert shard_map["a1.flac"].shard_name == "train-00000.tar"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_uploader.py::TestDatasetUploaderHelpers::test_create_media_archives_creates_tar_files -v`
Expected: FAIL with "'DatasetUploader' object has no attribute '_create_media_archives'"

**Step 3: Add the method**

```python
# Add to DatasetUploader class in dataset_gen/hub/uploader.py
# Also add import at top: from dataset_gen.hub.archiver import ShardInfo, create_sharded_archives

    def _create_media_archives(self, media_type: str, extension: str) -> dict[str, ShardInfo]:
        """
        Create sharded TAR archives for media files.

        Args:
            media_type: Either "audio" or "midi"
            extension: File extension (e.g., "flac", "mid")

        Returns:
            Dict mapping filename to ShardInfo
        """
        src_dir = self.config.dataset_dir / media_type
        dst_dir = self.config.staging_dir / media_type

        if not src_dir.exists():
            logger.warning(f"Source directory {src_dir} does not exist")
            return {}

        filenames_by_split = self._get_filenames_by_split(media_type)

        shard_map = create_sharded_archives(
            source_dir=src_dir,
            output_dir=dst_dir,
            filenames_by_split=filenames_by_split,
            target_shard_size_bytes=self.config.tar_shard_size_bytes,
            extension=extension,
        )

        # Update stats
        total_files = sum(len(files) for files in filenames_by_split.values())
        if media_type == "audio":
            self.stats.audio_files = total_files
        elif media_type == "midi":
            self.stats.midi_files = total_files

        return shard_map
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_uploader.py::TestDatasetUploaderHelpers::test_create_media_archives_creates_tar_files -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "feat(hub): add _create_media_archives method"
```

---

## Task 10: Update prepare() to use TAR sharding and add shard columns

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Modify: `tests/test_hub_uploader.py`

**Step 1: Write the failing test**

```python
# Add to TestDatasetUploaderHelpers class

    def test_prepare_with_tar_shards_adds_shard_columns(self, sample_dataset):
        """prepare() adds audio_shard and audio_filename columns when using TAR shards."""
        # Create actual audio and midi files
        audio_dir = sample_dataset / "audio"
        audio_dir.mkdir()
        for name in ["a1.flac", "a2.flac", "a3.flac", "a4.flac"]:
            (audio_dir / name).write_bytes(b"fake audio content")

        midi_dir = sample_dataset / "midi"
        midi_dir.mkdir()
        for name in ["m1.mid", "m2.mid", "m3.mid", "m4.mid"]:
            (midi_dir / name).write_bytes(b"fake midi content")

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo", use_tar_shards=True)
        uploader = DatasetUploader(config)

        staging_dir = uploader.prepare()

        # Check parquet has shard columns
        train_df = pd.read_parquet(staging_dir / "data" / "train-00000-of-00001.parquet")

        assert "audio_shard" in train_df.columns
        assert "audio_filename" in train_df.columns
        assert "midi_shard" in train_df.columns
        assert "midi_filename" in train_df.columns

        # Values should be set
        assert train_df["audio_shard"].iloc[0] == "train-00000.tar"
        assert train_df["audio_filename"].iloc[0] in ["a1.flac", "a2.flac"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_uploader.py::TestDatasetUploaderHelpers::test_prepare_with_tar_shards_adds_shard_columns -v`
Expected: FAIL (prepare() doesn't add shard columns yet)

**Step 3: Update prepare() method**

This requires significant changes to the prepare() method. Replace the media copying logic with archive creation and add shard columns to the dataframe:

```python
# In DatasetUploader.prepare(), replace the media handling section.
# After "merged_df = self._merge_dataframes(samples_df, exercises_df)"
# and before "# Create split assignment column"

        # Handle media files (TAR shards or individual files)
        audio_shard_map: dict[str, ShardInfo] = {}
        midi_shard_map: dict[str, ShardInfo] = {}

        if self.config.use_tar_shards:
            # Create TAR archives
            if self.config.include_audio and not skip_media_copy:
                audio_shard_map = self._create_media_archives("audio", "flac")
            if self.config.include_midi and not skip_media_copy:
                midi_shard_map = self._create_media_archives("midi", "mid")

            # Add shard columns to dataframe
            if self.config.include_audio:
                merged_df["audio_shard"] = merged_df["audio_path"].apply(
                    lambda p: audio_shard_map.get(Path(p).name, ShardInfo("", "")).shard_name
                    if pd.notna(p) and p else None
                )
                merged_df["audio_filename"] = merged_df["audio_path"].apply(
                    lambda p: Path(p).name if pd.notna(p) and p else None
                )
            if self.config.include_midi:
                merged_df["midi_shard"] = merged_df["midi_path"].apply(
                    lambda p: midi_shard_map.get(Path(p).name, ShardInfo("", "")).shard_name
                    if pd.notna(p) and p else None
                )
                merged_df["midi_filename"] = merged_df["midi_path"].apply(
                    lambda p: Path(p).name if pd.notna(p) and p else None
                )
        else:
            # Original behavior: copy/link individual files
            if self.config.include_audio:
                merged_df["audio"] = merged_df["audio_path"].apply(
                    lambda p: f"audio/{Path(p).name}" if pd.notna(p) and p else None
                )
            if self.config.include_midi:
                merged_df["midi"] = merged_df["midi_path"].apply(
                    lambda p: f"midi/{Path(p).name}" if pd.notna(p) and p else None
                )

# Then later in the method, update the media file handling:

        if not self.config.use_tar_shards:
            # Copy/link audio files (original behavior)
            if self.config.include_audio:
                if skip_media_copy:
                    self._count_media_files("audio", "flac")
                else:
                    self._copy_media_files("audio", "flac", use_symlinks=use_symlinks)

            # Copy/link MIDI files (original behavior)
            if self.config.include_midi:
                if skip_media_copy:
                    self._count_media_files("midi", "mid")
                else:
                    self._copy_media_files("midi", "mid", use_symlinks=use_symlinks)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_uploader.py::TestDatasetUploaderHelpers::test_prepare_with_tar_shards_adds_shard_columns -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "feat(hub): integrate TAR sharding into prepare()"
```

---

## Task 11: Update push_to_hub.py script with --no-sharding flag

**Files:**
- Modify: `scripts/push_to_hub.py`

**Step 1: Update the script**

Add argument after `--dry-run`:

```python
    parser.add_argument(
        "--no-sharding",
        action="store_true",
        help="Disable TAR sharding (not recommended for large datasets)",
    )
```

And pass it to HubConfig:

```python
    config = HubConfig(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        include_audio=not args.no_audio,
        include_midi=not args.no_midi,
        staging_dir=args.staging_dir,
        use_tar_shards=not args.no_sharding,
    )
```

**Step 2: Test manually**

Run: `python scripts/push_to_hub.py --help`
Expected: Shows `--no-sharding` option

**Step 3: Commit**

```bash
git add scripts/push_to_hub.py
git commit -m "feat(hub): add --no-sharding flag to push_to_hub script"
```

---

## Task 12: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Run all hub-related tests**

Run: `pytest tests/test_hub_archiver.py tests/test_hub_uploader.py -v`
Expected: All tests PASS

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 157+ tests PASS

**Step 3: Run linting**

Run: `ruff check dataset_gen/hub/ && black --check dataset_gen/hub/`
Expected: No errors

**Step 4: Final commit if any fixes needed**

---

## Task 13: Test with real dataset (manual integration test)

**Files:** None (manual test)

**Step 1: Copy a small sample of real data to worktree**

```bash
mkdir -p output/dataset/audio output/dataset/midi output/dataset/labels
cp /path/to/main/repo/output/dataset/labels/*.parquet output/dataset/labels/
cp /path/to/main/repo/output/dataset/splits.json output/dataset/
# Copy just 100 audio and midi files for testing
ls /path/to/main/repo/output/dataset/audio/*.flac | head -100 | xargs -I {} cp {} output/dataset/audio/
ls /path/to/main/repo/output/dataset/midi/*.mid | head -100 | xargs -I {} cp {} output/dataset/midi/
```

**Step 2: Run dry-run**

Run: `python scripts/push_to_hub.py zkeown/sousa-test --dry-run`
Expected: Creates TAR archives in `output/dataset/hf_staging/`

**Step 3: Verify structure**

```bash
ls -la output/dataset/hf_staging/
ls -la output/dataset/hf_staging/audio/
ls -la output/dataset/hf_staging/midi/
```
Expected: TAR files instead of individual audio/midi files

**Step 4: Verify parquet has shard columns**

```python
import pandas as pd
df = pd.read_parquet("output/dataset/hf_staging/data/train-00000-of-00001.parquet")
print(df[["audio_shard", "audio_filename", "midi_shard", "midi_filename"]].head())
```
Expected: Shard columns populated correctly
