"""Tests for HuggingFace Hub uploader."""

import json
import pytest
import pandas as pd

from dataset_gen.hub.uploader import HubConfig, DatasetUploader


class TestHubConfig:
    """Tests for HubConfig dataclass."""

    def test_staging_dir_defaults_to_hf_staging(self, tmp_path):
        """staging_dir defaults to dataset_dir/hf_staging."""
        config = HubConfig(
            dataset_dir=tmp_path,
            repo_id="test/repo",
        )
        assert config.staging_dir == tmp_path / "hf_staging"

    def test_custom_staging_dir(self, tmp_path):
        """staging_dir can be set to a custom path."""
        custom = tmp_path / "custom_staging"
        config = HubConfig(
            dataset_dir=tmp_path,
            repo_id="test/repo",
            staging_dir=custom,
        )
        assert config.staging_dir == custom


class TestDatasetUploaderHelpers:
    """Tests for DatasetUploader helper methods."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal dataset structure for testing."""
        # Create labels directory
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Create samples parquet with rudiment_slug
        samples_df = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "profile_id": ["p1", "p1", "p2", "p3"],
                "rudiment_slug": [
                    "single_stroke_roll",
                    "double_stroke_roll",
                    "single_stroke_roll",
                    "paradiddle",
                ],
                "audio_path": [
                    "audio/single_stroke_roll/a1.flac",
                    "audio/double_stroke_roll/a2.flac",
                    "audio/single_stroke_roll/a3.flac",
                    "audio/paradiddle/a4.flac",
                ],
                "midi_path": [
                    "midi/single_stroke_roll/m1.mid",
                    "midi/double_stroke_roll/m2.mid",
                    "midi/single_stroke_roll/m3.mid",
                    "midi/paradiddle/m4.mid",
                ],
            }
        )
        samples_df.to_parquet(labels_dir / "samples.parquet")

        # Create exercises parquet (minimal)
        exercises_df = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "overall_score": [80.0, 85.0, 75.0, 90.0],
            }
        )
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

    def test_prepare_organizes_audio_by_rudiment(self, sample_dataset):
        """prepare() creates audio/{rudiment_slug}/ subdirectories."""
        # Create actual audio files in rudiment subdirectories
        audio_dir = sample_dataset / "audio"
        audio_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "a1.flac"),
            ("double_stroke_roll", "a2.flac"),
            ("single_stroke_roll", "a3.flac"),
            ("paradiddle", "a4.flac"),
        ]:
            (audio_dir / slug).mkdir(exist_ok=True)
            (audio_dir / slug / name).write_bytes(b"fake audio content")

        midi_dir = sample_dataset / "midi"
        midi_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "m1.mid"),
            ("double_stroke_roll", "m2.mid"),
            ("single_stroke_roll", "m3.mid"),
            ("paradiddle", "m4.mid"),
        ]:
            (midi_dir / slug).mkdir(exist_ok=True)
            (midi_dir / slug / name).write_bytes(b"fake midi content")

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        staging_dir = uploader.prepare()

        # Check rudiment subdirectories exist
        assert (staging_dir / "audio" / "single_stroke_roll").is_dir()
        assert (staging_dir / "audio" / "double_stroke_roll").is_dir()
        assert (staging_dir / "audio" / "paradiddle").is_dir()

        # Check files are in correct subdirectories
        assert (staging_dir / "audio" / "single_stroke_roll" / "a1.flac").exists()
        assert (staging_dir / "audio" / "double_stroke_roll" / "a2.flac").exists()
        assert (staging_dir / "audio" / "single_stroke_roll" / "a3.flac").exists()
        assert (staging_dir / "audio" / "paradiddle" / "a4.flac").exists()

        # MIDI should also be organized
        assert (staging_dir / "midi" / "single_stroke_roll" / "m1.mid").exists()
        assert (staging_dir / "midi" / "double_stroke_roll" / "m2.mid").exists()

    def test_prepare_parquet_has_rudiment_paths(self, sample_dataset):
        """prepare() writes parquets with audio/{rudiment_slug}/{filename} paths."""
        audio_dir = sample_dataset / "audio"
        audio_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "a1.flac"),
            ("double_stroke_roll", "a2.flac"),
            ("single_stroke_roll", "a3.flac"),
            ("paradiddle", "a4.flac"),
        ]:
            (audio_dir / slug).mkdir(exist_ok=True)
            (audio_dir / slug / name).write_bytes(b"fake audio content")

        midi_dir = sample_dataset / "midi"
        midi_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "m1.mid"),
            ("double_stroke_roll", "m2.mid"),
            ("single_stroke_roll", "m3.mid"),
            ("paradiddle", "m4.mid"),
        ]:
            (midi_dir / slug).mkdir(exist_ok=True)
            (midi_dir / slug / name).write_bytes(b"fake midi content")

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        staging_dir = uploader.prepare()

        train_df = pd.read_parquet(staging_dir / "data" / "train-00000-of-00001.parquet")

        # Should have 'audio' and 'midi' columns with rudiment paths
        assert "audio" in train_df.columns
        assert "midi" in train_df.columns

        # Should NOT have shard columns
        assert "audio_shard" not in train_df.columns
        assert "audio_filename" not in train_df.columns
        assert "midi_shard" not in train_df.columns
        assert "midi_filename" not in train_df.columns

        # Paths should include rudiment slug
        audio_paths = train_df["audio"].tolist()
        assert any("single_stroke_roll" in p for p in audio_paths)

    def test_prepare_dry_run_skips_media_copy(self, sample_dataset):
        """prepare(skip_media_copy=True) counts files but doesn't copy."""
        audio_dir = sample_dataset / "audio"
        audio_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "a1.flac"),
            ("double_stroke_roll", "a2.flac"),
            ("single_stroke_roll", "a3.flac"),
            ("paradiddle", "a4.flac"),
        ]:
            (audio_dir / slug).mkdir(exist_ok=True)
            (audio_dir / slug / name).write_bytes(b"fake audio content")

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        staging_dir = uploader.prepare(skip_media_copy=True)

        # Parquet files should exist
        assert (staging_dir / "data" / "train-00000-of-00001.parquet").exists()

        # Audio directory should NOT exist in staging
        assert not (staging_dir / "audio").exists()

        # Stats should reflect counted files
        assert uploader.stats.audio_files == 4

    def test_prepare_uses_symlinks_by_default(self, sample_dataset):
        """prepare() creates symlinks rather than copies by default."""
        audio_dir = sample_dataset / "audio"
        audio_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "a1.flac"),
            ("double_stroke_roll", "a2.flac"),
            ("single_stroke_roll", "a3.flac"),
            ("paradiddle", "a4.flac"),
        ]:
            (audio_dir / slug).mkdir(exist_ok=True)
            (audio_dir / slug / name).write_bytes(b"fake audio content")

        midi_dir = sample_dataset / "midi"
        midi_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "m1.mid"),
            ("double_stroke_roll", "m2.mid"),
            ("single_stroke_roll", "m3.mid"),
            ("paradiddle", "m4.mid"),
        ]:
            (midi_dir / slug).mkdir(exist_ok=True)
            (midi_dir / slug / name).write_bytes(b"fake midi content")

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        staging_dir = uploader.prepare(use_symlinks=True)

        # Files should be symlinks
        audio_file = staging_dir / "audio" / "single_stroke_roll" / "a1.flac"
        assert audio_file.is_symlink()

    def test_prepare_stats(self, sample_dataset):
        """prepare() updates upload stats correctly."""
        audio_dir = sample_dataset / "audio"
        audio_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "a1.flac"),
            ("double_stroke_roll", "a2.flac"),
            ("single_stroke_roll", "a3.flac"),
            ("paradiddle", "a4.flac"),
        ]:
            (audio_dir / slug).mkdir(exist_ok=True)
            (audio_dir / slug / name).write_bytes(b"fake audio content")

        midi_dir = sample_dataset / "midi"
        midi_dir.mkdir()
        for slug, name in [
            ("single_stroke_roll", "m1.mid"),
            ("double_stroke_roll", "m2.mid"),
            ("single_stroke_roll", "m3.mid"),
            ("paradiddle", "m4.mid"),
        ]:
            (midi_dir / slug).mkdir(exist_ok=True)
            (midi_dir / slug / name).write_bytes(b"fake midi content")

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        uploader.prepare()

        assert uploader.stats.total_samples == 4
        assert uploader.stats.train_samples == 2
        assert uploader.stats.val_samples == 1
        assert uploader.stats.test_samples == 1
        assert uploader.stats.audio_files == 4
        assert uploader.stats.midi_files == 4
