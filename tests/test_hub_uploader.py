"""Tests for HuggingFace Hub uploader."""

import json
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

from dataset_gen.hub.uploader import HubConfig, DatasetUploader


class TestHubConfigParquet:
    """Tests for the rewritten HubConfig dataclass (Parquet-native upload)."""

    def test_configs_defaults_to_all_three(self, tmp_path):
        """configs defaults to ["audio", "midi_only", "labels_only"]."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert config.configs == ["audio", "midi_only", "labels_only"]

    def test_configs_custom_subset(self, tmp_path):
        """configs can be set to a custom subset."""
        config = HubConfig(
            dataset_dir=tmp_path,
            repo_id="test/repo",
            configs=["audio"],
        )
        assert config.configs == ["audio"]

    def test_max_shard_size_defaults_to_1gb(self, tmp_path):
        """max_shard_size defaults to '1GB'."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert config.max_shard_size == "1GB"

    def test_no_staging_dir_attribute(self, tmp_path):
        """staging_dir attribute does not exist on HubConfig."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert not hasattr(config, "staging_dir")

    def test_no_include_audio_attribute(self, tmp_path):
        """include_audio attribute does not exist on HubConfig."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert not hasattr(config, "include_audio")

    def test_no_include_midi_attribute(self, tmp_path):
        """include_midi attribute does not exist on HubConfig."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert not hasattr(config, "include_midi")


class TestBuildDatasetDict:
    """Tests for DatasetUploader.build_dataset_dict method."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal dataset structure for testing build_dataset_dict.

        Creates 4 samples across 3 profiles:
        - p1 has 2 samples (train split)
        - p2 has 1 sample (val split)
        - p3 has 1 sample (test split)
        """
        # Create labels directory
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        # Create samples parquet with rudiment_slug and media paths
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
                "tempo_bpm": [120, 140, 100, 160],
                "skill_tier": ["beginner", "intermediate", "advanced", "professional"],
                "audio_path": [
                    "audio/single_stroke_roll/s1.flac",
                    "audio/double_stroke_roll/s2.flac",
                    "audio/single_stroke_roll/s3.flac",
                    "audio/paradiddle/s4.flac",
                ],
                "midi_path": [
                    "midi/single_stroke_roll/s1.mid",
                    "midi/double_stroke_roll/s2.mid",
                    "midi/single_stroke_roll/s3.mid",
                    "midi/paradiddle/s4.mid",
                ],
            }
        )
        samples_df.to_parquet(labels_dir / "samples.parquet")

        # Create exercises parquet with scores
        exercises_df = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3", "s4"],
                "timing_accuracy": [70.0, 80.0, 85.0, 95.0],
                "overall_score": [72.0, 82.0, 84.0, 93.0],
            }
        )
        exercises_df.to_parquet(labels_dir / "exercises.parquet")

        # Create splits.json
        splits = {
            "train_profile_ids": ["p1"],
            "val_profile_ids": ["p2"],
            "test_profile_ids": ["p3"],
        }
        with open(tmp_path / "splits.json", "w") as f:
            json.dump(splits, f)

        # Create actual audio files (fake bytes) in rudiment subdirectories
        for slug, filename in [
            ("single_stroke_roll", "s1.flac"),
            ("double_stroke_roll", "s2.flac"),
            ("single_stroke_roll", "s3.flac"),
            ("paradiddle", "s4.flac"),
        ]:
            audio_dir = tmp_path / "audio" / slug
            audio_dir.mkdir(parents=True, exist_ok=True)
            (audio_dir / filename).write_bytes(b"fake audio data for " + filename.encode())

        # Create actual MIDI files (fake bytes) in rudiment subdirectories
        for slug, filename in [
            ("single_stroke_roll", "s1.mid"),
            ("double_stroke_roll", "s2.mid"),
            ("single_stroke_roll", "s3.mid"),
            ("paradiddle", "s4.mid"),
        ]:
            midi_dir = tmp_path / "midi" / slug
            midi_dir.mkdir(parents=True, exist_ok=True)
            (midi_dir / filename).write_bytes(b"fake midi data for " + filename.encode())

        return tmp_path

    def test_build_labels_only_dataset_dict(self, sample_dataset):
        """labels_only config returns DatasetDict with correct splits and no media columns."""
        from datasets import DatasetDict

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        dd = uploader.build_dataset_dict("labels_only")

        # Returns a DatasetDict
        assert isinstance(dd, DatasetDict)

        # Has correct split names
        assert set(dd.keys()) == {"train", "validation", "test"}

        # Correct row counts: p1 has 2 samples (train), p2 has 1 (val), p3 has 1 (test)
        assert len(dd["train"]) == 2
        assert len(dd["validation"]) == 1
        assert len(dd["test"]) == 1

        # Has expected metadata columns
        assert "sample_id" in dd["train"].column_names
        assert "overall_score" in dd["train"].column_names

        # No media columns
        assert "audio" not in dd["train"].column_names
        assert "midi" not in dd["train"].column_names

    def test_build_midi_only_dataset_dict(self, sample_dataset):
        """midi_only config has midi column (bytes) but no audio column."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        dd = uploader.build_dataset_dict("midi_only")

        # Has midi column
        assert "midi" in dd["train"].column_names

        # No audio column
        assert "audio" not in dd["train"].column_names

        # MIDI data is bytes
        midi_val = dd["train"][0]["midi"]
        assert isinstance(midi_val, bytes)
        assert len(midi_val) > 0

    def test_build_audio_dataset_dict(self, sample_dataset):
        """audio config has both audio and midi columns."""
        from datasets import Audio as AudioFeature

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        dd = uploader.build_dataset_dict("audio")

        # Has both audio and midi columns
        assert "audio" in dd["train"].column_names
        assert "midi" in dd["train"].column_names

        # Audio column is typed as Audio feature
        assert isinstance(dd["train"].features["audio"], AudioFeature)

        # MIDI data is bytes (access column directly to avoid triggering audio decode)
        midi_val = dd["train"]["midi"][0]
        assert isinstance(midi_val, bytes)

    def test_metadata_columns_present(self, sample_dataset):
        """All expected metadata and score columns are present in the output."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        dd = uploader.build_dataset_dict("labels_only")

        columns = dd["train"].column_names
        # Metadata columns from samples
        assert "sample_id" in columns
        assert "profile_id" in columns
        assert "rudiment_slug" in columns
        assert "tempo_bpm" in columns
        assert "skill_tier" in columns

        # Score columns from exercises
        assert "timing_accuracy" in columns
        assert "overall_score" in columns

    def test_internal_path_columns_excluded(self, sample_dataset):
        """audio_path and midi_path columns are NOT in the output."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)

        # Check all three config types
        for config_name in ["labels_only", "midi_only", "audio"]:
            dd = uploader.build_dataset_dict(config_name)
            for split_name in ["train", "validation", "test"]:
                columns = dd[split_name].column_names
                assert (
                    "audio_path" not in columns
                ), f"audio_path found in {config_name}/{split_name}"
                assert "midi_path" not in columns, f"midi_path found in {config_name}/{split_name}"


class TestPurgeRepo:
    """Tests for DatasetUploader.purge_repo method."""

    def test_purge_deletes_all_files_except_gitattributes(self):
        """purge_repo deletes all files except .gitattributes."""
        from pathlib import Path

        config = HubConfig(dataset_dir=Path("/fake"), repo_id="test/repo")
        uploader = DatasetUploader(config)

        mock_api = MagicMock()
        # Simulate list_repo_tree returning file-like objects
        file1 = MagicMock()
        file1.rfilename = "data/train-00000.parquet"
        file2 = MagicMock()
        file2.rfilename = "audio/single_stroke_roll/s1.flac"
        file3 = MagicMock()
        file3.rfilename = ".gitattributes"
        file4 = MagicMock()
        file4.rfilename = "README.md"
        mock_api.list_repo_tree.return_value = [file1, file2, file3, file4]

        uploader.purge_repo(api=mock_api)

        # Should call create_commit with delete operations
        mock_api.create_commit.assert_called_once()
        call_kwargs = mock_api.create_commit.call_args[1]
        assert call_kwargs["repo_type"] == "dataset"
        operations = call_kwargs["operations"]
        deleted_paths = [op.path_in_repo for op in operations]
        assert ".gitattributes" not in deleted_paths
        assert "data/train-00000.parquet" in deleted_paths
        assert "audio/single_stroke_roll/s1.flac" in deleted_paths
        assert "README.md" in deleted_paths

    def test_purge_skips_empty_repo(self):
        """purge_repo does nothing if only .gitattributes exists."""
        from pathlib import Path

        config = HubConfig(dataset_dir=Path("/fake"), repo_id="test/repo")
        uploader = DatasetUploader(config)

        mock_api = MagicMock()
        gitattr = MagicMock()
        gitattr.rfilename = ".gitattributes"
        mock_api.list_repo_tree.return_value = [gitattr]

        uploader.purge_repo(api=mock_api)
        mock_api.create_commit.assert_not_called()


class TestUploadParquet:
    """Tests for the rewritten DatasetUploader.upload method (Parquet-native)."""

    def test_upload_pushes_each_config(self, tmp_path):
        """upload() pushes each configured config to hub."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo", configs=["labels_only"])
        uploader = DatasetUploader(config)

        with (
            patch.object(uploader, "build_dataset_dict") as mock_build,
            patch("dataset_gen.hub.uploader.HfApi") as mock_hf,
        ):
            mock_dd = MagicMock()
            mock_dd.items.return_value = [("train", MagicMock(num_rows=10))]
            mock_build.return_value = mock_dd

            uploader.upload()

            mock_build.assert_called_once_with("labels_only")
            mock_dd.push_to_hub.assert_called_once()
            call_kwargs = mock_dd.push_to_hub.call_args
            assert call_kwargs[0][0] == "test/repo"
            assert call_kwargs[1]["config_name"] == "labels_only"

    def test_upload_dry_run_builds_but_does_not_push(self, tmp_path):
        """upload(dry_run=True) builds DatasetDict but doesn't push."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo", configs=["labels_only"])
        uploader = DatasetUploader(config)

        with (
            patch.object(uploader, "build_dataset_dict") as mock_build,
            patch("dataset_gen.hub.uploader.HfApi"),
        ):
            mock_dd = MagicMock()
            mock_dd.items.return_value = [("train", MagicMock(num_rows=10))]
            mock_build.return_value = mock_dd

            result = uploader.upload(dry_run=True)

            mock_build.assert_called_once()
            mock_dd.push_to_hub.assert_not_called()
            assert result is None

    def test_upload_pushes_auxiliary_tables(self, tmp_path):
        """upload() uploads strokes.parquet and measures.parquet to auxiliary/."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "strokes.parquet").write_bytes(b"fake strokes")
        (labels_dir / "measures.parquet").write_bytes(b"fake measures")

        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo", configs=["labels_only"])
        uploader = DatasetUploader(config)

        with (
            patch.object(uploader, "build_dataset_dict") as mock_build,
            patch("dataset_gen.hub.uploader.HfApi") as mock_hf_cls,
        ):
            mock_dd = MagicMock()
            mock_dd.items.return_value = []
            mock_build.return_value = mock_dd
            mock_api = mock_hf_cls.return_value

            uploader.upload()

            upload_calls = mock_api.upload_file.call_args_list
            uploaded_paths = [c[1]["path_in_repo"] for c in upload_calls]
            assert "auxiliary/strokes.parquet" in uploaded_paths
            assert "auxiliary/measures.parquet" in uploaded_paths

    def test_upload_with_purge_calls_purge_repo(self, tmp_path):
        """upload(purge=True) calls purge_repo before pushing."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo", configs=["labels_only"])
        uploader = DatasetUploader(config)

        with (
            patch.object(uploader, "build_dataset_dict") as mock_build,
            patch.object(uploader, "purge_repo") as mock_purge,
            patch("dataset_gen.hub.uploader.HfApi"),
        ):
            mock_dd = MagicMock()
            mock_dd.items.return_value = []
            mock_build.return_value = mock_dd

            uploader.upload(purge=True)
            mock_purge.assert_called_once()


class TestConvenienceFunctions:
    """Tests for module-level convenience functions and exports."""

    def test_push_to_hub_accepts_new_params(self):
        """push_to_hub accepts configs, purge, max_shard_size."""
        import inspect

        from dataset_gen.hub.uploader import push_to_hub

        sig = inspect.signature(push_to_hub)
        params = list(sig.parameters.keys())
        assert "configs" in params
        assert "purge" in params
        assert "max_shard_size" in params
        assert "include_audio" not in params
        assert "include_midi" not in params

    def test_prepare_hf_structure_removed(self):
        """prepare_hf_structure is no longer exported."""
        from dataset_gen.hub import __all__

        assert "prepare_hf_structure" not in __all__
