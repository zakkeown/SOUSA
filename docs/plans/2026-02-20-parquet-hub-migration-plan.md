# Parquet Hub Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 120K-file hub upload with Parquet-embedded media across three configs (audio, midi_only, labels_only), reducing to ~130 files.

**Architecture:** Rewrite `DatasetUploader` to build HuggingFace `DatasetDict` objects with Audio and binary MIDI columns, then push via the `datasets` API with `max_shard_size="1GB"`. Three configs share the same metadata/scores but differ in media columns. Auxiliary stroke/measure tables are uploaded separately.

**Tech Stack:** `datasets` library (Audio feature type, DatasetDict, push_to_hub), `huggingface_hub` (HfApi for repo management and auxiliary uploads), `pandas` (DataFrame manipulation), `pyarrow`.

---

### Task 1: Rewrite HubConfig dataclass

**Files:**
- Modify: `dataset_gen/hub/uploader.py:24-52`

**Step 1: Write the failing test**

Add to `tests/test_hub_uploader.py`:

```python
class TestHubConfigParquet:
    """Tests for updated HubConfig dataclass."""

    def test_default_configs_is_all_three(self, tmp_path):
        """configs defaults to all three: audio, midi_only, labels_only."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert config.configs == ["audio", "midi_only", "labels_only"]

    def test_custom_configs(self, tmp_path):
        """configs can be set to a subset."""
        config = HubConfig(
            dataset_dir=tmp_path,
            repo_id="test/repo",
            configs=["labels_only"],
        )
        assert config.configs == ["labels_only"]

    def test_max_shard_size_default(self, tmp_path):
        """max_shard_size defaults to 1GB."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert config.max_shard_size == "1GB"

    def test_no_staging_dir(self, tmp_path):
        """staging_dir concept is removed."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert not hasattr(config, "staging_dir")

    def test_no_include_audio_midi_flags(self, tmp_path):
        """include_audio/include_midi flags are removed."""
        config = HubConfig(dataset_dir=tmp_path, repo_id="test/repo")
        assert not hasattr(config, "include_audio")
        assert not hasattr(config, "include_midi")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_uploader.py::TestHubConfigParquet -v`
Expected: FAIL (old HubConfig doesn't have `configs` field, still has `staging_dir`/`include_audio`/`include_midi`)

**Step 3: Rewrite HubConfig**

Replace the `HubConfig` dataclass in `dataset_gen/hub/uploader.py`:

```python
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

    # Shard settings
    max_shard_size: str = "1GB"

    def __post_init__(self):
        self.dataset_dir = Path(self.dataset_dir)
        if self.configs is None:
            self.configs = ["audio", "midi_only", "labels_only"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_uploader.py::TestHubConfigParquet -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "refactor(hub): rewrite HubConfig for Parquet-native upload"
```

---

### Task 2: Build the DatasetDict construction logic

This is the core of the migration — a method that reads local parquets + media files and builds a HuggingFace `DatasetDict` with embedded binary data.

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Test: `tests/test_hub_uploader.py`

**Step 1: Write the failing test**

Add to `tests/test_hub_uploader.py`. This test uses the existing `sample_dataset` fixture but enhanced with actual audio/MIDI file bytes:

```python
class TestBuildDatasetDict:
    """Tests for DatasetDict construction with embedded media."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal dataset with actual media files."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        samples_df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3", "s4"],
            "profile_id": ["p1", "p1", "p2", "p3"],
            "rudiment_slug": ["single_stroke_roll", "double_stroke_roll",
                              "single_stroke_roll", "paradiddle"],
            "tempo_bpm": [120, 100, 120, 140],
            "duration_sec": [2.0, 2.5, 2.0, 1.5],
            "num_cycles": [2, 3, 2, 2],
            "skill_tier": ["beginner", "intermediate", "beginner", "advanced"],
            "skill_tier_binary": ["novice", "skilled", "novice", "skilled"],
            "dominant_hand": ["right", "right", "left", "right"],
            "num_strokes": [16, 24, 16, 20],
            "num_measures": [2, 3, 2, 2],
            "soundfont": ["sf1", "sf2", "sf1", "sf1"],
            "augmentation_preset": ["clean", "studio", "clean", "garage"],
            "augmentation_group_id": ["g1", "g2", "g1", "g3"],
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
        })
        samples_df.to_parquet(labels_dir / "samples.parquet")

        exercises_df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3", "s4"],
            "timing_accuracy": [80.0, 85.0, 75.0, 90.0],
            "timing_consistency": [78.0, 82.0, 72.0, 88.0],
            "tempo_stability": [82.0, 87.0, 77.0, 92.0],
            "subdivision_evenness": [76.0, 80.0, 70.0, 86.0],
            "velocity_control": [79.0, 84.0, 74.0, 89.0],
            "accent_differentiation": [75.0, 80.0, 70.0, 85.0],
            "accent_accuracy": [77.0, 82.0, 72.0, 87.0],
            "hand_balance": [80.0, 85.0, 75.0, 90.0],
            "weak_hand_index": [50.0, 55.0, 45.0, 60.0],
            "flam_quality": [None, None, None, None],
            "diddle_quality": [None, None, None, None],
            "roll_sustain": [None, None, None, None],
            "groove_feel_proxy": [0.7, 0.75, 0.65, 0.85],
            "overall_score": [78.0, 83.0, 73.0, 88.0],
        })
        exercises_df.to_parquet(labels_dir / "exercises.parquet")

        splits = {
            "train_profile_ids": ["p1"],
            "val_profile_ids": ["p2"],
            "test_profile_ids": ["p3"],
        }
        with open(tmp_path / "splits.json", "w") as f:
            json.dump(splits, f)

        # Create actual media files
        for subdir, ext, content in [
            ("audio", "flac", b"FAKE_FLAC_CONTENT_HERE"),
            ("midi", "mid", b"MThd\x00\x00\x00\x06"),
        ]:
            for slug, sid in [
                ("single_stroke_roll", "s1"),
                ("double_stroke_roll", "s2"),
                ("single_stroke_roll", "s3"),
                ("paradiddle", "s4"),
            ]:
                d = tmp_path / subdir / slug
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{sid}.{ext}").write_bytes(content)

        return tmp_path

    def test_build_labels_only_dataset_dict(self, sample_dataset):
        """build_dataset_dict('labels_only') returns DatasetDict without media columns."""
        from datasets import DatasetDict

        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)
        dd = uploader.build_dataset_dict("labels_only")

        assert isinstance(dd, DatasetDict)
        assert set(dd.keys()) == {"train", "validation", "test"}
        assert "audio" not in dd["train"].column_names
        assert "midi" not in dd["train"].column_names
        assert "sample_id" in dd["train"].column_names
        assert "overall_score" in dd["train"].column_names
        assert dd["train"].num_rows == 2
        assert dd["validation"].num_rows == 1
        assert dd["test"].num_rows == 1

    def test_build_midi_only_dataset_dict(self, sample_dataset):
        """build_dataset_dict('midi_only') includes midi bytes but no audio."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)
        dd = uploader.build_dataset_dict("midi_only")

        assert "midi" in dd["train"].column_names
        assert "audio" not in dd["train"].column_names
        # MIDI should be bytes
        midi_val = dd["train"][0]["midi"]
        assert isinstance(midi_val, bytes)
        assert len(midi_val) > 0

    def test_build_audio_dataset_dict(self, sample_dataset):
        """build_dataset_dict('audio') includes both audio and midi columns."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)
        dd = uploader.build_dataset_dict("audio")

        assert "audio" in dd["train"].column_names
        assert "midi" in dd["train"].column_names

    def test_metadata_columns_present(self, sample_dataset):
        """All expected metadata and score columns are present."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)
        dd = uploader.build_dataset_dict("labels_only")

        expected_cols = [
            "sample_id", "profile_id", "rudiment_slug", "skill_tier",
            "tempo_bpm", "duration_sec", "num_cycles", "dominant_hand",
            "soundfont", "augmentation_preset",
            "timing_accuracy", "overall_score",
        ]
        for col in expected_cols:
            assert col in dd["train"].column_names, f"Missing column: {col}"

    def test_internal_path_columns_excluded(self, sample_dataset):
        """audio_path and midi_path (internal filesystem paths) are not in output."""
        config = HubConfig(dataset_dir=sample_dataset, repo_id="test/repo")
        uploader = DatasetUploader(config)
        dd = uploader.build_dataset_dict("labels_only")

        assert "audio_path" not in dd["train"].column_names
        assert "midi_path" not in dd["train"].column_names
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hub_uploader.py::TestBuildDatasetDict -v`
Expected: FAIL (`build_dataset_dict` method doesn't exist)

**Step 3: Implement build_dataset_dict method**

Add to the `DatasetUploader` class in `dataset_gen/hub/uploader.py`:

```python
def build_dataset_dict(self, config_name: str) -> "DatasetDict":
    """
    Build a HuggingFace DatasetDict for the given config.

    Args:
        config_name: One of "audio", "midi_only", "labels_only"

    Returns:
        DatasetDict with train/validation/test splits
    """
    from datasets import Dataset, DatasetDict, Features, Value, Audio

    # Load source data
    samples_df = pd.read_parquet(self.config.dataset_dir / "labels" / "samples.parquet")
    exercises_df = pd.read_parquet(self.config.dataset_dir / "labels" / "exercises.parquet")

    with open(self.config.dataset_dir / "splits.json") as f:
        splits = json.load(f)

    # Merge samples with exercise scores
    merged_df = self._merge_dataframes(samples_df, exercises_df)

    # Add split assignment
    merged_df["split"] = merged_df["profile_id"].apply(
        lambda pid: self._get_split(pid, splits)
    )

    # Drop internal path columns — we'll resolve them to bytes/Audio
    path_cols_to_drop = ["audio_path", "midi_path"]
    # Keep the path values for media resolution before dropping
    audio_paths = merged_df["audio_path"] if "audio_path" in merged_df.columns else None
    midi_paths = merged_df["midi_path"] if "midi_path" in merged_df.columns else None
    merged_df = merged_df.drop(
        columns=[c for c in path_cols_to_drop if c in merged_df.columns]
    )

    # Build per-split datasets
    split_datasets = {}
    split_map = {"train": "train", "val": "validation", "test": "test"}

    for split_key, split_name in split_map.items():
        mask = merged_df["split"] == split_key
        split_df = merged_df[mask].drop(columns=["split"]).reset_index(drop=True)

        if len(split_df) == 0:
            continue

        data_dict = split_df.to_dict(orient="list")

        # Add media columns based on config
        if config_name in ("audio", "midi_only"):
            split_midi_paths = midi_paths[mask].reset_index(drop=True)
            midi_bytes = []
            for path_val in split_midi_paths:
                if pd.notna(path_val) and path_val:
                    slug = Path(path_val).parent.name
                    filename = Path(path_val).name
                    full_path = self.config.dataset_dir / "midi" / slug / filename
                    midi_bytes.append(full_path.read_bytes() if full_path.exists() else None)
                else:
                    midi_bytes.append(None)
            data_dict["midi"] = midi_bytes

        if config_name == "audio":
            split_audio_paths = audio_paths[mask].reset_index(drop=True)
            audio_file_paths = []
            for path_val in split_audio_paths:
                if pd.notna(path_val) and path_val:
                    slug = Path(path_val).parent.name
                    filename = Path(path_val).name
                    full_path = self.config.dataset_dir / "audio" / slug / filename
                    audio_file_paths.append(str(full_path) if full_path.exists() else None)
                else:
                    audio_file_paths.append(None)
            data_dict["audio"] = audio_file_paths

        ds = Dataset.from_dict(data_dict)

        # Cast audio column to Audio feature type
        if config_name == "audio" and "audio" in ds.column_names:
            ds = ds.cast_column("audio", Audio())

        split_datasets[split_name] = ds

    return DatasetDict(split_datasets)
```

Also keep the existing `_merge_dataframes` and `_get_split` helper methods unchanged — they are reused.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hub_uploader.py::TestBuildDatasetDict -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "feat(hub): add build_dataset_dict for Parquet-native configs"
```

---

### Task 3: Implement hub purge functionality

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Test: `tests/test_hub_uploader.py`

**Step 1: Write the failing test**

```python
from unittest.mock import MagicMock, patch

class TestPurgeRepo:
    """Tests for hub repository purge."""

    def test_purge_deletes_all_files(self):
        """purge_repo deletes all files except .gitattributes."""
        config = HubConfig(
            dataset_dir=Path("/fake"),
            repo_id="test/repo",
        )
        uploader = DatasetUploader(config)

        mock_api = MagicMock()
        mock_api.list_repo_tree.return_value = [
            MagicMock(rfilename="data/train-00000.parquet"),
            MagicMock(rfilename="audio/single_stroke_roll/s1.flac"),
            MagicMock(rfilename=".gitattributes"),
            MagicMock(rfilename="README.md"),
        ]

        uploader.purge_repo(api=mock_api)

        # Should call delete_files with all files except .gitattributes
        mock_api.delete_files.assert_called_once()
        deleted = mock_api.delete_files.call_args[1]["delete_patterns"]
        assert ".gitattributes" not in deleted
        assert "data/train-00000.parquet" in deleted
        assert "README.md" in deleted

    def test_purge_skips_empty_repo(self):
        """purge_repo does nothing for empty repo."""
        config = HubConfig(dataset_dir=Path("/fake"), repo_id="test/repo")
        uploader = DatasetUploader(config)

        mock_api = MagicMock()
        mock_api.list_repo_tree.return_value = [
            MagicMock(rfilename=".gitattributes"),
        ]

        uploader.purge_repo(api=mock_api)
        mock_api.delete_files.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hub_uploader.py::TestPurgeRepo -v`
Expected: FAIL (`purge_repo` doesn't exist)

**Step 3: Implement purge_repo**

Add to `DatasetUploader`:

```python
def purge_repo(self, api: "HfApi | None" = None) -> None:
    """
    Delete all files from the hub repo except .gitattributes.

    Args:
        api: HfApi instance (created if None)
    """
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=self.config.token)

    logger.info(f"Listing files in {self.config.repo_id}...")
    files = list(api.list_repo_tree(
        repo_id=self.config.repo_id,
        repo_type="dataset",
        recursive=True,
    ))

    to_delete = [
        f.rfilename for f in files
        if hasattr(f, "rfilename") and f.rfilename != ".gitattributes"
    ]

    if not to_delete:
        logger.info("Repository is empty, nothing to purge")
        return

    logger.info(f"Purging {len(to_delete)} files from {self.config.repo_id}")
    api.delete_files(
        repo_id=self.config.repo_id,
        repo_type="dataset",
        delete_patterns=to_delete,
        commit_message="chore: purge individual files for Parquet migration",
    )
    logger.info("Purge complete")
```

**Note:** The `HfApi.delete_files` method may not exist — check the actual huggingface_hub API. The alternative is to use `CommitOperationDelete` with `create_commit`. Adjust during implementation based on the actual API available. The test mocks the API, so the interface can be adjusted.

An alternative implementation using `create_commit`:

```python
from huggingface_hub import CommitOperationDelete

operations = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
api.create_commit(
    repo_id=self.config.repo_id,
    repo_type="dataset",
    operations=operations,
    commit_message="chore: purge individual files for Parquet migration",
)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hub_uploader.py::TestPurgeRepo -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "feat(hub): add purge_repo to clear hub before Parquet upload"
```

---

### Task 4: Implement the upload method

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Test: `tests/test_hub_uploader.py`

**Step 1: Write the failing test**

```python
class TestUploadParquet:
    """Tests for the Parquet-native upload method."""

    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Reuse the same fixture from TestBuildDatasetDict."""
        # (Same fixture content as Task 2 — copy it or extract to module-level fixture)
        ...  # See Task 2 for full fixture

    @patch("dataset_gen.hub.uploader.HfApi")
    def test_upload_calls_push_to_hub_for_each_config(self, mock_hf_api_cls, sample_dataset):
        """upload() pushes each configured config to hub."""
        config = HubConfig(
            dataset_dir=sample_dataset,
            repo_id="test/repo",
            configs=["labels_only"],
        )
        uploader = DatasetUploader(config)

        # Mock push_to_hub on the DatasetDict
        with patch.object(uploader, "build_dataset_dict") as mock_build:
            mock_dd = MagicMock()
            mock_build.return_value = mock_dd

            uploader.upload()

            mock_build.assert_called_once_with("labels_only")
            mock_dd.push_to_hub.assert_called_once_with(
                "test/repo",
                config_name="labels_only",
                max_shard_size="1GB",
                token=None,
                private=False,
            )

    @patch("dataset_gen.hub.uploader.HfApi")
    def test_upload_dry_run_does_not_push(self, mock_hf_api_cls, sample_dataset):
        """upload(dry_run=True) builds but doesn't push."""
        config = HubConfig(
            dataset_dir=sample_dataset,
            repo_id="test/repo",
            configs=["labels_only"],
        )
        uploader = DatasetUploader(config)

        with patch.object(uploader, "build_dataset_dict") as mock_build:
            mock_dd = MagicMock()
            mock_build.return_value = mock_dd

            uploader.upload(dry_run=True)

            mock_build.assert_called_once()
            mock_dd.push_to_hub.assert_not_called()

    @patch("dataset_gen.hub.uploader.HfApi")
    def test_upload_pushes_auxiliary_tables(self, mock_hf_api_cls, sample_dataset):
        """upload() uploads strokes.parquet and measures.parquet to auxiliary/."""
        # Create strokes and measures parquets
        labels_dir = sample_dataset / "labels"
        pd.DataFrame({"sample_id": ["s1"], "index": [0], "timing_error": [0.01]}).to_parquet(
            labels_dir / "strokes.parquet"
        )
        pd.DataFrame({"sample_id": ["s1"], "index": [0], "mean_timing": [0.02]}).to_parquet(
            labels_dir / "measures.parquet"
        )

        config = HubConfig(
            dataset_dir=sample_dataset,
            repo_id="test/repo",
            configs=["labels_only"],
        )
        uploader = DatasetUploader(config)

        with patch.object(uploader, "build_dataset_dict") as mock_build:
            mock_dd = MagicMock()
            mock_build.return_value = mock_dd

            uploader.upload()

            # Check that auxiliary files were uploaded
            mock_api = mock_hf_api_cls.return_value
            upload_calls = mock_api.upload_file.call_args_list
            uploaded_paths = [c[1]["path_in_repo"] for c in upload_calls]
            assert "auxiliary/strokes.parquet" in uploaded_paths
            assert "auxiliary/measures.parquet" in uploaded_paths
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hub_uploader.py::TestUploadParquet -v`
Expected: FAIL (old `upload` method has different signature/behavior)

**Step 3: Rewrite the upload method**

Replace the existing `upload` method in `DatasetUploader`:

```python
def upload(self, dry_run: bool = False, purge: bool = False) -> str | None:
    """
    Upload dataset to HuggingFace Hub as Parquet shards.

    Args:
        dry_run: If True, build DatasetDicts but don't push
        purge: If True, delete all existing files before upload

    Returns:
        URL to the uploaded dataset, or None if dry_run
    """
    from huggingface_hub import HfApi

    api = HfApi(token=self.config.token)

    if not dry_run:
        # Create repository if needed
        logger.info(f"Creating/verifying repository: {self.config.repo_id}")
        api.create_repo(
            repo_id=self.config.repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=self.config.private,
        )

        if purge:
            self.purge_repo(api=api)

    # Build and push each config
    for config_name in self.config.configs:
        logger.info(f"Building DatasetDict for config '{config_name}'...")
        dd = self.build_dataset_dict(config_name)

        # Log stats
        for split_name, ds in dd.items():
            logger.info(f"  {config_name}/{split_name}: {ds.num_rows} rows, "
                        f"columns: {ds.column_names}")

        if dry_run:
            logger.info(f"DRY RUN: would push config '{config_name}' to {self.config.repo_id}")
            continue

        logger.info(f"Pushing config '{config_name}' to {self.config.repo_id}...")
        dd.push_to_hub(
            self.config.repo_id,
            config_name=config_name,
            max_shard_size=self.config.max_shard_size,
            token=self.config.token,
            private=self.config.private,
        )
        logger.info(f"Config '{config_name}' pushed successfully")

    # Upload auxiliary tables
    if not dry_run:
        self._upload_auxiliary(api)

    # Upload README
    if not dry_run:
        readme_path = self.config.dataset_dir / "README.md"
        if readme_path.exists():
            logger.info("Uploading README.md...")
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=self.config.repo_id,
                repo_type="dataset",
                commit_message="docs: update dataset card",
            )

    url = f"https://huggingface.co/datasets/{self.config.repo_id}"
    if dry_run:
        logger.info(f"DRY RUN complete. Would upload to: {url}")
        return None

    logger.info(f"Upload complete: {url}")
    return url

def _upload_auxiliary(self, api: "HfApi") -> None:
    """Upload stroke and measure level parquet tables to auxiliary/."""
    labels_dir = self.config.dataset_dir / "labels"

    for filename in ["strokes.parquet", "measures.parquet"]:
        src = labels_dir / filename
        if src.exists():
            logger.info(f"Uploading auxiliary/{filename}...")
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=f"auxiliary/{filename}",
                repo_id=self.config.repo_id,
                repo_type="dataset",
                commit_message=f"data: upload auxiliary {filename}",
            )
        else:
            logger.warning(f"Auxiliary file not found: {src}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hub_uploader.py::TestUploadParquet -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "feat(hub): rewrite upload to push Parquet configs via datasets API"
```

---

### Task 5: Remove old staging code and update convenience functions

**Files:**
- Modify: `dataset_gen/hub/uploader.py`
- Modify: `dataset_gen/hub/__init__.py`

**Step 1: Write the failing test**

```python
class TestConvenienceFunctions:
    """Tests for updated module-level convenience functions."""

    def test_push_to_hub_signature(self):
        """push_to_hub accepts new kwargs (configs, purge)."""
        import inspect
        sig = inspect.signature(push_to_hub)
        params = list(sig.parameters.keys())
        assert "configs" in params
        assert "purge" in params
        assert "max_shard_size" in params
        # Old params should be gone
        assert "include_audio" not in params
        assert "include_midi" not in params

    def test_prepare_hf_structure_removed(self):
        """prepare_hf_structure convenience function is removed."""
        from dataset_gen.hub import __all__
        assert "prepare_hf_structure" not in __all__
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hub_uploader.py::TestConvenienceFunctions -v`
Expected: FAIL

**Step 3: Remove old code, update convenience functions**

In `dataset_gen/hub/uploader.py`:

1. Delete the `prepare` method entirely
2. Delete `_copy_media_by_rudiment` method
3. Delete `_count_media_files` method
4. Delete the `prepare_hf_structure` function
5. Update the `push_to_hub` convenience function:

```python
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
```

Update `dataset_gen/hub/__init__.py`:

```python
from dataset_gen.hub.uploader import (
    HubConfig,
    DatasetUploader,
    push_to_hub,
)

__all__ = [
    "HubConfig",
    "DatasetUploader",
    "push_to_hub",
]
```

Also update/remove `UploadStats` if it is no longer used by the new code. The new `upload` method doesn't use the stats tracking — the `datasets` library handles progress. You can keep `UploadStats` for the dry-run summary if useful, or remove it.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hub_uploader.py::TestConvenienceFunctions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/hub/uploader.py dataset_gen/hub/__init__.py
git commit -m "refactor(hub): remove staging/symlink code, update convenience API"
```

---

### Task 6: Update existing tests

The old tests in `TestDatasetUploaderHelpers` and `test_hub_integration.py` test the staging/symlink flow that no longer exists. They need to be rewritten to test the new DatasetDict flow.

**Files:**
- Modify: `tests/test_hub_uploader.py` (remove old `TestHubConfig`, `TestDatasetUploaderHelpers`)
- Modify: `tests/test_hub_integration.py` (rewrite for new API)

**Step 1: Remove old test classes**

In `tests/test_hub_uploader.py`, delete:
- `TestHubConfig` class (replaced by `TestHubConfigParquet`)
- `TestDatasetUploaderHelpers` class (replaced by `TestBuildDatasetDict`)

**Step 2: Rewrite integration test**

Replace `tests/test_hub_integration.py`:

```python
"""Integration test: generation output is compatible with Parquet hub uploader."""

import json
import numpy as np
import pandas as pd
import pytest

from dataset_gen.pipeline.storage import StorageConfig, DatasetWriter
from dataset_gen.hub.uploader import HubConfig, DatasetUploader
from dataset_gen.labels.schema import Sample, ExerciseScores


@pytest.fixture
def generated_dataset(tmp_path):
    """Generate a small dataset using the storage pipeline."""
    config = StorageConfig(output_dir=tmp_path)
    writer = DatasetWriter(config)

    rudiments = ["flam", "paradiddle", "single_stroke_roll"]
    profiles = {"p1": "train", "p2": "val", "p3": "test"}

    for profile_id in profiles:
        for slug in rudiments:
            sample_id = f"test_{slug}_{profile_id}"
            sample = Sample(
                sample_id=sample_id,
                profile_id=profile_id,
                rudiment_slug=slug,
                tempo_bpm=120,
                duration_sec=2.0,
                num_cycles=2,
                skill_tier="beginner",
                skill_tier_binary="novice",
                dominant_hand="right",
                strokes=[],
                measures=[],
                exercise_scores=ExerciseScores(
                    timing_accuracy=75.0,
                    timing_consistency=70.0,
                    tempo_stability=80.0,
                    subdivision_evenness=65.0,
                    velocity_control=72.0,
                    accent_differentiation=60.0,
                    accent_accuracy=68.0,
                    hand_balance=74.0,
                    weak_hand_index=50.0,
                    groove_feel_proxy=0.7,
                    overall_score=71.0,
                ),
            )
            audio = np.zeros(44100, dtype=np.float32)
            midi = b"\x00" * 100
            writer.write_sample(sample, midi_data=midi, audio_data=audio)

    writer.flush()

    splits = {
        "train_profile_ids": ["p1"],
        "val_profile_ids": ["p2"],
        "test_profile_ids": ["p3"],
    }
    with open(tmp_path / "splits.json", "w") as f:
        json.dump(splits, f)

    return tmp_path


def test_generated_dataset_builds_labels_only(generated_dataset):
    """Labels-only config builds successfully from generated data."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("labels_only")

    assert dd["train"].num_rows == 3
    assert dd["validation"].num_rows == 3
    assert dd["test"].num_rows == 3
    assert "overall_score" in dd["train"].column_names


def test_generated_dataset_builds_midi_only(generated_dataset):
    """MIDI-only config includes MIDI bytes from generated data."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("midi_only")

    assert "midi" in dd["train"].column_names
    midi_val = dd["train"][0]["midi"]
    assert isinstance(midi_val, bytes)
    assert len(midi_val) > 0


def test_generated_dataset_builds_audio(generated_dataset):
    """Audio config includes audio from generated data."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("audio")

    assert "audio" in dd["train"].column_names
    assert "midi" in dd["train"].column_names


def test_all_splits_have_correct_sample_count(generated_dataset):
    """Each split has the correct number of samples based on profile assignment."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)
    dd = uploader.build_dataset_dict("labels_only")

    # 3 profiles, 3 rudiments each = 9 total, 3 per split
    total = sum(ds.num_rows for ds in dd.values())
    assert total == 9
```

**Step 3: Run all hub tests**

Run: `pytest tests/test_hub_uploader.py tests/test_hub_integration.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/test_hub_uploader.py tests/test_hub_integration.py
git commit -m "test(hub): rewrite tests for Parquet-native upload"
```

---

### Task 7: Update push_to_hub.py CLI script

**Files:**
- Modify: `scripts/push_to_hub.py`

**Step 1: Rewrite the CLI script**

Replace the contents of `scripts/push_to_hub.py`:

```python
#!/usr/bin/env python3
"""
Push SOUSA dataset to HuggingFace Hub as Parquet shards.
========================================================

Upload the generated SOUSA dataset with embedded audio/MIDI in Parquet format.

Usage:
    python scripts/push_to_hub.py zkeown/sousa                       # Upload all configs
    python scripts/push_to_hub.py zkeown/sousa --dry-run              # Test without upload
    python scripts/push_to_hub.py zkeown/sousa --configs labels_only  # Labels only (~50MB)
    python scripts/push_to_hub.py zkeown/sousa --purge                # Clear repo first

Prerequisites:
    1. Install hub dependencies: pip install 'sousa[hub]'
    2. Login to HuggingFace: huggingface-cli login
    3. Generate dataset: python scripts/generate_dataset.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_gen.hub import HubConfig, DatasetUploader


VALID_CONFIGS = ["audio", "midi_only", "labels_only"]


def main():
    parser = argparse.ArgumentParser(
        description="Push SOUSA dataset to HuggingFace Hub as Parquet shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/push_to_hub.py zkeown/sousa                        # All configs
  python scripts/push_to_hub.py zkeown/sousa --configs labels_only   # Labels only
  python scripts/push_to_hub.py zkeown/sousa --configs midi_only audio  # MIDI + audio
  python scripts/push_to_hub.py zkeown/sousa --purge --yes           # Purge without prompt
  python scripts/push_to_hub.py zkeown/sousa --dry-run               # Test run

Configs:
  audio        Full dataset with FLAC audio + MIDI bytes + labels (~96GB)
  midi_only    MIDI bytes + labels, no audio (~2.5GB)
  labels_only  Pure tabular metadata + scores (~50MB)

Prerequisites:
  pip install 'sousa[hub]'
  huggingface-cli login
        """,
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repository ID (e.g., zkeown/sousa)",
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        type=Path,
        default=Path("output/dataset"),
        help="Path to generated dataset (default: output/dataset)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=VALID_CONFIGS,
        default=None,
        help="Which configs to upload (default: all three)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="1GB",
        help="Max Parquet shard size (default: 1GB)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete all existing files from the repo before upload",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="HuggingFace API token (uses cached token if not provided)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build DatasetDicts but don't upload (for testing)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate dataset directory
    if not args.dataset_dir.exists():
        logger.error(f"Dataset directory not found: {args.dataset_dir}")
        logger.error("Run 'python scripts/generate_dataset.py' first")
        sys.exit(1)

    labels_dir = args.dataset_dir / "labels"
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Check dependencies
    try:
        import huggingface_hub  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        logger.error("Required packages not installed")
        logger.error("Install with: pip install 'sousa[hub]'")
        sys.exit(1)

    configs = args.configs or VALID_CONFIGS

    # Print plan
    print("\n" + "=" * 60)
    print("HUGGINGFACE HUB UPLOAD (Parquet)")
    print("=" * 60)
    print(f"\nRepository: {args.repo_id}")
    print(f"Visibility: {'Private' if args.private else 'Public'}")
    print(f"Source:     {args.dataset_dir}")
    print(f"Configs:    {', '.join(configs)}")
    print(f"Shard size: {args.max_shard_size}")
    if args.purge:
        print("Purge:      YES - will delete all existing files first")
    if args.dry_run:
        print("Mode:       DRY RUN (no actual upload)")
    print("=" * 60 + "\n")

    # Confirm purge
    if args.purge and not args.dry_run and not args.yes:
        response = input(
            f"This will DELETE all files from {args.repo_id}. Continue? [y/N] "
        )
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Configure and upload
    config = HubConfig(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        configs=configs,
        max_shard_size=args.max_shard_size,
    )

    uploader = DatasetUploader(config)

    try:
        url = uploader.upload(dry_run=args.dry_run, purge=args.purge)

        if args.dry_run:
            print("\nDRY RUN complete. No files were uploaded.")
            print("Remove --dry-run to upload for real.")
        else:
            print("\n" + "=" * 60)
            print("UPLOAD COMPLETE")
            print("=" * 60)
            print(f"\nDataset URL: {url}")
            print("\nLoad the dataset:")
            print(f'  ds = load_dataset("{args.repo_id}")               # audio (default)')
            print(f'  ds = load_dataset("{args.repo_id}", "midi_only")  # MIDI + labels')
            print(f'  ds = load_dataset("{args.repo_id}", "labels_only") # labels only')
            print("=" * 60)

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
```

**Step 2: Run lint to verify**

Run: `ruff check scripts/push_to_hub.py && black --check scripts/push_to_hub.py`
Expected: PASS

**Step 3: Commit**

```bash
git add scripts/push_to_hub.py
git commit -m "feat(cli): rewrite push_to_hub for Parquet configs"
```

---

### Task 8: Update README.md YAML frontmatter for configs

**Files:**
- Modify: `output/dataset/README.md`

**Step 1: Update YAML frontmatter**

Replace the YAML frontmatter at the top of `output/dataset/README.md`:

```yaml
---
license: mit
task_categories:
  - audio-classification
  - tabular-classification
  - tabular-regression
tags:
  - audio
  - music
  - drum-rudiments
  - synthetic
  - snare-drum
  - midi
  - percussion
  - music-information-retrieval
size_categories:
  - 100K<n<1M
language:
  - en
pretty_name: SOUSA - Synthetic Open Unified Snare Assessment
configs:
  - config_name: audio
    data_files:
      - split: train
        path: audio/train-*.parquet
      - split: validation
        path: audio/validation-*.parquet
      - split: test
        path: audio/test-*.parquet
    default: true
  - config_name: midi_only
    data_files:
      - split: train
        path: midi_only/train-*.parquet
      - split: validation
        path: midi_only/validation-*.parquet
      - split: test
        path: midi_only/test-*.parquet
  - config_name: labels_only
    data_files:
      - split: train
        path: labels_only/train-*.parquet
      - split: validation
        path: labels_only/validation-*.parquet
      - split: test
        path: labels_only/test-*.parquet
---
```

**Step 2: Update the Usage section**

Replace the Usage section in the README body:

```markdown
## Usage

```python
from datasets import load_dataset

# Full dataset with audio (~96GB download)
ds = load_dataset("zkeown/sousa")

# MIDI + labels only (~2.5GB download)
ds = load_dataset("zkeown/sousa", "midi_only")

# Pure tabular labels (~50MB download)
ds = load_dataset("zkeown/sousa", "labels_only")

# Access a sample
sample = ds["train"][0]
print(sample["rudiment_slug"], sample["skill_tier"], sample["overall_score"])

# Access audio (decoded automatically)
audio_array = sample["audio"]["array"]     # numpy array
sample_rate = sample["audio"]["sampling_rate"]  # 44100

# Access MIDI bytes
midi_bytes = sample["midi"]  # raw .mid file bytes
```
```

**Step 3: Update the File Organization section**

Replace with:

```markdown
### File Organization

The dataset uses three configurations with different levels of content:

| Config | Contents | Size |
|--------|----------|------|
| `audio` (default) | FLAC audio + MIDI bytes + metadata + scores | ~96GB |
| `midi_only` | MIDI bytes + metadata + scores | ~2.5GB |
| `labels_only` | Metadata + scores only | ~50MB |

Stroke-level and measure-level labels are available as auxiliary files:

```python
import pandas as pd
strokes = pd.read_parquet("hf://datasets/zkeown/sousa/auxiliary/strokes.parquet")
measures = pd.read_parquet("hf://datasets/zkeown/sousa/auxiliary/measures.parquet")
```
```

**Step 4: Commit**

```bash
git add output/dataset/README.md
git commit -m "docs: update dataset card for Parquet configs"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run lint**

Run: `ruff check dataset_gen/ scripts/ && black --check dataset_gen/ scripts/`
Expected: PASS

**Step 3: Run a dry-run upload to verify the full flow**

Run: `python scripts/push_to_hub.py zkeown/sousa --dry-run --configs labels_only`
Expected: Prints schema info, shard counts, and "DRY RUN complete" without uploading

**Step 4: Commit any fixes**

If any fixes were needed, commit them:

```bash
git add -A
git commit -m "fix: address test/lint issues from Parquet migration"
```

---

### Task 10: Execute the actual upload

This is the manual execution step — not automated.

**Step 1: Upload labels_only first (validates pipeline)**

Run: `python scripts/push_to_hub.py zkeown/sousa --purge --configs labels_only --yes`
Expected: Purges existing repo, uploads ~50MB of labels-only Parquet shards

**Step 2: Verify on hub**

Visit `https://huggingface.co/datasets/zkeown/sousa` and confirm:
- Dataset viewer works
- `labels_only` config shows correct columns and data

**Step 3: Upload midi_only**

Run: `python scripts/push_to_hub.py zkeown/sousa --configs midi_only`
Expected: Uploads ~2.5GB of MIDI + labels Parquet shards

**Step 4: Upload audio**

Run: `python scripts/push_to_hub.py zkeown/sousa --configs audio`
Expected: Uploads ~96GB across ~96 Parquet shards (this will take a while)

**Step 5: Verify all configs on hub**

```python
from datasets import load_dataset
ds = load_dataset("zkeown/sousa", "labels_only", split="train[:5]")
print(ds)
ds = load_dataset("zkeown/sousa", "midi_only", split="train[:5]")
print(ds[0]["midi"][:20])
ds = load_dataset("zkeown/sousa", split="train[:1]")
print(ds[0]["audio"])
```

**Step 6: Commit final state**

```bash
git add -A
git commit -m "feat: complete Parquet hub migration"
```
