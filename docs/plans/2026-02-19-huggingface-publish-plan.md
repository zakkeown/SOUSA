# HuggingFace Publish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Publish the SOUSA dataset to HuggingFace Hub at `zkeown/sousa` with media organized by rudiment subdirectories.

**Architecture:** Merge the existing organize-by-rudiment uploader refactor, then update the generation pipeline to match that layout natively. Regenerate the dataset, create a dataset card, and upload via `upload_large_folder`.

**Tech Stack:** Python, pandas, pyarrow, huggingface_hub, soundfile, FluidSynth

---

### Task 1: Merge `refactor/organize-by-rudiment` into main

**Files:**
- Deleted: `dataset_gen/hub/archiver.py`
- Deleted: `tests/test_hub_archiver.py`
- Modified: `dataset_gen/hub/__init__.py`
- Modified: `dataset_gen/hub/uploader.py`
- Modified: `scripts/push_to_hub.py`
- Modified: `tests/test_hub_uploader.py`

**Step 1: Check for conflicts**

Run: `git merge --no-commit --no-ff refactor/organize-by-rudiment`
Expected: Clean merge (no conflicts)
Then: `git merge --abort` (we'll do the real merge next)

**Step 2: Run tests on the branch**

Run: `git stash && git checkout refactor/organize-by-rudiment && pytest tests/test_hub_uploader.py -v`
Expected: All tests pass

**Step 3: Merge into main**

```bash
git checkout main
git merge refactor/organize-by-rudiment --no-ff -m "refactor(hub): organize media by rudiment, remove TAR sharding"
```

**Step 4: Run full test suite on main**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 5: Delete the local branch**

Run: `git branch -d refactor/organize-by-rudiment`

**Step 6: Commit checkpoint**

The merge commit from step 3 is the checkpoint. Verify: `git log --oneline -3`

---

### Task 2: Update `storage.py` to write media into rudiment subdirectories

**Files:**
- Modify: `dataset_gen/pipeline/storage.py:88-150` (directory setup + write methods)
- Test: `tests/test_storage.py` (or existing tests that exercise storage)

**Step 1: Write a failing test for rudiment-organized output**

Create or update test in the appropriate test file. The test generates a sample with a known `rudiment_slug` and verifies the output path includes the rudiment subdirectory.

```python
def test_write_sample_organizes_by_rudiment(tmp_path):
    """write_sample should write audio/midi into rudiment subdirectories."""
    from dataset_gen.pipeline.storage import StorageConfig, DatasetWriter
    from dataset_gen.labels.schema import Sample, ExerciseScores

    config = StorageConfig(output_dir=tmp_path)
    writer = DatasetWriter(config)

    # Create minimal sample with rudiment_slug
    sample = Sample(
        sample_id="test001_flam_120bpm",
        profile_id="p1",
        rudiment_slug="flam",
        tempo_bpm=120,
        duration_sec=2.0,
        num_cycles=2,
        skill_tier="beginner",
        skill_tier_binary="developing",
        dominant_hand="right",
        strokes=[],
        measures=[],
        exercise_scores=ExerciseScores(),
    )

    import numpy as np
    audio_data = np.zeros(44100, dtype=np.float32)  # 1 second of silence
    midi_data = b"\x00" * 100  # Dummy MIDI bytes

    writer.write_sample(sample, midi_data=midi_data, audio_data=audio_data)

    # Verify files are in rudiment subdirectories
    assert (tmp_path / "midi" / "flam" / "test001_flam_120bpm.mid").exists()
    assert (tmp_path / "audio" / "flam" / "test001_flam_120bpm.flac").exists()

    # Verify stored paths in sample reflect rudiment subdirs
    assert sample.midi_path == "midi/flam/test001_flam_120bpm.mid"
    assert sample.audio_path == "audio/flam/test001_flam_120bpm.flac"
```

**Step 2: Run the test to verify it fails**

Run: `pytest tests/test_storage_rudiment.py::test_write_sample_organizes_by_rudiment -v`
Expected: FAIL - files written to flat `midi/test001_flam_120bpm.mid` instead of `midi/flam/...`

**Step 3: Implement the changes in `storage.py`**

In `dataset_gen/pipeline/storage.py`, make these changes:

1. `_setup_directories()` (line 88-93): Remove pre-creation of flat `audio/` and `midi/` dirs. Only create `labels/` dir.

```python
def _setup_directories(self) -> None:
    """Create output directory structure."""
    self.config.output_dir.mkdir(parents=True, exist_ok=True)
    (self.config.output_dir / self.config.labels_subdir).mkdir(exist_ok=True)
```

2. `write_sample()` (line 95-130): Pass `sample.rudiment_slug` to write methods.

```python
# Write MIDI
if midi_data is not None:
    midi_path = self._write_midi(sample.sample_id, sample.rudiment_slug, midi_data)
    paths["midi"] = midi_path
    sample.midi_path = str(midi_path.relative_to(self.config.output_dir))

# Write audio
if audio_data is not None:
    audio_path = self._write_audio(sample.sample_id, sample.rudiment_slug, audio_data)
    paths["audio"] = audio_path
    sample.audio_path = str(audio_path.relative_to(self.config.output_dir))
```

3. `_write_midi()` (line 132-136): Accept `rudiment_slug`, create subdirectory, write there.

```python
def _write_midi(self, sample_id: str, rudiment_slug: str, midi_data: bytes) -> Path:
    """Write MIDI data to file in rudiment subdirectory."""
    rudiment_dir = self.config.output_dir / self.config.midi_subdir / rudiment_slug
    rudiment_dir.mkdir(parents=True, exist_ok=True)
    midi_path = rudiment_dir / f"{sample_id}.mid"
    midi_path.write_bytes(midi_data)
    return midi_path
```

4. `_write_audio()` (line 138-150): Accept `rudiment_slug`, create subdirectory, write there.

```python
def _write_audio(self, sample_id: str, rudiment_slug: str, audio_data: np.ndarray) -> Path:
    """Write audio data to file in rudiment subdirectory."""
    ext = self.config.audio_format
    rudiment_dir = self.config.output_dir / self.config.audio_subdir / rudiment_slug
    rudiment_dir.mkdir(parents=True, exist_ok=True)
    audio_path = rudiment_dir / f"{sample_id}.{ext}"

    sf.write(
        str(audio_path),
        audio_data,
        self.config.sample_rate,
        format=self.config.audio_format.upper(),
        subtype=self.config.audio_subtype,
    )
    return audio_path
```

5. Update the docstring (line 51-67) to reflect the new directory structure:

```
output_dir/
├── midi/
│   └── {rudiment_slug}/
│       └── {sample_id}.mid
├── audio/
│   └── {rudiment_slug}/
│       └── {sample_id}.flac
├── labels/
│   ├── strokes.parquet
│   ├── measures.parquet
│   ├── exercises.parquet
│   └── samples.parquet
└── index.json
```

**Step 4: Run the test to verify it passes**

Run: `pytest tests/test_storage_rudiment.py::test_write_sample_organizes_by_rudiment -v`
Expected: PASS

**Step 5: Run the full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: Some existing tests may fail if they assume flat layout. Note which ones.

**Step 6: Commit**

```bash
git add dataset_gen/pipeline/storage.py tests/test_storage_rudiment.py
git commit -m "feat(storage): write media files into rudiment subdirectories"
```

---

### Task 3: Update `generate_dataset.py` worker merge to handle rudiment subdirs

**Files:**
- Modify: `scripts/generate_dataset.py:169-217` (merge_worker_outputs function)

**Step 1: Update `merge_worker_outputs()` to handle rudiment subdirectories**

The function currently creates flat `audio/` and `midi/` dirs and moves files directly. It needs to handle the nested `audio/{rudiment_slug}/` structure from each worker.

Replace lines 169-217 with logic that:
1. Does NOT pre-create flat `main_midi_dir` / `main_audio_dir`
2. Iterates over worker `audio/` and `midi/` dirs, preserving rudiment subdirectory structure
3. Moves files from `worker_N/audio/{slug}/file.flac` to `output/audio/{slug}/file.flac`

```python
main_labels_dir = output_dir / "labels"
main_labels_dir.mkdir(parents=True, exist_ok=True)

# ... (parquet merge logic stays the same) ...

# Move MIDI and audio files preserving rudiment subdirectories
for w in range(num_workers):
    worker_dir = output_dir / f"worker_{w}"

    for media_type in ["midi", "audio"]:
        worker_media = worker_dir / media_type
        if not worker_media.exists():
            continue
        main_media = output_dir / media_type
        # Walk rudiment subdirectories
        for rudiment_dir in worker_media.iterdir():
            if not rudiment_dir.is_dir():
                continue
            dest_dir = main_media / rudiment_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for media_file in rudiment_dir.iterdir():
                if media_file.is_file():
                    shutil.move(str(media_file), str(dest_dir / media_file.name))

    # Clean up worker directory
    shutil.rmtree(worker_dir, ignore_errors=True)
```

**Step 2: Run tests**

Run: `pytest tests/ -v`
Expected: Pass

**Step 3: Commit**

```bash
git add scripts/generate_dataset.py
git commit -m "feat(generate): preserve rudiment subdirectory structure in worker merge"
```

---

### Task 4: Update validation scripts to use metadata paths

**Files:**
- Modify: `dataset_gen/validation/verify.py:707` (MIDI path construction)
- Modify: `dataset_gen/validation/audio_check.py:101,244` (audio path construction)

**Context:** Both validators currently construct paths as `self.midi_dir / f"{sample_id}.mid"` (flat). They need to use the `audio_path` / `midi_path` columns from the metadata instead, since files are now in rudiment subdirectories.

**Step 1: Update `verify.py` MIDI alignment check**

At line 707, change:
```python
# Old: midi_path = self.midi_dir / f"{sample_id}.mid"
# New: use midi_path from metadata
midi_path_str = sample_row.get("midi_path")
if pd.isna(midi_path_str) or not midi_path_str:
    continue
midi_path = self.dataset_dir / midi_path_str
```

**Step 2: Update `audio_check.py` audio file resolution**

At line 244, change:
```python
# Old: audio_path = self.audio_dir / f"{sample_id}.flac"
# New: use audio_path from metadata
audio_path_str = row.get("audio_path")
if pd.isna(audio_path_str) or not audio_path_str:
    continue
audio_path = self.dataset_dir / audio_path_str
```

Also update line 101: remove `self.audio_dir = self.dataset_dir / "audio"` if no longer needed (check all usages first).

At lines 369-371 (`__main__` section), update the audio dir existence check to check for `audio/` with subdirectories rather than flat files.

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: Pass

**Step 4: Commit**

```bash
git add dataset_gen/validation/verify.py dataset_gen/validation/audio_check.py
git commit -m "fix(validation): resolve media paths from metadata instead of flat directory"
```

---

### Task 5: Update hub uploader to handle already-organized source

**Files:**
- Modify: `dataset_gen/hub/uploader.py` (`_copy_media_by_rudiment` method)

**Context:** After the merge in Task 1, the uploader reads from flat `audio/` and reorganizes into `audio/{slug}/`. Now that generation writes into `audio/{slug}/` natively, the uploader's source path construction needs updating. The source files are at `audio/{slug}/{filename}` not `audio/{filename}`.

**Step 1: Update `_copy_media_by_rudiment()`**

The method currently looks up source files as `src_dir / filename`. It needs to look them up as `src_dir / slug / filename` since the source is now organized by rudiment.

```python
# Change from:
# src_file = src_dir / filename
# To:
src_file = src_dir / slug / filename
```

The destination path stays the same (it's already `dst_base / slug / filename`).

**Step 2: Run uploader tests**

Run: `pytest tests/test_hub_uploader.py -v`

Update test fixtures if needed: source audio/midi files should be in rudiment subdirectories.

**Step 3: Commit**

```bash
git add dataset_gen/hub/uploader.py tests/test_hub_uploader.py
git commit -m "fix(hub): read media from rudiment subdirectories in source"
```

---

### Task 6: Integration test -- generation output matches uploader expectations

**Files:**
- Create: `tests/test_hub_integration.py`

**Step 1: Write the integration test**

```python
"""Integration test: generation output layout matches uploader staging."""
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

    for profile_id, split in profiles.items():
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
                skill_tier_binary="developing",
                dominant_hand="right",
                strokes=[],
                measures=[],
                exercise_scores=ExerciseScores(),
            )
            audio = np.zeros(44100, dtype=np.float32)
            midi = b"\x00" * 100
            writer.write_sample(sample, midi_data=midi, audio_data=audio)

    writer.flush()

    # Create splits.json
    splits = {
        "train_profile_ids": ["p1"],
        "val_profile_ids": ["p2"],
        "test_profile_ids": ["p3"],
    }
    with open(tmp_path / "splits.json", "w") as f:
        json.dump(splits, f)

    return tmp_path


def test_generation_layout_matches_uploader(generated_dataset):
    """Verify uploader can stage a dataset generated with rudiment subdirs."""
    config = HubConfig(dataset_dir=generated_dataset, repo_id="test/repo")
    uploader = DatasetUploader(config)

    staging_dir = uploader.prepare()

    # Check parquet has correct paths
    train_df = pd.read_parquet(staging_dir / "data" / "train-00000-of-00001.parquet")
    assert "audio" in train_df.columns

    # Every audio path in parquet should point to a real file in staging
    for _, row in train_df.iterrows():
        audio_path = row.get("audio")
        if audio_path:
            assert (staging_dir / audio_path).exists(), f"Missing: {audio_path}"

        midi_path = row.get("midi")
        if midi_path:
            assert (staging_dir / midi_path).exists(), f"Missing: {midi_path}"
```

**Step 2: Run the integration test**

Run: `pytest tests/test_hub_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_hub_integration.py
git commit -m "test: add integration test for generation-to-upload path consistency"
```

---

### Task 7: Regenerate the dataset

**Step 1: Delete old dataset output**

Run: `rm -rf output/dataset`

**Step 2: Regenerate with full preset and audio**

Run: `python scripts/generate_dataset.py --preset full --with-audio`
Expected: ~45 minutes. Output at `output/dataset/` with `audio/{slug}/` and `midi/{slug}/` subdirectories.

**Step 3: Quick validation**

Run: `python scripts/check_generation.py output/dataset`
Expected: Exit code 0, no critical issues.

**Step 4: Full test suite**

Run: `pytest tests/ -v`
Expected: All pass.

**Step 5: Dataset stats**

Run: `python scripts/dataset_stats.py output/dataset`
Expected: ~100K samples, ~100K audio files, ~20K MIDI files across 40 rudiment subdirectories.

**Step 6: Spot-check directory structure**

Run: `ls output/dataset/audio/ | head -10` -- should list rudiment slugs.
Run: `ls output/dataset/audio/single_stroke_roll/ | head -5` -- should list FLAC files.
Run: `ls output/dataset/midi/flam/ | head -5` -- should list MID files.

---

### Task 8: Create dataset card

**Files:**
- Create: `output/dataset/README.md`

**Step 1: Create the dataset card**

Write `output/dataset/README.md` with YAML frontmatter and markdown content.

The card should include:
- YAML frontmatter with: `license`, `task_categories`, `tags`, `size_categories`, `dataset_info` (features, splits)
- Description of SOUSA: what it is, 40 PAS rudiments, 100K samples, 4 skill tiers
- Dataset structure: splits (train/val/test by player profile), file organization
- Features table: all columns from the merged parquet with types and descriptions
- Usage example with `load_dataset`
- Generation info with link to GitHub
- Citation BibTeX

Refer to existing parquet schema for column names and types. Key columns:
- From `samples.parquet`: sample_id, profile_id, rudiment_slug, tempo_bpm, duration_sec, num_cycles, skill_tier, skill_tier_binary, dominant_hand, midi_path, audio_path, num_strokes, num_measures, soundfont, augmentation_preset, augmentation_group_id, aug_*
- From `exercises.parquet`: timing_accuracy, timing_consistency, tempo_stability, subdivision_evenness, velocity_control, accent_differentiation, accent_accuracy, hand_balance, weak_hand_index, flam_quality, diddle_quality, roll_sustain, groove_feel_proxy, overall_score

**Step 2: Verify the card renders**

Visually inspect the markdown. Check YAML is valid: `python -c "import yaml; yaml.safe_load(open('output/dataset/README.md').read().split('---')[1])"`

**Step 3: Commit code changes (not dataset output)**

```bash
git add -A  # Only staged code changes, NOT output/dataset
git commit -m "docs: create HuggingFace dataset card"
```

Note: The dataset card lives in `output/dataset/` which is gitignored. It will be uploaded to HF Hub.

---

### Task 9: Dry-run upload

**Step 1: Verify HuggingFace auth**

Run: `huggingface-cli whoami`
Expected: Shows `zkeown` username.

**Step 2: Install hub dependencies if needed**

Run: `pip install -e '.[hub]'`

**Step 3: Dry-run upload**

Run: `python scripts/push_to_hub.py zkeown/sousa --dry-run`

Expected output:
- Staging directory created at `output/dataset/hf_staging`
- Shows sample counts per split
- Shows file counts
- Says "DRY RUN complete"

**Step 4: Inspect staging directory**

Run: `ls output/dataset/hf_staging/` -- should show `data/`, `audio/`, `midi/`, `README.md`
Run: `ls output/dataset/hf_staging/audio/` -- should show 40 rudiment subdirectories
Run: `ls output/dataset/hf_staging/data/` -- should show 3 parquet files (train/val/test)

Verify parquet content:
```bash
python -c "
import pandas as pd
for split in ['train', 'validation', 'test']:
    df = pd.read_parquet(f'output/dataset/hf_staging/data/{split}-00000-of-00001.parquet')
    print(f'{split}: {len(df)} samples, columns: {len(df.columns)}')
    print(f'  audio paths sample: {df[\"audio\"].dropna().iloc[0]}')
"
```

---

### Task 10: Upload to HuggingFace Hub

**Step 1: Run the upload**

Run: `python scripts/push_to_hub.py zkeown/sousa`

This will:
1. Create the `zkeown/sousa` dataset repository on HF Hub
2. Prepare staging directory (parquet + symlinked audio/midi)
3. Upload via `upload_large_folder` (chunked, resumable)

Expected: Long-running process. Monitor terminal output for progress.

**Step 2: If upload fails partway, resume**

Simply rerun: `python scripts/push_to_hub.py zkeown/sousa`
The `upload_large_folder` API detects already-uploaded files and skips them.

---

### Task 11: Post-upload verification

**Step 1: Check HF web UI**

Visit `https://huggingface.co/datasets/zkeown/sousa` -- verify:
- Dataset card renders correctly
- File browser shows `audio/`, `midi/`, `data/` directories
- `audio/` contains rudiment subdirectories
- Parquet viewer shows data

**Step 2: Load dataset programmatically**

```python
from datasets import load_dataset
ds = load_dataset("zkeown/sousa")
print(ds)
print(ds["train"][0])
print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")
```

Expected: Matches local dataset counts.

**Step 3: Final commit**

If any fixes were needed, commit them:
```bash
git add -A
git commit -m "chore: post-upload fixes"
```

---

### Task 12: Update SOUSAphone (sibling project)

**Files:**
- Check: `/Users/zakkeown/Code/SOUSAphone/sousa/data/dataset.py:194`
- Check: `/Users/zakkeown/Code/SOUSAphone/tests/data/test_dataset.py:17-22`

**Context:** SOUSAphone's `SOUSADataset.__getitem__()` loads audio via `self.dataset_path / row["audio_path"]`. Since `audio_path` in the regenerated parquet now includes the rudiment slug (e.g., `audio/flam/test001.flac`), this should work automatically. But we need to verify and update test fixtures.

**Step 1: Verify SOUSAphone still loads correctly**

The metadata-driven path construction (`self.dataset_path / row["audio_path"]`) should work with the new paths. Verify by pointing SOUSAphone at the regenerated dataset and running its tests.

**Step 2: Update test fixtures if needed**

The mock CSV in `tests/data/test_dataset.py` uses flat paths like `audio/sample_001.flac`. Update these to include rudiment slugs: `audio/flam/sample_001.flac`.

**Step 3: Run SOUSAphone tests**

Run: `cd /Users/zakkeown/Code/SOUSAphone && pytest tests/ -v`

**Step 4: Commit SOUSAphone changes if any**

```bash
cd /Users/zakkeown/Code/SOUSAphone
git add -A
git commit -m "fix: update test fixtures for rudiment-organized audio paths"
```
