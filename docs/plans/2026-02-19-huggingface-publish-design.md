# SOUSA HuggingFace Publish Design

**Date:** 2026-02-19
**Target repo:** `zkeown/sousa`
**Dataset size:** ~121 GB (100K samples, 100K audio FLAC, 20K MIDI, 4 label parquets)

## Problem

SOUSA generates a synthetic drum rudiment dataset locally but it has never been published to HuggingFace Hub. The current generation pipeline writes media files into flat `audio/` and `midi/` directories (100K+ files total), which exceeds HuggingFace's per-directory limits and is poorly organized for browsing.

An existing branch (`refactor/organize-by-rudiment`) refactors the Hub uploader to organize media into `audio/{rudiment_slug}/` subdirectories at upload time. However, the generation pipeline itself still outputs flat directories, creating a mismatch between source and published layout.

## Decision

Align everything end-to-end: update the generation pipeline to write into rudiment subdirectories natively, regenerate the dataset, then publish to HuggingFace with a proper dataset card.

## Design

### Phase 1: Merge `refactor/organize-by-rudiment`

1. Check for merge conflicts against main
2. Run test suite on the branch
3. Fast-forward merge into main
4. Delete the local branch

### Phase 2: Update generation pipeline to write by rudiment

**`dataset_gen/pipeline/storage.py` changes:**

- `_setup_directories()`: Stop pre-creating flat `audio/` and `midi/` dirs. Rudiment subdirs are created on demand.
- `_write_midi(sample_id, midi_data)` -> `_write_midi(sample_id, rudiment_slug, midi_data)`: Write to `midi/{rudiment_slug}/{sample_id}.mid`
- `_write_audio(sample_id, audio_data)` -> `_write_audio(sample_id, rudiment_slug, audio_data)`: Write to `audio/{rudiment_slug}/{sample_id}.flac`
- `write_sample()`: Pass `sample.rudiment_slug` to the write methods
- Stored `midi_path`/`audio_path` in parquet reflect new paths (e.g., `midi/single_stroke_roll/beg001_single_stroke_roll_120bpm.mid`)

**`dataset_gen/hub/uploader.py` changes:**

- `_copy_media_by_rudiment()`: Update to handle already-organized source (read from `audio/{slug}/` subdirs instead of flat `audio/`)
- The uploader's `prepare()` creates the same `audio/{slug}/` layout in staging, but now source files are already organized identically

**Test and validation updates:**

- Update `tests/test_hub_uploader.py` fixtures to use organized source layout
- Update any validation scripts that assume flat audio/midi directories
- Add integration test: generate small sample set, run uploader `prepare()`, verify staged paths match generation output paths

**Sousaphone model (sibling directory):**

- Check and update any hardcoded audio/midi path references that assume flat layout

### Phase 3: Regenerate dataset

1. Run: `python scripts/generate_dataset.py --preset full --with-audio` (~45 min)
2. Validate:
   - `python scripts/check_generation.py output/dataset`
   - `pytest tests/ -v`
   - `python scripts/dataset_stats.py output/dataset`

### Phase 4: Create dataset card

Create `output/dataset/README.md` with:

- **YAML frontmatter**: `dataset_info` with features, splits, size estimates; `license: apache-2.0`; tags: `audio`, `music`, `drum-rudiments`, `synthetic`, `snare-drum`, `midi`; task categories
- **Dataset description**: SOUSA overview, 40 PAS rudiments, 100K samples, 4 skill tiers, hierarchical labels
- **Dataset structure**: Splits (train/val/test by player profile), columns, file organization (`audio/{rudiment_slug}/`, `midi/{rudiment_slug}/`)
- **Usage example**: `load_dataset("zkeown/sousa")` Python snippet
- **Features table**: Column names, types, descriptions
- **Generation info**: Link to GitHub repo, generation command
- **Citation**: BibTeX entry

### Phase 5: Upload to HuggingFace

1. Verify auth: `huggingface-cli whoami`
2. Dry run: `python scripts/push_to_hub.py zkeown/sousa --dry-run`
   - Confirm staging directory structure looks correct
   - Verify split parquet files have correct sample counts
3. Upload: `python scripts/push_to_hub.py zkeown/sousa`
   - Uses `upload_large_folder` (chunked, resumable)
   - If connection drops, rerun the same command to resume
4. Monitor progress in terminal output

### Phase 6: Post-upload verification

1. Check dataset page at `https://huggingface.co/datasets/zkeown/sousa`
2. Verify dataset card renders correctly
3. Smoke test:
   ```python
   from datasets import load_dataset
   ds = load_dataset("zkeown/sousa")
   print(ds)
   print(ds["train"][0])
   ```
4. Verify split sample counts match local dataset
5. Spot-check audio file accessibility

## Key Files

| File | Role |
|------|------|
| `dataset_gen/pipeline/storage.py` | Generation output layout (Phase 2) |
| `dataset_gen/hub/uploader.py` | HF upload staging + upload (Phase 2, 5) |
| `dataset_gen/hub/archiver.py` | Deleted by merge (Phase 1) |
| `scripts/push_to_hub.py` | Upload CLI entry point (Phase 5) |
| `scripts/generate_dataset.py` | Regeneration entry point (Phase 3) |
| `output/dataset/README.md` | Dataset card (Phase 4) |
| `tests/test_hub_uploader.py` | Uploader tests (Phase 2) |

## Risks

- **Upload time**: 121 GB over `upload_large_folder` could take hours. Mitigated by resume capability.
- **Regeneration variance**: Synthetic data is seeded but any code change could shift output. Mitigated by validation suite.
- **Sousaphone model compatibility**: Path changes could break sibling project. Needs explicit check.

## Out of scope

- Changing the dataset content or labels
- Audio augmentation changes
- HuggingFace dataset viewer configuration (auto-detected from parquet)
