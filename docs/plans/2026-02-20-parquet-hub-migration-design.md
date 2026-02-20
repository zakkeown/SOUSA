# Parquet Hub Migration Design

**Date**: 2026-02-20
**Status**: Approved
**Problem**: Hub upload produces 120K+ individual files (MIDI + FLAC), exceeding HuggingFace's 100K file limit.
**Solution**: Embed media as binary columns in Parquet shards, reduce to ~130 files total.

## Repository Structure

```
zkeown/sousa/
├── README.md
├── audio/
│   ├── train-00000-of-00096.parquet      # ~1GB each
│   ├── ...
│   ├── validation-00000-of-00019.parquet
│   └── test-00000-of-00027.parquet
├── midi_only/
│   ├── train-00000-of-00003.parquet
│   ├── ...
│   └── test-00000-of-00001.parquet
├── labels_only/
│   ├── train-00000-of-00001.parquet
│   ├── validation-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
└── auxiliary/
    ├── strokes.parquet
    └── measures.parquet
```

## Three Dataset Configurations

```python
from datasets import load_dataset

ds = load_dataset("zkeown/sousa")                  # "audio" config (default), ~96GB
ds = load_dataset("zkeown/sousa", "midi_only")      # MIDI + labels, ~2.5GB
ds = load_dataset("zkeown/sousa", "labels_only")    # Pure tabular, ~50MB
```

## Schema

### Main Configs (audio / midi_only / labels_only)

| Column | Type | In audio | In midi_only | In labels_only |
|--------|------|----------|--------------|----------------|
| sample_id | string | Y | Y | Y |
| profile_id | string | Y | Y | Y |
| rudiment_slug | string | Y | Y | Y |
| skill_tier | string | Y | Y | Y |
| tempo_bpm | float64 | Y | Y | Y |
| duration_sec | float64 | Y | Y | Y |
| num_cycles | int64 | Y | Y | Y |
| dominant_hand | string | Y | Y | Y |
| soundfont | string | Y | Y | Y |
| augmentation_preset | string | Y | Y | Y |
| audio | Audio (FLAC + sr) | Y | - | - |
| midi | binary | Y | Y | - |
| timing_accuracy | float64 | Y | Y | Y |
| timing_consistency | float64 | Y | Y | Y |
| tempo_stability | float64 | Y | Y | Y |
| subdivision_evenness | float64 | Y | Y | Y |
| velocity_control | float64 | Y | Y | Y |
| accent_differentiation | float64 | Y | Y | Y |
| accent_accuracy | float64 | Y | Y | Y |
| hand_balance | float64 | Y | Y | Y |
| weak_hand_index | float64 | Y | Y | Y |
| flam_quality | float64 | Y | Y | Y |
| diddle_quality | float64 | Y | Y | Y |
| roll_sustain | float64 | Y | Y | Y |
| groove_feel_proxy | float64 | Y | Y | Y |
| overall_score | float64 | Y | Y | Y |

- `audio` column uses HF `Audio` feature type (FLAC bytes + sampling_rate: 44100)
- `midi` column uses `Value("binary")` (raw .mid file bytes)

### Auxiliary Tables

Uploaded to `auxiliary/` directory, not part of any config. Loaded separately:

```python
import pandas as pd
strokes = pd.read_parquet("hf://datasets/zkeown/sousa/auxiliary/strokes.parquet")
measures = pd.read_parquet("hf://datasets/zkeown/sousa/auxiliary/measures.parquet")
```

These contain stroke-level (35M rows) and measure-level (10M rows) hierarchical labels, joinable on `sample_id`.

## Implementation Scope

### Modified Files

1. **`dataset_gen/hub/uploader.py`** — Major rewrite:
   - Replace staging/symlink approach with `datasets.DatasetDict` construction
   - Build three configs by iterating samples, reading audio/MIDI bytes
   - Push each config via `dataset_dict.push_to_hub(config_name=..., max_shard_size="1GB")`
   - Add hub purge step before upload
   - Upload auxiliary tables via `HfApi.upload_file()`
   - Remove: `_copy_media_by_rudiment()`, symlink logic, staging directory concept

2. **`scripts/push_to_hub.py`** — Update CLI:
   - Remove `--no-audio` / `--no-midi` flags
   - Add `--configs` (select which configs to upload, default: all)
   - Add `--purge` (clean existing repo before upload, default: prompt)
   - Add `--max-shard-size` (default: "1GB")

3. **`output/dataset/README.md`** — Update YAML frontmatter to declare configs

4. **Tests** — New test coverage for uploader schema and round-trip verification

### Unchanged

- `dataset_gen/pipeline/storage.py` — generation pipeline untouched
- `dataset_gen/pipeline/generate.py` — dataset generation untouched
- Existing test suite — generation tests unchanged

## Upload Process

1. **Purge**: Delete all existing files from `zkeown/sousa` (except `.gitattributes`)
2. **Upload `labels_only`** (~50MB, validates pipeline)
3. **Upload `midi_only`** (~2.5GB)
4. **Upload `audio`** (~96GB across ~96 shards)
5. **Upload auxiliary** (strokes + measures parquets)
6. **Upload README.md**

Each config is a separate commit. Partial failures leave the repo usable.

## Safety

- `--dry-run` builds DatasetDict and prints schema/shard count without uploading
- `push_to_hub` is resumable (re-run picks up where it left off)
- `--purge` prompts for confirmation unless `--yes` is passed

## Estimated Sizes

| Config | Size | Shards |
|--------|------|--------|
| audio | ~96GB | ~96 |
| midi_only | ~2.5GB | ~3 |
| labels_only | ~50MB | 3 |
| auxiliary | ~55MB | 2 |
| **Total** | **~99GB** | **~130 files** |
