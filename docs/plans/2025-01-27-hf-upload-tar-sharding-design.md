# HuggingFace Upload: TAR Sharding Design

**Date:** 2025-01-27
**Status:** Approved
**Problem:** Current upload script tries to push 240k individual files, exceeding HuggingFace's 100k file limit and causing rate limiting (128 commits/hour)

## Solution

Bundle audio/MIDI files into sharded TAR archives, reducing total file count to ~100.

## Archive Structure

```
output/dataset/hf_staging/
├── README.md
├── data/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
├── audio/
│   ├── train-00000.tar
│   ├── train-00001.tar
│   ├── ... (~96 shards at 1GB each)
│   ├── validation-00000.tar
│   └── test-00000.tar
└── midi/
    ├── train.tar
    ├── validation.tar
    └── test.tar
```

## Parquet Schema Changes

Current columns:
- `audio_path`: `"audio/adv000_double_drag_tap_112bpm_douglasn_practiceroom.flac"`
- `midi_path`: `"midi/adv000_double_drag_tap_112bpm.mid"`

New columns:
- `audio_shard`: `"train-00042.tar"`
- `audio_filename`: `"adv000_double_drag_tap_112bpm_douglasn_practiceroom.flac"`
- `midi_shard`: `"train.tar"`
- `midi_filename`: `"adv000_double_drag_tap_112bpm.mid"`

## Implementation

### New File: `dataset_gen/hub/archiver.py`

```python
@dataclass
class ShardInfo:
    shard_name: str
    filename: str

def create_sharded_archives(
    source_dir: Path,
    output_dir: Path,
    filenames_by_split: dict[str, list[str]],  # split -> list of filenames
    target_shard_size_bytes: int = 1_000_000_000,  # 1GB
    extension: str = "flac",
) -> dict[str, ShardInfo]:
    """
    Create TAR archives from source files, sharded by size.

    Args:
        source_dir: Directory containing source files
        output_dir: Directory to write TAR archives
        filenames_by_split: Mapping of split name to list of filenames
        target_shard_size_bytes: Target size per shard
        extension: File extension to process

    Returns:
        Mapping of original filename to ShardInfo(shard_name, filename)
    """
```

Key behaviors:
- Groups files by split first
- Creates shards when accumulated size exceeds target
- Returns mapping for parquet updates
- Uses deterministic ordering for reproducibility

### Changes to `dataset_gen/hub/uploader.py`

1. Add `use_tar_shards: bool = True` to `HubConfig`

2. Replace `_copy_media_files()` logic:
```python
def _create_media_archives(self, media_type: str, extension: str) -> dict[str, ShardInfo]:
    """Create sharded TAR archives for audio or MIDI files."""
    # Get filenames grouped by split from the merged dataframe
    # Call create_sharded_archives()
    # Return shard mapping
```

3. Update `_merge_dataframes()` to add shard columns:
```python
# After creating archives:
merged_df["audio_shard"] = merged_df["sample_id"].map(
    lambda sid: audio_shard_map[sid].shard_name
)
merged_df["audio_filename"] = merged_df["sample_id"].map(
    lambda sid: audio_shard_map[sid].filename
)
```

4. Keep `upload_large_folder()` - works fine with ~100 files

### Changes to `scripts/push_to_hub.py`

Add flag to disable sharding (for testing):
```python
parser.add_argument(
    "--no-sharding",
    action="store_true",
    help="Disable TAR sharding (not recommended for large datasets)",
)
```

## Consumer Usage

### Basic (with helper)

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import tarfile
import soundfile as sf
import io

def load_audio(sample, repo_id="zkeown/sousa"):
    """Load audio from a dataset sample."""
    shard_path = hf_hub_download(
        repo_id,
        f"audio/{sample['audio_shard']}",
        repo_type="dataset"
    )
    with tarfile.open(shard_path) as tar:
        audio_bytes = tar.extractfile(sample["audio_filename"]).read()
        return sf.read(io.BytesIO(audio_bytes))

ds = load_dataset("zkeown/sousa")
audio, sr = load_audio(ds["train"][0])
```

### Batch Processing (extract shards)

```python
# Download and extract all training audio
for shard in ds["train"]["audio_shard"].unique():
    shard_path = hf_hub_download("zkeown/sousa", f"audio/{shard}", repo_type="dataset")
    with tarfile.open(shard_path) as tar:
        tar.extractall("./audio/")
```

## File Count Estimate

| Content | Files |
|---------|-------|
| Parquet (3 splits) | 3 |
| Audio shards (~96GB / 1GB) | ~96 |
| Validation audio | ~1-2 |
| Test audio | ~1-2 |
| MIDI (3 splits, small) | 3 |
| README | 1 |
| **Total** | **~106** |

Well under 100k limit.

## Testing

1. Unit tests for `archiver.py`:
   - Shard size targeting
   - Split separation
   - Deterministic output

2. Integration test:
   - Generate small dataset
   - Run upload with `--dry-run`
   - Verify archive structure
   - Verify parquet shard columns

## Migration

No migration needed - this changes the upload format, not the generation format. Existing generated datasets work with the new uploader.
