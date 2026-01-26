# Loading the Dataset

This guide covers all methods for loading and accessing the SOUSA dataset.

## From HuggingFace Hub

The easiest way to use SOUSA is via the HuggingFace Hub. No local generation required.

### Basic Loading

```python
from datasets import load_dataset

# Load the full dataset (all splits)
dataset = load_dataset("zkeown/sousa")

# Access splits
train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

# Get a sample
sample = train[0]
print(f"Rudiment: {sample['rudiment_slug']}")
print(f"Overall Score: {sample['overall_score']:.1f}")
```

### Load Specific Split

```python
from datasets import load_dataset

# Load only the training split
train = load_dataset("zkeown/sousa", split="train")

# Load only validation
val = load_dataset("zkeown/sousa", split="validation")

# Load a subset (first 1000 samples)
subset = load_dataset("zkeown/sousa", split="train[:1000]")
```

### Streaming for Memory Efficiency

For large-scale training or limited memory, use streaming mode:

```python
from datasets import load_dataset

# Stream the dataset (loads samples on-demand)
dataset = load_dataset("zkeown/sousa", streaming=True)

# Iterate through samples
for sample in dataset["train"]:
    # Process sample
    print(sample["rudiment_slug"])
    break  # Just show first sample

# Take first N samples
from itertools import islice
first_100 = list(islice(dataset["train"], 100))
```

!!! tip "When to Use Streaming"
    - Training on the full 100K dataset
    - Limited RAM (< 16 GB)
    - Preprocessing on-the-fly
    - Distributed training

### Load Specific Columns

Reduce memory by loading only needed columns:

```python
from datasets import load_dataset

# Load only audio and labels
dataset = load_dataset(
    "zkeown/sousa",
    columns=["audio", "overall_score", "skill_tier", "rudiment_slug"]
)
```

## From Local Files

If you generated the dataset locally, load it from the output directory.

### Using Pandas

```python
import pandas as pd
from pathlib import Path

dataset_dir = Path("output/dataset")

# Load the main samples table
samples_df = pd.read_parquet(dataset_dir / "labels" / "samples.parquet")

# Load hierarchical labels
exercises_df = pd.read_parquet(dataset_dir / "labels" / "exercises.parquet")
measures_df = pd.read_parquet(dataset_dir / "labels" / "measures.parquet")
strokes_df = pd.read_parquet(dataset_dir / "labels" / "strokes.parquet")

print(f"Samples: {len(samples_df)}")
print(f"Exercises: {len(exercises_df)}")
print(f"Measures: {len(measures_df)}")
print(f"Strokes: {len(strokes_df)}")
```

### Using HuggingFace Datasets (Local)

Load local parquet files with the HuggingFace datasets library:

```python
from datasets import load_dataset

# Load from local parquet files
dataset = load_dataset(
    "parquet",
    data_dir="./output/dataset/labels",
    split="train"  # or specify a data_files dict
)
```

### Using PyArrow

For maximum performance with large datasets:

```python
import pyarrow.parquet as pq

# Read with PyArrow
table = pq.read_table("output/dataset/labels/samples.parquet")

# Convert to pandas if needed
df = table.to_pandas()

# Or iterate over row groups for memory efficiency
parquet_file = pq.ParquetFile("output/dataset/labels/samples.parquet")
for batch in parquet_file.iter_batches(batch_size=1000):
    df_batch = batch.to_pandas()
    # Process batch
```

## Accessing Audio

### From HuggingFace Dataset

Audio is automatically loaded as numpy arrays:

```python
sample = dataset["train"][0]
audio = sample["audio"]

# Audio data
audio_array = audio["array"]       # NumPy array (float32)
sample_rate = audio["sampling_rate"]  # 44100 Hz

print(f"Duration: {len(audio_array) / sample_rate:.2f} seconds")
print(f"Shape: {audio_array.shape}")
```

### Play Audio in Jupyter

```python
from IPython.display import Audio

sample = dataset["train"][0]
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

### Save Audio to File

```python
import soundfile as sf

sample = dataset["train"][0]
audio = sample["audio"]

sf.write(
    "output_sample.wav",
    audio["array"],
    audio["sampling_rate"]
)
```

### From Local Files

```python
import soundfile as sf
from pathlib import Path

# Load audio by sample ID
sample_id = "beg042_single_paradiddle_100bpm_marching_practicedry"
audio_path = Path("output/dataset/audio") / f"{sample_id}.flac"

audio_data, sample_rate = sf.read(audio_path)
print(f"Loaded {len(audio_data)} samples at {sample_rate} Hz")
```

### Batch Audio Loading

```python
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def load_audio(sample_id):
    """Load audio for a single sample."""
    audio_path = Path("output/dataset/audio") / f"{sample_id}.flac"
    if audio_path.exists():
        return sf.read(audio_path)
    return None, None

# Load multiple audio files in parallel
sample_ids = samples_df["sample_id"].head(100).tolist()

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(load_audio, sample_ids))

audio_data = [(data, sr) for data, sr in results if data is not None]
print(f"Loaded {len(audio_data)} audio files")
```

## Accessing MIDI

### From Local Files

```python
import mido
from pathlib import Path

# Load MIDI by sample ID
sample_id = "beg042_single_paradiddle_100bpm_marching_practicedry"
midi_path = Path("output/dataset/midi") / f"{sample_id}.mid"

mid = mido.MidiFile(midi_path)

# Get basic info
print(f"Tracks: {len(mid.tracks)}")
print(f"Ticks per beat: {mid.ticks_per_beat}")

# Iterate through messages
for track in mid.tracks:
    for msg in track:
        if msg.type == "note_on" and msg.velocity > 0:
            print(f"Note: {msg.note}, Velocity: {msg.velocity}, Time: {msg.time}")
```

### Extract Note Events

```python
def extract_notes(midi_path):
    """Extract note events from MIDI file."""
    mid = mido.MidiFile(midi_path)
    notes = []
    current_time = 0

    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append({
                    "time": current_time,
                    "note": msg.note,
                    "velocity": msg.velocity
                })

    return notes

notes = extract_notes("output/dataset/midi/sample.mid")
print(f"Total notes: {len(notes)}")
```

### Convert MIDI to DataFrame

```python
import pandas as pd
import mido

def midi_to_dataframe(midi_path):
    """Convert MIDI file to pandas DataFrame."""
    mid = mido.MidiFile(midi_path)
    events = []
    current_time = 0

    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == "note_on":
                events.append({
                    "time_ticks": current_time,
                    "time_seconds": mido.tick2second(
                        current_time, mid.ticks_per_beat, 500000
                    ),
                    "note": msg.note,
                    "velocity": msg.velocity,
                    "is_note_on": msg.velocity > 0
                })

    return pd.DataFrame(events)

df = midi_to_dataframe("output/dataset/midi/sample.mid")
print(df.head())
```

## Working with Splits

### Load Split Assignments

```python
import json

with open("output/dataset/splits.json") as f:
    splits = json.load(f)

print(f"Train profiles: {len(splits['train_profile_ids'])}")
print(f"Val profiles: {len(splits['val_profile_ids'])}")
print(f"Test profiles: {len(splits['test_profile_ids'])}")
```

### Filter by Split

```python
# Using pandas
train_df = samples_df[samples_df["split"] == "train"]
val_df = samples_df[samples_df["split"] == "val"]
test_df = samples_df[samples_df["split"] == "test"]

# Using HuggingFace datasets
train = dataset.filter(lambda x: x["split"] == "train")
```

## Creating PyTorch DataLoaders

### Basic DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from pathlib import Path

class SOUSADataset(Dataset):
    def __init__(self, samples_df, audio_dir, transform=None):
        self.samples = samples_df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        # Load audio
        audio_path = self.audio_dir / f"{row['sample_id']}.flac"
        audio, sr = sf.read(audio_path)

        # Get label
        label = row["overall_score"] / 100.0  # Normalize to 0-1

        if self.transform:
            audio = self.transform(audio)

        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label)

# Create dataset and dataloader
train_dataset = SOUSADataset(train_df, "output/dataset/audio")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
```

### With HuggingFace Datasets

```python
from datasets import load_dataset
from transformers import AutoFeatureExtractor

# Load dataset
dataset = load_dataset("zkeown/sousa")

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def preprocess(batch):
    audio = batch["audio"]
    inputs = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=True,
    )
    inputs["labels"] = batch["overall_score"] / 100.0
    return inputs

# Apply preprocessing
dataset = dataset.map(preprocess, remove_columns=["audio"])

# Convert to torch format
dataset.set_format("torch")

# Use with DataLoader
train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
```

## Caching Preprocessed Data

For faster iteration during model development:

```python
from datasets import load_dataset

# Map with caching (persists across sessions)
dataset = dataset.map(
    preprocess_function,
    cache_file_name="./cache/preprocessed_{split}",
    num_proc=4,  # Parallel processing
)
```

### Manual Feature Caching

```python
import pickle
from pathlib import Path

CACHE_DIR = Path("./feature_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_features(sample_id, audio_array, extract_fn):
    """Cache extracted features to disk."""
    cache_path = CACHE_DIR / f"{sample_id}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    features = extract_fn(audio_array)

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)

    return features
```

## Memory Management

### For Large Datasets

```python
# Use streaming mode
dataset = load_dataset("zkeown/sousa", streaming=True)

# Process in batches
def process_batch(batch):
    # Your processing logic
    return batch

# Use batched mapping
dataset = dataset.map(process_batch, batched=True, batch_size=100)
```

### Reduce Memory with Column Selection

```python
# Load only necessary columns
dataset = load_dataset(
    "zkeown/sousa",
    columns=["audio", "overall_score", "skill_tier"]
)
```

### Use Memory Mapping

```python
from datasets import load_dataset

# Keep data on disk, memory-map for access
dataset = load_dataset("zkeown/sousa", keep_in_memory=False)
```

## Next Steps

- [User Guide](../user-guide/index.md) - ML task examples and best practices
- [Data Format](../reference/data-format.md) - Full schema documentation
- [Limitations](../limitations.md) - Known constraints and recommendations
