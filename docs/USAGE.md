# SOUSA Dataset Usage Guide

This guide covers loading, exploring, and using the SOUSA dataset for machine learning.

## Loading the Dataset

### From HuggingFace Hub

The easiest way to use SOUSA is via HuggingFace Hub:

```python
from datasets import load_dataset

# Load full dataset (downloads all splits)
dataset = load_dataset("zkeown/sousa")

# Load specific split
train = load_dataset("zkeown/sousa", split="train")
val = load_dataset("zkeown/sousa", split="validation")
test = load_dataset("zkeown/sousa", split="test")

# Stream for memory efficiency (recommended for large datasets)
dataset = load_dataset("zkeown/sousa", streaming=True)
for sample in dataset["train"]:
    # Process sample
    break
```

### From Local Files

If you generated the dataset locally:

```python
from datasets import load_dataset

# Load from local parquet files
dataset = load_dataset(
    "parquet",
    data_dir="./output/dataset/hf_staging/data"
)

# Or use the custom loader
from dataset_gen.pipeline.storage import ParquetReader

reader = ParquetReader("./output/dataset")
samples_df = reader.load_samples()
exercises_df = reader.load_exercises()
```

## Accessing Audio

SOUSA includes 44.1kHz FLAC audio files:

```python
# Get audio from a sample
sample = dataset["train"][0]
audio = sample["audio"]

audio_array = audio["array"]      # NumPy array
sample_rate = audio["sampling_rate"]  # 44100

# Play audio (Jupyter/IPython)
from IPython.display import Audio
Audio(audio_array, rate=sample_rate)

# Save to file
import soundfile as sf
sf.write("output.wav", audio_array, sample_rate)
```

## Accessing MIDI

MIDI files provide raw performance data:

```python
import mido

# Get MIDI path from sample
midi_path = sample["midi_path"]  # Relative path like "midi/sample_id.mid"

# Load MIDI file
mid = mido.MidiFile(f"./output/dataset/{midi_path}")

# Iterate through messages
for track in mid.tracks:
    for msg in track:
        if msg.type == "note_on":
            print(f"Note {msg.note} velocity {msg.velocity}")
```

## Filtering Data

### By Skill Level

```python
# Filter by skill tier
beginners = dataset["train"].filter(lambda x: x["skill_tier"] == "beginner")
advanced = dataset["train"].filter(lambda x: x["skill_tier"] == "advanced")

# Count by skill
from collections import Counter
skills = Counter(dataset["train"]["skill_tier"])
print(skills)  # {'intermediate': 23000, 'beginner': 18000, ...}
```

### By Rudiment

```python
# Single rudiment
paradiddles = dataset["train"].filter(
    lambda x: x["rudiment_slug"] == "single_paradiddle"
)

# Rudiment category
flam_rudiments = dataset["train"].filter(
    lambda x: "flam" in x["rudiment_slug"]
)

# List all rudiments
rudiments = dataset["train"].unique("rudiment_slug")
print(f"Found {len(rudiments)} rudiments")
```

### By Score

```python
# High performers
high_score = dataset["train"].filter(lambda x: x["overall_score"] > 80)

# Specific score range
mid_timing = dataset["train"].filter(
    lambda x: 50 <= x["timing_accuracy"] <= 70
)
```

### By Tempo

```python
# Slow tempos (for beginners)
slow = dataset["train"].filter(lambda x: x["tempo_bpm"] < 90)

# Fast tempos
fast = dataset["train"].filter(lambda x: x["tempo_bpm"] > 140)
```

## ML Task Recommendations

Before training, review these task-specific recommendations based on dataset characteristics. See [LIMITATIONS.md](LIMITATIONS.md) for detailed rationale.

### Skill Classification

**Problem:** Adjacent skill tiers have 67-83% score overlap, creating an inherent accuracy ceiling (~70-80%).

**Recommended approaches:**

```python
# Option 1: Use 2-class binary labels (less overlap)
dataset = dataset.filter(lambda x: True)  # No filter needed
label = sample["skill_tier_binary"]  # "novice" or "skilled"

# Option 2: Filter by tier confidence (cleaner 4-class labels)
dataset = dataset.filter(lambda x: x["tier_confidence"] > 0.5)
label = sample["skill_tier"]  # "beginner", "intermediate", "advanced", "professional"
```

**Always use class weights** to handle 4.26:1 class imbalance:
```python
from collections import Counter
import torch

# Compute inverse frequency weights
labels = [sample["skill_tier"] for sample in dataset]
counts = Counter(labels)
total = sum(counts.values())
weights = {k: total / (len(counts) * v) for k, v in counts.items()}
class_weights = torch.tensor([weights[tier] for tier in sorted(weights.keys())])

# Use in loss function
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

**Report balanced accuracy**, not just accuracy:
```python
from sklearn.metrics import balanced_accuracy_score
balanced_acc = balanced_accuracy_score(y_true, y_pred)
```

### Score Regression (Assessment)

**Recommended target:** `overall_score`

```python
# Simple regression setup
target = sample["overall_score"] / 100.0  # Normalize to 0-1 range
criterion = torch.nn.MSELoss()
```

This captures the correlated cluster of timing/consistency scores in a single target.

### Multi-Task Learning

**Recommended minimal score set** (orthogonal targets):
```python
RECOMMENDED_SCORES = ["overall_score", "tempo_stability"]

# Multi-task output heads
targets = {
    "overall_score": sample["overall_score"] / 100.0,
    "tempo_stability": sample["tempo_stability"] / 100.0,
}
```

**Avoid using all scores** as separate targets—`timing_accuracy`, `timing_consistency`, and `overall_score` are highly correlated (r > 0.85) and provide redundant learning signal.

### Filtering by Augmentation

Control acoustic domain during training:

```python
# Clean samples only (no augmentation)
clean = dataset.filter(lambda x: x["augmentation_preset"] == "none")

# Specific soundfont
marching = dataset.filter(lambda x: x["soundfont"] == "marching")

# Link clean and augmented variants
groups = dataset.group_by("augmentation_group_id")
```

### Summary Table

| Task | Recommended Approach |
|------|---------------------|
| Skill classification (4-class) | Filter by `tier_confidence > 0.5`, use class weights, report balanced accuracy |
| Skill classification (2-class) | Use `skill_tier_binary` field |
| Assessment/regression | Use `overall_score` as target |
| Multi-task | Use `["overall_score", "tempo_stability"]` |
| Cross-soundfont generalization | Train on mixed soundfonts, evaluate per-soundfont |

## Working with Labels

### Exercise-Level Scores

All scores are 0-100 (higher = better):

```python
sample = dataset["train"][0]

# Timing metrics
print(f"Timing Accuracy: {sample['timing_accuracy']:.1f}")
print(f"Timing Consistency: {sample['timing_consistency']:.1f}")
print(f"Tempo Stability: {sample['tempo_stability']:.1f}")
print(f"Subdivision Evenness: {sample['subdivision_evenness']:.1f}")

# Dynamics metrics
print(f"Velocity Control: {sample['velocity_control']:.1f}")
print(f"Accent Differentiation: {sample['accent_differentiation']:.1f}")
print(f"Accent Accuracy: {sample['accent_accuracy']:.1f}")

# Balance metrics
print(f"Hand Balance: {sample['hand_balance']:.1f}")
print(f"Weak Hand Index: {sample['weak_hand_index']:.1f}")  # 50 = balanced

# Composite
print(f"Overall Score: {sample['overall_score']:.1f}")
```

### Rudiment-Specific Scores

Some scores only apply to certain rudiments:

```python
# Flam quality (only for flam rudiments)
if sample.get("flam_quality") is not None:
    print(f"Flam Quality: {sample['flam_quality']:.1f}")

# Diddle quality (paradiddles, etc.)
if sample.get("diddle_quality") is not None:
    print(f"Diddle Quality: {sample['diddle_quality']:.1f}")

# Roll sustain (roll rudiments)
if sample.get("roll_sustain") is not None:
    print(f"Roll Sustain: {sample['roll_sustain']:.1f}")
```

## Training Examples

### Performance Score Regression

Predict overall performance score from audio:

```python
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np

# Load dataset
dataset = load_dataset("zkeown/sousa")

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base"
)

def preprocess(batch):
    audio = batch["audio"]
    inputs = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=True,
    )
    # Regression target
    inputs["labels"] = batch["overall_score"] / 100.0  # Normalize to 0-1
    return inputs

# Preprocess dataset
dataset = dataset.map(preprocess, remove_columns=["audio"])

# Train model
# ... (standard HuggingFace training loop)
```

### Skill Level Classification

4-class classification (beginner, intermediate, advanced, professional):

```python
from datasets import load_dataset

dataset = load_dataset("zkeown/sousa")

# Get label mapping
label2id = {label: i for i, label in enumerate(
    ["beginner", "intermediate", "advanced", "professional"]
)}
id2label = {i: label for label, i in label2id.items()}

def preprocess(batch):
    # ... audio preprocessing ...
    batch["labels"] = label2id[batch["skill_tier"]]
    return batch

dataset = dataset.map(preprocess)
```

### Rudiment Classification

40-class classification across all PAS rudiments:

```python
# Get all unique rudiments
rudiments = sorted(dataset["train"].unique("rudiment_slug"))
label2id = {r: i for i, r in enumerate(rudiments)}

def preprocess(batch):
    # ... audio preprocessing ...
    batch["labels"] = label2id[batch["rudiment_slug"]]
    return batch
```

## Data Analysis

### Score Distributions

```python
import matplotlib.pyplot as plt
import pandas as pd

# Convert to pandas for analysis
df = dataset["train"].to_pandas()

# Score distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(df["overall_score"], bins=50)
axes[0, 0].set_title("Overall Score Distribution")

axes[0, 1].hist(df["timing_accuracy"], bins=50)
axes[0, 1].set_title("Timing Accuracy Distribution")

axes[1, 0].hist(df["velocity_control"], bins=50)
axes[1, 0].set_title("Velocity Control Distribution")

axes[1, 1].hist(df["hand_balance"], bins=50)
axes[1, 1].set_title("Hand Balance Distribution")

plt.tight_layout()
plt.savefig("score_distributions.png")
```

### Skill Tier Analysis

```python
# Average scores by skill tier
skill_stats = df.groupby("skill_tier")[
    ["overall_score", "timing_accuracy", "velocity_control"]
].mean()
print(skill_stats)

# Expected output:
#                overall_score  timing_accuracy  velocity_control
# beginner              45.2             42.1              48.3
# intermediate          65.7             63.4              67.2
# advanced              82.3             84.1              81.5
# professional          93.1             95.2              92.8
```

### Correlation Analysis

```python
# Score correlations
score_cols = [
    "overall_score", "timing_accuracy", "timing_consistency",
    "velocity_control", "hand_balance"
]
correlations = df[score_cols].corr()
print(correlations)

# Visualize
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap="coolwarm")
plt.savefig("score_correlations.png")
```

## Best Practices

### Memory Management

For large-scale training:

```python
# Use streaming to avoid loading entire dataset
dataset = load_dataset("zkeown/sousa", streaming=True)

# Use smaller batches
dataloader = DataLoader(dataset, batch_size=8)

# Process in chunks
for i, batch in enumerate(dataloader):
    if i % 1000 == 0:
        print(f"Processed {i * 8} samples")
```

### Reproducibility

```python
# Set seeds for reproducibility
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

### Data Augmentation

The dataset includes multiple augmentation presets. You can further augment:

```python
import torchaudio.transforms as T

# Additional augmentation during training
augment = T.Compose([
    T.TimeMasking(time_mask_param=80),
    T.FrequencyMasking(freq_mask_param=27),
])

def augment_audio(batch):
    waveform = torch.tensor(batch["audio"]["array"])
    batch["audio"]["array"] = augment(waveform).numpy()
    return batch
```

## Audio Preprocessing Recommendations

This section provides guidance on preparing audio for ML models.

### Resampling

SOUSA audio is 44.1kHz, but many pretrained models expect 16kHz:

```python
import torchaudio
import torchaudio.transforms as T

# Resample to 16kHz (required for Wav2Vec2, HuBERT, etc.)
resampler = T.Resample(orig_freq=44100, new_freq=16000)

def preprocess_audio(batch):
    waveform = torch.tensor(batch["audio"]["array"])
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dim
    resampled = resampler(waveform).squeeze(0)
    batch["audio"]["array"] = resampled.numpy()
    batch["audio"]["sampling_rate"] = 16000
    return batch

# Apply to dataset
dataset = dataset.map(preprocess_audio)
```

**Sample rate by model:**

| Model | Expected Sample Rate |
|-------|---------------------|
| Wav2Vec2 | 16,000 Hz |
| HuBERT | 16,000 Hz |
| Whisper | 16,000 Hz |
| AST (Audio Spectrogram Transformer) | 16,000 Hz |
| CLAP | 48,000 Hz |
| Custom CNN on spectrograms | Any (use native 44,100 Hz) |

### Mel-Spectrogram Parameters

For spectrogram-based models, recommended parameters for drum audio:

```python
import torchaudio.transforms as T

# Recommended mel-spectrogram for drum analysis
mel_transform = T.MelSpectrogram(
    sample_rate=16000,      # After resampling
    n_fft=1024,             # ~64ms window at 16kHz
    hop_length=256,         # ~16ms hop, 75% overlap
    n_mels=128,             # Good balance of resolution
    f_min=20,               # Capture low-end fundamentals
    f_max=8000,             # Most drum energy is below 8kHz
)

# Convert to log scale (dB)
amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

def extract_mel_spectrogram(audio_array):
    waveform = torch.tensor(audio_array).unsqueeze(0)
    mel = mel_transform(waveform)
    mel_db = amplitude_to_db(mel)
    return mel_db.squeeze(0).numpy()
```

**Parameter trade-offs:**

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `n_fft` | Better time resolution | Better frequency resolution |
| `hop_length` | More time frames, slower | Fewer frames, faster |
| `n_mels` | Faster, less detail | More frequency detail |

**For onset detection tasks**, use smaller `hop_length` (128-160):
```python
onset_mel = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=128,  # ~8ms for precise onset timing
    n_mels=64,
)
```

### Normalization Strategies

**Per-sample normalization** (recommended for most tasks):
```python
def normalize_audio(audio_array):
    """Normalize to [-1, 1] range."""
    max_val = np.abs(audio_array).max()
    if max_val > 0:
        return audio_array / max_val
    return audio_array
```

**Global normalization** (use with dataset-wide stats):
```python
# Compute stats on training set
train_mean = np.mean([s["audio"]["array"].mean() for s in train_subset])
train_std = np.mean([s["audio"]["array"].std() for s in train_subset])

def normalize_global(audio_array):
    return (audio_array - train_mean) / train_std
```

**For spectrograms**, normalize per-frequency-bin:
```python
def normalize_spectrogram(mel_db):
    """Per-channel (frequency bin) normalization."""
    mean = mel_db.mean(axis=-1, keepdims=True)
    std = mel_db.std(axis=-1, keepdims=True) + 1e-6
    return (mel_db - mean) / std
```

### Handling Variable-Length Audio

SOUSA samples vary in duration. Common strategies:

**Strategy 1: Padding/Truncation (simple)**
```python
TARGET_LENGTH = 5 * 16000  # 5 seconds at 16kHz

def pad_or_truncate(audio_array):
    if len(audio_array) > TARGET_LENGTH:
        return audio_array[:TARGET_LENGTH]
    elif len(audio_array) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(audio_array)
        return np.pad(audio_array, (0, padding), mode='constant')
    return audio_array
```

**Strategy 2: Chunking (for very long samples)**
```python
CHUNK_SIZE = 3 * 16000  # 3 second chunks
OVERLAP = 0.5          # 50% overlap

def chunk_audio(audio_array):
    """Split into overlapping chunks."""
    hop = int(CHUNK_SIZE * (1 - OVERLAP))
    chunks = []
    for start in range(0, len(audio_array) - CHUNK_SIZE + 1, hop):
        chunks.append(audio_array[start:start + CHUNK_SIZE])
    return chunks
```

**Strategy 3: Dynamic batching (advanced)**
```python
from torch.nn.utils.rnn import pad_sequence

def collate_variable_length(batch):
    """Custom collate function for variable-length audio."""
    waveforms = [torch.tensor(s["audio"]["array"]) for s in batch]
    lengths = torch.tensor([len(w) for w in waveforms])

    # Pad to max length in batch
    padded = pad_sequence(waveforms, batch_first=True)

    labels = torch.tensor([s["overall_score"] for s in batch])

    return {
        "waveforms": padded,
        "lengths": lengths,
        "labels": labels,
    }

dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_variable_length)
```

### Pretrained Model Integration

**Using Wav2Vec2 feature extractor:**
```python
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base"
)

def preprocess_for_wav2vec2(batch):
    # Resample to 16kHz first
    audio = batch["audio"]["array"]

    # Feature extractor handles padding and normalization
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=80000,  # 5 seconds
        truncation=True,
    )

    batch["input_values"] = inputs.input_values.squeeze(0)
    return batch
```

**Using HuggingFace Audio pipeline:**
```python
from transformers import pipeline

# Zero-shot classification (good for quick experiments)
classifier = pipeline(
    "audio-classification",
    model="facebook/wav2vec2-base",
    device=0,  # GPU
)

# Classify audio samples
results = classifier(audio_array)
```

### Feature Caching

For faster iteration, cache preprocessed features:

```python
import hashlib
import pickle
from pathlib import Path

CACHE_DIR = Path("./feature_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_features(sample_id, audio_array, extract_fn):
    """Cache extracted features to disk."""
    cache_key = hashlib.md5(sample_id.encode()).hexdigest()
    cache_path = CACHE_DIR / f"{cache_key}.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    features = extract_fn(audio_array)

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)

    return features
```

**Using HuggingFace dataset caching:**
```python
# Map with caching (persists across sessions)
dataset = dataset.map(
    preprocess_audio,
    cache_file_name="./cache/preprocessed",
    num_proc=4,
)
```

## Troubleshooting

### Audio Not Loading

```python
# Check audio feature is enabled
print(dataset["train"].features)

# Manually load audio if needed
import soundfile as sf
audio_path = f"./output/dataset/{sample['audio_path']}"
audio_array, sr = sf.read(audio_path)
```

### Out of Memory

```python
# Use streaming mode
dataset = load_dataset("zkeown/sousa", streaming=True)

# Or load specific columns only
dataset = load_dataset(
    "zkeown/sousa",
    columns=["audio", "overall_score", "skill_tier"]
)
```

### Slow Loading

```python
# Use multiple workers
dataset = load_dataset("zkeown/sousa", num_proc=4)

# Cache preprocessed data
dataset = dataset.map(preprocess, cache_file_name="preprocessed.arrow")
```

## Validation

### Generate Validation Report

```python
from dataset_gen.validation.report import generate_report

# Generate comprehensive report
report = generate_report(
    dataset_dir='output/dataset',
    output_path='validation_report.json',
    include_realism=True,
    include_midi_checks=True
)

# Print summary
print(report.summary())
```

### Quick Validation Check

```python
from dataset_gen.validation.report import quick_validate

# Returns True if all critical checks pass
is_valid = quick_validate('output/dataset')
print(f"Dataset valid: {is_valid}")
```

### Individual Validation Components

```python
from dataset_gen.validation import (
    analyze_dataset,
    verify_labels,
    validate_realism
)

# Statistical analysis
stats = analyze_dataset('output/dataset')
print(f"Total samples: {stats['num_samples']}")
print(f"Mean timing error: {stats['timing_error']['mean']:.2f}ms")
print(f"Skill tier distribution: {stats['skill_tier_counts']}")

# Data integrity verification
result = verify_labels('output/dataset')
print(f"Checks passed: {result.num_passed}/{result.num_passed + result.num_failed}")
for check in result.checks:
    status = "✓" if check.passed else "✗"
    print(f"  {status} {check.name}: {check.message}")

# Realism validation (comparison to literature)
realism = validate_realism('output/dataset')
print(f"Literature pass rate: {realism['literature_pass_rate']}%")
print(f"Correlation pass rate: {realism['correlation_pass_rate']}%")
```

### Reading Validation Report

```python
import json

with open('output/dataset/validation_report.json') as f:
    report = json.load(f)

# Check if all verifications passed
print(f"All checks passed: {report['verification']['all_passed']}")

# Get timing accuracy by skill tier
timing = report['stats']['exercise_timing_accuracy']['by_group']
for tier, stats in timing.items():
    print(f"{tier}: {stats['mean']:.1f} ± {stats['std']:.1f}")

# Check realism validation
print(f"Literature comparisons: {report['realism']['literature_pass_rate']}% pass")
print(f"Correlation checks: {report['realism']['correlation_pass_rate']}% pass")
```

### Command Line Validation

```bash
# Check dataset health during/after generation
python scripts/check_generation.py output/dataset

# Generate dataset with automatic validation
python scripts/generate_dataset.py --preset full
# Validation runs automatically at the end
```

For detailed validation documentation, see [VALIDATION.md](VALIDATION.md).
