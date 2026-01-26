# Feature Extraction

This guide covers extracting machine learning features from SOUSA audio files. We provide utilities for mel spectrograms, onset detection, RMS energy, and pretrained model features (Wav2Vec2, HuBERT).

!!! info "Source Code"
    Complete implementation: [`examples/feature_extraction.py`](https://github.com/zakkeown/SOUSA/blob/main/examples/feature_extraction.py)

## Prerequisites

=== "Basic (Mel/RMS)"

    ```bash
    pip install torch torchaudio
    # or
    pip install librosa
    ```

=== "Onset Detection"

    ```bash
    pip install librosa
    ```

=== "Pretrained Models"

    ```bash
    pip install torch transformers
    ```

## Quick Start

```python
from examples.feature_extraction import FeatureExtractor

# Create unified extractor
extractor = FeatureExtractor(
    sample_rate=16000,
    feature_type="mel_spectrogram",
)

# Extract from file
features = extractor.extract("output/dataset/audio/sample.flac")

# Or from array
import numpy as np
audio = np.random.randn(16000 * 3)  # 3 seconds at 16kHz
features = extractor.extract_from_array(audio)

print(features["mel_spectrogram"].shape)  # (128, 188) - (n_mels, time_frames)
```

## Mel Spectrogram Extraction

Mel spectrograms are the most common input representation for audio neural networks.

### Basic Usage

```python
from examples.feature_extraction import MelSpectrogramExtractor

extractor = MelSpectrogramExtractor(
    sample_rate=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=256,
    f_min=20.0,
    f_max=8000.0,
    normalized=True,
)

# Extract from audio array
mel = extractor(audio_array)
print(mel.shape)  # (128, time_frames)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | `16000` | Audio sample rate |
| `n_mels` | `128` | Number of mel frequency bands |
| `n_fft` | `1024` | FFT window size |
| `hop_length` | `256` | Hop length between frames |
| `f_min` | `20.0` | Minimum frequency (Hz) |
| `f_max` | `8000.0` | Maximum frequency (Hz) |
| `power` | `2.0` | Spectrogram power (2=power, 1=magnitude) |
| `normalized` | `True` | Per-channel normalization |
| `use_torchaudio` | `True` | Use torchaudio (faster) or librosa |

### Recommended Settings for Drums

```python
# Settings optimized for drum/percussion audio
extractor = MelSpectrogramExtractor(
    sample_rate=16000,
    n_mels=128,
    n_fft=1024,       # ~64ms window at 16kHz
    hop_length=256,   # ~16ms hop = 62.5 fps
    f_min=20.0,       # Capture low-frequency body
    f_max=8000.0,     # Drums don't need higher frequencies
)
```

!!! tip "Frame Rate Calculation"
    Frame rate = sample_rate / hop_length

    With default settings: 16000 / 256 = 62.5 frames per second

## MFCC Features

For traditional ML models, MFCCs provide compact representations:

```python
import librosa

def extract_mfcc(audio: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13):
    """Extract MFCC features."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=1024,
        hop_length=256,
    )

    # Add delta and delta-delta features
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack: (39, time_frames) = 13 MFCCs + 13 deltas + 13 delta-deltas
    return np.vstack([mfcc, mfcc_delta, mfcc_delta2])
```

## Onset Detection Features

Onset detection is particularly relevant for drum performance analysis:

```python
from examples.feature_extraction import OnsetDetector

detector = OnsetDetector(sample_rate=16000, hop_length=128)

# Detect onsets
result = detector.detect_onsets(audio_array)

print(f"Number of onsets: {result['num_onsets']}")
print(f"Onset times (seconds): {result['onset_times'][:5]}")
print(f"Onset strength shape: {result['onset_strength'].shape}")

# Compute inter-onset interval features
ioi_features = detector.compute_ioi_features(result["onset_times"])

print(f"Mean IOI: {ioi_features['ioi_mean']:.3f}s")
print(f"IOI std: {ioi_features['ioi_std']:.3f}s")
print(f"Estimated tempo: {ioi_features['tempo_estimate']:.1f} BPM")
```

### IOI Features for Timing Analysis

Inter-onset intervals (IOIs) are useful for analyzing timing consistency:

| Feature | Description |
|---------|-------------|
| `ioi_mean` | Mean time between strokes (seconds) |
| `ioi_std` | Standard deviation of IOIs |
| `ioi_cv` | Coefficient of variation (std/mean) - timing consistency |
| `tempo_estimate` | Estimated tempo from mean IOI |

## RMS Energy Features

RMS (Root Mean Square) energy captures dynamics:

```python
from examples.feature_extraction import RMSExtractor

extractor = RMSExtractor(frame_length=2048, hop_length=512)

# Extract RMS curve
rms = extractor(audio_array)

# Compute dynamics statistics
dynamics = extractor.compute_dynamics_features(rms)

print(f"Mean RMS: {dynamics['rms_mean']:.4f}")
print(f"Dynamic range: {dynamics['dynamic_range']:.4f}")
print(f"RMS CV: {dynamics['rms_cv']:.4f}")  # Dynamics consistency
```

### Dynamics Features

| Feature | Description |
|---------|-------------|
| `rms_mean` | Average loudness |
| `rms_std` | Loudness variation |
| `rms_max` | Peak loudness |
| `rms_min` | Minimum loudness |
| `dynamic_range` | Max - min RMS |
| `rms_cv` | Coefficient of variation |

## Pretrained Model Features

### Wav2Vec2

Wav2Vec2 provides powerful self-supervised features:

```python
from examples.feature_extraction import Wav2Vec2Extractor

# Load pretrained model
extractor = Wav2Vec2Extractor(
    model_name="facebook/wav2vec2-base",
    layer=-1,      # Last layer (-1) or specific layer (0-11)
    device="cuda",
    freeze=True,
)

# Extract sequence features
features = extractor(audio_array)
print(features.shape)  # (time_frames, 768)

# Extract pooled (single vector) features
pooled = extractor.extract_pooled(audio_array, pooling="mean")
print(pooled.shape)  # (768,)
```

### HuBERT

HuBERT works similarly (uses same API):

```python
extractor = Wav2Vec2Extractor(
    model_name="facebook/hubert-base-ls960",
    device="cuda",
)

features = extractor(audio_array)
```

### Available Pooling Strategies

```python
# Mean pooling (recommended for most tasks)
pooled = extractor.extract_pooled(audio, pooling="mean")

# Max pooling
pooled = extractor.extract_pooled(audio, pooling="max")

# First token (CLS-like)
pooled = extractor.extract_pooled(audio, pooling="first")

# Last token
pooled = extractor.extract_pooled(audio, pooling="last")
```

### Multi-Layer Features

For probing or concatenation:

```python
# Get features from all transformer layers
all_layers = extractor(audio_array, return_all_layers=True)

for layer_name, features in all_layers.items():
    print(f"{layer_name}: {features.shape}")

# layer_0: (time_frames, 768)  - CNN output
# layer_1: (time_frames, 768)  - Transformer layer 1
# ...
# layer_12: (time_frames, 768) - Last transformer layer
```

## Unified Feature Extractor

The `FeatureExtractor` class combines multiple extractors:

```python
from examples.feature_extraction import FeatureExtractor

# Extract all available features
extractor = FeatureExtractor(
    sample_rate=16000,
    feature_type="all",  # "mel_spectrogram", "onset", "rms", "wav2vec2", or "all"
    cache_dir="feature_cache",  # Optional caching
)

features = extractor.extract("audio.flac")

# Features dict contains:
# - mel_spectrogram: (n_mels, time_frames)
# - onset_times: (num_onsets,)
# - onset_strength: (time_frames,)
# - num_onsets: int
# - ioi_mean, ioi_std, ioi_cv, tempo_estimate: floats
# - rms: (time_frames,)
# - rms_mean, rms_std, rms_max, rms_min, dynamic_range, rms_cv: floats
# - wav2vec2: (time_frames, 768)
# - wav2vec2_pooled: (768,)
```

## Feature Caching

For large-scale experiments, enable caching:

```python
extractor = FeatureExtractor(
    sample_rate=16000,
    feature_type="mel_spectrogram",
    cache_dir="./feature_cache",
)

# First call: computes and caches
features1 = extractor.extract("audio.flac")

# Second call: loads from cache (instant)
features2 = extractor.extract("audio.flac")
```

!!! warning "Cache Invalidation"
    The cache key includes the audio filename, feature type, and sample rate.
    If you change extractor parameters (e.g., `n_mels`), clear the cache:

    ```bash
    rm -rf ./feature_cache
    ```

## Batch Processing

Process multiple files in parallel:

```python
from examples.feature_extraction import extract_batch_features

audio_paths = [
    "output/dataset/audio/sample1.flac",
    "output/dataset/audio/sample2.flac",
    "output/dataset/audio/sample3.flac",
]

features_list = extract_batch_features(
    audio_paths,
    feature_type="mel_spectrogram",
    sample_rate=16000,
    num_workers=4,
    cache_dir="./feature_cache",
)

for path, features in zip(audio_paths, features_list):
    print(f"{path}: {features['mel_spectrogram'].shape}")
```

!!! note "Wav2Vec2 and Multiprocessing"
    Wav2Vec2 extraction may not work well with multiprocessing due to CUDA/model loading issues. Use `num_workers=1` for pretrained features, or extract them separately.

## Integration with PyTorch

### As a Transform

```python
import torch
from examples.feature_extraction import MelSpectrogramExtractor
from examples.pytorch_dataloader import SOUSADataset


class MelTransform:
    """Transform waveform to mel spectrogram."""

    def __init__(self, sample_rate: int = 16000):
        self.extractor = MelSpectrogramExtractor(sample_rate=sample_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.extractor(waveform.numpy())
        return torch.from_numpy(mel)


# Use with dataset
dataset = SOUSADataset(
    data_dir="output/dataset",
    split="train",
    target="skill_tier",
    transform=MelTransform(),
)

sample = dataset[0]
print(sample["waveform"].shape)  # Now a mel spectrogram!
```

### Pre-computed Features Dataset

For faster training, pre-compute features:

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path


class PrecomputedFeaturesDataset(Dataset):
    """Dataset with pre-computed features."""

    def __init__(
        self,
        data_dir: str,
        features_dir: str,
        split: str = "train",
    ):
        self.features_dir = Path(features_dir)

        # Load metadata
        data_dir = Path(data_dir)
        samples = pd.read_parquet(data_dir / "labels" / "samples.parquet")
        exercises = pd.read_parquet(data_dir / "labels" / "exercises.parquet")
        self.data = samples.merge(exercises, on="sample_id")

        # Filter by split (implement your own split logic)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row["sample_id"]

        # Load pre-computed features
        features = torch.load(self.features_dir / f"{sample_id}.pt")

        return {
            "features": features,
            "label": row["overall_score"] / 100.0,
            "sample_id": sample_id,
        }
```

## Example: Complete Feature Extraction Pipeline

```python
"""Extract features for all samples in the dataset."""
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

from examples.feature_extraction import FeatureExtractor


def extract_all_features(
    data_dir: str,
    output_dir: str,
    feature_type: str = "mel_spectrogram",
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sample list
    samples = pd.read_parquet(data_dir / "labels" / "samples.parquet")

    # Create extractor
    extractor = FeatureExtractor(
        sample_rate=16000,
        feature_type=feature_type,
    )

    # Process all samples
    for _, row in tqdm(samples.iterrows(), total=len(samples)):
        sample_id = row["sample_id"]
        audio_path = data_dir / row["audio_path"]

        if not audio_path.exists():
            continue

        # Extract features
        features = extractor.extract(audio_path)

        # Save as tensor
        torch.save(
            {k: torch.from_numpy(v) if hasattr(v, "shape") else v
             for k, v in features.items()},
            output_dir / f"{sample_id}.pt"
        )


if __name__ == "__main__":
    extract_all_features(
        data_dir="output/dataset",
        output_dir="output/features/mel",
        feature_type="mel_spectrogram",
    )
```

## Next Steps

- [PyTorch DataLoader](pytorch-dataloader.md) - Combine features with DataLoader
- [Filtering Samples](filtering.md) - Filter dataset before feature extraction
- [Hierarchical Labels](hierarchical-labels.md) - Align features with stroke-level labels
