# ML Training Recommendations

This guide provides best practices for training machine learning models on SOUSA, including task-specific recommendations, audio preprocessing, and code examples.

## Task Overview

| Task | Target | Difficulty | Recommended Approach |
|------|--------|------------|---------------------|
| Skill Classification (4-class) | `skill_tier` | Medium | Filter by `tier_confidence`, use class weights |
| Skill Classification (Binary) | `skill_tier_binary` | Easy | Less overlap, simpler model |
| Score Regression | `overall_score` | Easy | MSE loss, normalized target |
| Multi-task Learning | Multiple scores | Hard | Use orthogonal targets |
| Rudiment Classification | `rudiment_slug` | Medium | 40-class classification |

## Skill Classification

### 4-Class Classification

Classify samples into beginner, intermediate, advanced, or professional.

!!! warning "Classification Ceiling"
    Adjacent skill tiers have 67-83% score overlap, creating an inherent accuracy ceiling of ~70-80%. This is realistic (skill exists on a continuum) but affects model evaluation.

#### Recommended Setup

```python
from datasets import load_dataset
from collections import Counter
import torch
import torch.nn as nn

# Load dataset
dataset = load_dataset("zkeown/sousa")

# Filter for confident tier assignments
train = dataset["train"].filter(lambda x: x["tier_confidence"] > 0.5)
val = dataset["validation"].filter(lambda x: x["tier_confidence"] > 0.5)

# Label mapping
label2id = {
    "beginner": 0,
    "intermediate": 1,
    "advanced": 2,
    "professional": 3,
}
id2label = {v: k for k, v in label2id.items()}
num_labels = len(label2id)
```

#### Class Weights (Required)

The 4.26:1 class imbalance requires class weights:

```python
# Compute inverse frequency weights
labels = [sample["skill_tier"] for sample in train]
counts = Counter(labels)
total = sum(counts.values())

weights = {k: total / (len(counts) * v) for k, v in counts.items()}
class_weights = torch.tensor([
    weights["beginner"],
    weights["intermediate"],
    weights["advanced"],
    weights["professional"],
])

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

#### Evaluation Metrics

Always report **balanced accuracy**, not just accuracy:

```python
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Predict on test set
y_true = [sample["skill_tier"] for sample in test]
y_pred = model.predict(test)

# Balanced accuracy (handles class imbalance)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.3f}")

# Per-class metrics
print(classification_report(y_true, y_pred, target_names=list(label2id.keys())))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(label2id.keys()))
```

### Binary Classification (Recommended)

For cleaner labels with less overlap:

```python
# Use binary labels
train = dataset["train"]

# Binary: novice (beginner + intermediate) vs skilled (advanced + professional)
def add_binary_label(sample):
    sample["label"] = 0 if sample["skill_tier_binary"] == "novice" else 1
    return sample

train = train.map(add_binary_label)

# Simpler loss (still use class weights if imbalanced)
criterion = nn.BCEWithLogitsLoss()
```

**Expected Performance:**

| Approach | Balanced Accuracy | Notes |
|----------|-------------------|-------|
| 4-class, no filtering | ~65-70% | Overlap limits performance |
| 4-class, tier_confidence > 0.5 | ~75-80% | Cleaner labels |
| Binary classification | ~85-90% | Less overlap |

## Score Regression

### Overall Score as Target

The simplest and most effective regression target:

```python
import torch
import torch.nn as nn

# Normalize to 0-1 range
def preprocess(sample):
    sample["target"] = sample["overall_score"] / 100.0
    return sample

train = dataset["train"].map(preprocess)

# MSE loss
criterion = nn.MSELoss()

# Or use Huber loss for robustness to outliers
criterion = nn.HuberLoss(delta=0.1)
```

### Why Overall Score?

The `overall_score` captures the correlated cluster of timing/consistency scores:

```
timing_accuracy ↔ timing_consistency: r = 0.89
timing_accuracy ↔ overall_score: r = 0.88
```

Using individual scores as separate targets provides redundant signal.

### Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = [sample["overall_score"] for sample in test]
y_pred = model.predict(test) * 100  # Denormalize

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.3f}")
```

## Multi-Task Learning

### Minimal Orthogonal Score Set

Avoid using all scores (high redundancy). Use these orthogonal targets:

```python
# Recommended minimal set
TARGETS = ["overall_score", "tempo_stability"]

def preprocess_multitask(sample):
    sample["targets"] = {
        "overall_score": sample["overall_score"] / 100.0,
        "tempo_stability": sample["tempo_stability"] / 100.0,
    }
    return sample
```

### Multi-Head Architecture

```python
import torch
import torch.nn as nn

class MultiTaskHead(nn.Module):
    def __init__(self, input_dim, num_tasks=2):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        return [head(x).squeeze(-1) for head in self.heads]

# Combined loss
def multi_task_loss(predictions, targets, weights=None):
    if weights is None:
        weights = [1.0] * len(predictions)

    loss = 0
    for pred, target, weight in zip(predictions, targets, weights):
        loss += weight * nn.functional.mse_loss(pred, target)

    return loss
```

### Handling Rudiment-Specific Scores

Some scores are null for non-applicable rudiments:

```python
# Mask-based loss for optional scores
def masked_mse_loss(predictions, targets, mask):
    """Compute MSE only where mask is True."""
    if mask.sum() == 0:
        return torch.tensor(0.0)

    masked_pred = predictions[mask]
    masked_target = targets[mask]
    return nn.functional.mse_loss(masked_pred, masked_target)

# Example: flam_quality is only valid for flam rudiments
flam_mask = torch.tensor([
    sample["flam_quality"] is not None for sample in batch
])
if flam_mask.any():
    flam_loss = masked_mse_loss(flam_pred, flam_target, flam_mask)
```

## Audio Preprocessing

### Resampling

SOUSA audio is 44.1kHz. Many pretrained models expect 16kHz:

```python
import torchaudio
import torchaudio.transforms as T

# Resample to 16kHz
resampler = T.Resample(orig_freq=44100, new_freq=16000)

def resample_audio(sample):
    waveform = torch.tensor(sample["audio"]["array"])
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    resampled = resampler(waveform).squeeze(0)
    sample["audio"]["array"] = resampled.numpy()
    sample["audio"]["sampling_rate"] = 16000
    return sample
```

**Sample Rate by Model:**

| Model | Expected Rate |
|-------|---------------|
| Wav2Vec2 | 16,000 Hz |
| HuBERT | 16,000 Hz |
| Whisper | 16,000 Hz |
| AST | 16,000 Hz |
| CLAP | 48,000 Hz |
| Custom CNN | Any (use native 44,100 Hz) |

### Mel-Spectrogram Features

For spectrogram-based models:

```python
import torchaudio.transforms as T

# Recommended parameters for drum audio
mel_transform = T.MelSpectrogram(
    sample_rate=16000,      # After resampling
    n_fft=1024,             # ~64ms window
    hop_length=256,         # ~16ms hop
    n_mels=128,             # Frequency bins
    f_min=20,               # Capture low fundamentals
    f_max=8000,             # Most drum energy < 8kHz
)

# Convert to log scale
amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

def extract_mel_spectrogram(audio_array):
    waveform = torch.tensor(audio_array).unsqueeze(0)
    mel = mel_transform(waveform)
    mel_db = amplitude_to_db(mel)
    return mel_db.squeeze(0).numpy()
```

**For onset detection** (precise stroke timing):

```python
onset_mel = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,
    hop_length=128,  # ~8ms for precise timing
    n_mels=64,
)
```

### Normalization

**Per-sample normalization:**

```python
def normalize_audio(audio_array):
    """Normalize to [-1, 1] range."""
    max_val = np.abs(audio_array).max()
    if max_val > 0:
        return audio_array / max_val
    return audio_array
```

**Per-frequency-bin normalization (spectrograms):**

```python
def normalize_spectrogram(mel_db):
    """Per-channel (frequency bin) normalization."""
    mean = mel_db.mean(axis=-1, keepdims=True)
    std = mel_db.std(axis=-1, keepdims=True) + 1e-6
    return (mel_db - mean) / std
```

### Handling Variable Length

SOUSA samples vary in duration (4-12 seconds typical).

**Padding/Truncation:**

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

**Dynamic batching:**

```python
from torch.nn.utils.rnn import pad_sequence

def collate_variable_length(batch):
    waveforms = [torch.tensor(s["audio"]["array"]) for s in batch]
    lengths = torch.tensor([len(w) for w in waveforms])

    # Pad to max length in batch
    padded = pad_sequence(waveforms, batch_first=True)

    labels = torch.tensor([s["overall_score"] / 100.0 for s in batch])

    return {
        "waveforms": padded,
        "lengths": lengths,
        "labels": labels,
    }

dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=collate_variable_length
)
```

## PyTorch DataLoader Examples

### Basic DataLoader

```python
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class SOUSADataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract audio
        audio = torch.tensor(sample["audio"]["array"], dtype=torch.float32)

        # Extract label
        label = torch.tensor(sample["overall_score"] / 100.0, dtype=torch.float32)

        if self.transform:
            audio = self.transform(audio)

        return {"audio": audio, "label": label}

# Create dataset
hf_dataset = load_dataset("zkeown/sousa", split="train")
train_dataset = SOUSADataset(hf_dataset)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

### With Preprocessing

```python
import torchaudio.transforms as T

class SOUSAMelDataset(Dataset):
    def __init__(self, hf_dataset, sample_rate=16000):
        self.data = hf_dataset
        self.sample_rate = sample_rate

        # Transforms
        self.resampler = T.Resample(orig_freq=44100, new_freq=sample_rate)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
        )
        self.to_db = T.AmplitudeToDB()

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load and resample audio
        audio = torch.tensor(sample["audio"]["array"]).unsqueeze(0)
        audio = self.resampler(audio)

        # Extract mel spectrogram
        mel = self.mel_transform(audio)
        mel_db = self.to_db(mel)

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # Label
        label = torch.tensor(sample["overall_score"] / 100.0)

        return {"mel": mel_db.squeeze(0), "label": label}
```

### Streaming DataLoader

For large datasets that don't fit in memory:

```python
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

class StreamingSOUSADataset(IterableDataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("zkeown/sousa", split=split, streaming=True)

    def __iter__(self):
        for sample in self.dataset:
            audio = torch.tensor(sample["audio"]["array"], dtype=torch.float32)
            label = torch.tensor(sample["overall_score"] / 100.0)
            yield {"audio": audio, "label": label}

# Streaming dataloader
stream_loader = DataLoader(
    StreamingSOUSADataset("train"),
    batch_size=32,
)

# Iterate
for batch in stream_loader:
    # Process batch
    pass
```

## Training Loop Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(audio)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            outputs = model(audio)
            loss = criterion(outputs.squeeze(), labels)

            total_loss += loss.item()
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(val_loader), all_preds, all_labels

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, preds, labels = evaluate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
```

## Best Practices Summary

| Aspect | Recommendation |
|--------|----------------|
| Skill classification | Use binary labels or filter by `tier_confidence > 0.5` |
| Class imbalance | Always use class weights for 4-class classification |
| Regression target | Use `overall_score` (captures correlated cluster) |
| Multi-task | Use `["overall_score", "tempo_stability"]` (orthogonal) |
| Audio preprocessing | Resample to 16kHz for pretrained models |
| Evaluation | Report balanced accuracy for classification |
| Augmentation | Train on mixed presets, validate on clean |
| Data splits | Use provided profile-based splits (prevents leakage) |
