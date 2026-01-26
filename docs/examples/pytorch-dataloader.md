# PyTorch DataLoader Integration

This guide shows how to integrate the SOUSA dataset with PyTorch for training neural networks. We cover custom `Dataset` classes, collate functions for variable-length audio, and efficient DataLoader configurations.

!!! info "Source Code"
    Complete implementation: [`examples/pytorch_dataloader.py`](https://github.com/zakkeown/SOUSA/blob/main/examples/pytorch_dataloader.py)

## Prerequisites

```bash
pip install torch torchaudio
```

## Basic Usage

### Loading the Dataset

```python
from examples.pytorch_dataloader import SOUSADataset, create_dataloader

# Create dataset for training
dataset = SOUSADataset(
    data_dir="output/dataset",
    split="train",
    target="skill_tier",      # Classification target
    resample_rate=16000,      # Resample to 16kHz
    max_length_sec=5.0,       # Truncate to 5 seconds
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
```

### Creating a DataLoader

```python
# Create DataLoader with custom collate function
dataloader = create_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    fixed_length=80000,  # 5 seconds at 16kHz (optional)
)

# Iterate over batches
for batch in dataloader:
    waveforms = batch["waveforms"]       # (B, max_length)
    lengths = batch["lengths"]           # (B,)
    labels = batch["labels"]             # (B,)
    attention_mask = batch["attention_mask"]  # (B, max_length)

    # Your training code here
    break
```

## SOUSADataset Class

The `SOUSADataset` class wraps the SOUSA parquet files and provides a standard PyTorch `Dataset` interface.

### Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `data_dir` | `str \| Path` | required | Path to SOUSA dataset directory |
| `split` | `str \| None` | `None` | `"train"`, `"validation"`, `"test"`, or `None` for all |
| `target` | `str \| list[str]` | `"overall_score"` | Target variable(s) for labels |
| `resample_rate` | `int \| None` | `16000` | Target sample rate (None keeps 44.1kHz) |
| `max_length_sec` | `float \| None` | `None` | Maximum audio length in seconds |
| `transform` | `Callable \| None` | `None` | Optional audio transform |
| `normalize_audio` | `bool` | `True` | Normalize audio to [-1, 1] |

### Target Options

=== "Classification"

    ```python
    # Skill tier classification (4 classes)
    dataset = SOUSADataset(
        data_dir="output/dataset",
        target="skill_tier",
    )
    # Labels: 0=beginner, 1=intermediate, 2=advanced, 3=professional

    # Rudiment classification (40 classes)
    dataset = SOUSADataset(
        data_dir="output/dataset",
        target="rudiment_slug",
    )
    ```

=== "Regression"

    ```python
    # Single score regression
    dataset = SOUSADataset(
        data_dir="output/dataset",
        target="overall_score",
    )
    # Labels: normalized to [0, 1]

    # Any score column works
    dataset = SOUSADataset(
        data_dir="output/dataset",
        target="timing_accuracy",
    )
    ```

=== "Multi-Task"

    ```python
    # Multiple targets for multi-task learning
    dataset = SOUSADataset(
        data_dir="output/dataset",
        target=["overall_score", "timing_accuracy", "tempo_stability"],
    )
    # Labels: tensor of shape (3,) with normalized scores
    ```

### Accessing Samples

```python
# Get a single sample
sample = dataset[0]

print(sample["waveform"].shape)    # torch.Size([80000]) - 5 sec @ 16kHz
print(sample["label"])             # 2 (for skill_tier target)
print(sample["sample_id"])         # "adv042_single_paradiddle_100bpm_..."
print(sample["skill_tier"])        # "advanced"
print(sample["rudiment_slug"])     # "single_paradiddle"
print(sample["tempo_bpm"])         # 100
```

## Collate Functions

SOUSA audio samples have variable lengths. We provide two collate strategies:

### Variable-Length Collation

Pads each batch to the maximum length within that batch. Memory-efficient for diverse lengths.

```python
from examples.pytorch_dataloader import collate_variable_length

dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=collate_variable_length,
)

batch = next(iter(dataloader))
# batch["waveforms"].shape varies per batch
# batch["attention_mask"] indicates valid vs padded positions
```

### Fixed-Length Collation

Pads/truncates all samples to a fixed length. More efficient for training (consistent tensor shapes).

```python
from examples.pytorch_dataloader import collate_fixed_length
from functools import partial

# Create collate function with fixed length
collate_fn = partial(collate_fixed_length, target_length=80000)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=collate_fn,
)

batch = next(iter(dataloader))
# batch["waveforms"].shape is always (B, 80000)
```

### Batch Dictionary

Both collate functions return the same structure:

```python
{
    "waveforms": torch.Tensor,      # (B, max_length) - padded audio
    "lengths": torch.Tensor,        # (B,) - original lengths
    "labels": torch.Tensor,         # (B,) or (B, num_targets)
    "attention_mask": torch.Tensor, # (B, max_length) - 1=valid, 0=padding
    "sample_ids": list[str],        # Sample identifiers
    "metadata": {
        "skill_tiers": list[str],
        "rudiment_slugs": list[str],
        "tempo_bpms": list[int],
    }
}
```

## Stratified Batch Sampler

For classification tasks, use stratified sampling to ensure balanced batches:

```python
from examples.pytorch_dataloader import StratifiedBatchSampler

sampler = StratifiedBatchSampler(
    dataset,
    batch_size=16,
    drop_last=False,
)

dataloader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_variable_length,
)

# Each batch contains ~equal samples from each skill tier
```

## Training Loop Example

Here's a complete training loop for skill tier classification:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from examples.pytorch_dataloader import (
    SOUSADataset,
    create_dataloader,
    SKILL_TIER_TO_ID,
    ID_TO_SKILL_TIER,
)


class SimpleClassifier(nn.Module):
    """Simple 1D CNN classifier for demonstration."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, attention_mask=None):
        # x: (B, T) -> (B, 1, T)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.classifier(x)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        waveforms = batch["waveforms"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(waveforms, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    train_dataset = SOUSADataset(
        data_dir="output/dataset",
        split="train",
        target="skill_tier",
        resample_rate=16000,
        max_length_sec=5.0,
    )

    val_dataset = SOUSADataset(
        data_dir="output/dataset",
        split="validation",
        target="skill_tier",
        resample_rate=16000,
        max_length_sec=5.0,
    )

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        fixed_length=80000,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        fixed_length=80000,
    )

    # Model, optimizer, loss
    model = SimpleClassifier(num_classes=4).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2%}")


if __name__ == "__main__":
    main()
```

## Custom Transforms

Add audio augmentation during training:

```python
import torchaudio.transforms as T


class AudioTransform:
    """Compose multiple audio transforms."""

    def __init__(self, sample_rate: int = 16000):
        self.time_stretch = T.TimeStretch()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=30)
        self.time_mask = T.TimeMasking(time_mask_param=100)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Add your augmentations here
        # Note: Some transforms work on spectrograms, not waveforms
        return waveform


# Use with dataset
dataset = SOUSADataset(
    data_dir="output/dataset",
    split="train",
    target="skill_tier",
    transform=AudioTransform(),
)
```

## Performance Tips

!!! tip "Optimization Recommendations"

    1. **Use `fixed_length` collation** for faster training (consistent tensor shapes)

    2. **Increase `num_workers`** based on your CPU cores:
       ```python
       dataloader = create_dataloader(dataset, num_workers=8)
       ```

    3. **Enable `pin_memory`** for faster GPU transfer (enabled by default)

    4. **Resample to 16kHz** - sufficient for drum audio, faster processing

    5. **Use smaller batch sizes** if memory-limited:
       ```python
       dataloader = create_dataloader(dataset, batch_size=8)
       ```

## Label Mappings

```python
from examples.pytorch_dataloader import (
    SKILL_TIER_LABELS,      # ["beginner", "intermediate", "advanced", "professional"]
    SKILL_TIER_TO_ID,       # {"beginner": 0, "intermediate": 1, ...}
    ID_TO_SKILL_TIER,       # {0: "beginner", 1: "intermediate", ...}
)

# Convert predictions back to labels
predictions = model(batch["waveforms"])
predicted_ids = predictions.argmax(dim=1)
predicted_labels = [ID_TO_SKILL_TIER[i.item()] for i in predicted_ids]
```

## Next Steps

- [Feature Extraction](feature-extraction.md) - Extract mel spectrograms and pretrained features
- [Filtering Samples](filtering.md) - Filter dataset before creating DataLoader
- [Hierarchical Labels](hierarchical-labels.md) - Use stroke/measure-level labels for sequence models
