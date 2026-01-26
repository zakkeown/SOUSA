# Hierarchical Labels

SOUSA provides labels at three hierarchical levels: **exercise**, **measure**, and **stroke**. This guide shows how to access and use these multi-level labels for various ML tasks.

!!! info "Source Code"
    Complete implementation: [`examples/hierarchical_labels.py`](https://github.com/zakkeown/SOUSA/blob/main/examples/hierarchical_labels.py)

## Label Hierarchy

```
Exercise Level (1 per sample)
├── Overall score, timing accuracy, etc.
│
├── Measure Level (N per sample)
│   ├── Per-measure timing statistics
│   ├── Per-measure velocity statistics
│   └── Hand balance metrics
│
└── Stroke Level (M per sample)
    ├── Intended vs actual timing
    ├── Intended vs actual velocity
    ├── Hand (L/R)
    └── Articulation type
```

## Prerequisites

```bash
pip install pandas numpy
# Optional for visualization:
pip install matplotlib
# Optional for PyTorch integration:
pip install torch
```

## Loading Hierarchical Labels

```python
from examples.hierarchical_labels import HierarchicalLabels

# Load all label levels
labels = HierarchicalLabels("output/dataset")

# Get sample IDs
sample_ids = labels.get_sample_ids()
print(f"Total samples: {len(sample_ids)}")

sample_id = sample_ids[0]
```

## Exercise-Level Labels

Exercise-level labels provide overall performance scores:

```python
# Get exercise scores
exercise = labels.get_exercise(sample_id)

print(f"Overall score: {exercise['overall_score']:.1f}")
print(f"Timing accuracy: {exercise['timing_accuracy']:.1f}")
print(f"Timing consistency: {exercise['timing_consistency']:.1f}")
print(f"Velocity accuracy: {exercise['velocity_accuracy']:.1f}")
print(f"Tempo stability: {exercise['tempo_stability']:.1f}")
```

### Exercise Score Columns

| Column | Description |
|--------|-------------|
| `overall_score` | Composite quality score (0-100) |
| `timing_accuracy` | Mean timing error score (0-100) |
| `timing_consistency` | Timing error variance score (0-100) |
| `velocity_accuracy` | Mean velocity error score (0-100) |
| `velocity_consistency` | Velocity variance score (0-100) |
| `hand_balance` | L/R hand consistency (0-100) |
| `tempo_stability` | Tempo drift score (0-100) |
| `flam_quality` | Flam spacing quality (if applicable) |
| `diddle_quality` | Diddle evenness (if applicable) |

## Measure-Level Labels

Measure-level labels aggregate statistics per measure:

```python
# Get measures for a sample
measures = labels.get_measures(sample_id)

print(f"Number of measures: {len(measures)}")
print(f"Columns: {list(measures.columns)}")

# Examine first measure
first_measure = measures.iloc[0]
print(f"\nMeasure 0:")
print(f"  Strokes {first_measure['stroke_start']} to {first_measure['stroke_end']}")
print(f"  Timing mean error: {first_measure['timing_mean_error_ms']:.2f} ms")
print(f"  Timing std: {first_measure['timing_std_ms']:.2f} ms")
```

### Measure Columns

| Column | Description |
|--------|-------------|
| `index` | Measure number (0-based) |
| `stroke_start` | First stroke index in measure |
| `stroke_end` | Last stroke index (exclusive) |
| `timing_mean_error_ms` | Mean timing error in ms |
| `timing_std_ms` | Timing error standard deviation |
| `velocity_mean` | Mean MIDI velocity |
| `velocity_std` | Velocity standard deviation |
| `lr_velocity_ratio` | Left/right hand velocity ratio |

## Stroke-Level Labels

Stroke-level labels provide the finest granularity:

```python
# Get all strokes for a sample
strokes = labels.get_strokes(sample_id)

print(f"Number of strokes: {len(strokes)}")
print(f"Columns: {list(strokes.columns)}")

# Examine first few strokes
for i, stroke in strokes.head().iterrows():
    print(f"Stroke {i}: {stroke['hand']} hand, "
          f"intended={stroke['intended_time_ms']:.1f}ms, "
          f"actual={stroke['actual_time_ms']:.1f}ms, "
          f"error={stroke['timing_error_ms']:.1f}ms")
```

### Stroke Columns

| Column | Description |
|--------|-------------|
| `index` | Stroke number (0-based) |
| `hand` | `"L"` or `"R"` |
| `stroke_type` | `"tap"`, `"accent"`, `"grace"`, `"diddle"` |
| `intended_time_ms` | Target timing (milliseconds) |
| `actual_time_ms` | Performed timing (milliseconds) |
| `timing_error_ms` | Actual - intended (positive = late) |
| `intended_velocity` | Target MIDI velocity (0-127) |
| `actual_velocity` | Performed MIDI velocity |
| `is_grace_note` | Boolean flag |
| `is_accent` | Boolean flag |

## Accessing Strokes Within a Measure

```python
# Get strokes for measure 2
measure_strokes = labels.get_strokes_for_measure(sample_id, measure_index=2)

print(f"Strokes in measure 2: {len(measure_strokes)}")
for _, stroke in measure_strokes.iterrows():
    print(f"  {stroke['hand']} at {stroke['actual_time_ms']:.1f}ms")
```

## Aligning Strokes with Audio

For sequence models, align stroke times with audio sample indices:

```python
# Align strokes with audio at 16kHz
aligned = labels.align_strokes_with_audio(
    sample_id,
    sample_rate=16000,
)

print("First 5 strokes with sample indices:")
for _, stroke in aligned.head().iterrows():
    print(f"  Stroke at {stroke['actual_time_ms']:.1f}ms -> "
          f"sample {stroke['actual_sample_idx']}")
```

## Creating Stroke Sequences for RNNs/Transformers

### Fixed-Length Sequences

```python
# Create padded stroke sequence
sequence = labels.create_stroke_sequence(
    sample_id,
    max_strokes=128,
    features=["timing_error_ms", "actual_velocity", "is_accent", "is_grace_note"],
)

print(f"Sequence shape: {sequence['sequence'].shape}")  # (128, 4)
print(f"Valid strokes: {sequence['mask'].sum()}")
print(f"Feature names: {sequence['feature_names']}")
```

### PyTorch Tensors

```python
# Get as PyTorch tensors
sequence_torch = labels.create_stroke_sequence_torch(
    sample_id,
    max_strokes=128,
)

print(f"Sequence: {sequence_torch['sequence'].shape}")  # torch.Size([128, 4])
print(f"Mask: {sequence_torch['mask'].shape}")          # torch.Size([128])
```

## Frame-Level Targets

### Onset Detection Targets

Create binary onset labels aligned with audio frames:

```python
# Create onset targets at 10ms resolution
duration_ms = strokes["actual_time_ms"].max() + 500  # Add buffer
onset_targets = labels.create_onset_targets(
    sample_id,
    duration_ms=duration_ms,
    resolution_ms=10.0,
)

print(f"Onset target shape: {onset_targets.shape}")
print(f"Number of onsets: {onset_targets.sum()}")
```

### Hand Classification Targets

```python
# Create hand (L/R) targets for each frame
hand_targets = labels.create_hand_targets(
    sample_id,
    duration_ms=duration_ms,
    resolution_ms=10.0,
)

print(f"Hand target shape: {hand_targets.shape}")
print(f"Unique values: {np.unique(hand_targets)}")
# 0 = no stroke, 1 = left hand, 2 = right hand
```

## Timing Error Sequences

```python
# Get normalized timing errors for all strokes
timing_errors = labels.get_timing_error_sequence(
    sample_id,
    normalize=True,  # Normalized to ~[-1, 1]
)

print(f"Timing errors shape: {timing_errors.shape}")
print(f"Range: [{timing_errors.min():.3f}, {timing_errors.max():.3f}]")
```

## Measure-Level Summary

Compute extended measure statistics:

```python
# Get detailed measure summary
summary = labels.compute_measure_summary(sample_id)

print("Measure summary:")
for _, measure in summary.iterrows():
    print(f"  Measure {measure['index']}: "
          f"{measure['num_strokes']} strokes, "
          f"{measure['num_accents']} accents, "
          f"{measure['pct_right_hand']:.0%} right hand")
```

## Visualization

Visualize stroke timeline with audio:

```python
import torchaudio

# Load audio
audio_path = f"output/dataset/{labels.get_sample_metadata(sample_id)['audio_path']}"
waveform, sr = torchaudio.load(audio_path)
audio = waveform.squeeze().numpy()

# Resample if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(waveform).squeeze().numpy()

# Create visualization
fig, axes = labels.visualize_strokes(
    sample_id,
    audio_array=audio,
    sample_rate=16000,
    figsize=(14, 10),
    save_path="stroke_visualization.png",
)
```

The visualization includes:

1. Waveform with stroke markers (red = right hand, blue = left hand)
2. Timing errors over time
3. Velocity pattern
4. Sticking pattern (L/R alternation)

## Multi-Task Learning Example

Use hierarchical labels for multi-task targets:

```python
import torch
from torch.utils.data import Dataset


class HierarchicalLabelDataset(Dataset):
    """Dataset with multi-level targets."""

    def __init__(
        self,
        data_dir: str,
        max_strokes: int = 128,
    ):
        self.labels = HierarchicalLabels(data_dir)
        self.sample_ids = self.labels.get_sample_ids()
        self.max_strokes = max_strokes

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Exercise-level targets
        exercise = self.labels.get_exercise(sample_id)
        exercise_target = torch.tensor([
            exercise["overall_score"] / 100.0,
            exercise["timing_accuracy"] / 100.0,
            exercise["tempo_stability"] / 100.0,
        ])

        # Stroke-level sequence
        stroke_seq = self.labels.create_stroke_sequence_torch(
            sample_id,
            max_strokes=self.max_strokes,
        )

        # Measure count (for auxiliary task)
        measures = self.labels.get_measures(sample_id)
        num_measures = len(measures)

        return {
            "sample_id": sample_id,
            "exercise_target": exercise_target,       # (3,)
            "stroke_sequence": stroke_seq["sequence"], # (max_strokes, 4)
            "stroke_mask": stroke_seq["mask"],         # (max_strokes,)
            "num_measures": num_measures,
        }


# Usage
dataset = HierarchicalLabelDataset("output/dataset")
sample = dataset[0]
print(f"Exercise target: {sample['exercise_target']}")
print(f"Stroke sequence: {sample['stroke_sequence'].shape}")
```

## Regression vs Classification Tasks

### Stroke-Level Regression

Predict timing errors for each stroke:

```python
# Target: timing error per stroke
stroke_seq = labels.create_stroke_sequence(
    sample_id,
    features=["timing_error_ms"],
)
# Use stroke_seq["sequence"][:, 0] as regression target
```

### Measure-Level Classification

Classify measures as "good" or "needs work":

```python
measures = labels.get_measures(sample_id)

# Create binary labels based on timing std
threshold_ms = 15.0
measure_labels = (measures["timing_std_ms"] < threshold_ms).astype(int)
# 1 = good timing consistency, 0 = needs work
```

### Exercise-Level Ordinal Regression

Predict skill tier from ordinal labels:

```python
SKILL_TIER_ORDINAL = {
    "beginner": 0,
    "intermediate": 1,
    "advanced": 2,
    "professional": 3,
}

metadata = labels.get_sample_metadata(sample_id)
ordinal_label = SKILL_TIER_ORDINAL[metadata["skill_tier"]]
```

## Example: Stroke-Level Transformer

```python
import torch
import torch.nn as nn


class StrokeTransformer(nn.Module):
    """Transformer for stroke sequence analysis."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 4,  # Skill tiers
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, stroke_seq, mask):
        # stroke_seq: (B, max_strokes, input_dim)
        # mask: (B, max_strokes)

        x = self.input_proj(stroke_seq)

        # Create attention mask (True = ignore)
        attn_mask = ~mask

        x = self.encoder(x, src_key_padding_mask=attn_mask)

        # Pool over valid strokes
        x = x * mask.unsqueeze(-1)
        x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

        return self.classifier(x)


# Usage
model = StrokeTransformer()

sample = dataset[0]
stroke_seq = sample["stroke_sequence"].unsqueeze(0)  # Add batch dim
mask = sample["stroke_mask"].unsqueeze(0)

logits = model(stroke_seq, mask)
print(f"Output shape: {logits.shape}")  # (1, 4)
```

## Next Steps

- [PyTorch DataLoader](pytorch-dataloader.md) - Combine with audio features
- [Feature Extraction](feature-extraction.md) - Extract mel spectrograms aligned with strokes
- [Filtering Samples](filtering.md) - Filter by exercise-level scores
