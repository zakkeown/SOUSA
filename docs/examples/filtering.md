# Filtering Samples

This guide shows how to filter the SOUSA dataset by skill tier, rudiment type, tempo, score ranges, soundfont, and augmentation settings. Filtering helps create focused subsets for specific experiments or training scenarios.

!!! info "Source Code"
    Complete implementation: [`examples/filtering.py`](https://github.com/zakkeown/SOUSA/blob/main/examples/filtering.py)

## Prerequisites

```bash
pip install pandas pyarrow
```

## Loading Dataset Metadata

All filtering operates on the parquet metadata files:

```python
import pandas as pd
from pathlib import Path


def load_metadata(data_dir: str) -> pd.DataFrame:
    """Load and merge sample metadata."""
    data_dir = Path(data_dir)

    samples = pd.read_parquet(data_dir / "labels" / "samples.parquet")
    exercises = pd.read_parquet(data_dir / "labels" / "exercises.parquet")

    return samples.merge(exercises, on="sample_id", how="left")


# Load all metadata
df = load_metadata("output/dataset")
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
```

## Filter by Skill Tier

```python
def filter_by_skill_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Filter to specific skill tier."""
    return df[df["skill_tier"] == tier].reset_index(drop=True)


# Get only advanced players
advanced_df = filter_by_skill_tier(df, "advanced")
print(f"Advanced samples: {len(advanced_df)}")

# Available tiers
SKILL_TIERS = ["beginner", "intermediate", "advanced", "professional"]
```

### Multiple Skill Tiers

```python
# Get beginner and intermediate (novice group)
novice_df = df[df["skill_tier"].isin(["beginner", "intermediate"])]

# Get advanced and professional (skilled group)
skilled_df = df[df["skill_tier"].isin(["advanced", "professional"])]
```

## Filter by Rudiment

```python
def filter_by_rudiment(df: pd.DataFrame, rudiment_slug: str) -> pd.DataFrame:
    """Filter to specific rudiment."""
    return df[df["rudiment_slug"] == rudiment_slug].reset_index(drop=True)


# Get single paradiddle samples
paradiddle_df = filter_by_rudiment(df, "single_paradiddle")

# List all available rudiments
rudiments = sorted(df["rudiment_slug"].unique())
print(f"Available rudiments ({len(rudiments)}):")
for r in rudiments[:10]:
    print(f"  - {r}")
```

### Filter by Rudiment Category

```python
# Get all paradiddle variants
paradiddle_variants = df[df["rudiment_slug"].str.contains("paradiddle")]

# Get all roll types
rolls = df[df["rudiment_slug"].str.contains("roll")]

# Get all flam rudiments
flams = df[df["rudiment_slug"].str.contains("flam")]
```

## Filter by Tempo

```python
def filter_by_tempo_range(
    df: pd.DataFrame,
    min_tempo: int = 60,
    max_tempo: int = 200,
) -> pd.DataFrame:
    """Filter to tempo range (BPM)."""
    mask = (df["tempo_bpm"] >= min_tempo) & (df["tempo_bpm"] <= max_tempo)
    return df[mask].reset_index(drop=True)


# Get slow tempos only (60-100 BPM)
slow_df = filter_by_tempo_range(df, 60, 100)

# Get fast tempos only (140-200 BPM)
fast_df = filter_by_tempo_range(df, 140, 200)
```

## Filter by Score

```python
def filter_by_score_range(
    df: pd.DataFrame,
    min_score: float = 0,
    max_score: float = 100,
    score_column: str = "overall_score",
) -> pd.DataFrame:
    """Filter to samples within score range."""
    mask = (df[score_column] >= min_score) & (df[score_column] <= max_score)
    return df[mask].reset_index(drop=True)


# High-quality performances only
high_quality_df = filter_by_score_range(df, min_score=80)

# Low-quality performances (for contrastive learning)
low_quality_df = filter_by_score_range(df, max_score=40)

# Filter by timing accuracy specifically
accurate_df = filter_by_score_range(df, min_score=75, score_column="timing_accuracy")
```

### Available Score Columns

| Column | Description | Range |
|--------|-------------|-------|
| `overall_score` | Composite quality score | 0-100 |
| `timing_accuracy` | How close to intended timing | 0-100 |
| `timing_consistency` | Consistency of timing errors | 0-100 |
| `velocity_accuracy` | How close to intended velocity | 0-100 |
| `velocity_consistency` | Consistency of velocity | 0-100 |
| `hand_balance` | L/R hand consistency | 0-100 |
| `tempo_stability` | Tempo drift throughout exercise | 0-100 |
| `tier_confidence` | Confidence in skill tier label | 0-1 |

## Filter by Augmentation

### Clean Samples Only

```python
def filter_clean_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to clean (non-augmented) samples only."""
    if "augmentation_preset" not in df.columns:
        return df

    mask = df["augmentation_preset"].isna() | (df["augmentation_preset"] == "none")
    return df[mask].reset_index(drop=True)


clean_df = filter_clean_samples(df)
print(f"Clean samples: {len(clean_df)}")
```

### Augmented Samples Only

```python
def filter_augmented_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to augmented samples only."""
    if "augmentation_preset" not in df.columns:
        return df

    mask = df["augmentation_preset"].notna() & (df["augmentation_preset"] != "none")
    return df[mask].reset_index(drop=True)


augmented_df = filter_augmented_samples(df)
print(f"Augmented samples: {len(augmented_df)}")
```

### Specific Augmentation Preset

```python
def filter_by_augmentation_preset(df: pd.DataFrame, preset: str) -> pd.DataFrame:
    """Filter to specific augmentation preset."""
    return df[df["augmentation_preset"] == preset].reset_index(drop=True)


# Get moderately augmented samples
moderate_df = filter_by_augmentation_preset(df, "moderate")

# Available presets (check your dataset)
if "augmentation_preset" in df.columns:
    presets = df["augmentation_preset"].unique()
    print(f"Available presets: {presets}")
```

## Filter by Soundfont

```python
def filter_by_soundfont(df: pd.DataFrame, soundfont: str) -> pd.DataFrame:
    """Filter to specific soundfont."""
    return df[df["soundfont"] == soundfont].reset_index(drop=True)


# Get practice pad samples only
practice_pad_df = filter_by_soundfont(df, "practice_pad")

# Get marching snare samples
marching_df = filter_by_soundfont(df, "marching_snare")

# List available soundfonts
if "soundfont" in df.columns:
    soundfonts = df["soundfont"].unique()
    print(f"Available soundfonts: {soundfonts}")
```

## Combined Filters

Chain multiple filters for specific subsets:

```python
# Advanced players, single paradiddle, fast tempo, high quality
subset = df[
    (df["skill_tier"] == "advanced") &
    (df["rudiment_slug"] == "single_paradiddle") &
    (df["tempo_bpm"] >= 140) &
    (df["overall_score"] >= 75)
]

print(f"Subset size: {len(subset)}")
```

### Using FilteredSOUSADataset

The `FilteredSOUSADataset` class combines filtering with PyTorch DataLoader:

```python
from examples.filtering import FilteredSOUSADataset

# Create filtered dataset
dataset = FilteredSOUSADataset(
    data_dir="output/dataset",
    split="train",
    target="skill_tier",
    filters={
        "augmentation_preset": None,    # None = clean/null
        "soundfont": "practice_pad",
        "skill_tier": "advanced",
    }
)

print(f"Filtered dataset size: {len(dataset)}")
```

### Filter Values

| Filter | Value Type | Example |
|--------|------------|---------|
| Single value | `str` | `{"skill_tier": "advanced"}` |
| Multiple values | `list` | `{"skill_tier": ["advanced", "professional"]}` |
| Null/clean | `None` | `{"augmentation_preset": None}` |

## Augmentation Groups

Link clean samples with their augmented variants:

```python
def group_by_augmentation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group samples by augmentation_group_id."""
    if "augmentation_group_id" not in df.columns:
        return {}

    return {
        group_id: group_df.reset_index(drop=True)
        for group_id, group_df in df.groupby("augmentation_group_id")
    }


def get_clean_augmented_pairs(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Get pairs of (clean_sample_id, [augmented_sample_ids])."""
    if "augmentation_group_id" not in df.columns:
        return []

    pairs = []
    for group_id, group_df in df.groupby("augmentation_group_id"):
        clean_mask = (
            group_df["augmentation_preset"].isna() |
            (group_df["augmentation_preset"] == "none")
        )
        clean_ids = group_df[clean_mask]["sample_id"].tolist()
        augmented_ids = group_df[~clean_mask]["sample_id"].tolist()

        if clean_ids and augmented_ids:
            for clean_id in clean_ids:
                pairs.append((clean_id, augmented_ids))

    return pairs


# Get pairs for contrastive learning
pairs = get_clean_augmented_pairs(df)
print(f"Found {len(pairs)} clean-augmented pairs")

if pairs:
    clean_id, aug_ids = pairs[0]
    print(f"Clean: {clean_id}")
    print(f"Augmented: {aug_ids}")
```

## Dataset Statistics

Print summary statistics for any filtered subset:

```python
def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset"):
    """Print summary statistics for a filtered dataset."""
    print(f"\n{name}")
    print("=" * 50)
    print(f"Total samples: {len(df)}")

    if "skill_tier" in df.columns:
        print(f"\nBy skill tier:")
        for tier, count in df["skill_tier"].value_counts().items():
            pct = count / len(df) * 100
            print(f"  {tier}: {count} ({pct:.1f}%)")

    if "soundfont" in df.columns:
        print(f"\nBy soundfont:")
        for sf, count in df["soundfont"].value_counts().items():
            print(f"  {sf}: {count}")

    if "overall_score" in df.columns:
        print(f"\nScore statistics:")
        print(f"  Mean: {df['overall_score'].mean():.1f}")
        print(f"  Std:  {df['overall_score'].std():.1f}")
        print(f"  Min:  {df['overall_score'].min():.1f}")
        print(f"  Max:  {df['overall_score'].max():.1f}")


# Use it
print_dataset_summary(df, "Full Dataset")
print_dataset_summary(filter_clean_samples(df), "Clean Samples Only")
```

## Example: Balanced Skill Tier Subset

Create a balanced subset for classification:

```python
def create_balanced_subset(
    df: pd.DataFrame,
    samples_per_tier: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create balanced subset with equal samples per skill tier."""
    balanced_dfs = []

    for tier in ["beginner", "intermediate", "advanced", "professional"]:
        tier_df = df[df["skill_tier"] == tier]

        if len(tier_df) >= samples_per_tier:
            sampled = tier_df.sample(n=samples_per_tier, random_state=random_state)
        else:
            sampled = tier_df  # Take all if not enough

        balanced_dfs.append(sampled)

    return pd.concat(balanced_dfs, ignore_index=True)


balanced_df = create_balanced_subset(df, samples_per_tier=500)
print_dataset_summary(balanced_df, "Balanced Dataset")
```

## Example: Filter for Specific Experiment

```python
def create_experiment_subset(
    data_dir: str,
    output_path: str,
):
    """Create a filtered subset for a timing accuracy experiment."""
    df = load_metadata(data_dir)

    # Filter criteria:
    # - Clean audio only (no augmentation)
    # - Single paradiddle rudiment
    # - Medium tempo range (100-140 BPM)
    # - Exclude extreme scores (keep 20-90 range)

    subset = df[
        # Clean audio
        (df["augmentation_preset"].isna() | (df["augmentation_preset"] == "none")) &
        # Single paradiddle
        (df["rudiment_slug"] == "single_paradiddle") &
        # Medium tempo
        (df["tempo_bpm"] >= 100) & (df["tempo_bpm"] <= 140) &
        # Exclude extremes
        (df["overall_score"] >= 20) & (df["overall_score"] <= 90)
    ]

    print(f"Filtered to {len(subset)} samples")

    # Save sample IDs for reproducibility
    subset[["sample_id", "skill_tier", "tempo_bpm", "overall_score"]].to_csv(
        output_path, index=False
    )
    print(f"Saved to {output_path}")

    return subset


# Run
subset = create_experiment_subset(
    "output/dataset",
    "experiment_samples.csv"
)
```

## Using Filtered Indices with DataLoader

```python
from torch.utils.data import Subset
from examples.pytorch_dataloader import SOUSADataset, create_dataloader


# Load full dataset
full_dataset = SOUSADataset(
    data_dir="output/dataset",
    split="train",
    target="skill_tier",
)

# Get filtered indices
df = load_metadata("output/dataset")
filtered_df = df[
    (df["skill_tier"].isin(["advanced", "professional"])) &
    (df["overall_score"] >= 70)
]

# Map sample_ids to dataset indices
sample_id_to_idx = {
    row["sample_id"]: idx
    for idx, row in full_dataset.data.iterrows()
}

filtered_indices = [
    sample_id_to_idx[sid]
    for sid in filtered_df["sample_id"]
    if sid in sample_id_to_idx
]

# Create subset
filtered_dataset = Subset(full_dataset, filtered_indices)
print(f"Filtered dataset size: {len(filtered_dataset)}")

# Create dataloader
dataloader = create_dataloader(filtered_dataset, batch_size=16)
```

## Next Steps

- [PyTorch DataLoader](pytorch-dataloader.md) - Use filtered data in training
- [Feature Extraction](feature-extraction.md) - Extract features from filtered subsets
- [Hierarchical Labels](hierarchical-labels.md) - Access detailed stroke-level labels
