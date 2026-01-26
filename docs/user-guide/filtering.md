# Filtering and Preprocessing

This guide covers filtering SOUSA data by various criteria and understanding the dataset distribution for effective preprocessing.

## Filtering by Skill Tier

### Basic Skill Filtering

```python
from datasets import load_dataset

dataset = load_dataset("zkeown/sousa")

# Filter by single skill tier
beginners = dataset["train"].filter(lambda x: x["skill_tier"] == "beginner")
advanced = dataset["train"].filter(lambda x: x["skill_tier"] == "advanced")

# Multiple skill tiers
novice_players = dataset["train"].filter(
    lambda x: x["skill_tier"] in ["beginner", "intermediate"]
)
```

### Skill Tier Distribution

| Tier | Proportion | Full Dataset (~100K) | Characteristics |
|------|------------|---------------------|-----------------|
| Beginner | 25% | ~25,000 | High timing variance (50ms std), learning basics |
| Intermediate | 35% | ~35,000 | Moderate consistency, developing technique |
| Advanced | 25% | ~25,000 | Strong technique, minor timing errors |
| Professional | 15% | ~15,000 | Near-perfect (7ms std), consistent execution |

!!! info "Class imbalance"
    Professional samples are ~2.3x less frequent than intermediate. Use class weights for classification tasks.

### Binary Skill Labels

For reduced overlap between classes:

```python
# Use binary labels (novice vs skilled)
novice = dataset["train"].filter(lambda x: x["skill_tier_binary"] == "novice")
skilled = dataset["train"].filter(lambda x: x["skill_tier_binary"] == "skilled")

# Binary mapping:
# novice = beginner + intermediate
# skilled = advanced + professional
```

### Filtering by Tier Confidence

High-confidence samples are more clearly representative of their tier:

```python
# Filter for clear tier assignments (reduces overlap)
confident = dataset["train"].filter(lambda x: x["tier_confidence"] > 0.5)

# Very confident samples only
very_confident = dataset["train"].filter(lambda x: x["tier_confidence"] > 0.7)
```

| Tier Confidence | Description | Use Case |
|-----------------|-------------|----------|
| > 0.7 | Clear tier assignment | Classification with clean labels |
| 0.5 - 0.7 | Moderate confidence | General training |
| < 0.5 | Borderline cases | May hurt classification accuracy |

## Filtering by Rudiment

### Single Rudiment

```python
# Specific rudiment
paradiddles = dataset["train"].filter(
    lambda x: x["rudiment_slug"] == "single_paradiddle"
)

# By rudiment name pattern
flams = dataset["train"].filter(lambda x: "flam" in x["rudiment_slug"])
rolls = dataset["train"].filter(lambda x: "roll" in x["rudiment_slug"])
```

### Rudiment Categories

All 40 PAS rudiments organized by category:

=== "Roll Rudiments (15)"

    | Slug | Name |
    |------|------|
    | `single_stroke_roll` | Single Stroke Roll |
    | `single_stroke_four` | Single Stroke Four |
    | `single_stroke_seven` | Single Stroke Seven |
    | `multiple_bounce_roll` | Multiple Bounce Roll |
    | `triple_stroke_roll` | Triple Stroke Roll |
    | `double_stroke_roll` | Double Stroke Open Roll |
    | `five_stroke_roll` | Five Stroke Roll |
    | `six_stroke_roll` | Six Stroke Roll |
    | `seven_stroke_roll` | Seven Stroke Roll |
    | `nine_stroke_roll` | Nine Stroke Roll |
    | `ten_stroke_roll` | Ten Stroke Roll |
    | `eleven_stroke_roll` | Eleven Stroke Roll |
    | `thirteen_stroke_roll` | Thirteen Stroke Roll |
    | `fifteen_stroke_roll` | Fifteen Stroke Roll |
    | `seventeen_stroke_roll` | Seventeen Stroke Roll |

=== "Paradiddle Rudiments (5)"

    | Slug | Name |
    |------|------|
    | `single_paradiddle` | Single Paradiddle |
    | `double_paradiddle` | Double Paradiddle |
    | `triple_paradiddle` | Triple Paradiddle |
    | `paradiddle_diddle` | Paradiddle-Diddle |
    | `single_paradiddle_diddle` | Single Paradiddle-Diddle |

=== "Flam Rudiments (11)"

    | Slug | Name |
    |------|------|
    | `flam` | Flam |
    | `flam_accent` | Flam Accent |
    | `flam_tap` | Flam Tap |
    | `flamacue` | Flamacue |
    | `flam_paradiddle` | Flam Paradiddle |
    | `single_flammed_mill` | Single Flammed Mill |
    | `flam_paradiddle_diddle` | Flam Paradiddle-Diddle |
    | `pataflafla` | Pataflafla |
    | `swiss_army_triplet` | Swiss Army Triplet |
    | `inverted_flam_tap` | Inverted Flam Tap |
    | `flam_drag` | Flam Drag |

=== "Drag Rudiments (9)"

    | Slug | Name |
    |------|------|
    | `drag` | Drag |
    | `single_drag_tap` | Single Drag Tap |
    | `double_drag_tap` | Double Drag Tap |
    | `lesson_25` | Lesson 25 |
    | `single_dragadiddle` | Single Dragadiddle |
    | `drag_paradiddle_1` | Drag Paradiddle #1 |
    | `drag_paradiddle_2` | Drag Paradiddle #2 |
    | `single_ratamacue` | Single Ratamacue |
    | `double_ratamacue` | Double Ratamacue |
    | `triple_ratamacue` | Triple Ratamacue |

### Filter by Rudiment Category

```python
# Define category sets
ROLL_RUDIMENTS = [
    "single_stroke_roll", "single_stroke_four", "single_stroke_seven",
    "multiple_bounce_roll", "triple_stroke_roll", "double_stroke_roll",
    "five_stroke_roll", "six_stroke_roll", "seven_stroke_roll",
    "nine_stroke_roll", "ten_stroke_roll", "eleven_stroke_roll",
    "thirteen_stroke_roll", "fifteen_stroke_roll", "seventeen_stroke_roll",
]

FLAM_RUDIMENTS = [
    "flam", "flam_accent", "flam_tap", "flamacue", "flam_paradiddle",
    "single_flammed_mill", "flam_paradiddle_diddle", "pataflafla",
    "swiss_army_triplet", "inverted_flam_tap", "flam_drag",
]

# Filter by category
rolls_only = dataset["train"].filter(lambda x: x["rudiment_slug"] in ROLL_RUDIMENTS)
flams_only = dataset["train"].filter(lambda x: x["rudiment_slug"] in FLAM_RUDIMENTS)
```

## Filtering by Tempo

```python
# Slow tempos (beginner-friendly)
slow = dataset["train"].filter(lambda x: x["tempo_bpm"] < 90)

# Medium tempos
medium = dataset["train"].filter(lambda x: 90 <= x["tempo_bpm"] <= 130)

# Fast tempos
fast = dataset["train"].filter(lambda x: x["tempo_bpm"] > 140)

# Specific tempo range
target_tempo = dataset["train"].filter(lambda x: 100 <= x["tempo_bpm"] <= 120)
```

### Tempo Distribution

| Tempo Range | BPM | Typical Skill Level |
|-------------|-----|---------------------|
| Slow | 60-90 | All levels |
| Medium | 90-130 | Intermediate+ |
| Fast | 130-160 | Advanced+ |
| Very Fast | 160-180 | Professional |

## Filtering by Score

### Overall Score

```python
# High performers (strong technique)
high_score = dataset["train"].filter(lambda x: x["overall_score"] > 80)

# Low performers (beginners)
low_score = dataset["train"].filter(lambda x: x["overall_score"] < 40)

# Score range
mid_range = dataset["train"].filter(
    lambda x: 40 <= x["overall_score"] <= 70
)
```

### Specific Score Metrics

```python
# By timing accuracy
good_timing = dataset["train"].filter(lambda x: x["timing_accuracy"] > 70)

# By hand balance
balanced = dataset["train"].filter(lambda x: x["hand_balance"] > 90)

# By velocity control
controlled = dataset["train"].filter(lambda x: x["velocity_control"] > 75)

# Combined criteria
elite = dataset["train"].filter(
    lambda x: x["timing_accuracy"] > 80
    and x["hand_balance"] > 95
    and x["overall_score"] > 85
)
```

### Available Scores

| Score | Range | Description |
|-------|-------|-------------|
| `overall_score` | 0-100 | Weighted composite score |
| `timing_accuracy` | 0-100 | How close to intended timing |
| `timing_consistency` | 0-100 | Variance in timing errors |
| `tempo_stability` | 0-100 | Consistency throughout exercise |
| `velocity_control` | 0-100 | Control over stroke dynamics |
| `hand_balance` | 0-100 | Evenness between L/R hands |
| `accent_differentiation` | 0-100 | Clarity of accented strokes |
| `accent_accuracy` | 0-100 | Correct accent placement |

### Rudiment-Specific Scores

These scores are only available for applicable rudiments:

```python
# Flam quality (flam rudiments only)
flam_samples = dataset["train"].filter(
    lambda x: x.get("flam_quality") is not None
)
good_flams = flam_samples.filter(lambda x: x["flam_quality"] > 80)

# Diddle quality (paradiddles, rolls with diddles)
diddle_samples = dataset["train"].filter(
    lambda x: x.get("diddle_quality") is not None
)

# Roll sustain (roll rudiments)
roll_samples = dataset["train"].filter(
    lambda x: x.get("roll_sustain") is not None
)
```

## Dataset Statistics

### Computing Statistics

```python
import pandas as pd
from collections import Counter

# Convert to pandas for analysis
df = dataset["train"].to_pandas()

# Basic counts
print(f"Total samples: {len(df)}")
print(f"Unique profiles: {df['profile_id'].nunique()}")
print(f"Unique rudiments: {df['rudiment_slug'].nunique()}")

# Skill tier distribution
skill_counts = Counter(df["skill_tier"])
for tier, count in skill_counts.items():
    print(f"{tier}: {count} ({count/len(df)*100:.1f}%)")

# Score statistics
print(f"\nOverall Score: mean={df['overall_score'].mean():.1f}, "
      f"std={df['overall_score'].std():.1f}")
print(f"Timing Accuracy: mean={df['timing_accuracy'].mean():.1f}, "
      f"std={df['timing_accuracy'].std():.1f}")
```

### Expected Statistics (Full Dataset)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Tempo (BPM) | 118 | 34 | 60 | 180 |
| Duration (sec) | 5.9 | 2.9 | 1.6 | 20 |
| Strokes/Sample | 42 | 18 | 16 | 96 |
| Overall Score | 37 | 31 | 0 | 95 |
| Timing Accuracy | 37 | 31 | 0 | 95 |
| Hand Balance | 88 | 11 | 35 | 100 |

### Score Distribution by Skill Tier

| Tier | Overall Score | Timing Accuracy | Hand Balance |
|------|---------------|-----------------|--------------|
| Beginner | 15-45 | 5-35 | 75-85 |
| Intermediate | 35-65 | 30-60 | 85-92 |
| Advanced | 60-85 | 55-85 | 92-96 |
| Professional | 80-95 | 75-95 | 95-100 |

## Preprocessing Recommendations

### For Classification Tasks

```python
# Recommended preprocessing for skill classification
from sklearn.model_selection import train_test_split

# Filter for confident tier assignments
df_confident = df[df["tier_confidence"] > 0.5]

# Ensure balanced classes with stratification
X = df_confident["sample_id"].values
y = df_confident["skill_tier"].values

# Note: SOUSA already has profile-based splits
# Use the provided splits rather than random splitting
train_ids = set(splits["train_profile_ids"])
val_ids = set(splits["val_profile_ids"])
test_ids = set(splits["test_profile_ids"])
```

### For Regression Tasks

```python
# Normalize scores to 0-1 range
df["overall_score_norm"] = df["overall_score"] / 100.0
df["timing_accuracy_norm"] = df["timing_accuracy"] / 100.0

# Remove outliers (optional)
q_low = df["overall_score"].quantile(0.01)
q_high = df["overall_score"].quantile(0.99)
df_filtered = df[(df["overall_score"] >= q_low) & (df["overall_score"] <= q_high)]
```

### Handling Null Values

Some scores are null for non-applicable rudiments:

```python
# Option 1: Filter to non-null
flam_data = df[df["flam_quality"].notna()]

# Option 2: Fill with sentinel value (for masking)
df["flam_quality_filled"] = df["flam_quality"].fillna(-1)

# Option 3: Separate handling by rudiment type
has_flams = df["rudiment_slug"].isin(FLAM_RUDIMENTS)
has_diddles = df["rudiment_slug"].isin(DIDDLE_RUDIMENTS)
```

## Common Filter Combinations

### Clean High-Quality Training Data

```python
# High quality + clean augmentation + confident labels
quality_data = dataset["train"].filter(
    lambda x: x["overall_score"] > 60
    and x["augmentation_preset"] == "clean_studio"
    and x["tier_confidence"] > 0.6
)
```

### Beginner-Focused Subset

```python
# Beginner samples at slow tempos
beginner_practice = dataset["train"].filter(
    lambda x: x["skill_tier"] == "beginner"
    and x["tempo_bpm"] < 100
)
```

### Specific Rudiment Study

```python
# All paradiddles across skill levels
paradiddle_study = dataset["train"].filter(
    lambda x: "paradiddle" in x["rudiment_slug"]
    and x["diddle_quality"] is not None
)
```

### Cross-Soundfont Generalization

```python
# Train on some soundfonts, test on others
train_soundfonts = ["generalu", "marching", "mtpowerd"]
test_soundfonts = ["douglasn", "fluidr3"]

train_data = dataset["train"].filter(
    lambda x: x["aug_soundfont"] in train_soundfonts
)
test_data = dataset["test"].filter(
    lambda x: x["aug_soundfont"] in test_soundfonts
)
```

## Data Export

### Export Filtered Data

```python
# Export to CSV
filtered_df = df[df["skill_tier"] == "professional"]
filtered_df.to_csv("professional_samples.csv", index=False)

# Export to Parquet (recommended for large datasets)
filtered_df.to_parquet("professional_samples.parquet")

# Export sample IDs for reproducibility
with open("filtered_ids.txt", "w") as f:
    for sample_id in filtered_df["sample_id"]:
        f.write(f"{sample_id}\n")
```

### Creating Subsets

```python
from datasets import Dataset

# Create a HuggingFace dataset from filtered data
subset_df = df[df["skill_tier"].isin(["advanced", "professional"])]
subset_dataset = Dataset.from_pandas(subset_df)

# Push to Hub (optional)
subset_dataset.push_to_hub("username/sousa-advanced-only")
```
