# User Guide

This guide covers everything you need to use the SOUSA dataset effectively, from generation to ML training.

## Quick Start

```bash
# Install SOUSA
pip install -e .

# Generate a small test dataset
python scripts/generate_dataset.py --preset small

# Generate with audio (requires soundfonts)
python scripts/setup_soundfonts.py
python scripts/generate_dataset.py --preset medium --with-audio
```

## Guide Sections

### [Dataset Generation](generation.md)

Learn how to generate SOUSA datasets with different configurations:

- **Generation presets** - small, medium, and full configurations with storage estimates
- **Command line options** - customize profiles, tempos, augmentations, and more
- **Reproducibility** - seeding for deterministic generation
- **Parallel generation** - multi-worker support for faster generation
- **Checkpoints** - resumable generation for large datasets

### [Audio Augmentation](augmentation.md)

Understand the audio augmentation pipeline:

- **Augmentation presets** - 10+ presets from clean studio to lo-fi
- **Signal chain** - room simulation, mic modeling, compression, degradation
- **Training impact** - how augmentation affects model robustness
- **Custom configurations** - create your own augmentation profiles

### [Filtering and Preprocessing](filtering.md)

Filter and prepare data for your specific use case:

- **Skill tier filtering** - beginner, intermediate, advanced, professional
- **Rudiment selection** - 40 PAS rudiments across 4 categories
- **Score-based filtering** - filter by timing accuracy, overall score, etc.
- **Dataset statistics** - understand your data distribution

### [ML Training Recommendations](ml-training.md)

Best practices for training models on SOUSA:

- **Skill classification** - binary vs 4-class, handling class imbalance
- **Score regression** - overall_score as primary target
- **Multi-task learning** - minimal orthogonal score sets
- **Audio preprocessing** - mel spectrograms, resampling, normalization
- **PyTorch examples** - DataLoader and training code

### [Validation](validation.md)

Validate your generated datasets:

- **Running validation reports** - comprehensive quality checks
- **Data integrity** - 13 verification checks
- **Realism validation** - comparison to literature benchmarks
- **Test suite** - 26 test cases for validation infrastructure

## Dataset Splits

SOUSA uses profile-based splits to prevent data leakage:

| Split | Profiles | Samples (~100K) |
|-------|----------|-----------------|
| Train | 70% | ~70,000 |
| Validation | 15% | ~15,000 |
| Test | 15% | ~15,000 |

!!! info "Why profile-based splits?"
    All samples from a single player profile stay in the same split. This tests generalization to "new players" not seen during training, which is more realistic for deployment scenarios.

## Skill Tier Distribution

| Tier | Proportion | Description |
|------|------------|-------------|
| Beginner | 25% | Learning basics, high timing variance |
| Intermediate | 35% | Developing consistency |
| Advanced | 25% | Strong technique, minor errors |
| Professional | 15% | Near-perfect execution |

## Loading the Dataset

=== "HuggingFace Hub"

    ```python
    from datasets import load_dataset

    # Load full dataset
    dataset = load_dataset("zkeown/sousa")

    # Load specific split
    train = load_dataset("zkeown/sousa", split="train")

    # Stream for memory efficiency
    dataset = load_dataset("zkeown/sousa", streaming=True)
    ```

=== "Local Files"

    ```python
    from datasets import load_dataset

    # From local parquet files
    dataset = load_dataset(
        "parquet",
        data_dir="./output/dataset/hf_staging/data"
    )
    ```

## Related Documentation

- [Data Format Reference](../reference/data-format.md) - Complete schema documentation
- [Limitations](../limitations.md) - Known limitations and biases
- [Validation Guide](validation.md) - Validation infrastructure details
