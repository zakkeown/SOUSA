# Examples

This section provides practical examples for working with the SOUSA dataset in machine learning workflows. Each example includes complete, runnable code with detailed explanations.

## Overview

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| [PyTorch DataLoader](pytorch-dataloader.md) | Load SOUSA into PyTorch training pipelines | `Dataset`, `DataLoader`, collate functions, batching |
| [Feature Extraction](feature-extraction.md) | Extract ML features from audio | Mel spectrograms, MFCCs, Wav2Vec2, caching |
| [Filtering Samples](filtering.md) | Filter dataset by various criteria | Skill tiers, rudiments, scores, augmentation |
| [Hierarchical Labels](hierarchical-labels.md) | Work with stroke/measure/exercise labels | Multi-level targets, sequence modeling |

## Prerequisites

All examples assume you have:

1. **Installed SOUSA** with development dependencies:

    ```bash
    pip install -e ".[dev]"
    ```

2. **Generated a dataset** (at minimum, the small preset):

    ```bash
    python scripts/generate_dataset.py --preset small --with-audio
    ```

3. **Optional ML dependencies** for specific examples:

    === "PyTorch"

        ```bash
        pip install torch torchaudio
        ```

    === "Feature Extraction"

        ```bash
        pip install librosa transformers
        ```

## Running the Examples

Each example can be run directly from the command line:

```bash
# PyTorch DataLoader example
python -m examples.pytorch_dataloader output/dataset

# Feature extraction example
python -m examples.feature_extraction

# Filtering example
python -m examples.filtering output/dataset

# Hierarchical labels example
python -m examples.hierarchical_labels output/dataset
```

## Quick Start

Here's a minimal example to get you started:

```python
import pandas as pd
from pathlib import Path

# Load dataset metadata
data_dir = Path("output/dataset")
samples = pd.read_parquet(data_dir / "labels" / "samples.parquet")
exercises = pd.read_parquet(data_dir / "labels" / "exercises.parquet")

# Merge for easy access
data = samples.merge(exercises, on="sample_id")

# Filter to advanced players playing paradiddles
subset = data[
    (data["skill_tier"] == "advanced") &
    (data["rudiment_slug"].str.contains("paradiddle"))
]

print(f"Found {len(subset)} samples")
print(f"Mean overall score: {subset['overall_score'].mean():.1f}")
```

## Source Code

All example source files are available in the [`examples/`](https://github.com/zakkeown/SOUSA/tree/main/examples) directory:

- `pytorch_dataloader.py` - Complete PyTorch integration
- `feature_extraction.py` - Feature extraction utilities
- `filtering.py` - Filtering functions and examples
- `hierarchical_labels.py` - Hierarchical label access

## Contributing Examples

Have an interesting use case? We welcome example contributions! See the [Contributing Guide](../contributing/index.md) for how to submit new examples.

!!! tip "Example Requirements"
    Good examples should:

    - Be self-contained and runnable
    - Include docstrings and comments
    - Handle missing optional dependencies gracefully
    - Work with the `--preset small` dataset for quick testing
