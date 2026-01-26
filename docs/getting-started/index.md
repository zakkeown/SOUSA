# Getting Started

Welcome to SOUSA! This section will help you get up and running with the synthetic drum rudiment dataset generator.

## Overview

SOUSA (Synthetic Open Unified Snare Assessment) generates synthetic drum rudiment datasets for machine learning training. You can either:

1. **Use the pre-generated dataset** from HuggingFace Hub (easiest)
2. **Generate your own dataset** locally with custom configurations

## Quick Navigation

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Set up Python, install SOUSA, and configure FluidSynth for audio synthesis.

    [:octicons-arrow-right-24: Installation](installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Generate a test dataset and explore the samples in 5 minutes.

    [:octicons-arrow-right-24: Quick Start](quickstart.md)

-   :material-database:{ .lg .middle } **Loading Data**

    ---

    Load the dataset from HuggingFace Hub or local files.

    [:octicons-arrow-right-24: Loading Data](loading.md)

</div>

## Recommended Path

### For ML Practitioners

If you want to use SOUSA for training models, the fastest path is:

```python
from datasets import load_dataset

# Load from HuggingFace (no local installation needed)
dataset = load_dataset("zkeown/sousa")
```

See the [Loading Data](loading.md) guide for more options.

### For Dataset Customization

If you want to generate custom datasets with different configurations:

1. Follow the [Installation](installation.md) guide
2. Work through the [Quick Start](quickstart.md) tutorial
3. Explore the [User Guide](../user-guide/index.md) for advanced options

## Requirements at a Glance

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11+ |
| RAM | 4 GB | 8+ GB |
| Disk (small preset) | 1 GB | 2 GB |
| Disk (full preset) | 100 GB | 150 GB |
| FluidSynth | Required for audio | - |

## What You'll Learn

By the end of this section, you will be able to:

- Install SOUSA and its dependencies
- Generate a test dataset with MIDI and audio
- Load and explore samples programmatically
- Filter samples by skill level, rudiment, or score
- Prepare data for ML training pipelines

## Need Help?

- Check the [User Guide](../user-guide/index.md) for detailed examples
- See [Limitations](../limitations.md) for known constraints
- Review [Validation](../user-guide/validation.md) for quality assurance details
- Open an [issue on GitHub](https://github.com/zakkeown/SOUSA/issues) for bugs or questions
