"""
HuggingFace Hub integration for SOUSA dataset.

This module provides utilities to:
- Prepare the dataset for HuggingFace Hub format
- Upload the dataset to HuggingFace Hub
- Generate consolidated parquet files for efficient loading
- Organize media files into rudiment subdirectories
"""

from dataset_gen.hub.uploader import (
    HubConfig,
    DatasetUploader,
    prepare_hf_structure,
    push_to_hub,
)

__all__ = [
    "HubConfig",
    "DatasetUploader",
    "prepare_hf_structure",
    "push_to_hub",
]
