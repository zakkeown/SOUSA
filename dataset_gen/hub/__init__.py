"""
HuggingFace Hub integration for SOUSA dataset.

This module provides utilities to:
- Build HuggingFace DatasetDict objects with embedded media
- Upload Parquet-native datasets to HuggingFace Hub
- Manage multi-config datasets (audio, midi_only, labels_only)
"""

from dataset_gen.hub.uploader import (
    HubConfig,
    DatasetUploader,
    push_to_hub,
)

__all__ = [
    "HubConfig",
    "DatasetUploader",
    "push_to_hub",
]
