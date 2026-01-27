"""
HuggingFace Hub integration for SOUSA dataset.

This module provides utilities to:
- Prepare the dataset for HuggingFace Hub format
- Upload the dataset to HuggingFace Hub
- Generate consolidated parquet files for efficient loading
- Create sharded TAR archives for large media files
"""

from dataset_gen.hub.archiver import (
    ShardInfo,
    create_sharded_archives,
)
from dataset_gen.hub.uploader import (
    HubConfig,
    DatasetUploader,
    prepare_hf_structure,
    push_to_hub,
)

__all__ = [
    # Archiver
    "ShardInfo",
    "create_sharded_archives",
    # Uploader
    "HubConfig",
    "DatasetUploader",
    "prepare_hf_structure",
    "push_to_hub",
]
