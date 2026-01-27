"""
TAR archive creation for HuggingFace Hub uploads.

Creates sharded TAR archives from audio/MIDI files, grouped by split,
to reduce total file count for HuggingFace's 100k file limit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShardInfo:
    """Information about which shard contains a file."""

    shard_name: str  # e.g., "train-00001.tar"
    filename: str  # e.g., "sample_001.flac"
