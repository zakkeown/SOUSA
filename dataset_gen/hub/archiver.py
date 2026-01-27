"""
TAR archive creation for HuggingFace Hub uploads.

Creates sharded TAR archives from audio/MIDI files, grouped by split,
to reduce total file count for HuggingFace's 100k file limit.
"""

from __future__ import annotations

import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ShardInfo:
    """Information about which shard contains a file."""

    shard_name: str  # e.g., "train-00001.tar"
    filename: str  # e.g., "sample_001.flac"


def create_sharded_archives(
    source_dir: Path,
    output_dir: Path,
    filenames_by_split: dict[str, list[str]],
    target_shard_size_bytes: int = 1_000_000_000,  # 1GB
    extension: str = "flac",
) -> dict[str, ShardInfo]:
    """
    Create TAR archives from source files, sharded by size.

    Args:
        source_dir: Directory containing source files
        output_dir: Directory to write TAR archives
        filenames_by_split: Mapping of split name to list of filenames
        target_shard_size_bytes: Target size per shard (default 1GB)
        extension: File extension being processed (for logging)

    Returns:
        Mapping of original filename to ShardInfo(shard_name, filename)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, ShardInfo] = {}

    for split_name, filenames in filenames_by_split.items():
        shard_index = 0
        current_shard_size = 0
        current_shard_path = output_dir / f"{split_name}-{shard_index:05d}.tar"
        current_tar = tarfile.open(current_shard_path, "w")

        for filename in sorted(filenames):  # Sort for determinism
            src_file = source_dir / filename
            if not src_file.exists():
                logger.warning(f"Source file not found: {src_file}")
                continue

            file_size = src_file.stat().st_size

            # Check if we need a new shard (but always put at least one file per shard)
            if current_shard_size > 0 and current_shard_size + file_size > target_shard_size_bytes:
                current_tar.close()
                logger.info(
                    f"Closed {current_shard_path.name} with {current_shard_size / (1024**2):.1f} MB"
                )
                shard_index += 1
                current_shard_size = 0
                current_shard_path = output_dir / f"{split_name}-{shard_index:05d}.tar"
                current_tar = tarfile.open(current_shard_path, "w")

            # Add file to current shard
            current_tar.add(src_file, arcname=filename)
            current_shard_size += file_size

            shard_name = f"{split_name}-{shard_index:05d}.tar"
            result[filename] = ShardInfo(shard_name=shard_name, filename=filename)

        # Close final shard
        current_tar.close()
        if current_shard_size > 0:
            logger.info(
                f"Closed {current_shard_path.name} with {current_shard_size / (1024**2):.1f} MB"
            )

    return result
