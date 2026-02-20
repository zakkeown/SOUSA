#!/usr/bin/env python3
"""
Push SOUSA dataset to HuggingFace Hub as Parquet shards.
========================================================

Upload the generated SOUSA dataset with embedded audio/MIDI in Parquet format.

Usage:
    python scripts/push_to_hub.py zkeown/sousa                       # Upload all configs
    python scripts/push_to_hub.py zkeown/sousa --dry-run              # Test without upload
    python scripts/push_to_hub.py zkeown/sousa --configs labels_only  # Labels only (~50MB)
    python scripts/push_to_hub.py zkeown/sousa --purge                # Clear repo first

Prerequisites:
    1. Install hub dependencies: pip install 'sousa[hub]'
    2. Login to HuggingFace: huggingface-cli login
    3. Generate dataset: python scripts/generate_dataset.py
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_gen.hub import HubConfig, DatasetUploader

VALID_CONFIGS = ["audio", "midi_only", "labels_only"]


def main():
    parser = argparse.ArgumentParser(
        description="Push SOUSA dataset to HuggingFace Hub as Parquet shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/push_to_hub.py zkeown/sousa                        # All configs
  python scripts/push_to_hub.py zkeown/sousa --configs labels_only   # Labels only
  python scripts/push_to_hub.py zkeown/sousa --configs midi_only audio  # MIDI + audio
  python scripts/push_to_hub.py zkeown/sousa --purge --yes           # Purge without prompt
  python scripts/push_to_hub.py zkeown/sousa --dry-run               # Test run

Configs:
  audio        Full dataset with FLAC audio + MIDI bytes + labels (~96GB)
  midi_only    MIDI bytes + labels, no audio (~2.5GB)
  labels_only  Pure tabular metadata + scores (~50MB)

Prerequisites:
  pip install 'sousa[hub]'
  huggingface-cli login
        """,
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repository ID (e.g., zkeown/sousa)",
    )
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        default=Path("output/dataset"),
        help="Path to generated dataset (default: output/dataset)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=VALID_CONFIGS,
        default=None,
        help="Which configs to upload (default: all three)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="1GB",
        help="Max Parquet shard size (default: 1GB)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete all existing files from the repo before upload",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--token",
        "-t",
        type=str,
        default=None,
        help="HuggingFace API token (uses cached token if not provided)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build DatasetDicts but don't upload (for testing)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate dataset directory
    if not args.dataset_dir.exists():
        logger.error(f"Dataset directory not found: {args.dataset_dir}")
        logger.error("Run 'python scripts/generate_dataset.py' first")
        sys.exit(1)

    labels_dir = args.dataset_dir / "labels"
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        sys.exit(1)

    # Check dependencies
    try:
        import huggingface_hub  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        logger.error("Required packages not installed")
        logger.error("Install with: pip install 'sousa[hub]'")
        sys.exit(1)

    configs = args.configs or VALID_CONFIGS

    # Print plan
    print("\n" + "=" * 60)
    print("HUGGINGFACE HUB UPLOAD (Parquet)")
    print("=" * 60)
    print(f"\nRepository: {args.repo_id}")
    print(f"Visibility: {'Private' if args.private else 'Public'}")
    print(f"Source:     {args.dataset_dir}")
    print(f"Configs:    {', '.join(configs)}")
    print(f"Shard size: {args.max_shard_size}")
    if args.purge:
        print("Purge:      YES - will delete all existing files first")
    if args.dry_run:
        print("Mode:       DRY RUN (no actual upload)")
    print("=" * 60 + "\n")

    # Confirm purge
    if args.purge and not args.dry_run and not args.yes:
        response = input(f"This will DELETE all files from {args.repo_id}. Continue? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    # Configure and upload
    config = HubConfig(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        configs=configs,
        max_shard_size=args.max_shard_size,
    )

    uploader = DatasetUploader(config)

    try:
        url = uploader.upload(dry_run=args.dry_run, purge=args.purge)

        if args.dry_run:
            print("\nDRY RUN complete. No files were uploaded.")
            print("Remove --dry-run to upload for real.")
        else:
            print("\n" + "=" * 60)
            print("UPLOAD COMPLETE")
            print("=" * 60)
            print(f"\nDataset URL: {url}")
            print("\nLoad the dataset:")
            print(f'  ds = load_dataset("{args.repo_id}")               # audio (default)')
            print(f'  ds = load_dataset("{args.repo_id}", "midi_only")  # MIDI + labels')
            print(f'  ds = load_dataset("{args.repo_id}", "labels_only") # labels only')
            print("=" * 60)

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
