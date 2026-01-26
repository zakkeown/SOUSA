#!/usr/bin/env python3
"""
Push SOUSA dataset to HuggingFace Hub.
======================================

Upload the generated SOUSA dataset to HuggingFace Hub for public distribution.

Usage:
    python scripts/push_to_hub.py username/sousa                    # Upload to Hub
    python scripts/push_to_hub.py username/sousa --dry-run          # Test without upload
    python scripts/push_to_hub.py username/sousa --no-audio         # Labels + MIDI only
    python scripts/push_to_hub.py username/sousa --private          # Private repository

Prerequisites:
    1. Install hub dependencies: pip install 'rudimentary[hub]'
    2. Login to HuggingFace: huggingface-cli login
    3. Generate dataset: python scripts/generate_dataset.py

The script will:
    1. Prepare the dataset in HuggingFace-compatible format
    2. Create consolidated parquet files for each split
    3. Copy audio/MIDI files
    4. Upload everything to HuggingFace Hub
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_gen.hub import HubConfig, DatasetUploader


def main():
    parser = argparse.ArgumentParser(
        description="Push SOUSA dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/push_to_hub.py myuser/sousa              # Public upload
  python scripts/push_to_hub.py myuser/sousa --private    # Private repo
  python scripts/push_to_hub.py myuser/sousa --dry-run    # Test run
  python scripts/push_to_hub.py myuser/sousa --no-audio   # Skip audio files

Prerequisites:
  pip install 'rudimentary[hub]'
  huggingface-cli login
        """,
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repository ID (e.g., username/sousa)",
    )
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=Path,
        default=Path("output/dataset"),
        help="Path to generated dataset (default: output/dataset)",
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
        "--no-audio",
        action="store_true",
        help="Skip audio files (upload labels and MIDI only)",
    )
    parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Skip MIDI files (upload labels and audio only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare files but don't upload (for testing)",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Custom staging directory for HF format",
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
        logger.error("Dataset appears incomplete")
        sys.exit(1)

    # Check for dataset card
    readme = args.dataset_dir / "README.md"
    if not readme.exists():
        logger.warning("No README.md (dataset card) found in dataset directory")
        logger.warning("Consider creating one for better discoverability on HuggingFace")

    # Check for HuggingFace dependencies
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        logger.error("huggingface_hub not installed")
        logger.error("Install with: pip install 'rudimentary[hub]'")
        sys.exit(1)

    # Configure upload
    config = HubConfig(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        include_audio=not args.no_audio,
        include_midi=not args.no_midi,
        staging_dir=args.staging_dir,
    )

    # Print plan
    print("\n" + "=" * 60)
    print("HUGGINGFACE HUB UPLOAD")
    print("=" * 60)
    print(f"\nRepository: {config.repo_id}")
    print(f"Visibility: {'Private' if config.private else 'Public'}")
    print(f"\nSource: {config.dataset_dir}")
    print("\nContent:")
    print("  Labels (parquet): Yes")
    print(f"  Audio files:      {'Yes' if config.include_audio else 'No'}")
    print(f"  MIDI files:       {'Yes' if config.include_midi else 'No'}")
    if args.dry_run:
        print("\nMode: DRY RUN (no actual upload)")
    print("=" * 60 + "\n")

    # Execute upload
    uploader = DatasetUploader(config)

    try:
        # Prepare staging directory
        logger.info("Preparing HuggingFace format...")
        if args.dry_run:
            # Skip media copy for dry-run to avoid duplicating 96GB of files
            staging_dir = uploader.prepare(skip_media_copy=True)
        else:
            # Use symlinks to avoid duplicating large files
            staging_dir = uploader.prepare(skip_media_copy=False, use_symlinks=True)
        print(f"\nStaging directory: {staging_dir}")
        print(uploader.stats.summary())

        if args.dry_run:
            print(f"\nDRY RUN complete. Parquet files prepared in: {staging_dir}")
            print("Audio/MIDI files were counted but not copied.")
            print("To upload for real, remove --dry-run flag")
            return

        # Upload
        logger.info("Uploading to HuggingFace Hub...")
        url = uploader.upload(dry_run=False)

        print("\n" + "=" * 60)
        print("UPLOAD COMPLETE")
        print("=" * 60)
        print(f"\nDataset URL: {url}")
        print("\nYou can now load the dataset with:")
        print("  from datasets import load_dataset")
        print(f'  dataset = load_dataset("{args.repo_id}")')
        print("=" * 60)

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
