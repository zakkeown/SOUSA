#!/usr/bin/env python3
"""
SOUSA: Synthetic Open Unified Snare Assessment
==============================================

Generate the full synthetic drum rudiment dataset for ML training.

Usage:
    python scripts/generate_dataset.py                    # Full 100K dataset (MIDI + labels)
    python scripts/generate_dataset.py --preset small     # Quick test (~1,200 samples)
    python scripts/generate_dataset.py --preset medium    # Medium (~12,000 samples)
    python scripts/generate_dataset.py --with-audio       # Include audio (requires soundfont)
    python scripts/generate_dataset.py --output my_dataset

Presets:
    small   - 10 profiles × 3 tempos × 1 aug = ~1,200 samples (quick testing)
    medium  - 50 profiles × 3 tempos × 2 aug = ~12,000 samples (development)
    full    - 100 profiles × 5 tempos × 5 aug = ~100,000 samples (production)
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_gen.pipeline.generate import DatasetGenerator, GenerationConfig
from dataset_gen.profiles.archetypes import generate_profiles_batch, SkillTier
from dataset_gen.pipeline.splits import SplitGenerator, SplitConfig, SplitAssignment
from dataset_gen.rudiments.loader import load_all_rudiments
from dataset_gen.validation.report import generate_report

# Preset configurations for different use cases
PRESETS = {
    "small": {
        "profiles": 10,
        "tempos": 3,
        "augmentations": 1,
        "description": "Quick testing (~1,200 samples)",
    },
    "medium": {
        "profiles": 50,
        "tempos": 3,
        "augmentations": 2,
        "description": "Development (~12,000 samples)",
    },
    "full": {
        "profiles": 100,
        "tempos": 5,
        "augmentations": 5,
        "description": "Production (~100,000 samples)",
    },
}


def worker_generate(
    worker_id: int,
    profile_indices: list[int],
    all_profiles: list,
    rudiments: list,
    config_dict: dict,
    output_dir: Path,
) -> dict:
    """
    Worker function for parallel generation.

    Each worker generates samples for a subset of profiles to its own output directory.

    Args:
        worker_id: Worker identifier
        profile_indices: Indices of profiles this worker should process
        all_profiles: Full list of profiles (worker picks from indices)
        rudiments: List of rudiments
        config_dict: Serializable config parameters
        output_dir: Base output directory (worker appends worker_id)

    Returns:
        Statistics dict with samples generated
    """
    # Re-import in worker process (required for multiprocessing)
    from dataset_gen.pipeline.generate import DatasetGenerator, GenerationConfig

    # Create worker-specific output directory
    worker_dir = output_dir / f"worker_{worker_id}"

    # Reconstruct config with worker-specific settings
    config = GenerationConfig(
        output_dir=worker_dir,
        num_profiles=len(profile_indices),
        tempos_per_rudiment=config_dict["tempos_per_rudiment"],
        augmentations_per_sample=config_dict["augmentations_per_sample"],
        generate_audio=config_dict["generate_audio"],
        soundfont_path=(
            Path(config_dict["soundfont_path"]) if config_dict["soundfont_path"] else None
        ),
        seed=config_dict["seed"] + worker_id,  # Different seed per worker
        verbose=config_dict["verbose"],
        target_duration_sec=config_dict.get("target_duration_sec"),
    )

    # Get this worker's profiles with their global indices
    worker_profiles = [(i, all_profiles[i]) for i in profile_indices]

    # Create a dummy splits assignment (profiles already assigned globally)
    # Worker doesn't need to track splits - we merge later
    generator = DatasetGenerator(config)

    # Pre-populate profile numbers with GLOBAL indices to avoid ID collisions
    for global_idx, profile in worker_profiles:
        generator._profile_numbers[profile.id] = global_idx

    # Create a minimal split assignment for the worker
    dummy_splits = SplitAssignment(
        train_profile_ids=[p.id for _, p in worker_profiles],
        val_profile_ids=[],
        test_profile_ids=[],
    )

    try:
        # Generate samples for this worker's profiles
        for _, profile in worker_profiles:
            generator._generate_profile_samples(profile, rudiments, dummy_splits)

        generator._writer.flush()
        return {
            "worker_id": worker_id,
            "samples": generator.progress.completed_samples,
            "failed": generator.progress.failed_samples,
            "output_dir": str(worker_dir),
        }
    finally:
        generator.close()


def merge_worker_outputs(output_dir: Path, num_workers: int, logger) -> None:
    """
    Merge parquet files from worker directories into main output.

    Args:
        output_dir: Base output directory
        num_workers: Number of workers
        logger: Logger instance
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil

    main_labels_dir = output_dir / "labels"
    main_midi_dir = output_dir / "midi"
    main_audio_dir = output_dir / "audio"

    main_labels_dir.mkdir(parents=True, exist_ok=True)
    main_midi_dir.mkdir(parents=True, exist_ok=True)
    main_audio_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = ["strokes.parquet", "measures.parquet", "exercises.parquet", "samples.parquet"]

    for pq_file in parquet_files:
        dfs = []
        for w in range(num_workers):
            worker_path = output_dir / f"worker_{w}" / "labels" / pq_file
            if worker_path.exists():
                dfs.append(pd.read_parquet(worker_path))

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            output_path = main_labels_dir / pq_file
            table = pa.Table.from_pandas(combined)
            pq.write_table(table, str(output_path), compression="snappy")
            logger.info(f"Merged {pq_file}: {len(combined)} records")

    # Move MIDI and audio files
    for w in range(num_workers):
        worker_dir = output_dir / f"worker_{w}"

        # Move MIDI files
        worker_midi = worker_dir / "midi"
        if worker_midi.exists():
            for midi_file in worker_midi.glob("*.mid"):
                shutil.move(str(midi_file), str(main_midi_dir / midi_file.name))

        # Move audio files
        worker_audio = worker_dir / "audio"
        if worker_audio.exists():
            for audio_file in worker_audio.glob("*"):
                if audio_file.is_file():
                    shutil.move(str(audio_file), str(main_audio_dir / audio_file.name))

        # Clean up worker directory
        shutil.rmtree(worker_dir, ignore_errors=True)

    logger.info("Worker outputs merged successfully")


def main():
    parser = argparse.ArgumentParser(
        description="SOUSA: Generate synthetic drum rudiment dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  small   Quick testing (~1,200 samples)
  medium  Development (~12,000 samples)
  full    Production (~100,000 samples)

Examples:
  python scripts/generate_dataset.py --preset small       # Quick test
  python scripts/generate_dataset.py                      # Full 100K dataset
  python scripts/generate_dataset.py --with-audio         # With audio synthesis
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output/dataset"),
        help="Output directory (default: output/dataset)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="Use a preset configuration (small/medium/full)",
    )
    parser.add_argument(
        "--profiles",
        "-p",
        type=int,
        default=None,
        help="Number of player profiles (default: 100, or from preset)",
    )
    parser.add_argument(
        "--tempos",
        "-t",
        type=int,
        default=None,
        help="Tempos per rudiment (default: 5, or from preset)",
    )
    parser.add_argument(
        "--augmentations",
        "-a",
        type=int,
        default=None,
        help="Augmented versions per sample (default: 5, or from preset)",
    )
    parser.add_argument(
        "--with-audio",
        action="store_true",
        help="Generate audio files (requires soundfont)",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=None,
        help="Path to soundfont file (.sf2) or directory containing .sf2 files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation report generation",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1, 0 = auto-detect)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--fixed-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Generate samples with fixed duration (e.g., 4.0 for 4 seconds). "
        "Recommended for ML training to prevent models learning note count instead of patterns.",
    )

    args = parser.parse_args()

    # Apply preset if specified, then allow individual overrides
    if args.preset:
        preset = PRESETS[args.preset]
        if args.profiles is None:
            args.profiles = preset["profiles"]
        if args.tempos is None:
            args.tempos = preset["tempos"]
        if args.augmentations is None:
            args.augmentations = preset["augmentations"]
    else:
        # Default to full preset values if no preset specified
        if args.profiles is None:
            args.profiles = PRESETS["full"]["profiles"]
        if args.tempos is None:
            args.tempos = PRESETS["full"]["tempos"]
        if args.augmentations is None:
            args.augmentations = PRESETS["full"]["augmentations"]

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load rudiments
    rudiment_dir = Path(__file__).parent.parent / "dataset_gen" / "rudiments" / "definitions"
    rudiments = list(load_all_rudiments(rudiment_dir).values())
    logger.info(f"Loaded {len(rudiments)} rudiments")

    # Calculate expected samples and storage estimates
    total_samples = args.profiles * len(rudiments) * args.tempos * args.augmentations

    # Estimate storage (rough estimates based on typical sizes)
    midi_size_kb = 2  # ~2KB per MIDI file
    label_overhead_kb = 0.5  # ~0.5KB per sample in parquet (compressed)
    audio_size_mb = 1.5  # ~1.5MB per FLAC file (8 seconds @ 44.1kHz)

    midi_total_mb = (total_samples * midi_size_kb) / 1024
    labels_total_mb = (total_samples * label_overhead_kb) / 1024
    audio_total_gb = (total_samples * audio_size_mb) / 1024 if args.with_audio else 0

    print("\n" + "=" * 60)
    print("DATASET GENERATION PLAN")
    print("=" * 60)
    if args.preset:
        print(f"Preset: {args.preset} - {PRESETS[args.preset]['description']}")
    print("\nConfiguration:")
    print(f"  {args.profiles} player profiles")
    print(f"  {len(rudiments)} rudiments")
    print(f"  {args.tempos} tempos per rudiment")
    print(f"  {args.augmentations} augmentations per sample")
    if args.fixed_duration:
        print(f"  Fixed duration: {args.fixed_duration}s per sample (ML-optimized)")
    else:
        print("  Variable duration: 4 cycles per sample (original behavior)")
    print(f"\nTotal samples: {total_samples:,}")
    print("\nEstimated storage:")
    print(f"  MIDI files:  ~{midi_total_mb:.1f} MB")
    print(f"  Labels:      ~{labels_total_mb:.1f} MB")
    if args.with_audio:
        print(f"  Audio files: ~{audio_total_gb:.1f} GB")
        print(f"  TOTAL:       ~{audio_total_gb + (midi_total_mb + labels_total_mb) / 1024:.1f} GB")
    else:
        print(f"  TOTAL:       ~{midi_total_mb + labels_total_mb:.1f} MB (no audio)")
    print("=" * 60 + "\n")

    # Validate audio settings
    if args.with_audio and not args.soundfont:
        # Check for default soundfont directory
        default_sf = Path(__file__).parent.parent / "data" / "soundfonts"
        sf_files = sorted(default_sf.glob("*.sf2")) if default_sf.exists() else []
        if sf_files:
            # Use the directory so all soundfonts are loaded
            args.soundfont = default_sf
            print(f"\nSoundfonts found ({len(sf_files)}):")
            for sf in sf_files:
                size_mb = sf.stat().st_size / (1024 * 1024)
                print(f"  {sf.name} ({size_mb:.1f} MB)")
        else:
            logger.error("Audio generation requested but no soundfont found.")
            logger.error("Run: python scripts/setup_soundfonts.py")
            logger.error("Or provide --soundfont path to .sf2 file or directory")
            sys.exit(1)

    # Auto-detect workers if set to 0
    if args.workers == 0:
        args.workers = max(1, mp.cpu_count() - 1)
        print(f"Auto-detected {args.workers} workers")

    start_time = time.time()

    if args.workers > 1:
        # Parallel generation
        print(f"\nUsing {args.workers} parallel workers")

        # Generate profiles upfront (need to share across workers)
        skill_distribution = {
            SkillTier.BEGINNER: 0.25,
            SkillTier.INTERMEDIATE: 0.35,
            SkillTier.ADVANCED: 0.25,
            SkillTier.PROFESSIONAL: 0.15,
        }
        all_profiles = generate_profiles_batch(
            args.profiles,
            skill_distribution=skill_distribution,
            seed=args.seed,
        )

        # Generate splits
        split_config = SplitConfig(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
        )
        split_generator = SplitGenerator(split_config)
        splits = split_generator.generate_splits(all_profiles)
        split_generator.save_splits(splits, args.output / "splits.json")

        # Partition profiles across workers
        profile_indices = list(range(len(all_profiles)))
        chunks = [profile_indices[i :: args.workers] for i in range(args.workers)]

        # Serialize config for workers
        config_dict = {
            "tempos_per_rudiment": args.tempos,
            "augmentations_per_sample": args.augmentations,
            "generate_audio": args.with_audio,
            "soundfont_path": str(args.soundfont) if args.soundfont else None,
            "seed": args.seed,
            "verbose": not args.quiet,
            "target_duration_sec": args.fixed_duration,
        }

        # Create worker tasks
        worker_args = [
            (worker_id, chunks[worker_id], all_profiles, rudiments, config_dict, args.output)
            for worker_id in range(args.workers)
        ]

        # Run workers in parallel
        logger.info(f"Starting {args.workers} worker processes...")
        with mp.Pool(processes=args.workers) as pool:
            results = pool.starmap(worker_generate, worker_args)

        # Report worker results
        total_samples = sum(r["samples"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        logger.info(f"Workers completed: {total_samples} samples, {total_failed} failed")

        # Merge worker outputs
        logger.info("Merging worker outputs...")
        merge_worker_outputs(args.output, args.workers, logger)

        generation_time = time.time() - start_time
        logger.info(f"Generation complete in {generation_time / 60:.1f} minutes")

    else:
        # Sequential generation (original path)
        config = GenerationConfig(
            output_dir=args.output,
            num_profiles=args.profiles,
            tempos_per_rudiment=args.tempos,
            augmentations_per_sample=args.augmentations,
            generate_audio=args.with_audio,
            soundfont_path=args.soundfont,
            seed=args.seed,
            verbose=not args.quiet,
            target_duration_sec=args.fixed_duration,
        )

        logger.info(f"Generating dataset to {args.output}")
        generator = DatasetGenerator(config)

        try:
            splits = generator.generate(rudiments=rudiments)
            generation_time = time.time() - start_time
            logger.info(f"Generation complete in {generation_time / 60:.1f} minutes")
        finally:
            generator.close()

    # Report split statistics
    print("\nDataset splits:")
    print(f"  train: {len(splits.train_profile_ids)} profiles")
    print(f"  val: {len(splits.val_profile_ids)} profiles")
    print(f"  test: {len(splits.test_profile_ids)} profiles")

    # Run validation
    if not args.skip_validation:
        logger.info("Running validation...")
        validation_start = time.time()
        report_path = args.output / "validation_report.json"
        report = generate_report(args.output, report_path)
        validation_time = time.time() - validation_start

        print("\n" + "=" * 60)
        print(report.summary())
        print(f"\nValidation completed in {validation_time:.1f} seconds")

        if not report.verification.all_passed:
            logger.warning("Some validation checks failed - review report")
            sys.exit(1)
    else:
        logger.info("Skipping validation (--skip-validation)")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Dataset saved to: {args.output}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Rate: {total_samples / total_time:.1f} samples/second")
    print("=" * 60)


if __name__ == "__main__":
    main()
