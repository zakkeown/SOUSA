#!/usr/bin/env python3
"""Generate a large clean dataset for ML training.

Fixed 120 BPM, no augmentation, single soundfont (FluidR3_GM_GS).
This produces the cleanest possible training data.

Usage:
    python scripts/generate_clean_dataset.py --profiles 250  # 10K samples
    python scripts/generate_clean_dataset.py --profiles 500  # 20K samples
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_gen.profiles.archetypes import generate_profiles_batch, SkillTier
from dataset_gen.pipeline.splits import SplitGenerator, SplitConfig, SplitAssignment
from dataset_gen.rudiments.loader import load_all_rudiments


def worker_generate(worker_id, profile_indices, all_profiles, rudiments, config_dict, output_dir):
    """Worker function for parallel generation."""
    import traceback
    from dataset_gen.pipeline.generate import GenerationConfig, DatasetGenerator

    worker_dir = output_dir / f"worker_{worker_id}"

    config = GenerationConfig(
        output_dir=worker_dir,
        num_profiles=len(profile_indices),
        tempos_per_rudiment=config_dict["tempos_per_rudiment"],
        augmentations_per_sample=config_dict["augmentations_per_sample"],
        generate_audio=config_dict["generate_audio"],
        soundfont_path=(
            Path(config_dict["soundfont_path"]) if config_dict["soundfont_path"] else None
        ),
        seed=config_dict["seed"] + worker_id,
        verbose=config_dict["verbose"],
        target_duration_sec=config_dict.get("target_duration_sec"),
    )

    # Override tempo range and augmentation
    config.tempo_range = tuple(config_dict["tempo_range"])
    config.apply_augmentation = config_dict["apply_augmentation"]

    worker_profiles = [(i, all_profiles[i]) for i in profile_indices]

    generator = None
    try:
        generator = DatasetGenerator(config)

        for global_idx, profile in worker_profiles:
            generator._profile_numbers[profile.id] = global_idx

        dummy_splits = SplitAssignment(
            train_profile_ids=[p.id for _, p in worker_profiles],
            val_profile_ids=[],
            test_profile_ids=[],
        )

        for _, profile in worker_profiles:
            generator._generate_profile_samples(profile, rudiments, dummy_splits)

        generator._writer.flush()
        return {
            "worker_id": worker_id,
            "samples": generator.progress.completed_samples,
            "failed": generator.progress.failed_samples,
            "output_dir": str(worker_dir),
            "error": None,
        }
    except Exception as e:
        return {
            "worker_id": worker_id,
            "samples": generator.progress.completed_samples if generator else 0,
            "failed": generator.progress.failed_samples if generator else len(profile_indices),
            "output_dir": str(worker_dir),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
    finally:
        if generator:
            generator.close()


def merge_worker_outputs(output_dir, num_workers, logger):
    """Merge parquet files from worker directories."""
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
                try:
                    dfs.append(pd.read_parquet(worker_path))
                except Exception as e:
                    logger.warning(f"Could not read {worker_path}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            table = pa.Table.from_pandas(combined)
            pq.write_table(table, str(main_labels_dir / pq_file), compression="snappy")
            logger.info(f"Merged {pq_file}: {len(combined)} records")

    for w in range(num_workers):
        worker_dir = output_dir / f"worker_{w}"
        worker_midi = worker_dir / "midi"
        if worker_midi.exists():
            for midi_file in worker_midi.glob("*.mid"):
                shutil.move(str(midi_file), str(main_midi_dir / midi_file.name))
        worker_audio = worker_dir / "audio"
        if worker_audio.exists():
            for audio_file in worker_audio.glob("*"):
                if audio_file.is_file():
                    shutil.move(str(audio_file), str(main_audio_dir / audio_file.name))
        shutil.rmtree(worker_dir, ignore_errors=True)

    logger.info("Worker outputs merged successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean dataset (fixed 120 BPM, no augmentation)"
    )
    parser.add_argument(
        "--profiles",
        "-p",
        type=int,
        default=250,
        help="Number of profiles (default: 250 = 10K samples)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("output/clean_10k"), help="Output directory"
    )
    parser.add_argument("--workers", "-w", type=int, default=0, help="Workers (0=auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-duration", type=float, default=4.0, help="Fixed duration in seconds"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    rudiment_dir = Path(__file__).parent.parent / "dataset_gen" / "rudiments" / "definitions"
    rudiments = list(load_all_rudiments(rudiment_dir).values())

    # Single soundfont only
    soundfont_path = Path(__file__).parent.parent / "data" / "soundfonts" / "FluidR3_GM_GS.sf2"
    if not soundfont_path.exists():
        logger.error(f"Soundfont not found: {soundfont_path}")
        sys.exit(1)

    total_samples = args.profiles * len(rudiments) * 1 * 1  # 1 tempo, 1 aug
    print(f"\nGenerating {total_samples:,} clean samples")
    print(
        f"  {args.profiles} profiles × {len(rudiments)} rudiments × 1 tempo (120 BPM) × 1 (no augmentation)"
    )
    print(f"  Fixed duration: {args.fixed_duration}s, Soundfont: FluidR3_GM_GS")

    if args.workers == 0:
        args.workers = max(1, mp.cpu_count() - 1)
    print(f"  Workers: {args.workers}\n")

    start_time = time.time()

    # Generate profiles
    skill_distribution = {
        SkillTier.BEGINNER: 0.25,
        SkillTier.INTERMEDIATE: 0.35,
        SkillTier.ADVANCED: 0.25,
        SkillTier.PROFESSIONAL: 0.15,
    }
    all_profiles = generate_profiles_batch(
        args.profiles, skill_distribution=skill_distribution, seed=args.seed
    )

    # Generate splits
    split_config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    split_generator = SplitGenerator(split_config)
    splits = split_generator.generate_splits(all_profiles)
    args.output.mkdir(parents=True, exist_ok=True)
    split_generator.save_splits(splits, args.output / "splits.json")

    # Config for workers
    config_dict = {
        "tempos_per_rudiment": 1,
        "augmentations_per_sample": 1,
        "generate_audio": True,
        "soundfont_path": str(soundfont_path),
        "seed": args.seed,
        "verbose": False,
        "target_duration_sec": args.fixed_duration,
        "tempo_range": (120, 120),
        "apply_augmentation": False,
    }

    profile_indices = list(range(len(all_profiles)))
    chunks = [profile_indices[i :: args.workers] for i in range(args.workers)]

    worker_args = [
        (wid, chunks[wid], all_profiles, rudiments, config_dict, args.output)
        for wid in range(args.workers)
    ]

    logger.info(f"Starting {args.workers} worker processes...")
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(worker_generate, worker_args)

    total_generated = sum(r["samples"] for r in results)
    total_failed = sum(r["failed"] for r in results)
    failed_workers = [r for r in results if r.get("error")]

    if failed_workers:
        for r in failed_workers:
            logger.warning(f"Worker {r['worker_id']} error: {r['error'].split(chr(10))[0]}")

    logger.info(f"Generated: {total_generated} samples, {total_failed} failed")

    logger.info("Merging worker outputs...")
    merge_worker_outputs(args.output, args.workers, logger)

    # Create metadata.csv from parquet
    logger.info("Creating metadata.csv...")
    import pandas as pd
    import json

    samples_pq = args.output / "labels" / "samples.parquet"
    splits_json = args.output / "splits.json"

    df = pd.read_parquet(samples_pq)
    with open(splits_json) as f:
        split_data = json.load(f)

    split_map = {}
    for split_name in ["train", "val", "test"]:
        for pid in split_data.get(f"{split_name}_profile_ids", []):
            split_map[pid] = split_name

    df["split"] = df["profile_id"].map(split_map)
    df.to_csv(args.output / "metadata.csv", index=False)

    gen_time = time.time() - start_time
    print(f"\nDone! {total_generated:,} samples in {gen_time / 60:.1f} minutes")
    print(f"Saved to: {args.output}")

    # Quick validation
    print("\nSplit distribution:")
    print(df["split"].value_counts().to_string())
    print(f"\nClasses: {df['rudiment_slug'].nunique()}")
    print(f"Samples per class: {df['rudiment_slug'].value_counts().mean():.0f}")
    print(f"Tempo: {df['tempo_bpm'].unique()}")


if __name__ == "__main__":
    main()
