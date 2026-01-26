"""
Parallel batch processing for dataset generation.

This module provides concurrent sample generation using multiprocessing
to maximize throughput on multi-core machines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import logging
import time

import numpy as np

from dataset_gen.rudiments.schema import Rudiment
from dataset_gen.profiles.archetypes import PlayerProfile
from dataset_gen.midi_gen.generator import MIDIGenerator
from dataset_gen.midi_gen.articulations import ArticulationEngine
from dataset_gen.labels.compute import compute_sample_labels
from dataset_gen.labels.schema import Sample
from dataset_gen.audio_aug.pipeline import AugmentationConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """A single generation job."""

    job_id: str
    profile: PlayerProfile
    rudiment: Rudiment
    tempo: int
    augmentation_idx: int = 0
    augmentation_config: AugmentationConfig | None = None

    def to_dict(self) -> dict:
        """Serialize for multiprocessing."""
        return {
            "job_id": self.job_id,
            "profile_dict": self.profile.to_dict(),
            "rudiment_dict": self.rudiment.model_dump(),
            "tempo": self.tempo,
            "augmentation_idx": self.augmentation_idx,
            "augmentation_config": (
                self.augmentation_config.model_dump() if self.augmentation_config else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchJob":
        """Deserialize from multiprocessing."""
        from dataset_gen.profiles.archetypes import PlayerProfile
        from dataset_gen.rudiments.schema import Rudiment

        profile = PlayerProfile.from_dict(data["profile_dict"])
        rudiment = Rudiment.model_validate(data["rudiment_dict"])
        aug_config = (
            AugmentationConfig.model_validate(data["augmentation_config"])
            if data["augmentation_config"]
            else None
        )

        return cls(
            job_id=data["job_id"],
            profile=profile,
            rudiment=rudiment,
            tempo=data["tempo"],
            augmentation_idx=data["augmentation_idx"],
            augmentation_config=aug_config,
        )


@dataclass
class BatchResult:
    """Result from a batch job."""

    job_id: str
    sample: Sample | None = None
    midi_data: bytes | None = None
    audio_data: np.ndarray | None = None
    error: str | None = None
    duration_sec: float = 0.0

    @property
    def success(self) -> bool:
        return self.sample is not None and self.error is None


@dataclass
class BatchProgress:
    """Track batch processing progress."""

    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_sec(self) -> float:
        return time.time() - self.start_time

    @property
    def jobs_per_sec(self) -> float:
        if self.elapsed_sec > 0:
            return self.completed_jobs / self.elapsed_sec
        return 0.0

    @property
    def eta_sec(self) -> float:
        if self.jobs_per_sec > 0:
            remaining = self.total_jobs - self.completed_jobs
            return remaining / self.jobs_per_sec
        return float("inf")

    @property
    def percent_complete(self) -> float:
        if self.total_jobs > 0:
            return 100 * self.completed_jobs / self.total_jobs
        return 0.0

    def __str__(self) -> str:
        return (
            f"Progress: {self.completed_jobs}/{self.total_jobs} "
            f"({self.percent_complete:.1f}%) - "
            f"{self.jobs_per_sec:.1f} jobs/sec - "
            f"ETA: {self.eta_sec:.0f}s"
        )


def _worker_generate_sample(
    job_dict: dict,
    num_cycles: int,
    seed_offset: int,
) -> dict:
    """
    Worker function for parallel sample generation.

    This runs in a separate process, so it receives serialized data.
    """
    start_time = time.time()

    try:
        job = BatchJob.from_dict(job_dict)

        # Create worker-local generators with unique seeds
        worker_seed = hash(job.job_id) % (2**31) + seed_offset
        midi_gen = MIDIGenerator(seed=worker_seed)
        articulation_engine = ArticulationEngine(seed=worker_seed + 1)

        # Generate MIDI performance
        performance = midi_gen.generate(
            rudiment=job.rudiment,
            profile=job.profile,
            tempo_bpm=job.tempo,
            num_cycles=num_cycles,
            include_midi=True,
        )

        # Apply articulations
        performance.strokes = articulation_engine.process(
            performance.strokes,
            job.rudiment,
            job.profile,
        )

        # Compute labels
        sample = compute_sample_labels(performance, job.rudiment, job.profile)
        sample.sample_id = job.job_id

        duration = time.time() - start_time

        return {
            "job_id": job.job_id,
            "sample_dict": sample.model_dump(),
            "midi_data": performance.midi_data,
            "error": None,
            "duration_sec": duration,
        }

    except Exception as e:
        return {
            "job_id": job_dict.get("job_id", "unknown"),
            "sample_dict": None,
            "midi_data": None,
            "error": str(e),
            "duration_sec": time.time() - start_time,
        }


class BatchGenerator:
    """
    Generate samples in parallel batches.

    Uses multiprocessing for CPU-bound MIDI/label generation,
    with optional threading for I/O-bound audio synthesis.
    """

    def __init__(
        self,
        num_workers: int | None = None,
        num_cycles: int = 4,
        seed: int = 42,
    ):
        """
        Initialize batch generator.

        Args:
            num_workers: Number of parallel workers (default: CPU count - 1)
            num_cycles: Number of rudiment cycles per sample
            seed: Base random seed
        """
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.num_cycles = num_cycles
        self.seed = seed

        self.progress = BatchProgress()
        self._progress_callback: Callable[[BatchProgress], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[BatchProgress], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def generate_jobs(
        self,
        profiles: list[PlayerProfile],
        rudiments: list[Rudiment],
        tempos: list[int],
        augmentations_per_sample: int = 1,
        augmentation_configs: list[AugmentationConfig] | None = None,
    ) -> Iterator[BatchJob]:
        """
        Generate batch jobs for all combinations.

        Args:
            profiles: Player profiles
            rudiments: Rudiments to generate
            tempos: Tempos for each profile/rudiment combo
            augmentations_per_sample: Number of augmented versions per MIDI
            augmentation_configs: Augmentation configs (optional)

        Yields:
            BatchJob instances
        """
        for profile in profiles:
            for rudiment in rudiments:
                for tempo in tempos:
                    for aug_idx in range(augmentations_per_sample):
                        job_id = f"{profile.id}_{rudiment.slug}_{tempo}bpm_{aug_idx}"

                        aug_config = None
                        if augmentation_configs and aug_idx < len(augmentation_configs):
                            aug_config = augmentation_configs[aug_idx]

                        yield BatchJob(
                            job_id=job_id,
                            profile=profile,
                            rudiment=rudiment,
                            tempo=tempo,
                            augmentation_idx=aug_idx,
                            augmentation_config=aug_config,
                        )

    def process_batch(
        self,
        jobs: list[BatchJob],
        chunk_size: int = 100,
    ) -> Iterator[BatchResult]:
        """
        Process a batch of jobs in parallel.

        Args:
            jobs: List of jobs to process
            chunk_size: Jobs to submit at once

        Yields:
            BatchResult for each completed job
        """
        self.progress = BatchProgress(total_jobs=len(jobs))
        self.progress.start_time = time.time()

        logger.info(f"Processing {len(jobs)} jobs with {self.num_workers} workers")

        # Serialize jobs for multiprocessing
        job_dicts = [job.to_dict() for job in jobs]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit jobs in chunks to avoid memory issues
            futures = {}

            for i in range(0, len(job_dicts), chunk_size):
                chunk = job_dicts[i : i + chunk_size]

                for job_dict in chunk:
                    future = executor.submit(
                        _worker_generate_sample,
                        job_dict,
                        self.num_cycles,
                        self.seed,
                    )
                    futures[future] = job_dict["job_id"]

            # Collect results as they complete
            for future in as_completed(futures):
                job_id = futures[future]

                try:
                    result_dict = future.result()

                    # Reconstruct Sample from dict
                    sample = None
                    if result_dict["sample_dict"]:
                        sample = Sample.model_validate(result_dict["sample_dict"])

                    result = BatchResult(
                        job_id=result_dict["job_id"],
                        sample=sample,
                        midi_data=result_dict["midi_data"],
                        error=result_dict["error"],
                        duration_sec=result_dict["duration_sec"],
                    )

                except Exception as e:
                    result = BatchResult(
                        job_id=job_id,
                        error=f"Future error: {e}",
                    )

                # Update progress
                if result.success:
                    self.progress.completed_jobs += 1
                else:
                    self.progress.failed_jobs += 1

                if self._progress_callback:
                    self._progress_callback(self.progress)

                yield result

        logger.info(
            f"Batch complete: {self.progress.completed_jobs} succeeded, "
            f"{self.progress.failed_jobs} failed in {self.progress.elapsed_sec:.1f}s"
        )


class StreamingBatchProcessor:
    """
    Process jobs in streaming fashion with bounded memory.

    Uses a producer-consumer pattern with queues to limit
    memory usage for large datasets.
    """

    def __init__(
        self,
        generator: BatchGenerator,
        max_pending: int = 1000,
    ):
        """
        Initialize streaming processor.

        Args:
            generator: Batch generator instance
            max_pending: Maximum jobs in flight
        """
        self.generator = generator
        self.max_pending = max_pending

    def process_stream(
        self,
        job_iterator: Iterator[BatchJob],
        result_callback: Callable[[BatchResult], None],
    ) -> BatchProgress:
        """
        Process jobs from iterator in streaming fashion.

        Args:
            job_iterator: Iterator yielding jobs
            result_callback: Called for each completed result

        Returns:
            Final progress statistics
        """
        # Collect jobs in batches
        batch: list[BatchJob] = []
        total_processed = 0

        for job in job_iterator:
            batch.append(job)

            if len(batch) >= self.max_pending:
                # Process batch
                for result in self.generator.process_batch(batch):
                    result_callback(result)
                    total_processed += 1

                batch = []

        # Process remaining
        if batch:
            for result in self.generator.process_batch(batch):
                result_callback(result)
                total_processed += 1

        return self.generator.progress


def generate_batch(
    profiles: list[PlayerProfile],
    rudiments: list[Rudiment],
    tempos: list[int],
    num_workers: int | None = None,
    num_cycles: int = 4,
    seed: int = 42,
) -> list[BatchResult]:
    """
    Convenience function to generate a batch of samples.

    Args:
        profiles: Player profiles
        rudiments: Rudiments to generate
        tempos: Tempos to use
        num_workers: Number of parallel workers
        num_cycles: Cycles per sample
        seed: Random seed

    Returns:
        List of BatchResult objects
    """
    generator = BatchGenerator(
        num_workers=num_workers,
        num_cycles=num_cycles,
        seed=seed,
    )

    jobs = list(generator.generate_jobs(profiles, rudiments, tempos))
    return list(generator.process_batch(jobs))
