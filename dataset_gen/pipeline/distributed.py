"""
Distributed processing support using Ray.

This module provides cloud-scale generation using Ray for:
- Distributed sample generation across clusters
- Automatic scaling and fault tolerance
- Progress tracking across workers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Callable, Any
import logging
import time


# Ray is optional - only import if available
try:
    import ray
    from ray import ObjectRef

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None  # type: ignore
    ObjectRef = Any  # type: ignore

from dataset_gen.rudiments.schema import Rudiment
from dataset_gen.profiles.archetypes import PlayerProfile
from dataset_gen.midi_gen.generator import MIDIGenerator
from dataset_gen.midi_gen.articulations import ArticulationEngine
from dataset_gen.labels.compute import compute_sample_labels
from dataset_gen.labels.schema import Sample
from dataset_gen.pipeline.parallel import BatchJob, BatchResult, BatchProgress

logger = logging.getLogger(__name__)


def check_ray_available() -> bool:
    """Check if Ray is available."""
    return RAY_AVAILABLE


def init_ray(
    address: str | None = None,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    **kwargs,
) -> None:
    """
    Initialize Ray runtime.

    Args:
        address: Ray cluster address (None for local)
        num_cpus: Number of CPUs to use
        num_gpus: Number of GPUs to use
        **kwargs: Additional ray.init arguments
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not installed. Install with: pip install ray")

    init_kwargs = {}

    if address:
        init_kwargs["address"] = address
    if num_cpus:
        init_kwargs["num_cpus"] = num_cpus
    if num_gpus:
        init_kwargs["num_gpus"] = num_gpus

    init_kwargs.update(kwargs)

    if not ray.is_initialized():
        ray.init(**init_kwargs)
        logger.info(f"Ray initialized: {ray.cluster_resources()}")


def shutdown_ray() -> None:
    """Shutdown Ray runtime."""
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")


if RAY_AVAILABLE:

    @ray.remote
    def _ray_generate_sample(
        job_dict: dict,
        num_cycles: int,
        seed_offset: int,
    ) -> dict:
        """
        Ray remote function for sample generation.

        This runs on Ray workers, potentially across machines.
        """
        start_time = time.time()

        try:
            job = BatchJob.from_dict(job_dict)

            # Create worker-local generators
            worker_seed = hash(job.job_id) % (2**31) + seed_offset
            midi_gen = MIDIGenerator(seed=worker_seed)
            articulation_engine = ArticulationEngine(seed=worker_seed + 1)

            # Generate MIDI
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


@dataclass
class RayConfig:
    """Configuration for Ray distributed processing."""

    # Cluster settings
    address: str | None = None  # None = local, "auto" = auto-detect
    num_cpus: int | None = None
    num_gpus: int | None = None

    # Task settings
    max_pending_tasks: int = 1000
    task_timeout_sec: float = 300.0

    # Retry settings
    max_retries: int = 3
    retry_exceptions: bool = True


class RayBatchGenerator:
    """
    Generate samples using Ray for distributed processing.

    Provides the same interface as BatchGenerator but uses
    Ray for scaling across machines.
    """

    def __init__(
        self,
        config: RayConfig | None = None,
        num_cycles: int = 4,
        seed: int = 42,
    ):
        """
        Initialize Ray batch generator.

        Args:
            config: Ray configuration
            num_cycles: Number of cycles per sample
            seed: Random seed
        """
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not installed. Install with: pip install ray")

        self.config = config or RayConfig()
        self.num_cycles = num_cycles
        self.seed = seed

        self.progress = BatchProgress()
        self._progress_callback: Callable[[BatchProgress], None] | None = None

        self._initialized = False

    def initialize(self) -> None:
        """Initialize Ray runtime."""
        if not self._initialized:
            init_ray(
                address=self.config.address,
                num_cpus=self.config.num_cpus,
                num_gpus=self.config.num_gpus,
            )
            self._initialized = True

    def shutdown(self) -> None:
        """Shutdown Ray runtime."""
        if self._initialized:
            shutdown_ray()
            self._initialized = False

    def set_progress_callback(
        self,
        callback: Callable[[BatchProgress], None],
    ) -> None:
        """Set progress callback."""
        self._progress_callback = callback

    def process_batch(
        self,
        jobs: list[BatchJob],
    ) -> Iterator[BatchResult]:
        """
        Process jobs using Ray.

        Args:
            jobs: List of jobs to process

        Yields:
            BatchResult for each completed job
        """
        self.initialize()

        self.progress = BatchProgress(total_jobs=len(jobs))
        self.progress.start_time = time.time()

        logger.info(f"Submitting {len(jobs)} jobs to Ray")

        # Serialize jobs
        job_dicts = [job.to_dict() for job in jobs]

        # Submit tasks with bounded pending count
        pending: dict[ObjectRef, str] = {}
        job_iter = iter(enumerate(job_dicts))
        completed_count = 0

        while completed_count < len(jobs):
            # Submit more tasks if under limit
            while len(pending) < self.config.max_pending_tasks:
                try:
                    idx, job_dict = next(job_iter)
                    ref = _ray_generate_sample.options(
                        max_retries=self.config.max_retries,
                    ).remote(
                        job_dict,
                        self.num_cycles,
                        self.seed,
                    )
                    pending[ref] = job_dict["job_id"]
                except StopIteration:
                    break

            if not pending:
                break

            # Wait for at least one to complete
            ready, _ = ray.wait(list(pending.keys()), num_returns=1)

            for ref in ready:
                job_id = pending.pop(ref)

                try:
                    result_dict = ray.get(ref)

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
                        error=f"Ray error: {e}",
                    )

                # Update progress
                completed_count += 1
                if result.success:
                    self.progress.completed_jobs += 1
                else:
                    self.progress.failed_jobs += 1

                if self._progress_callback:
                    self._progress_callback(self.progress)

                yield result

        logger.info(
            f"Ray batch complete: {self.progress.completed_jobs} succeeded, "
            f"{self.progress.failed_jobs} failed"
        )


class DistributedDatasetGenerator:
    """
    High-level distributed dataset generator.

    Combines Ray processing with checkpointing for
    robust large-scale generation.
    """

    def __init__(
        self,
        output_dir: Path | str,
        ray_config: RayConfig | None = None,
        num_cycles: int = 4,
        seed: int = 42,
        checkpoint_interval: int = 100,
    ):
        """
        Initialize distributed generator.

        Args:
            output_dir: Output directory
            ray_config: Ray configuration
            num_cycles: Cycles per sample
            seed: Random seed
            checkpoint_interval: Checkpoint every N samples
        """
        from dataset_gen.pipeline.checkpoint import ResumableGenerator
        from dataset_gen.pipeline.storage import DatasetWriter, StorageConfig

        self.output_dir = Path(output_dir)
        self.num_cycles = num_cycles
        self.seed = seed

        # Initialize components
        self.ray_generator = RayBatchGenerator(
            config=ray_config,
            num_cycles=num_cycles,
            seed=seed,
        )

        self.resumable = ResumableGenerator(
            output_dir=output_dir,
            checkpoint_interval=checkpoint_interval,
        )

        storage_config = StorageConfig(output_dir=output_dir)
        self.writer = DatasetWriter(storage_config)

        self._progress_callback: Callable[[BatchProgress], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[BatchProgress], None],
    ) -> None:
        """Set progress callback."""
        self._progress_callback = callback
        self.ray_generator.set_progress_callback(callback)

    def generate(
        self,
        profiles: list[PlayerProfile],
        rudiments: list[Rudiment],
        tempos: list[int],
        splits: Any,  # SplitAssignment
        config_dict: dict,
        batch_size: int = 500,
    ) -> None:
        """
        Generate dataset using Ray with checkpointing.

        Args:
            profiles: Player profiles
            rudiments: Rudiments to generate
            tempos: Tempos to use
            splits: Split assignment
            config_dict: Config for checkpointing
            batch_size: Jobs per batch
        """
        # Calculate total
        total_expected = len(profiles) * len(rudiments) * len(tempos)

        # Initialize or resume
        self.resumable.initialize(
            config_dict=config_dict,
            profiles=profiles,
            splits=splits,
            total_expected=total_expected,
        )

        # Generate all job IDs
        all_jobs = []
        for profile in profiles:
            for rudiment in rudiments:
                for tempo in tempos:
                    job_id = f"{profile.id}_{rudiment.slug}_{tempo}bpm_0"

                    if not self.resumable.should_skip(job_id):
                        all_jobs.append(
                            BatchJob(
                                job_id=job_id,
                                profile=profile,
                                rudiment=rudiment,
                                tempo=tempo,
                            )
                        )

        logger.info(f"Processing {len(all_jobs)} remaining jobs")

        # Process in batches
        for i in range(0, len(all_jobs), batch_size):
            batch = all_jobs[i : i + batch_size]

            for result in self.ray_generator.process_batch(batch):
                if result.success:
                    self.writer.write_sample(
                        sample=result.sample,
                        midi_data=result.midi_data,
                        audio_data=result.audio_data,
                    )
                    self.resumable.mark_completed(result.job_id)
                else:
                    logger.warning(f"Job {result.job_id} failed: {result.error}")
                    self.resumable.mark_failed(result.job_id)

            # Flush periodically
            self.writer.flush()

        # Finalize
        self.resumable.finalize()
        self.ray_generator.shutdown()

        logger.info("Distributed generation complete")
