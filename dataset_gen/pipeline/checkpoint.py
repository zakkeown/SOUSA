"""
Checkpointing and resumability for dataset generation.

This module provides state persistence to enable:
- Resuming interrupted generation runs
- Incremental dataset updates
- Progress recovery after crashes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib
import logging
import shutil

from dataset_gen.profiles.archetypes import PlayerProfile
from dataset_gen.pipeline.splits import SplitAssignment

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """State of a generation run that can be checkpointed."""

    # Run identification
    run_id: str
    started_at: str
    config_hash: str

    # Profile state
    profile_ids: list[str] = field(default_factory=list)
    profiles_data: list[dict] = field(default_factory=list)

    # Split state
    splits_data: dict | None = None

    # Progress tracking
    completed_sample_ids: set[str] = field(default_factory=set)
    failed_sample_ids: set[str] = field(default_factory=set)

    # Statistics
    total_expected: int = 0
    last_checkpoint_at: str = ""

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "config_hash": self.config_hash,
            "profile_ids": self.profile_ids,
            "profiles_data": self.profiles_data,
            "splits_data": self.splits_data,
            "completed_sample_ids": list(self.completed_sample_ids),
            "failed_sample_ids": list(self.failed_sample_ids),
            "total_expected": self.total_expected,
            "last_checkpoint_at": self.last_checkpoint_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        """Deserialize state from dictionary."""
        return cls(
            run_id=data["run_id"],
            started_at=data["started_at"],
            config_hash=data["config_hash"],
            profile_ids=data.get("profile_ids", []),
            profiles_data=data.get("profiles_data", []),
            splits_data=data.get("splits_data"),
            completed_sample_ids=set(data.get("completed_sample_ids", [])),
            failed_sample_ids=set(data.get("failed_sample_ids", [])),
            total_expected=data.get("total_expected", 0),
            last_checkpoint_at=data.get("last_checkpoint_at", ""),
        )

    @property
    def num_completed(self) -> int:
        return len(self.completed_sample_ids)

    @property
    def num_failed(self) -> int:
        return len(self.failed_sample_ids)

    @property
    def num_remaining(self) -> int:
        return self.total_expected - self.num_completed - self.num_failed

    @property
    def percent_complete(self) -> float:
        if self.total_expected > 0:
            return 100 * self.num_completed / self.total_expected
        return 0.0


class CheckpointManager:
    """
    Manage checkpoints for generation runs.

    Provides save/load functionality and atomic checkpoint updates.
    """

    CHECKPOINT_FILE = "checkpoint.json"
    CHECKPOINT_BACKUP = "checkpoint.backup.json"

    def __init__(self, output_dir: Path | str):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Output directory for checkpoints
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        self.backup_path = self.output_dir / self.CHECKPOINT_BACKUP

    def exists(self) -> bool:
        """Check if a checkpoint exists."""
        return self.checkpoint_path.exists()

    def load(self) -> CheckpointState | None:
        """
        Load checkpoint state if it exists.

        Returns:
            CheckpointState or None if no checkpoint exists
        """
        if not self.exists():
            return None

        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            state = CheckpointState.from_dict(data)
            logger.info(
                f"Loaded checkpoint: {state.num_completed}/{state.total_expected} "
                f"({state.percent_complete:.1f}% complete)"
            )
            return state

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

            # Try backup
            if self.backup_path.exists():
                try:
                    with open(self.backup_path) as f:
                        data = json.load(f)
                    state = CheckpointState.from_dict(data)
                    logger.info("Loaded checkpoint from backup")
                    return state
                except Exception as e2:
                    logger.error(f"Failed to load backup checkpoint: {e2}")

            return None

    def save(self, state: CheckpointState) -> None:
        """
        Save checkpoint state atomically.

        Uses backup file to ensure atomic updates.

        Args:
            state: State to save
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        state.last_checkpoint_at = datetime.now().isoformat()

        # Backup existing checkpoint
        if self.checkpoint_path.exists():
            shutil.copy(self.checkpoint_path, self.backup_path)

        # Write new checkpoint to temp file first
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename
            temp_path.replace(self.checkpoint_path)

            logger.debug(f"Saved checkpoint: {state.num_completed} completed")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def clear(self) -> None:
        """Remove checkpoint files."""
        for path in [self.checkpoint_path, self.backup_path]:
            if path.exists():
                path.unlink()

        logger.info("Cleared checkpoints")


class ResumableGenerator:
    """
    Wrapper that adds resumability to any generation process.

    Tracks completed samples and skips them on resume.
    """

    def __init__(
        self,
        output_dir: Path | str,
        checkpoint_interval: int = 100,
    ):
        """
        Initialize resumable generator.

        Args:
            output_dir: Output directory
            checkpoint_interval: Save checkpoint every N samples
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_manager = CheckpointManager(output_dir)

        self._state: CheckpointState | None = None
        self._samples_since_checkpoint = 0

    def initialize(
        self,
        config_dict: dict,
        profiles: list[PlayerProfile],
        splits: SplitAssignment,
        total_expected: int,
    ) -> CheckpointState:
        """
        Initialize or resume a generation run.

        Args:
            config_dict: Generation config as dict (for hashing)
            profiles: Generated profiles
            splits: Split assignment
            total_expected: Total expected samples

        Returns:
            CheckpointState (new or resumed)
        """
        # Hash config for change detection
        config_hash = self._hash_config(config_dict)

        # Try to resume
        existing_state = self.checkpoint_manager.load()

        if existing_state and existing_state.config_hash == config_hash:
            logger.info(
                f"Resuming run {existing_state.run_id} - "
                f"{existing_state.num_remaining} samples remaining"
            )
            self._state = existing_state
            return self._state

        # Start fresh
        if existing_state:
            logger.warning("Config changed since last checkpoint - starting fresh")
            self.checkpoint_manager.clear()

        self._state = CheckpointState(
            run_id=self._generate_run_id(),
            started_at=datetime.now().isoformat(),
            config_hash=config_hash,
            profile_ids=[p.id for p in profiles],
            profiles_data=[p.to_dict() for p in profiles],
            splits_data=splits.to_dict(),
            total_expected=total_expected,
        )

        self.checkpoint_manager.save(self._state)
        logger.info(f"Started new run {self._state.run_id}")

        return self._state

    @property
    def state(self) -> CheckpointState:
        """Get current state."""
        if self._state is None:
            raise RuntimeError("Generator not initialized")
        return self._state

    def should_skip(self, sample_id: str) -> bool:
        """Check if a sample should be skipped (already completed)."""
        return sample_id in self.state.completed_sample_ids

    def mark_completed(self, sample_id: str) -> None:
        """Mark a sample as completed."""
        self.state.completed_sample_ids.add(sample_id)
        self._samples_since_checkpoint += 1
        self._maybe_checkpoint()

    def mark_failed(self, sample_id: str) -> None:
        """Mark a sample as failed."""
        self.state.failed_sample_ids.add(sample_id)
        self._samples_since_checkpoint += 1
        self._maybe_checkpoint()

    def get_remaining_sample_ids(
        self,
        all_sample_ids: list[str],
    ) -> list[str]:
        """
        Filter sample IDs to only those not yet completed.

        Args:
            all_sample_ids: All expected sample IDs

        Returns:
            List of sample IDs still needing generation
        """
        completed = self.state.completed_sample_ids
        failed = self.state.failed_sample_ids
        return [sid for sid in all_sample_ids if sid not in completed and sid not in failed]

    def finalize(self) -> None:
        """Finalize generation and save final state."""
        if self._state:
            self.checkpoint_manager.save(self._state)
            logger.info(
                f"Generation complete: {self.state.num_completed} samples, "
                f"{self.state.num_failed} failed"
            )

    def _maybe_checkpoint(self) -> None:
        """Save checkpoint if interval reached."""
        if self._samples_since_checkpoint >= self.checkpoint_interval:
            self.checkpoint_manager.save(self.state)
            self._samples_since_checkpoint = 0

    def _hash_config(self, config_dict: dict) -> str:
        """Generate hash of config for change detection."""
        # Sort keys for deterministic hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{random_suffix}"


def get_profiles_from_checkpoint(
    checkpoint_path: Path | str,
) -> list[PlayerProfile] | None:
    """
    Load profiles from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        List of PlayerProfile or None if no checkpoint
    """
    manager = CheckpointManager(checkpoint_path)
    state = manager.load()

    if state is None or not state.profiles_data:
        return None

    return [PlayerProfile.from_dict(d) for d in state.profiles_data]


def get_splits_from_checkpoint(
    checkpoint_path: Path | str,
) -> SplitAssignment | None:
    """
    Load split assignment from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        SplitAssignment or None if no checkpoint
    """
    manager = CheckpointManager(checkpoint_path)
    state = manager.load()

    if state is None or state.splits_data is None:
        return None

    return SplitAssignment.from_dict(state.splits_data)
