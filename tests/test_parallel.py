"""Tests for parallel processing and checkpointing."""

import pytest
import tempfile
import numpy as np

from dataset_gen.profiles.archetypes import (
    PlayerProfile,
    SkillTier,
    generate_profile,
    generate_profiles_batch,
)
from dataset_gen.rudiments.schema import (
    Rudiment,
    RudimentCategory,
    parse_sticking_string,
)
from dataset_gen.pipeline.parallel import (
    BatchGenerator,
    BatchJob,
    BatchProgress,
    StreamingBatchProcessor,
    generate_batch,
)
from dataset_gen.pipeline.checkpoint import (
    CheckpointManager,
    CheckpointState,
    ResumableGenerator,
    get_profiles_from_checkpoint,
    get_splits_from_checkpoint,
)
from dataset_gen.pipeline.splits import generate_splits
from dataset_gen.pipeline.distributed import check_ray_available


@pytest.fixture
def sample_rudiment():
    """Create a simple rudiment for testing."""
    pattern = parse_sticking_string("RLRL", ">..>")
    return Rudiment(
        name="Test Rudiment",
        slug="test_rudiment",
        category=RudimentCategory.ROLL,
        pattern=pattern,
    )


@pytest.fixture
def sample_profile():
    """Create a sample profile for testing."""
    return generate_profile(SkillTier.INTERMEDIATE, rng=np.random.default_rng(42))


@pytest.fixture
def sample_profiles():
    """Create multiple profiles."""
    return generate_profiles_batch(5, seed=42)


class TestBatchJob:
    """Tests for BatchJob serialization."""

    def test_job_serialization(self, sample_rudiment, sample_profile):
        """Test BatchJob round-trip serialization."""
        job = BatchJob(
            job_id="test_job_1",
            profile=sample_profile,
            rudiment=sample_rudiment,
            tempo=120,
            augmentation_idx=0,
        )

        # Serialize
        job_dict = job.to_dict()
        assert job_dict["job_id"] == "test_job_1"
        assert job_dict["tempo"] == 120

        # Deserialize
        restored = BatchJob.from_dict(job_dict)
        assert restored.job_id == job.job_id
        assert restored.tempo == job.tempo
        assert restored.profile.id == job.profile.id
        assert restored.rudiment.slug == job.rudiment.slug


class TestBatchProgress:
    """Tests for progress tracking."""

    def test_progress_metrics(self):
        """Test progress calculations."""
        progress = BatchProgress(
            total_jobs=100,
            completed_jobs=50,
            failed_jobs=5,
        )

        assert progress.percent_complete == 50.0
        assert progress.jobs_per_sec >= 0
        assert progress.eta_sec >= 0

    def test_progress_str(self):
        """Test progress string representation."""
        progress = BatchProgress(total_jobs=100, completed_jobs=25)
        s = str(progress)
        assert "25/100" in s
        assert "25.0%" in s


class TestBatchGenerator:
    """Tests for parallel batch generation."""

    def test_generate_jobs(self, sample_rudiment, sample_profiles):
        """Test job generation from parameters."""
        generator = BatchGenerator(num_workers=2, seed=42)

        jobs = list(
            generator.generate_jobs(
                profiles=sample_profiles,
                rudiments=[sample_rudiment],
                tempos=[100, 120],
            )
        )

        # 5 profiles * 1 rudiment * 2 tempos = 10 jobs
        assert len(jobs) == 10

        # Check job IDs are unique
        job_ids = [j.job_id for j in jobs]
        assert len(set(job_ids)) == 10

    def test_process_batch(self, sample_rudiment, sample_profile):
        """Test processing a small batch."""
        generator = BatchGenerator(num_workers=2, num_cycles=2, seed=42)

        jobs = [
            BatchJob(
                job_id=f"test_{i}",
                profile=sample_profile,
                rudiment=sample_rudiment,
                tempo=120,
            )
            for i in range(3)
        ]

        results = list(generator.process_batch(jobs))

        assert len(results) == 3
        # At least some should succeed
        successful = [r for r in results if r.success]
        assert len(successful) >= 1

        # Check result has expected data
        for result in successful:
            assert result.sample is not None
            assert result.midi_data is not None
            assert result.error is None

    def test_progress_callback(self, sample_rudiment, sample_profile):
        """Test progress callback is called."""
        generator = BatchGenerator(num_workers=1, num_cycles=2, seed=42)

        progress_updates = []
        generator.set_progress_callback(lambda p: progress_updates.append(p.completed_jobs))

        jobs = [
            BatchJob(
                job_id="callback_test",
                profile=sample_profile,
                rudiment=sample_rudiment,
                tempo=100,
            )
        ]

        list(generator.process_batch(jobs))

        # Should have at least one progress update
        assert len(progress_updates) >= 1


class TestGenerateBatchConvenience:
    """Tests for generate_batch convenience function."""

    def test_generate_batch(self, sample_rudiment, sample_profile):
        """Test generate_batch convenience function."""
        results = generate_batch(
            profiles=[sample_profile],
            rudiments=[sample_rudiment],
            tempos=[100],
            num_workers=1,
            num_cycles=2,
            seed=42,
        )

        assert len(results) == 1
        assert results[0].sample is not None


class TestCheckpointState:
    """Tests for checkpoint state."""

    def test_state_serialization(self):
        """Test CheckpointState round-trip."""
        state = CheckpointState(
            run_id="test_run",
            started_at="2024-01-01T00:00:00",
            config_hash="abc123",
            profile_ids=["p1", "p2"],
            completed_sample_ids={"s1", "s2"},
            total_expected=10,
        )

        data = state.to_dict()
        restored = CheckpointState.from_dict(data)

        assert restored.run_id == state.run_id
        assert restored.profile_ids == state.profile_ids
        assert restored.completed_sample_ids == state.completed_sample_ids

    def test_progress_properties(self):
        """Test progress calculation properties."""
        state = CheckpointState(
            run_id="test",
            started_at="",
            config_hash="",
            total_expected=100,
        )
        state.completed_sample_ids = {"s1", "s2", "s3"}
        state.failed_sample_ids = {"f1"}

        assert state.num_completed == 3
        assert state.num_failed == 1
        assert state.num_remaining == 96
        assert state.percent_complete == 3.0


class TestCheckpointManager:
    """Tests for checkpoint manager."""

    def test_save_and_load(self):
        """Test saving and loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            state = CheckpointState(
                run_id="test_run",
                started_at="2024-01-01T00:00:00",
                config_hash="abc123",
                profile_ids=["p1", "p2"],
                total_expected=10,
            )
            state.completed_sample_ids.add("sample1")

            manager.save(state)
            assert manager.exists()

            loaded = manager.load()
            assert loaded is not None
            assert loaded.run_id == "test_run"
            assert "sample1" in loaded.completed_sample_ids

    def test_clear(self):
        """Test clearing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir)

            state = CheckpointState(
                run_id="test",
                started_at="",
                config_hash="",
            )
            manager.save(state)
            assert manager.exists()

            manager.clear()
            assert not manager.exists()


class TestResumableGenerator:
    """Tests for resumable generation."""

    def test_fresh_start(self, sample_profiles):
        """Test initializing a fresh run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            resumable = ResumableGenerator(tmpdir, checkpoint_interval=10)

            splits = generate_splits(sample_profiles, seed=42)
            state = resumable.initialize(
                config_dict={"seed": 42},
                profiles=sample_profiles,
                splits=splits,
                total_expected=100,
            )

            assert state.run_id is not None
            assert state.total_expected == 100
            assert len(state.profile_ids) == len(sample_profiles)

    def test_resume(self, sample_profiles):
        """Test resuming from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run
            resumable1 = ResumableGenerator(tmpdir, checkpoint_interval=10)
            splits = generate_splits(sample_profiles, seed=42)
            state1 = resumable1.initialize(
                config_dict={"seed": 42},
                profiles=sample_profiles,
                splits=splits,
                total_expected=100,
            )

            # Mark some samples complete
            resumable1.mark_completed("sample1")
            resumable1.mark_completed("sample2")
            resumable1.finalize()

            # Resume
            resumable2 = ResumableGenerator(tmpdir, checkpoint_interval=10)
            state2 = resumable2.initialize(
                config_dict={"seed": 42},
                profiles=sample_profiles,
                splits=splits,
                total_expected=100,
            )

            # Should have same run_id and completed samples
            assert state2.run_id == state1.run_id
            assert resumable2.should_skip("sample1")
            assert resumable2.should_skip("sample2")
            assert not resumable2.should_skip("sample3")

    def test_config_change_restarts(self, sample_profiles):
        """Test that config change causes fresh start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run
            resumable1 = ResumableGenerator(tmpdir)
            splits = generate_splits(sample_profiles, seed=42)
            state1 = resumable1.initialize(
                config_dict={"seed": 42},
                profiles=sample_profiles,
                splits=splits,
                total_expected=100,
            )
            resumable1.mark_completed("sample1")
            resumable1.finalize()

            # New run with different config
            resumable2 = ResumableGenerator(tmpdir)
            state2 = resumable2.initialize(
                config_dict={"seed": 99},  # Different config
                profiles=sample_profiles,
                splits=splits,
                total_expected=100,
            )

            # Should be fresh start
            assert state2.run_id != state1.run_id
            assert not resumable2.should_skip("sample1")

    def test_get_remaining_sample_ids(self, sample_profiles):
        """Test filtering remaining samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            resumable = ResumableGenerator(tmpdir)
            splits = generate_splits(sample_profiles, seed=42)
            resumable.initialize(
                config_dict={},
                profiles=sample_profiles,
                splits=splits,
                total_expected=5,
            )

            all_ids = ["s1", "s2", "s3", "s4", "s5"]
            resumable.mark_completed("s1")
            resumable.mark_failed("s3")

            remaining = resumable.get_remaining_sample_ids(all_ids)
            assert remaining == ["s2", "s4", "s5"]


class TestCheckpointHelpers:
    """Tests for checkpoint helper functions."""

    def test_get_profiles_from_checkpoint(self, sample_profiles):
        """Test loading profiles from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            resumable = ResumableGenerator(tmpdir)
            splits = generate_splits(sample_profiles, seed=42)
            resumable.initialize(
                config_dict={},
                profiles=sample_profiles,
                splits=splits,
                total_expected=10,
            )
            resumable.finalize()

            loaded = get_profiles_from_checkpoint(tmpdir)
            assert loaded is not None
            assert len(loaded) == len(sample_profiles)
            assert loaded[0].id == sample_profiles[0].id

    def test_get_splits_from_checkpoint(self, sample_profiles):
        """Test loading splits from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            resumable = ResumableGenerator(tmpdir)
            splits = generate_splits(sample_profiles, seed=42)
            resumable.initialize(
                config_dict={},
                profiles=sample_profiles,
                splits=splits,
                total_expected=10,
            )
            resumable.finalize()

            loaded = get_splits_from_checkpoint(tmpdir)
            assert loaded is not None
            assert loaded.train_profile_ids == splits.train_profile_ids


class TestDistributedAvailability:
    """Tests for distributed module availability check."""

    def test_check_ray_available(self):
        """Test Ray availability check function."""
        # This should run without error regardless of Ray availability
        available = check_ray_available()
        assert isinstance(available, bool)


class TestStreamingBatchProcessor:
    """Tests for streaming batch processor."""

    def test_process_stream(self, sample_rudiment, sample_profile):
        """Test streaming batch processing."""
        generator = BatchGenerator(num_workers=1, num_cycles=2, seed=42)
        processor = StreamingBatchProcessor(generator, max_pending=2)

        def job_iterator():
            for i in range(3):
                yield BatchJob(
                    job_id=f"stream_{i}",
                    profile=sample_profile,
                    rudiment=sample_rudiment,
                    tempo=100,
                )

        results = []
        processor.process_stream(
            job_iterator(),
            result_callback=lambda r: results.append(r),
        )

        # Should have processed all 3 jobs
        assert len(results) == 3
        # At least some should be successful
        successful = [r for r in results if r.success]
        assert len(successful) >= 1


class TestPlayerProfileSerialization:
    """Tests for PlayerProfile serialization methods."""

    def test_to_dict(self, sample_profile):
        """Test PlayerProfile.to_dict()."""
        data = sample_profile.to_dict()

        assert data["id"] == sample_profile.id
        assert data["skill_tier"] == sample_profile.skill_tier.value
        assert "dimensions" in data

    def test_from_dict(self, sample_profile):
        """Test PlayerProfile.from_dict()."""
        data = sample_profile.to_dict()
        restored = PlayerProfile.from_dict(data)

        assert restored.id == sample_profile.id
        assert restored.skill_tier == sample_profile.skill_tier
        assert (
            restored.dimensions.timing.timing_accuracy
            == sample_profile.dimensions.timing.timing_accuracy
        )

    def test_round_trip(self, sample_profile):
        """Test full round-trip serialization."""
        data = sample_profile.to_dict()
        restored = PlayerProfile.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2
