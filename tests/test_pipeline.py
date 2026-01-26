"""Tests for dataset generation pipeline."""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from dataset_gen.profiles.archetypes import (
    SkillTier,
    generate_profile,
    generate_profiles_batch,
)
from dataset_gen.rudiments.schema import (
    Rudiment,
    RudimentCategory,
    parse_sticking_string,
)
from dataset_gen.midi_gen.generator import MIDIGenerator
from dataset_gen.labels.compute import compute_sample_labels

from dataset_gen.pipeline.storage import (
    DatasetWriter,
    StorageConfig,
    ParquetReader,
    write_sample,
)
from dataset_gen.pipeline.splits import (
    SplitGenerator,
    SplitConfig,
    DatasetSplit,
    SampleSplitter,
    generate_splits,
)
from dataset_gen.pipeline.generate import (
    DatasetGenerator,
    GenerationConfig,
)


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
def sample_sample(sample_rudiment, sample_profile):
    """Create a sample Sample object for testing."""
    generator = MIDIGenerator(seed=42)
    performance = generator.generate(
        rudiment=sample_rudiment,
        profile=sample_profile,
        tempo_bpm=120,
        num_cycles=2,
        include_midi=False,
    )
    return compute_sample_labels(performance, sample_rudiment, sample_profile)


@pytest.fixture
def multiple_profiles():
    """Create multiple profiles for split testing."""
    return generate_profiles_batch(20, seed=42)


class TestDatasetWriter:
    """Tests for dataset storage."""

    def test_write_single_sample(self, sample_sample):
        """Test writing a single sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(output_dir=Path(tmpdir))
            writer = DatasetWriter(config)

            paths = writer.write_sample(sample_sample)
            writer.flush()

            # Check parquet files were created
            labels_dir = Path(tmpdir) / "labels"
            assert (labels_dir / "samples.parquet").exists()
            assert (labels_dir / "strokes.parquet").exists()
            assert (labels_dir / "measures.parquet").exists()
            assert (labels_dir / "exercises.parquet").exists()

    def test_write_with_audio(self, sample_sample):
        """Test writing a sample with audio data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(output_dir=Path(tmpdir))
            writer = DatasetWriter(config)

            # Create fake audio data
            audio_data = np.random.randn(44100, 2).astype(np.float32) * 0.5

            paths = writer.write_sample(
                sample_sample,
                audio_data=audio_data,
            )
            writer.flush()

            # Check audio file was created
            audio_dir = Path(tmpdir) / "audio"
            audio_files = list(audio_dir.glob("*.flac"))
            assert len(audio_files) == 1

    def test_write_with_midi(self, sample_rudiment, sample_profile):
        """Test writing a sample with MIDI data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate with MIDI
            generator = MIDIGenerator(seed=42)
            performance = generator.generate(
                rudiment=sample_rudiment,
                profile=sample_profile,
                tempo_bpm=120,
                num_cycles=2,
                include_midi=True,
            )
            sample = compute_sample_labels(performance, sample_rudiment, sample_profile)

            config = StorageConfig(output_dir=Path(tmpdir))
            writer = DatasetWriter(config)

            paths = writer.write_sample(
                sample,
                midi_data=performance.midi_data,
            )
            writer.flush()

            # Check MIDI file was created
            midi_dir = Path(tmpdir) / "midi"
            midi_files = list(midi_dir.glob("*.mid"))
            assert len(midi_files) == 1

    def test_batch_writing(self, sample_rudiment, sample_profile):
        """Test writing multiple samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(output_dir=Path(tmpdir))
            writer = DatasetWriter(config)

            # Generate multiple samples
            generator = MIDIGenerator(seed=42)
            for i in range(5):
                performance = generator.generate(
                    rudiment=sample_rudiment,
                    profile=sample_profile,
                    tempo_bpm=100 + i * 10,
                    num_cycles=2,
                    include_midi=False,
                )
                sample = compute_sample_labels(performance, sample_rudiment, sample_profile)
                writer.write_sample(sample)

            writer.flush()

            # Check we have all samples
            reader = ParquetReader(tmpdir)
            samples_df = reader.load_samples()
            assert len(samples_df) == 5

    def test_parquet_reader(self, sample_sample):
        """Test reading back parquet data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(output_dir=Path(tmpdir))
            writer = DatasetWriter(config)
            writer.write_sample(sample_sample)
            writer.flush()

            # Read back
            reader = ParquetReader(tmpdir)

            samples = reader.load_samples()
            assert len(samples) == 1
            assert samples.iloc[0]["sample_id"] == sample_sample.sample_id

            strokes = reader.load_strokes()
            assert len(strokes) == len(sample_sample.strokes)

            measures = reader.load_measures()
            assert len(measures) == len(sample_sample.measures)

            exercises = reader.load_exercises()
            assert len(exercises) == 1

    def test_convenience_functions(self, sample_sample):
        """Test write_sample and write_batch convenience functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test write_sample
            write_sample(sample_sample, tmpdir)

            # Verify data was written
            reader = ParquetReader(tmpdir)
            samples = reader.load_samples()
            assert len(samples) == 1


class TestSplitGenerator:
    """Tests for dataset split generation."""

    def test_basic_split(self, multiple_profiles):
        """Test basic profile splitting."""
        config = SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        generator = SplitGenerator(config)
        assignment = generator.generate_splits(multiple_profiles)

        # Check all profiles are assigned
        total_assigned = (
            len(assignment.train_profile_ids)
            + len(assignment.val_profile_ids)
            + len(assignment.test_profile_ids)
        )
        assert total_assigned == len(multiple_profiles)

        # Check no overlap
        train_set = set(assignment.train_profile_ids)
        val_set = set(assignment.val_profile_ids)
        test_set = set(assignment.test_profile_ids)
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_stratified_split(self, multiple_profiles):
        """Test stratified splitting by skill tier."""
        config = SplitConfig(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_by_skill=True,
            seed=42,
        )
        generator = SplitGenerator(config)
        assignment = generator.generate_splits(multiple_profiles)

        # Check we have stats
        assert "train_skill_distribution" in assignment.stats

    def test_split_lookup(self, multiple_profiles):
        """Test looking up splits for profiles."""
        assignment = generate_splits(multiple_profiles, seed=42)

        # Each profile should map to exactly one split
        for profile in multiple_profiles:
            split = assignment.get_split(profile.id)
            assert split in [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]

    def test_save_and_load_splits(self, multiple_profiles):
        """Test saving and loading split assignments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SplitConfig(seed=42)
            generator = SplitGenerator(config)
            assignment = generator.generate_splits(multiple_profiles)

            # Save
            split_path = Path(tmpdir) / "splits.json"
            generator.save_splits(assignment, split_path)
            assert split_path.exists()

            # Load
            loaded = generator.load_splits(split_path)
            assert loaded.train_profile_ids == assignment.train_profile_ids
            assert loaded.val_profile_ids == assignment.val_profile_ids
            assert loaded.test_profile_ids == assignment.test_profile_ids

    def test_sample_splitter(self, multiple_profiles):
        """Test SampleSplitter for splitting sample lists."""
        assignment = generate_splits(multiple_profiles, seed=42)
        splitter = SampleSplitter(assignment)

        # Create fake sample list
        samples = [{"profile_id": p.id, "data": i} for i, p in enumerate(multiple_profiles)]

        split_samples = splitter.split_samples(samples)

        # Check all samples are assigned
        total = (
            len(split_samples[DatasetSplit.TRAIN])
            + len(split_samples[DatasetSplit.VALIDATION])
            + len(split_samples[DatasetSplit.TEST])
        )
        assert total == len(samples)

    def test_ratios_respected(self):
        """Test that split ratios are approximately respected."""
        # Create many profiles for statistical testing
        profiles = generate_profiles_batch(100, seed=42)

        assignment = generate_splits(
            profiles,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        train_pct = len(assignment.train_profile_ids) / 100
        val_pct = len(assignment.val_profile_ids) / 100
        test_pct = len(assignment.test_profile_ids) / 100

        # Allow some tolerance
        assert 0.6 < train_pct < 0.8
        assert 0.1 < val_pct < 0.25
        assert 0.1 < test_pct < 0.25


class TestGenerationConfig:
    """Tests for generation configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GenerationConfig()

        assert config.num_profiles == 100
        assert config.train_ratio == 0.70
        assert config.seed == 42

    def test_skill_distribution(self):
        """Test skill distribution sums to 1."""
        config = GenerationConfig()

        total = sum(config.skill_distribution.values())
        assert abs(total - 1.0) < 1e-6


class TestDatasetGenerator:
    """Tests for the main generator."""

    def test_generator_initialization(self, sample_rudiment):
        """Test generator initializes without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                output_dir=Path(tmpdir),
                num_profiles=2,
                generate_audio=False,
            )
            generator = DatasetGenerator(config)
            assert generator is not None
            generator.close()

    def test_generate_single(self, sample_rudiment):
        """Test generating a single sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                output_dir=Path(tmpdir),
                num_profiles=2,
                generate_audio=False,
            )
            generator = DatasetGenerator(config)

            sample, midi_data, audio_data = generator.generate_single(
                rudiment=sample_rudiment,
                tempo=120,
            )

            assert sample is not None
            assert len(sample.strokes) > 0
            assert midi_data is not None  # MIDI should always be generated

            generator.close()

    def test_small_dataset_generation(self, sample_rudiment):
        """Test generating a small dataset (no audio)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                output_dir=Path(tmpdir),
                num_profiles=2,
                tempos_per_rudiment=1,
                augmentations_per_sample=1,
                generate_audio=False,
                verbose=False,
            )
            generator = DatasetGenerator(config)

            # Generate with just one rudiment
            splits = generator.generate(rudiments=[sample_rudiment])

            assert splits is not None
            assert (
                len(splits.train_profile_ids)
                + len(splits.val_profile_ids)
                + len(splits.test_profile_ids)
                == 2
            )

            # Check files were created
            labels_dir = Path(tmpdir) / "labels"
            assert (labels_dir / "samples.parquet").exists()

            # Read back and verify
            reader = ParquetReader(tmpdir)
            samples = reader.load_samples()
            assert len(samples) == 2  # 2 profiles * 1 rudiment * 1 tempo * 1 augmentation

            generator.close()


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_without_audio(self, sample_rudiment):
        """Test complete pipeline without audio synthesis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate dataset
            config = GenerationConfig(
                output_dir=Path(tmpdir),
                num_profiles=3,
                tempos_per_rudiment=2,
                augmentations_per_sample=1,
                generate_audio=False,
                verbose=False,
            )
            generator = DatasetGenerator(config)
            splits = generator.generate(rudiments=[sample_rudiment])
            generator.close()

            # Verify structure
            assert (Path(tmpdir) / "splits.json").exists()
            assert (Path(tmpdir) / "labels" / "samples.parquet").exists()

            # Load and verify data
            reader = ParquetReader(tmpdir)
            samples = reader.load_samples()

            # 3 profiles * 1 rudiment * 2 tempos * 1 augmentation = 6 samples
            assert len(samples) == 6

            strokes = reader.load_strokes()
            assert len(strokes) > 0

            # Verify splits work
            for _, row in samples.iterrows():
                profile_id = row["profile_id"]
                split = splits.get_split(profile_id)
                assert split in [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]

    def test_sample_profile_consistency(self, sample_rudiment):
        """Test that all samples from a profile are in the same split."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                output_dir=Path(tmpdir),
                num_profiles=5,
                tempos_per_rudiment=3,
                generate_audio=False,
                verbose=False,
            )
            generator = DatasetGenerator(config)
            splits = generator.generate(rudiments=[sample_rudiment])
            generator.close()

            # Load samples
            reader = ParquetReader(tmpdir)
            samples = reader.load_samples()

            # Group by profile
            profile_splits = {}
            for _, row in samples.iterrows():
                profile_id = row["profile_id"]
                split = splits.get_split(profile_id)

                if profile_id not in profile_splits:
                    profile_splits[profile_id] = split
                else:
                    # All samples from this profile should be in the same split
                    assert profile_splits[profile_id] == split
