"""
SOUSA Dataset Generation Pipeline.

Main orchestration for generating the Synthetic Open Unified Snare Assessment dataset:
1. Generate player profiles with correlated skill dimensions
2. Generate MIDI performances with realistic timing/velocity
3. Render audio via FluidSynth (multiple soundfonts)
4. Apply audio augmentation (rooms, mics, compression, noise)
5. Compute hierarchical labels (stroke, measure, exercise)
6. Save to disk with profile-based train/val/test splits
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import logging

import numpy as np

from dataset_gen.rudiments.schema import Rudiment
from dataset_gen.rudiments.loader import load_all_rudiments
from dataset_gen.profiles.archetypes import (
    PlayerProfile,
    SkillTier,
    generate_profile,
    generate_profiles_batch,
)
from dataset_gen.midi_gen.generator import MIDIGenerator, regenerate_midi
from dataset_gen.midi_gen.articulations import ArticulationEngine
from dataset_gen.labels.compute import compute_sample_labels
from dataset_gen.labels.schema import Sample, AudioAugmentation
from dataset_gen.audio_aug.pipeline import (
    AugmentationPipeline,
    AugmentationConfig,
    AugmentationPreset,
)
from dataset_gen.pipeline.storage import DatasetWriter, StorageConfig
from dataset_gen.pipeline.splits import (
    SplitGenerator,
    SplitConfig,
    SplitAssignment,
)

# Try to import audio synthesis (may not be available)
try:
    from dataset_gen.audio_synth.synthesizer import AudioSynthesizer, SynthConfig

    SYNTH_AVAILABLE = True
except ImportError:
    SYNTH_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("output/dataset"))

    # Scale settings
    num_profiles: int = 100
    samples_per_profile: int = 40  # One per rudiment
    num_cycles_per_sample: int = 4

    # Profile distribution
    skill_distribution: dict[SkillTier, float] = field(
        default_factory=lambda: {
            SkillTier.BEGINNER: 0.25,
            SkillTier.INTERMEDIATE: 0.35,
            SkillTier.ADVANCED: 0.25,
            SkillTier.PROFESSIONAL: 0.15,
        }
    )

    # Tempo settings
    tempo_range: tuple[int, int] = (60, 180)
    tempos_per_rudiment: int = 3  # Different tempos for each rudiment

    # Audio settings
    generate_audio: bool = True
    soundfont_path: Path | None = None  # Single soundfont or directory containing .sf2 files
    soundfont_paths: list[Path] | None = None  # Multiple soundfonts (overrides soundfont_path)
    sample_rate: int = 44100
    audio_format: str = "flac"

    # Augmentation settings
    apply_augmentation: bool = True
    augmentation_presets: list[AugmentationPreset] | None = None
    augmentations_per_sample: int = 1  # How many augmented versions per MIDI

    # Split settings
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Processing settings
    seed: int = 42
    batch_size: int = 100  # Flush to disk every N samples
    verbose: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

        # Handle soundfont paths - support single file, directory, or list
        if self.soundfont_paths is None and self.soundfont_path is not None:
            sf_path = Path(self.soundfont_path)
            if sf_path.is_dir():
                # Load all .sf2 files from directory
                self.soundfont_paths = sorted(sf_path.glob("*.sf2"))
            elif sf_path.is_file():
                self.soundfont_paths = [sf_path]

        # Default augmentation presets
        if self.augmentation_presets is None:
            self.augmentation_presets = [
                AugmentationPreset.CLEAN_STUDIO,
                AugmentationPreset.PRACTICE_ROOM,
                AugmentationPreset.GYM,
                AugmentationPreset.GARAGE,
                AugmentationPreset.MARCHING_FIELD,
            ]


@dataclass
class GenerationProgress:
    """Track generation progress."""

    total_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0

    current_profile: str = ""
    current_rudiment: str = ""

    def __str__(self) -> str:
        pct = 100 * self.completed_samples / max(self.total_samples, 1)
        return f"Progress: {self.completed_samples}/{self.total_samples} ({pct:.1f}%)"


class DatasetGenerator:
    """
    Generate complete synthetic drum rudiment dataset.

    Orchestrates the full pipeline from profiles to stored samples.
    """

    def __init__(self, config: GenerationConfig):
        """
        Initialize generator.

        Args:
            config: Generation configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        # Initialize components
        self._midi_gen = MIDIGenerator(seed=config.seed)
        self._articulation_engine = ArticulationEngine(seed=config.seed + 1)

        # Audio synthesis (if available and configured)
        self._synth: AudioSynthesizer | None = None
        self._soundfont_names: list[str] = []
        if config.generate_audio and config.soundfont_paths:
            if SYNTH_AVAILABLE:
                synth_config = SynthConfig(sample_rate=config.sample_rate)
                self._synth = AudioSynthesizer(soundfont_path=None, config=synth_config)
                # Load all soundfonts
                for sf_path in config.soundfont_paths:
                    try:
                        name = self._synth.load_soundfont(sf_path)
                        self._soundfont_names.append(name)
                        logger.info(f"Loaded soundfont: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to load soundfont {sf_path}: {e}")
                if not self._soundfont_names:
                    logger.warning("No soundfonts loaded successfully")
                    self._synth.close()
                    self._synth = None
            else:
                logger.warning("Audio synthesis not available (FluidSynth not installed)")

        # Augmentation pipeline
        self._augmenter: AugmentationPipeline | None = None
        if config.apply_augmentation:
            self._augmenter = AugmentationPipeline(sample_rate=config.sample_rate)

        # Storage
        storage_config = StorageConfig(
            output_dir=config.output_dir,
            audio_format=config.audio_format,
            sample_rate=config.sample_rate,
        )
        self._writer = DatasetWriter(storage_config)

        # Progress tracking
        self.progress = GenerationProgress()

        # Profile numbering for readable filenames
        self._profile_numbers: dict[str, int] = {}
        self._profile_counter = 0

        # Callbacks
        self._progress_callback: Callable[[GenerationProgress], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[GenerationProgress], None],
    ) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def generate(
        self,
        rudiments: list[Rudiment] | None = None,
        rudiment_dir: Path | str | None = None,
    ) -> SplitAssignment:
        """
        Generate the complete dataset.

        Args:
            rudiments: List of rudiments to use (optional)
            rudiment_dir: Directory containing rudiment YAML files (optional)

        Returns:
            SplitAssignment for the generated profiles
        """
        # Load rudiments
        if rudiments is None:
            if rudiment_dir:
                rudiments_dict = load_all_rudiments(Path(rudiment_dir))
            else:
                # Try default location
                default_dir = Path(__file__).parent.parent / "rudiments" / "definitions"
                if default_dir.exists():
                    rudiments_dict = load_all_rudiments(default_dir)
                else:
                    raise ValueError("No rudiments provided and default directory not found")
            # Convert dict to list (load_all_rudiments returns dict[slug, Rudiment])
            rudiments = list(rudiments_dict.values())

        logger.info(f"Loaded {len(rudiments)} rudiments")

        # Generate profiles
        profiles = self._generate_profiles()
        logger.info(f"Generated {len(profiles)} player profiles")

        # Generate splits
        split_config = SplitConfig(
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            seed=self.config.seed,
        )
        split_generator = SplitGenerator(split_config)
        splits = split_generator.generate_splits(profiles)

        # Save splits
        split_generator.save_splits(
            splits,
            self.config.output_dir / "splits.json",
        )

        # Calculate total samples
        self.progress.total_samples = (
            len(profiles)
            * len(rudiments)
            * self.config.tempos_per_rudiment
            * self.config.augmentations_per_sample
        )

        logger.info(f"Generating {self.progress.total_samples} samples")

        # Generate samples
        for profile in profiles:
            self._generate_profile_samples(profile, rudiments, splits)

        # Flush remaining samples
        self._writer.flush()

        logger.info(f"Generation complete: {self.progress.completed_samples} samples")

        return splits

    def _generate_profiles(self) -> list[PlayerProfile]:
        """Generate player profiles according to distribution."""
        return generate_profiles_batch(
            self.config.num_profiles,
            skill_distribution=self.config.skill_distribution,
            seed=self.config.seed,
        )

    def _get_profile_number(self, profile: PlayerProfile) -> int:
        """Get or assign a sequential number for a profile."""
        if profile.id not in self._profile_numbers:
            self._profile_numbers[profile.id] = self._profile_counter
            self._profile_counter += 1
        return self._profile_numbers[profile.id]

    def _make_sample_id(
        self,
        profile: PlayerProfile,
        rudiment: Rudiment,
        tempo: int,
        soundfont_name: str | None,
        aug_preset: str | None,
    ) -> str:
        """Create a readable sample ID."""
        # Skill tier abbreviation
        tier_abbrev = {
            SkillTier.BEGINNER: "beg",
            SkillTier.INTERMEDIATE: "int",
            SkillTier.ADVANCED: "adv",
            SkillTier.PROFESSIONAL: "pro",
        }.get(profile.skill_tier, "unk")

        # Profile number
        profile_num = self._get_profile_number(profile)

        # Soundfont short name (remove common suffixes)
        sf_short = "raw"
        if soundfont_name:
            sf_short = soundfont_name.replace("_GM_GS", "").replace("_GS", "")
            sf_short = sf_short.replace("_", "").replace("-", "")[:8].lower()

        # Augmentation preset short name
        aug_short = aug_preset.lower().replace("_", "") if aug_preset else "clean"

        return f"{tier_abbrev}{profile_num:03d}_{rudiment.slug}_{tempo}bpm_{sf_short}_{aug_short}"

    def _generate_profile_samples(
        self,
        profile: PlayerProfile,
        rudiments: list[Rudiment],
        splits: SplitAssignment,
    ) -> None:
        """Generate all samples for a single profile."""
        self.progress.current_profile = profile.id

        # Determine tempos for this profile
        tempo_min, tempo_max = self.config.tempo_range
        tempos = self.rng.integers(
            tempo_min,
            tempo_max + 1,
            size=self.config.tempos_per_rudiment,
        )

        for rudiment in rudiments:
            self.progress.current_rudiment = rudiment.slug

            for tempo in tempos:
                try:
                    self._generate_sample(profile, rudiment, int(tempo), splits)
                except Exception as e:
                    logger.error(f"Failed to generate sample: {e}")
                    self.progress.failed_samples += 1

                # Check if we should flush
                stats = self._writer.get_stats()
                if stats["samples_pending"] >= self.config.batch_size:
                    self._writer.flush()

    def _generate_sample(
        self,
        profile: PlayerProfile,
        rudiment: Rudiment,
        tempo: int,
        splits: SplitAssignment,
    ) -> None:
        """Generate a single sample (MIDI + audio variations)."""
        # Generate MIDI performance
        performance = self._midi_gen.generate(
            rudiment=rudiment,
            profile=profile,
            tempo_bpm=tempo,
            num_cycles=self.config.num_cycles_per_sample,
            include_midi=True,
        )

        # Apply articulation refinements
        performance.strokes = self._articulation_engine.process(
            performance.strokes,
            rudiment,
            profile,
        )

        # Regenerate MIDI to match modified strokes
        # (articulation engine changes timing/velocity, so MIDI must be updated)
        performance.midi_data = regenerate_midi(performance)

        # Generate augmentation variations (each can use a different soundfont)
        augmentation_configs = self._get_augmentation_configs()

        for aug_idx, aug_config in enumerate(augmentation_configs):
            # Select soundfont for this variation (cycle through available soundfonts)
            soundfont_name = None
            if self._soundfont_names:
                soundfont_name = self._soundfont_names[aug_idx % len(self._soundfont_names)]

            # Create readable sample ID
            sample_id = self._make_sample_id(
                profile, rudiment, tempo, soundfont_name, aug_config.preset_name
            )

            # Render audio with selected soundfont
            base_audio = None
            if self._synth is not None and performance.midi_data:
                try:
                    base_audio = self._synth.render(
                        midi_data=performance.midi_data,
                        duration_hint_sec=performance.duration_sec,
                        soundfont_name=soundfont_name,
                    )
                except Exception as e:
                    logger.warning(f"Audio synthesis failed: {e}")

            # Apply augmentation to audio
            augmented_audio = None
            audio_aug_params = None

            if base_audio is not None and self._augmenter is not None:
                try:
                    augmented_audio = self._augmenter.process(
                        base_audio,
                        config=aug_config,
                        seed=hash(sample_id) % (2**31),
                    )
                    audio_aug_params = AudioAugmentation(
                        soundfont=soundfont_name or "default",
                        room_type=aug_config.room_type.value if aug_config.room_enabled else None,
                        room_wet_dry=aug_config.room_wet_dry,
                        mic_distance=aug_config.mic_distance if aug_config.mic_enabled else None,
                        mic_type=aug_config.mic_type.value if aug_config.mic_enabled else None,
                        compression_ratio=(
                            aug_config.compression_ratio if aug_config.compression_enabled else None
                        ),
                        noise_level_db=(
                            aug_config.noise_level_db if aug_config.degradation_enabled else None
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")
                    augmented_audio = base_audio

            elif base_audio is not None:
                augmented_audio = base_audio

            # Compute labels
            sample = compute_sample_labels(performance, rudiment, profile)
            sample.sample_id = sample_id
            sample.audio_augmentation = audio_aug_params

            # Set augmentation tracking fields (always present for ML filtering)
            sample.soundfont = soundfont_name
            sample.augmentation_preset = (
                aug_config.preset_name if aug_config.preset_name else "none"
            )
            # augmentation_group_id links clean sample to its augmented variants
            # All samples from the same (profile, rudiment, tempo) share the same group
            sample.augmentation_group_id = f"{profile.id[:8]}_{rudiment.slug}_{tempo}"

            # Write sample
            self._writer.write_sample(
                sample=sample,
                midi_data=performance.midi_data if aug_idx == 0 else None,  # Only write MIDI once
                audio_data=augmented_audio,
            )

            self.progress.completed_samples += 1

            if self._progress_callback:
                self._progress_callback(self.progress)

            if self.config.verbose and self.progress.completed_samples % 100 == 0:
                logger.info(str(self.progress))

    def _get_augmentation_configs(self) -> list[AugmentationConfig]:
        """Get augmentation configurations for sample variations."""
        if not self.config.apply_augmentation:
            return [AugmentationConfig()]  # Single unaugmented version

        configs = []
        presets = self.config.augmentation_presets or []

        n_configs = min(self.config.augmentations_per_sample, len(presets))

        if n_configs == 0:
            return [AugmentationConfig()]

        # Select presets
        indices = list(range(len(presets)))
        self.rng.shuffle(indices)
        selected_indices = indices[:n_configs]

        for idx in selected_indices:
            configs.append(AugmentationConfig.from_preset(presets[idx]))

        return configs

    def generate_single(
        self,
        rudiment: Rudiment,
        profile: PlayerProfile | None = None,
        tempo: int = 120,
        preset: AugmentationPreset | None = None,
    ) -> tuple[Sample, bytes | None, np.ndarray | None]:
        """
        Generate a single sample for testing/inspection.

        Args:
            rudiment: Rudiment to perform
            profile: Player profile (generated if None)
            tempo: Tempo in BPM
            preset: Augmentation preset

        Returns:
            Tuple of (sample, midi_data, audio_data)
        """
        if profile is None:
            profile = generate_profile(SkillTier.INTERMEDIATE, rng=self.rng)

        # Generate MIDI
        performance = self._midi_gen.generate(
            rudiment=rudiment,
            profile=profile,
            tempo_bpm=tempo,
            num_cycles=self.config.num_cycles_per_sample,
            include_midi=True,
        )

        # Apply articulations
        performance.strokes = self._articulation_engine.process(
            performance.strokes,
            rudiment,
            profile,
        )

        # Regenerate MIDI to match modified strokes
        performance.midi_data = regenerate_midi(performance)

        # Compute labels
        sample = compute_sample_labels(performance, rudiment, profile)

        # Render audio (use first soundfont if available)
        audio_data = None
        soundfont_name = self._soundfont_names[0] if self._soundfont_names else None
        if self._synth is not None and performance.midi_data:
            try:
                audio_data = self._synth.render(
                    midi_data=performance.midi_data,
                    duration_hint_sec=performance.duration_sec,
                    soundfont_name=soundfont_name,
                )

                # Apply augmentation
                if preset and self._augmenter:
                    aug_config = AugmentationConfig.from_preset(preset)
                    audio_data = self._augmenter.process(audio_data, aug_config)

            except Exception as e:
                logger.warning(f"Audio generation failed: {e}")

        return sample, performance.midi_data, audio_data

    def close(self) -> None:
        """Clean up resources."""
        if self._synth:
            self._synth.close()


def generate_dataset(
    output_dir: Path | str,
    num_profiles: int = 100,
    soundfont_path: Path | str | None = None,
    rudiment_dir: Path | str | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> SplitAssignment:
    """
    Convenience function to generate a dataset.

    Args:
        output_dir: Output directory for dataset
        num_profiles: Number of player profiles to generate
        soundfont_path: Path to soundfont for audio synthesis
        rudiment_dir: Directory containing rudiment definitions
        seed: Random seed
        verbose: Whether to log progress

    Returns:
        SplitAssignment for the generated profiles
    """
    config = GenerationConfig(
        output_dir=Path(output_dir),
        num_profiles=num_profiles,
        soundfont_path=Path(soundfont_path) if soundfont_path else None,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        logging.basicConfig(level=logging.INFO)

    generator = DatasetGenerator(config)

    try:
        return generator.generate(rudiment_dir=rudiment_dir)
    finally:
        generator.close()
