"""Tests for audio augmentation pipeline."""

import pytest
import numpy as np

from dataset_gen.audio_aug.room import (
    RoomSimulator,
    RoomType,
    RoomConfig,
    apply_room,
)
from dataset_gen.audio_aug.mic import (
    MicSimulator,
    MicType,
    MicPosition,
    MicConfig,
    apply_mic,
)
from dataset_gen.audio_aug.chain import (
    RecordingChain,
    ChainConfig,
    PreampConfig,
    PreampType,
    CompressorConfig,
    apply_chain,
)
from dataset_gen.audio_aug.degradation import (
    AudioDegrader,
    DegradationConfig,
    NoiseConfig,
    NoiseType,
    BitDepthConfig,
    add_noise,
)
from dataset_gen.audio_aug.pipeline import (
    AugmentationPipeline,
    AugmentationConfig,
    AugmentationPreset,
    augment_audio,
    random_augmentation,
)


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing (1 second of sine wave)."""
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate)
    # Mix of frequencies to test filtering
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t)  # A4
        + 0.3 * np.sin(2 * np.pi * 880 * t)  # A5
        + 0.2 * np.sin(2 * np.pi * 220 * t)  # A3
    )
    return audio


@pytest.fixture
def sample_audio_stereo():
    """Generate stereo sample audio."""
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 445 * t)  # Slight detuning
    return np.column_stack([left, right])


class TestRoomSimulator:
    """Tests for room simulation."""

    def test_room_process_mono(self, sample_audio):
        """Test room processing with mono input."""
        config = RoomConfig(room_type=RoomType.STUDIO, wet_dry_mix=0.3)
        simulator = RoomSimulator(config)

        output = simulator.process(sample_audio)

        assert output.shape == sample_audio.shape
        assert not np.allclose(output, sample_audio)  # Should be modified

    def test_room_process_stereo(self, sample_audio_stereo):
        """Test room processing with stereo input."""
        config = RoomConfig(room_type=RoomType.STUDIO, wet_dry_mix=0.3)
        simulator = RoomSimulator(config)

        output = simulator.process(sample_audio_stereo)

        assert output.shape == sample_audio_stereo.shape

    def test_room_types(self, sample_audio):
        """Test different room types produce different outputs."""
        outputs = {}
        for room_type in RoomType:
            config = RoomConfig(room_type=room_type, wet_dry_mix=0.5)
            simulator = RoomSimulator(config)
            outputs[room_type] = simulator.process(sample_audio)

        # Different room types should produce different outputs
        assert not np.allclose(
            outputs[RoomType.STUDIO],
            outputs[RoomType.CONCERT_HALL],
        )

    def test_wet_dry_mix(self, sample_audio):
        """Test wet/dry mix affects output."""
        simulator = RoomSimulator()

        dry_config = RoomConfig(room_type=RoomType.STUDIO, wet_dry_mix=0.0)
        wet_config = RoomConfig(room_type=RoomType.STUDIO, wet_dry_mix=1.0)

        dry_output = simulator.process(sample_audio, dry_config)
        wet_output = simulator.process(sample_audio, wet_config)

        # Dry should be closer to original
        dry_diff = np.mean(np.abs(dry_output - sample_audio))
        wet_diff = np.mean(np.abs(wet_output - sample_audio))
        assert dry_diff < wet_diff

    def test_synthetic_ir_direct_sound_at_zero(self):
        """Direct sound should be at ir[0], not delayed by pre_delay."""
        sim = RoomSimulator(sample_rate=44100)
        for room_type in [RoomType.PRACTICE_ROOM, RoomType.GYM, RoomType.GARAGE]:
            config = RoomConfig(room_type=room_type)
            ir = sim._generate_synthetic_ir(config)
            # Direct sound should be the first significant energy
            # Check that ir[0] has significant amplitude (the direct sound)
            assert abs(ir[0, 0]) > 0.1, f"{room_type}: direct sound not at ir[0]"

    def test_convenience_function(self, sample_audio):
        """Test apply_room convenience function."""
        output = apply_room(sample_audio, RoomType.STUDIO, wet_dry_mix=0.3)
        assert output.shape == sample_audio.shape


class TestMicSimulator:
    """Tests for microphone simulation."""

    def test_mic_process_mono(self, sample_audio):
        """Test mic processing with mono input."""
        config = MicConfig(mic_type=MicType.CONDENSER)
        simulator = MicSimulator(config)

        output = simulator.process(sample_audio)

        assert output.shape == sample_audio.shape

    def test_mic_types(self, sample_audio):
        """Test different mic types produce different outputs."""
        outputs = {}
        for mic_type in MicType:
            config = MicConfig(mic_type=mic_type)
            simulator = MicSimulator(config)
            outputs[mic_type] = simulator.process(sample_audio)

        # Different mic types should produce different frequency responses
        assert not np.allclose(
            outputs[MicType.CONDENSER],
            outputs[MicType.RIBBON],
        )

    def test_mic_positions(self, sample_audio):
        """Test different mic positions affect output."""
        simulator = MicSimulator()

        center_config = MicConfig(position=MicPosition.CENTER)
        distant_config = MicConfig(position=MicPosition.DISTANT)

        center_output = simulator.process(sample_audio, center_config)
        distant_output = simulator.process(sample_audio, distant_config)

        # Distant should be quieter
        center_rms = np.sqrt(np.mean(center_output**2))
        distant_rms = np.sqrt(np.mean(distant_output**2))
        assert distant_rms < center_rms

    def test_convenience_function(self, sample_audio):
        """Test apply_mic convenience function."""
        output = apply_mic(sample_audio, MicType.DYNAMIC, MicPosition.CENTER)
        assert output.shape == sample_audio.shape


class TestRecordingChain:
    """Tests for recording chain simulation."""

    def test_chain_process_mono(self, sample_audio):
        """Test chain processing with mono input."""
        config = ChainConfig()
        chain = RecordingChain(config)

        output = chain.process(sample_audio)

        assert output.shape == sample_audio.shape

    def test_preamp_saturation(self, sample_audio):
        """Test preamp saturation changes the signal."""
        chain = RecordingChain()

        clean_config = ChainConfig(preamp=PreampConfig(preamp_type=PreampType.CLEAN, drive=0.0))
        saturated_config = ChainConfig(preamp=PreampConfig(preamp_type=PreampType.WARM, drive=0.5))

        clean_output = chain.process(sample_audio, clean_config)
        saturated_output = chain.process(sample_audio, saturated_config)

        # Saturation adds harmonics, changing the signal
        assert not np.allclose(clean_output, saturated_output)

    def test_compression(self, sample_audio):
        """Test compression reduces dynamic range."""
        chain = RecordingChain()

        # Create signal with varying amplitude
        modulated = sample_audio * (
            0.5 + 0.5 * np.sin(2 * np.pi * 2 * np.linspace(0, 1, len(sample_audio)))
        )

        no_comp_config = ChainConfig(compressor=CompressorConfig(enabled=False))
        comp_config = ChainConfig(
            compressor=CompressorConfig(
                enabled=True,
                ratio=8.0,
                threshold_db=-20.0,
            )
        )

        no_comp_output = chain.process(modulated, no_comp_config)
        comp_output = chain.process(modulated, comp_config)

        # Compressed signal should have smaller dynamic range
        no_comp_range = np.max(np.abs(no_comp_output)) - np.min(np.abs(no_comp_output))
        comp_range = np.max(np.abs(comp_output)) - np.min(np.abs(comp_output))
        # This is a rough test; compression should reduce range
        assert comp_range <= no_comp_range * 1.1  # Allow some tolerance

    def test_convenience_function(self, sample_audio):
        """Test apply_chain convenience function."""
        output = apply_chain(
            sample_audio,
            preamp_type=PreampType.WARM,
            drive=0.2,
        )
        assert output.shape == sample_audio.shape


class TestAudioDegrader:
    """Tests for audio degradation."""

    def test_add_noise(self, sample_audio):
        """Test noise addition."""
        config = DegradationConfig(
            noise=NoiseConfig(
                enabled=True,
                noise_type=NoiseType.PINK,
                level_db=-30.0,
            )
        )
        degrader = AudioDegrader(config)

        output = degrader.process(sample_audio, seed=42)

        # Output should be different due to noise
        assert not np.allclose(output, sample_audio)

    def test_noise_types(self, sample_audio):
        """Test different noise types."""
        outputs = {}
        for noise_type in [NoiseType.WHITE, NoiseType.PINK, NoiseType.BROWN]:
            config = DegradationConfig(
                noise=NoiseConfig(
                    enabled=True,
                    noise_type=noise_type,
                    level_db=-20.0,
                )
            )
            degrader = AudioDegrader(config)
            outputs[noise_type] = degrader.process(sample_audio, seed=42)

        # Different noise types should produce different outputs
        assert not np.allclose(outputs[NoiseType.WHITE], outputs[NoiseType.PINK])

    def test_bit_depth_reduction(self, sample_audio):
        """Test bit depth reduction."""
        config = DegradationConfig(
            bit_depth=BitDepthConfig(
                enabled=True,
                bit_depth=8,
                dither=False,
            )
        )
        degrader = AudioDegrader(config)

        output = degrader.process(sample_audio, seed=42)

        # 8-bit has fewer unique values
        unique_input = len(np.unique(np.round(sample_audio * 127)))
        unique_output = len(np.unique(np.round(output * 127)))
        assert unique_output <= unique_input

    def test_reproducibility(self, sample_audio):
        """Test that same seed produces same output."""
        config = DegradationConfig(noise=NoiseConfig(enabled=True, level_db=-30.0))
        degrader = AudioDegrader(config)

        output1 = degrader.process(sample_audio, seed=42)
        output2 = degrader.process(sample_audio, seed=42)

        assert np.allclose(output1, output2)

    def test_convenience_function(self, sample_audio):
        """Test add_noise convenience function."""
        output = add_noise(sample_audio, NoiseType.PINK, level_db=-40.0)
        assert output.shape == sample_audio.shape


class TestAugmentationPipeline:
    """Tests for the full augmentation pipeline."""

    def test_pipeline_process(self, sample_audio):
        """Test full pipeline processing."""
        config = AugmentationConfig(
            room_enabled=True,
            mic_enabled=True,
            chain_enabled=True,
            degradation_enabled=True,
        )
        pipeline = AugmentationPipeline(config)

        output = pipeline.process(sample_audio, seed=42)

        assert output.shape == sample_audio.shape
        assert not np.allclose(output, sample_audio)

    def test_pipeline_presets(self, sample_audio):
        """Test all presets work."""
        for preset in AugmentationPreset:
            pipeline = AugmentationPipeline(preset=preset)
            output = pipeline.process(sample_audio, seed=42)

            assert output.shape == sample_audio.shape
            assert np.isfinite(output).all()  # No NaN or inf

    def test_pipeline_stereo(self, sample_audio_stereo):
        """Test pipeline with stereo input."""
        pipeline = AugmentationPipeline(preset=AugmentationPreset.CLEAN_STUDIO)

        output = pipeline.process(sample_audio_stereo, seed=42)

        assert output.shape == sample_audio_stereo.shape

    def test_pipeline_normalization(self, sample_audio):
        """Test output normalization."""
        config = AugmentationConfig(
            normalize=True,
            target_peak_db=-3.0,
        )
        pipeline = AugmentationPipeline(config)

        output = pipeline.process(sample_audio)

        peak_db = 20 * np.log10(np.max(np.abs(output)) + 1e-10)
        assert abs(peak_db - (-3.0)) < 0.5  # Within 0.5 dB

    def test_convenience_function(self, sample_audio):
        """Test augment_audio convenience function."""
        output = augment_audio(
            sample_audio,
            preset=AugmentationPreset.PRACTICE_ROOM,
            seed=42,
        )
        assert output.shape == sample_audio.shape

    def test_random_augmentation(self, sample_audio):
        """Test random augmentation function."""
        output, config = random_augmentation(sample_audio, seed=42)

        assert output.shape == sample_audio.shape
        assert isinstance(config, AugmentationConfig)

    def test_get_augmentation_params(self, sample_audio):
        """Test getting augmentation parameters."""
        pipeline = AugmentationPipeline(preset=AugmentationPreset.CLEAN_STUDIO)
        params = pipeline.get_augmentation_params()

        assert "room_type" in params
        assert "mic_type" in params
        assert "compression_ratio" in params


class TestOutputRanges:
    """Tests to ensure outputs stay within valid ranges."""

    def test_room_output_range(self, sample_audio):
        """Test room output stays in valid range."""
        for room_type in RoomType:
            output = apply_room(sample_audio, room_type, wet_dry_mix=0.8)
            assert np.max(np.abs(output)) <= 1.0

    def test_mic_output_range(self, sample_audio):
        """Test mic output stays in valid range."""
        for mic_type in MicType:
            output = apply_mic(sample_audio, mic_type)
            assert np.max(np.abs(output)) <= 1.0

    def test_chain_output_range(self, sample_audio):
        """Test chain output stays in valid range."""
        output = apply_chain(sample_audio, PreampType.AGGRESSIVE, drive=0.8)
        assert np.max(np.abs(output)) <= 1.0

    def test_pipeline_output_range(self, sample_audio):
        """Test full pipeline output stays in valid range."""
        for preset in AugmentationPreset:
            output = augment_audio(sample_audio, preset=preset)
            assert np.max(np.abs(output)) <= 1.0
