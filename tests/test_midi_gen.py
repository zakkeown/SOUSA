"""Tests for MIDI generation engine."""

import pytest
from pathlib import Path
import tempfile
import numpy as np

from dataset_gen.rudiments.schema import (
    Rudiment,
    StickingPattern,
    Stroke,
    StrokeType,
    Hand,
    RudimentCategory,
    parse_sticking_string,
)
from dataset_gen.rudiments.loader import load_rudiment
from dataset_gen.profiles.archetypes import generate_profile, SkillTier
from dataset_gen.midi_gen.generator import (
    MIDIGenerator,
    StrokeEvent,
    GeneratedPerformance,
    generate_performance,
)


@pytest.fixture
def simple_rudiment():
    """Create a simple rudiment for testing."""
    pattern = parse_sticking_string("RLRL", ">..>")
    return Rudiment(
        name="Test Rudiment",
        slug="test_rudiment",
        category=RudimentCategory.ROLL,
        pattern=pattern,
    )


@pytest.fixture
def paradiddle_rudiment():
    """Load the single paradiddle for testing."""
    yaml_path = (
        Path(__file__).parent.parent
        / "dataset_gen"
        / "rudiments"
        / "definitions"
        / "16_single_paradiddle.yaml"
    )
    if yaml_path.exists():
        return load_rudiment(yaml_path)
    else:
        # Create a mock paradiddle
        strokes = [
            Stroke(hand=Hand.RIGHT, stroke_type=StrokeType.ACCENT),
            Stroke(hand=Hand.LEFT, stroke_type=StrokeType.TAP),
            Stroke(hand=Hand.RIGHT, stroke_type=StrokeType.DIDDLE, diddle_position=1),
            Stroke(hand=Hand.RIGHT, stroke_type=StrokeType.DIDDLE, diddle_position=2),
            Stroke(hand=Hand.LEFT, stroke_type=StrokeType.ACCENT),
            Stroke(hand=Hand.RIGHT, stroke_type=StrokeType.TAP),
            Stroke(hand=Hand.LEFT, stroke_type=StrokeType.DIDDLE, diddle_position=1),
            Stroke(hand=Hand.LEFT, stroke_type=StrokeType.DIDDLE, diddle_position=2),
        ]
        return Rudiment(
            name="Single Paradiddle",
            slug="single_paradiddle",
            category=RudimentCategory.DIDDLE,
            pattern=StickingPattern(strokes=strokes, beats_per_cycle=2),
        )


class TestMIDIGenerator:
    """Tests for the MIDI generator."""

    def test_generate_simple_performance(self, simple_rudiment):
        """Test generating a simple performance."""
        profile = generate_profile(SkillTier.INTERMEDIATE, rng=np.random.default_rng(42))
        generator = MIDIGenerator(seed=42)

        performance = generator.generate(
            rudiment=simple_rudiment,
            profile=profile,
            tempo_bpm=120,
            num_cycles=4,
        )

        assert performance.rudiment_slug == "test_rudiment"
        assert performance.profile_id == profile.id
        assert performance.tempo_bpm == 120
        assert len(performance.strokes) == 16  # 4 strokes * 4 cycles

    def test_stroke_events_have_valid_times(self, simple_rudiment):
        """Test that stroke events have monotonically increasing times."""
        profile = generate_profile(SkillTier.PROFESSIONAL, rng=np.random.default_rng(42))
        generator = MIDIGenerator(seed=42)

        performance = generator.generate(
            rudiment=simple_rudiment,
            profile=profile,
            tempo_bpm=120,
            num_cycles=4,
        )

        # Intended times should be monotonically increasing
        intended_times = [s.intended_time_ms for s in performance.strokes]
        assert intended_times == sorted(intended_times)

        # Actual times might have small deviations but should generally increase
        actual_times = [s.actual_time_ms for s in performance.strokes]
        for i in range(1, len(actual_times)):
            # Allow some overlap due to deviations
            assert actual_times[i] > actual_times[0]

    def test_stroke_events_have_valid_velocities(self, simple_rudiment):
        """Test that velocities are within MIDI range."""
        profile = generate_profile(SkillTier.BEGINNER, rng=np.random.default_rng(42))
        generator = MIDIGenerator(seed=42)

        performance = generator.generate(
            rudiment=simple_rudiment,
            profile=profile,
            tempo_bpm=120,
            num_cycles=4,
        )

        for stroke in performance.strokes:
            assert 1 <= stroke.actual_velocity <= 127
            assert 1 <= stroke.intended_velocity <= 127

    def test_midi_data_generated(self, simple_rudiment):
        """Test that MIDI bytes are generated."""
        profile = generate_profile(SkillTier.INTERMEDIATE, rng=np.random.default_rng(42))
        generator = MIDIGenerator(seed=42)

        performance = generator.generate(
            rudiment=simple_rudiment,
            profile=profile,
            tempo_bpm=120,
            num_cycles=4,
            include_midi=True,
        )

        assert performance.midi_data is not None
        assert len(performance.midi_data) > 0
        # Check MIDI header
        assert performance.midi_data[:4] == b"MThd"

    def test_midi_file_writeable(self, simple_rudiment):
        """Test that MIDI data can be written to file."""
        profile = generate_profile(SkillTier.INTERMEDIATE, rng=np.random.default_rng(42))
        generator = MIDIGenerator(seed=42)

        performance = generator.generate(
            rudiment=simple_rudiment,
            profile=profile,
            tempo_bpm=120,
            num_cycles=4,
        )

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            performance.to_midi_file(f.name)
            # Verify file was written
            written_data = Path(f.name).read_bytes()
            assert written_data == performance.midi_data

    def test_skill_affects_timing_errors(self, paradiddle_rudiment):
        """Test that skill level affects timing errors."""
        generator = MIDIGenerator(seed=42)

        beginner_profile = generate_profile(SkillTier.BEGINNER, rng=np.random.default_rng(100))
        pro_profile = generate_profile(SkillTier.PROFESSIONAL, rng=np.random.default_rng(101))

        beginner_perf = generator.generate(
            rudiment=paradiddle_rudiment,
            profile=beginner_profile,
            tempo_bpm=100,
            num_cycles=8,
        )

        pro_perf = generator.generate(
            rudiment=paradiddle_rudiment,
            profile=pro_profile,
            tempo_bpm=100,
            num_cycles=8,
        )

        # Calculate mean absolute timing error
        beginner_errors = [abs(s.timing_error_ms) for s in beginner_perf.strokes]
        pro_errors = [abs(s.timing_error_ms) for s in pro_perf.strokes]

        beginner_mean = np.mean(beginner_errors)
        pro_mean = np.mean(pro_errors)

        # Beginner should have larger errors
        assert beginner_mean > pro_mean

    def test_hand_balance_affects_velocity(self, paradiddle_rudiment):
        """Test that hand balance affects left/right velocity difference."""
        generator = MIDIGenerator(seed=42)

        # Beginner has worse hand balance
        beginner_profile = generate_profile(SkillTier.BEGINNER, rng=np.random.default_rng(200))

        performance = generator.generate(
            rudiment=paradiddle_rudiment,
            profile=beginner_profile,
            tempo_bpm=100,
            num_cycles=8,
        )

        # Separate left and right hand velocities
        left_vels = [s.actual_velocity for s in performance.strokes if s.hand == Hand.LEFT]
        right_vels = [s.actual_velocity for s in performance.strokes if s.hand == Hand.RIGHT]

        if left_vels and right_vels:
            left_mean = np.mean(left_vels)
            right_mean = np.mean(right_vels)

            # Should show some imbalance (depends on dominant hand)
            ratio = min(left_mean, right_mean) / max(left_mean, right_mean)
            # Ratio should be < 1 (some imbalance for beginner)
            assert ratio < 0.98


class TestVelocitySpread:
    """Tests for ideal velocity distribution."""

    def test_ideal_velocity_spread(self):
        """Ideal velocities should span at least 70 MIDI units for ML-useful dynamic range."""
        gen = MIDIGenerator(seed=42)
        profile = generate_profile(SkillTier.PROFESSIONAL, rng=np.random.default_rng(42))
        # Use flam paradiddle: has all 4 stroke types (grace, accent, tap, diddle)
        rudiment = load_rudiment(
            Path(__file__).parent.parent
            / "dataset_gen"
            / "rudiments"
            / "definitions"
            / "24_flam_paradiddle.yaml"
        )
        performance = gen.generate(rudiment, profile, tempo_bpm=120, num_cycles=4)

        velocities = [e.intended_velocity for e in performance.strokes]
        vel_range = max(velocities) - min(velocities)
        assert vel_range >= 70, f"Velocity range {vel_range} too narrow (need >= 70)"


class TestConvenienceFunction:
    """Tests for the convenience generate_performance function."""

    def test_generate_performance_function(self, simple_rudiment):
        """Test the standalone generate_performance function."""
        profile = generate_profile(SkillTier.INTERMEDIATE, rng=np.random.default_rng(42))

        performance = generate_performance(
            rudiment=simple_rudiment,
            profile=profile,
            tempo_bpm=120,
            num_cycles=4,
            seed=42,
        )

        assert isinstance(performance, GeneratedPerformance)
        assert len(performance.strokes) > 0
        assert performance.midi_data is not None


class TestStrokeEvent:
    """Tests for StrokeEvent dataclass."""

    def test_timing_error_calculation(self):
        """Test timing error property."""
        event = StrokeEvent(
            index=0,
            hand=Hand.RIGHT,
            stroke_type=StrokeType.TAP,
            intended_time_ms=100.0,
            actual_time_ms=105.0,
            intended_velocity=80,
            actual_velocity=85,
        )

        assert event.timing_error_ms == 5.0

    def test_velocity_error_calculation(self):
        """Test velocity error property."""
        event = StrokeEvent(
            index=0,
            hand=Hand.RIGHT,
            stroke_type=StrokeType.TAP,
            intended_time_ms=100.0,
            actual_time_ms=100.0,
            intended_velocity=80,
            actual_velocity=75,
        )

        assert event.velocity_error == -5
