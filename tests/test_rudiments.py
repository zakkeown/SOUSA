"""Tests for rudiment schema and loading."""

import pytest
from pathlib import Path

from dataset_gen.rudiments.schema import (
    Rudiment,
    StickingPattern,
    Stroke,
    StrokeType,
    Hand,
    RudimentCategory,
    parse_sticking_string,
    create_flam_pattern,
    create_diddle,
)
from dataset_gen.rudiments.loader import load_rudiment, load_all_rudiments


class TestStickingPattern:
    """Tests for sticking pattern parsing."""

    def test_parse_simple_sticking(self):
        """Test parsing a simple sticking string."""
        pattern = parse_sticking_string("RLRL")
        assert len(pattern.strokes) == 4
        assert pattern.strokes[0].hand == Hand.RIGHT
        assert pattern.strokes[1].hand == Hand.LEFT
        assert pattern.strokes[2].hand == Hand.RIGHT
        assert pattern.strokes[3].hand == Hand.LEFT

    def test_parse_sticking_with_accents(self):
        """Test parsing sticking with accent markers."""
        pattern = parse_sticking_string("RLRL", ">..>")
        assert pattern.strokes[0].stroke_type == StrokeType.ACCENT
        assert pattern.strokes[1].stroke_type == StrokeType.TAP
        assert pattern.strokes[2].stroke_type == StrokeType.TAP
        assert pattern.strokes[3].stroke_type == StrokeType.ACCENT

    def test_parse_sticking_with_spaces(self):
        """Test that spaces in sticking are handled."""
        pattern = parse_sticking_string("RL RR LR LL")
        assert len(pattern.strokes) == 8

    def test_parse_buzz_notation(self):
        """Test that B in simple pattern produces BUZZ stroke type."""
        from dataset_gen.rudiments.loader import _parse_pattern_from_yaml

        pattern = _parse_pattern_from_yaml(
            {
                "simple": "B B B B",
                "beats_per_cycle": 1,
            }
        )
        assert len(pattern.strokes) == 4
        # B alternates hands like R/L
        assert all(s.stroke_type == StrokeType.BUZZ for s in pattern.strokes)
        assert pattern.strokes[0].hand == Hand.RIGHT
        assert pattern.strokes[1].hand == Hand.LEFT

    def test_accent_positions(self):
        """Test accent position detection."""
        pattern = parse_sticking_string("RLRR LRLL", ">... >...")
        positions = pattern.accent_positions()
        assert positions == [0, 4]


class TestHelperFunctions:
    """Tests for pattern creation helpers."""

    def test_create_flam_pattern(self):
        """Test flam pattern creation."""
        flam = create_flam_pattern(Hand.RIGHT, accent=True)
        assert len(flam) == 2
        assert flam[0].hand == Hand.LEFT  # Grace note on opposite hand
        assert flam[0].stroke_type == StrokeType.GRACE
        assert flam[1].hand == Hand.RIGHT
        assert flam[1].stroke_type == StrokeType.ACCENT

    def test_create_diddle(self):
        """Test diddle creation."""
        diddle = create_diddle(Hand.RIGHT)
        assert len(diddle) == 2
        assert diddle[0].hand == Hand.RIGHT
        assert diddle[0].diddle_position == 1
        assert diddle[1].hand == Hand.RIGHT
        assert diddle[1].diddle_position == 2


class TestRudimentLoading:
    """Tests for loading rudiment definitions from YAML."""

    @pytest.fixture
    def definitions_dir(self):
        """Get path to rudiment definitions."""
        return Path(__file__).parent.parent / "dataset_gen" / "rudiments" / "definitions"

    def test_load_single_paradiddle(self, definitions_dir):
        """Test loading the single paradiddle rudiment."""
        yaml_path = definitions_dir / "16_single_paradiddle.yaml"
        if not yaml_path.exists():
            pytest.skip("Rudiment definition not found")

        rudiment = load_rudiment(yaml_path)
        assert rudiment.name == "Single Paradiddle"
        assert rudiment.slug == "single_paradiddle"
        assert rudiment.category == RudimentCategory.DIDDLE
        assert rudiment.pas_number == 16

        # Check pattern
        assert len(rudiment.pattern.strokes) == 8

    def test_load_all_rudiments(self, definitions_dir):
        """Test loading all rudiment definitions."""
        if not definitions_dir.exists():
            pytest.skip("Definitions directory not found")

        rudiments = load_all_rudiments(definitions_dir)
        assert len(rudiments) > 0

        # Check that we have all categories
        categories = {r.category for r in rudiments.values()}
        assert RudimentCategory.ROLL in categories
        assert RudimentCategory.DIDDLE in categories
        assert RudimentCategory.FLAM in categories
        assert RudimentCategory.DRAG in categories

    def test_double_drag_tap_has_12_strokes(self, definitions_dir):
        """Double drag tap needs 2 drags (4 grace notes) + accent + tap per half-cycle = 12 total."""
        rudiment = load_rudiment(definitions_dir / "33_double_drag_tap.yaml")
        assert rudiment.slug == "double_drag_tap"
        assert len(rudiment.pattern.strokes) == 12

        # First half: drag1(LL) drag2(LL) accent(R) tap(L)
        types = [s.stroke_type for s in rudiment.pattern.strokes[:6]]
        assert types == [
            StrokeType.GRACE,
            StrokeType.GRACE,  # drag 1
            StrokeType.GRACE,
            StrokeType.GRACE,  # drag 2
            StrokeType.ACCENT,
            StrokeType.TAP,
        ]
        # Second half mirrors
        types2 = [s.stroke_type for s in rudiment.pattern.strokes[6:]]
        assert types2 == [
            StrokeType.GRACE,
            StrokeType.GRACE,
            StrokeType.GRACE,
            StrokeType.GRACE,
            StrokeType.ACCENT,
            StrokeType.TAP,
        ]

    def test_swiss_army_triplet_sticking_differs_from_flam_accent(self, definitions_dir):
        """Swiss army triplet has doubled last stroke: R L L | L R R (not R L R | L R L)."""
        swiss = load_rudiment(definitions_dir / "28_swiss_army_triplet.yaml")
        flam_accent = load_rudiment(definitions_dir / "21_flam_accent.yaml")

        # Extract non-grace sticking from each
        def get_sticking(rudiment):
            return [s.hand for s in rudiment.pattern.strokes if s.stroke_type != StrokeType.GRACE]

        swiss_sticking = get_sticking(swiss)
        flam_sticking = get_sticking(flam_accent)

        # They must differ
        assert (
            swiss_sticking != flam_sticking
        ), "Swiss army triplet sticking should differ from flam accent"

        # Swiss army: R L L | L R R
        assert swiss_sticking == [
            Hand.RIGHT,
            Hand.LEFT,
            Hand.LEFT,
            Hand.LEFT,
            Hand.RIGHT,
            Hand.RIGHT,
        ]

    def test_multiple_bounce_roll_has_buzz_strokes(self, definitions_dir):
        """Multiple bounce roll strokes should be BUZZ type, not TAP."""
        rudiment = load_rudiment(definitions_dir / "04_multiple_bounce_roll.yaml")
        assert rudiment.slug == "multiple_bounce_roll"
        assert rudiment.params.roll_type == "buzz"
        assert rudiment.params.buzz_detail == "sub_strokes"

        # All strokes should be BUZZ type
        for stroke in rudiment.pattern.strokes:
            assert (
                stroke.stroke_type == StrokeType.BUZZ
            ), f"Expected BUZZ but got {stroke.stroke_type}"

    def test_rudiment_duration_calculation(self, definitions_dir):
        """Test duration calculation for rudiments."""
        yaml_path = definitions_dir / "16_single_paradiddle.yaml"
        if not yaml_path.exists():
            pytest.skip("Rudiment definition not found")

        rudiment = load_rudiment(yaml_path)
        duration = rudiment.duration_at_tempo(tempo_bpm=120, num_cycles=4)

        # At 120 BPM, 2 beats per cycle, 4 cycles = 8 beats = 4 seconds
        assert abs(duration - 4.0) < 0.1


class TestRudimentValidation:
    """Tests for rudiment validation logic."""

    def test_valid_slug(self):
        """Test that slug validation works."""
        # Valid slug
        rudiment = Rudiment(
            name="Test",
            slug="test_rudiment",
            category=RudimentCategory.ROLL,
            pattern=StickingPattern(
                strokes=[
                    Stroke(hand=Hand.RIGHT),
                    Stroke(hand=Hand.LEFT),
                ]
            ),
        )
        assert rudiment.slug == "test_rudiment"

    def test_invalid_slug_rejected(self):
        """Test that invalid slugs are rejected."""
        with pytest.raises(ValueError):
            Rudiment(
                name="Test",
                slug="Test Rudiment",  # Has space and capitals
                category=RudimentCategory.ROLL,
                pattern=StickingPattern(
                    strokes=[
                        Stroke(hand=Hand.RIGHT),
                    ]
                ),
            )

    def test_alternating_pattern(self):
        """Test getting alternating (hand-swapped) pattern."""
        pattern = parse_sticking_string("RLRR")
        rudiment = Rudiment(
            name="Test",
            slug="test",
            category=RudimentCategory.DIDDLE,
            pattern=pattern,
        )

        alt_pattern = rudiment.get_alternating_pattern()
        assert alt_pattern.strokes[0].hand == Hand.LEFT
        assert alt_pattern.strokes[1].hand == Hand.RIGHT
        assert alt_pattern.strokes[2].hand == Hand.LEFT
        assert alt_pattern.strokes[3].hand == Hand.LEFT
