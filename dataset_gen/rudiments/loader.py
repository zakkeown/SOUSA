"""
Load rudiment definitions from YAML files.
"""

from __future__ import annotations

from pathlib import Path
import yaml

from dataset_gen.rudiments.schema import (
    Rudiment,
    StickingPattern,
    Stroke,
    StrokeType,
    Hand,
    RudimentCategory,
    RudimentParams,
    Subdivision,
)

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


def _parse_stroke_notation(notation: str) -> Stroke:
    """
    Parse a single stroke notation character.

    Notation:
        R/L = right/left tap
        r/l = right/left tap (same as R/L for backward compat)
        R>/L> or >R/>L = right/left accent
        (R)/(L) = grace note
        RR/LL = first stroke of diddle (paired with next same-hand)
        rr/ll = second stroke of diddle
    """
    notation = notation.strip()

    # Grace notes: (R) or (L)
    if notation.startswith("(") and notation.endswith(")"):
        hand_char = notation[1]
        hand = Hand.RIGHT if hand_char.upper() == "R" else Hand.LEFT
        return Stroke(hand=hand, stroke_type=StrokeType.GRACE, grace_offset=-0.05)

    # Check for accent marker
    is_accent = ">" in notation
    hand_char = notation.replace(">", "").upper()

    if hand_char not in ("R", "L"):
        raise ValueError(f"Invalid hand character in notation: {notation}")

    hand = Hand.RIGHT if hand_char == "R" else Hand.LEFT
    stroke_type = StrokeType.ACCENT if is_accent else StrokeType.TAP

    return Stroke(hand=hand, stroke_type=stroke_type)


def _parse_pattern_from_yaml(pattern_data: dict) -> StickingPattern:
    """
    Parse a pattern definition from YAML data.

    Supports two formats:
    1. Simple string: "R L R R L R L L" with optional accents
    2. Detailed list with stroke types
    """
    if "simple" in pattern_data:
        # Simple string format
        sticking = pattern_data["simple"]
        accents = pattern_data.get("accents")
        strokes = []

        sticking_chars = sticking.replace(" ", "")
        accent_chars = accents.replace(" ", "") if accents else None

        if accent_chars and len(accent_chars) != len(sticking_chars):
            raise ValueError("Accent pattern must match sticking pattern length")

        hand_toggle = True  # Start with RIGHT for buzz strokes
        for i, char in enumerate(sticking_chars):
            upper = char.upper()
            if upper == "B":
                hand = Hand.RIGHT if hand_toggle else Hand.LEFT
                hand_toggle = not hand_toggle
                stroke_type = StrokeType.BUZZ
            else:
                hand = Hand.RIGHT if upper == "R" else Hand.LEFT
                is_accent = accent_chars and accent_chars[i] == ">"
                stroke_type = StrokeType.ACCENT if is_accent else StrokeType.TAP
            strokes.append(Stroke(hand=hand, stroke_type=stroke_type))

        beats = pattern_data.get("beats_per_cycle", len(strokes) / 4)
        return StickingPattern(strokes=strokes, beats_per_cycle=beats)

    elif "strokes" in pattern_data:
        # Detailed stroke list format
        strokes = []
        for stroke_data in pattern_data["strokes"]:
            if isinstance(stroke_data, str):
                strokes.append(_parse_stroke_notation(stroke_data))
            else:
                # Dict format with full stroke definition
                hand = Hand(stroke_data["hand"])
                stroke_type = StrokeType(stroke_data.get("type", "tap"))
                stroke = Stroke(
                    hand=hand,
                    stroke_type=stroke_type,
                    grace_offset=stroke_data.get("grace_offset"),
                    diddle_position=stroke_data.get("diddle_position"),
                )
                strokes.append(stroke)

        beats = pattern_data.get("beats_per_cycle", len(strokes) / 4)
        return StickingPattern(strokes=strokes, beats_per_cycle=beats)

    else:
        raise ValueError("Pattern must have 'simple' or 'strokes' key")


def _parse_rudiment_params(params_data: dict | None) -> RudimentParams:
    """Parse rudiment-specific parameters from YAML data."""
    if not params_data:
        return RudimentParams()

    return RudimentParams(
        flam_spacing_range=(
            tuple(params_data["flam_spacing_range"])
            if "flam_spacing_range" in params_data
            else None
        ),
        diddle_ratio_range=(
            tuple(params_data["diddle_ratio_range"])
            if "diddle_ratio_range" in params_data
            else None
        ),
        roll_type=params_data.get("roll_type"),
        roll_strokes_per_beat=params_data.get("roll_strokes_per_beat"),
        buzz_strokes_range=(
            tuple(params_data["buzz_strokes_range"])
            if "buzz_strokes_range" in params_data
            else None
        ),
        buzz_detail=params_data.get("buzz_detail"),
        drag_spacing_range=(
            tuple(params_data["drag_spacing_range"])
            if "drag_spacing_range" in params_data
            else None
        ),
    )


def load_rudiment(path: Path | str) -> Rudiment:
    """
    Load a single rudiment definition from a YAML file.

    Args:
        path: Path to the YAML file

    Returns:
        Parsed Rudiment object
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    pattern = _parse_pattern_from_yaml(data["pattern"])
    params = _parse_rudiment_params(data.get("params"))

    return Rudiment(
        name=data["name"],
        slug=data["slug"],
        category=RudimentCategory(data["category"]),
        pattern=pattern,
        subdivision=Subdivision(data.get("subdivision", "sixteenth")),
        tempo_range=tuple(data.get("tempo_range", [60, 180])),
        params=params,
        pas_number=data.get("pas_number"),
        description=data.get("description"),
        starts_on_left=data.get("starts_on_left", False),
    )


def load_all_rudiments(directory: Path | str | None = None) -> dict[str, Rudiment]:
    """
    Load all rudiment definitions from a directory.

    Args:
        directory: Path to directory containing YAML files.
                   Defaults to the built-in definitions directory.

    Returns:
        Dict mapping slug to Rudiment object
    """
    if directory is None:
        directory = DEFINITIONS_DIR
    directory = Path(directory)

    if not directory.exists():
        return {}

    rudiments = {}
    for yaml_file in directory.glob("*.yaml"):
        try:
            rudiment = load_rudiment(yaml_file)
            rudiments[rudiment.slug] = rudiment
        except Exception as e:
            raise ValueError(f"Failed to load {yaml_file}: {e}") from e

    return rudiments


def get_rudiments_by_category(
    rudiments: dict[str, Rudiment] | None = None,
) -> dict[RudimentCategory, list[Rudiment]]:
    """
    Group rudiments by category.

    Args:
        rudiments: Dict of rudiments. If None, loads all definitions.

    Returns:
        Dict mapping category to list of rudiments
    """
    if rudiments is None:
        rudiments = load_all_rudiments()

    by_category: dict[RudimentCategory, list[Rudiment]] = {cat: [] for cat in RudimentCategory}

    for rudiment in rudiments.values():
        by_category[rudiment.category].append(rudiment)

    return by_category
