"""
SFZ file parser for drum soundfonts.

Parses SFZ text format into structured region objects with resolved opcodes,
supporting the subset of SFZ features needed for drum sample playback:
- Velocity layers (lovel/hivel)
- Key mapping (lokey/hikey/key)
- Round robin (sequential via seq_position/seq_length, random via lorand/hirand)
- Volume/velocity curves (volume, amp_veltrack, amp_velcurve)
- Hierarchy resolution (control/global/master/group/region)
- Preprocessor directives (#define, #include)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

# SFZ hierarchy levels in order of precedence (region overrides all)
HEADER_NAMES = ("control", "global", "master", "group", "region")
HEADER_PATTERN = re.compile(r"<(" + "|".join(HEADER_NAMES) + r")>")

# Opcodes we care about for drum rendering
NUMERIC_OPCODES = {
    "lovel",
    "hivel",
    "lokey",
    "hikey",
    "key",
    "pitch_keycenter",
    "seq_position",
    "seq_length",
    "volume",
    "amp_veltrack",
    "tune",
    "transpose",
    "offset",
    "end",
    "loop_start",
    "loop_end",
    "group",
    "off_by",
    "output",
}

FLOAT_OPCODES = {
    "lorand",
    "hirand",
    "volume",
    "amp_veltrack",
    "tune",
}


@dataclass
class SfzRegion:
    """A single SFZ region with resolved opcodes."""

    sample: str = ""
    lovel: int = 0
    hivel: int = 127
    lokey: int = 0
    hikey: int = 127
    key: int | None = None  # Shorthand that sets lokey=hikey=pitch_keycenter
    pitch_keycenter: int | None = None
    seq_position: int = 0  # 0 = not using sequential RR
    seq_length: int = 0
    lorand: float = 0.0
    hirand: float = 1.0
    volume: float = 0.0  # dB
    amp_veltrack: float = 100.0  # 0-100, sensitivity to velocity
    amp_velcurve: dict[int, float] = field(default_factory=dict)
    # Raw opcodes for anything we don't explicitly handle
    opcodes: dict[str, str] = field(default_factory=dict)

    @property
    def effective_lokey(self) -> int:
        if self.key is not None:
            return self.key
        return self.lokey

    @property
    def effective_hikey(self) -> int:
        if self.key is not None:
            return self.key
        return self.hikey

    def matches(self, note: int, velocity: int) -> bool:
        """Check if this region matches the given note and velocity."""
        if not (self.effective_lokey <= note <= self.effective_hikey):
            return False
        if not (self.lovel <= velocity <= self.hivel):
            return False
        return True

    def matches_seq(self, seq_counter: int) -> bool:
        """Check if this region matches the sequential round-robin counter."""
        if self.seq_length == 0 or self.seq_position == 0:
            return True  # No sequential RR
        return self.seq_position == ((seq_counter % self.seq_length) + 1)

    def matches_rand(self, rand_value: float) -> bool:
        """Check if this region matches the random round-robin value."""
        return self.lorand <= rand_value < self.hirand


@dataclass
class SfzInstrument:
    """Parsed SFZ instrument with all regions and metadata."""

    regions: list[SfzRegion]
    base_path: Path  # Directory containing the .sfz file
    default_path: str = ""  # From <control> default_path opcode
    source_file: Path | None = None

    def get_regions_for_note(
        self,
        note: int,
        velocity: int,
        seq_counter: int = 0,
        rand_value: float = 0.0,
    ) -> list[SfzRegion]:
        """Get all regions that match the given note event."""
        matches = []
        for region in self.regions:
            if not region.sample:
                continue
            if not region.matches(note, velocity):
                continue
            if not region.matches_seq(seq_counter):
                continue
            if not region.matches_rand(rand_value):
                continue
            matches.append(region)
        return matches

    def resolve_sample_path(self, region: SfzRegion) -> Path:
        """Resolve a region's sample path to an absolute filesystem path."""
        # Normalize Windows backslashes
        sample = region.sample.replace("\\", "/")
        # Prepend default_path if sample is relative
        if self.default_path and not Path(sample).is_absolute():
            sample = self.default_path.replace("\\", "/").rstrip("/") + "/" + sample
        return self.base_path / sample

    def get_mapped_notes(self) -> set[int]:
        """Get all MIDI notes that have at least one region mapped."""
        notes = set()
        for region in self.regions:
            if not region.sample:
                continue
            lo = region.effective_lokey
            hi = region.effective_hikey
            for n in range(lo, hi + 1):
                notes.add(n)
        return notes


class SfzParser:
    """
    Parser for SFZ soundfont files.

    Handles:
    - Comment stripping (//)
    - Macro expansion (#define $VAR value)
    - File inclusion (#include "path")
    - Hierarchy resolution (control > global > master > group > region)
    - Opcode parsing including amp_velcurve_N=V
    """

    def parse(self, sfz_path: Path | str) -> SfzInstrument:
        """Parse an SFZ file into an SfzInstrument."""
        sfz_path = Path(sfz_path)
        if not sfz_path.exists():
            raise FileNotFoundError(f"SFZ file not found: {sfz_path}")

        base_path = sfz_path.parent
        text = self._preprocess(sfz_path, base_path)
        sections = self._tokenize(text)
        regions = self._resolve_hierarchy(sections)

        # Extract default_path from control section
        default_path = ""
        for header, opcodes in sections:
            if header == "control" and "default_path" in opcodes:
                default_path = opcodes["default_path"]
                break

        instrument = SfzInstrument(
            regions=regions,
            base_path=base_path,
            default_path=default_path,
            source_file=sfz_path,
        )

        logger.info(
            f"Parsed {sfz_path.name}: {len(regions)} regions, "
            f"notes: {sorted(instrument.get_mapped_notes())}"
        )
        return instrument

    def _preprocess(
        self,
        sfz_path: Path,
        base_path: Path,
        macros: dict[str, str] | None = None,
        _depth: int = 0,
    ) -> str:
        """Preprocess SFZ text: strip comments, expand macros, inline includes."""
        if _depth > 20:
            raise RecursionError(f"Too many nested #include levels (>{_depth})")

        if macros is None:
            macros = {}

        text = sfz_path.read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        output_lines = []

        for line in lines:
            # Strip // comments (but not inside quoted strings for sample paths)
            comment_idx = line.find("//")
            if comment_idx >= 0:
                line = line[:comment_idx]

            stripped = line.strip()

            # Handle #define
            if stripped.startswith("#define"):
                parts = stripped.split(None, 2)
                if len(parts) >= 3:
                    macro_name = parts[1]
                    macro_value = parts[2]
                    macros[macro_name] = macro_value
                continue

            # Handle #include
            if stripped.startswith("#include"):
                match = re.match(r'#include\s+"([^"]+)"', stripped)
                if match:
                    include_rel = match.group(1).replace("\\", "/")
                    include_path = sfz_path.parent / include_rel
                    if include_path.exists():
                        included = self._preprocess(include_path, base_path, macros, _depth + 1)
                        output_lines.append(included)
                    else:
                        logger.warning(f"Include file not found: {include_path}")
                continue

            # Expand macros
            for macro_name, macro_value in macros.items():
                if macro_name in line:
                    line = line.replace(macro_name, macro_value)

            output_lines.append(line)

        return "\n".join(output_lines)

    def _tokenize(self, text: str) -> list[tuple[str, dict[str, str]]]:
        """
        Split preprocessed text into (header, opcodes) sections.

        Returns list of (header_name, {opcode: value}) tuples.
        """
        sections: list[tuple[str, dict[str, str]]] = []
        current_header: str | None = None
        current_opcodes: dict[str, str] = {}

        # Split text into tokens, preserving header boundaries
        # We need to handle opcodes that may span the same line as headers
        pos = 0
        while pos < len(text):
            # Look for next header
            match = HEADER_PATTERN.search(text, pos)
            if match is None:
                # Parse remaining text as opcodes for current section
                remaining = text[pos:]
                self._parse_opcodes(remaining, current_opcodes)
                break

            # Parse opcodes between previous position and this header
            before = text[pos : match.start()]
            if before.strip():
                self._parse_opcodes(before, current_opcodes)

            # Save previous section
            if current_header is not None:
                sections.append((current_header, current_opcodes))

            # Start new section
            current_header = match.group(1)
            current_opcodes = {}
            pos = match.end()

        # Save final section
        if current_header is not None:
            sections.append((current_header, current_opcodes))

        return sections

    def _parse_opcodes(self, text: str, opcodes: dict[str, str]) -> None:
        """Parse key=value opcode pairs from text into the opcodes dict."""
        # SFZ opcodes: key=value separated by whitespace
        # Special case: sample= can contain spaces (it runs to the next opcode or EOL)
        # We handle this by finding all key= patterns and splitting between them

        # Find all opcode positions (key=)
        opcode_positions = list(re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=", text))

        for i, match in enumerate(opcode_positions):
            key = match.group(1)
            value_start = match.end()

            # Value extends to the next opcode key= or end of text
            if i + 1 < len(opcode_positions):
                value_end = opcode_positions[i + 1].start()
            else:
                value_end = len(text)

            value = text[value_start:value_end].strip()

            # Strip trailing whitespace and newlines from value
            value = value.strip()

            opcodes[key] = value

    def _resolve_hierarchy(self, sections: list[tuple[str, dict[str, str]]]) -> list[SfzRegion]:
        """
        Resolve SFZ hierarchy: each <region> inherits from parent levels.

        Hierarchy (most to least specific):
        region > group > master > global > control
        """
        # Context stack: accumulated opcodes at each level
        context: dict[str, dict[str, str]] = {
            "control": {},
            "global": {},
            "master": {},
            "group": {},
        }

        regions: list[SfzRegion] = []

        for header, opcodes in sections:
            if header == "region":
                # Merge all parent contexts + region opcodes
                merged = {}
                for level in ("control", "global", "master", "group"):
                    merged.update(context[level])
                merged.update(opcodes)

                region = self._opcodes_to_region(merged)
                if region.sample:
                    regions.append(region)
            else:
                # Update context at this level (and clear lower levels)
                context[header] = opcodes
                # When a higher-level header appears, reset lower levels
                levels = list(context.keys())
                header_idx = levels.index(header) if header in levels else -1
                if header_idx >= 0:
                    for lower in levels[header_idx + 1 :]:
                        context[lower] = {}

        return regions

    def _opcodes_to_region(self, opcodes: dict[str, str]) -> SfzRegion:
        """Convert a flat opcode dict to an SfzRegion."""
        region = SfzRegion()
        velcurve: dict[int, float] = {}

        for key, value in opcodes.items():
            # Handle amp_velcurve_N=V specially
            velcurve_match = re.match(r"amp_velcurve_(\d+)", key)
            if velcurve_match:
                vel = int(velcurve_match.group(1))
                try:
                    velcurve[vel] = float(value)
                except ValueError:
                    pass
                continue

            # Skip CC-controlled opcodes (we override to velocity-based)
            if key.startswith("amplitude_oncc") or key.startswith("amplitude_curvecc"):
                continue

            # Map known opcodes
            try:
                if key == "sample":
                    region.sample = value
                elif key == "lovel":
                    region.lovel = int(value)
                elif key == "hivel":
                    region.hivel = int(value)
                elif key == "lokey":
                    region.lokey = self._parse_note(value)
                elif key == "hikey":
                    region.hikey = self._parse_note(value)
                elif key == "key":
                    region.key = self._parse_note(value)
                elif key == "pitch_keycenter":
                    region.pitch_keycenter = self._parse_note(value)
                elif key == "seq_position":
                    region.seq_position = int(value)
                elif key == "seq_length":
                    region.seq_length = int(value)
                elif key == "lorand":
                    region.lorand = float(value)
                elif key == "hirand":
                    region.hirand = float(value)
                elif key == "volume":
                    region.volume = float(value)
                elif key == "amp_veltrack":
                    region.amp_veltrack = float(value)
                else:
                    region.opcodes[key] = value
            except (ValueError, TypeError):
                region.opcodes[key] = value

        if velcurve:
            region.amp_velcurve = velcurve

        return region

    def _parse_note(self, value: str) -> int:
        """Parse a MIDI note value (numeric or note name like c#4)."""
        # Try numeric first
        try:
            return int(value)
        except ValueError:
            pass

        # Parse note name (e.g., c4, c#4, db4)
        value = value.strip().lower()
        note_names = {
            "c": 0,
            "d": 2,
            "e": 4,
            "f": 5,
            "g": 7,
            "a": 9,
            "b": 11,
        }

        match = re.match(r"([a-g])(#|b)?(-?\d+)", value)
        if match:
            name, accidental, octave = match.groups()
            midi = note_names[name] + (int(octave) + 1) * 12
            if accidental == "#":
                midi += 1
            elif accidental == "b":
                midi -= 1
            return midi

        logger.warning(f"Could not parse note value: '{value}', defaulting to 0")
        return 0  # MIDI note 0 (C-1) — effectively unmapped for drums
