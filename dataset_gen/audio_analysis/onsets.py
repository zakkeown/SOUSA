"""Audio onset detection data types."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DetectedOnset"]


@dataclass
class DetectedOnset:
    """A detected audio onset event.

    Attributes:
        time_sec: Onset time in seconds from the start of the audio.
        strength: Onset detection strength/confidence (0.0-1.0).
    """

    time_sec: float
    strength: float
