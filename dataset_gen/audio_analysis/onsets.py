"""Audio onset detection with madmom (preferred) and librosa fallback."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from madmom.features.onsets import OnsetPeakPickingProcessor, RNNOnsetProcessor

    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False

__all__ = ["DetectedOnset", "detect_onsets", "get_onset_activation"]


@dataclass
class DetectedOnset:
    """A detected audio onset event.

    Attributes:
        time_sec: Onset time in seconds from the start of the audio.
        strength: Onset detection strength/confidence (0.0-1.0).
    """

    time_sec: float
    strength: float


def _load_audio_path(
    audio: str | Path | np.ndarray,
    sample_rate: int,
) -> tuple[Path | None, np.ndarray, int, bool]:
    """Resolve audio input to a file path and/or numpy array.

    Returns:
        (path_or_none, audio_array, sample_rate, is_temp_file)
    """
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        import soundfile as sf

        y, sr = sf.read(str(path), always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return path, y, sr, False

    # numpy array input -- no file on disk yet
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return None, y, sample_rate, False


def _write_temp_wav(y: np.ndarray, sr: int) -> Path:
    """Write a numpy array to a temporary WAV file (needed by madmom)."""
    import soundfile as sf

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    tmp.close()
    return Path(tmp.name)


def _combine_onsets(
    onsets: list[DetectedOnset],
    combine_ms: float,
) -> list[DetectedOnset]:
    """Merge onsets within *combine_ms* of each other, keeping the stronger one."""
    if not onsets:
        return []
    sorted_onsets = sorted(onsets, key=lambda o: o.time_sec)
    combined: list[DetectedOnset] = [sorted_onsets[0]]
    for onset in sorted_onsets[1:]:
        gap_sec = onset.time_sec - combined[-1].time_sec
        if gap_sec * 1000.0 < combine_ms:
            # Keep the stronger of the two
            if onset.strength > combined[-1].strength:
                combined[-1] = onset
        else:
            combined.append(onset)
    return combined


# ---------------------------------------------------------------------------
# madmom backend
# ---------------------------------------------------------------------------


def _detect_onsets_madmom(
    path: Path | None,
    y: np.ndarray,
    sr: int,
    threshold: float,
    combine_ms: float,
) -> list[DetectedOnset]:
    """Detect onsets using madmom RNNOnsetProcessor."""
    temp_file: Path | None = None
    try:
        if path is None:
            temp_file = _write_temp_wav(y, sr)
            path = temp_file

        proc = RNNOnsetProcessor()
        activation = proc(str(path))

        picker = OnsetPeakPickingProcessor(
            threshold=threshold,
            combine=combine_ms / 1000.0,
            fps=100,
        )
        onset_times = picker(activation)

        fps = 100
        onsets: list[DetectedOnset] = []
        for t in onset_times:
            frame_idx = int(round(t * fps))
            frame_idx = min(frame_idx, len(activation) - 1)
            strength = float(activation[frame_idx])
            onsets.append(DetectedOnset(time_sec=float(t), strength=strength))
        return onsets
    finally:
        if temp_file is not None:
            temp_file.unlink(missing_ok=True)


def _activation_madmom(
    path: Path | None,
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, int]:
    """Return raw onset activation from madmom (100 fps)."""
    temp_file: Path | None = None
    try:
        if path is None:
            temp_file = _write_temp_wav(y, sr)
            path = temp_file

        proc = RNNOnsetProcessor()
        activation = proc(str(path))
        return np.asarray(activation, dtype=np.float64), 100
    finally:
        if temp_file is not None:
            temp_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# librosa backend
# ---------------------------------------------------------------------------

_LIBROSA_HOP = 64


def _detect_onsets_librosa(
    path: Path | None,
    y: np.ndarray,
    sr: int,
    threshold: float,
    combine_ms: float,
) -> list[DetectedOnset]:
    """Detect onsets using librosa with tight hop length."""
    import librosa

    if path is not None:
        y, sr = librosa.load(str(path), sr=None, mono=True)

    hop = _LIBROSA_HOP
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop,
        onset_envelope=onset_env,
        delta=threshold,
        backtrack=False,
    )

    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)

    # Normalise envelope to 0-1 for strength values
    env_max = onset_env.max() if onset_env.max() > 0 else 1.0
    onsets: list[DetectedOnset] = []
    for frame, t in zip(frames, times):
        strength = float(onset_env[frame] / env_max)
        onsets.append(DetectedOnset(time_sec=float(t), strength=strength))

    return _combine_onsets(onsets, combine_ms)


def _activation_librosa(
    path: Path | None,
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, int]:
    """Return onset strength envelope from librosa."""
    import librosa

    if path is not None:
        y, sr = librosa.load(str(path), sr=None, mono=True)

    hop = _LIBROSA_HOP
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    fps = sr // hop
    return np.asarray(onset_env, dtype=np.float64), fps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_onsets(
    audio: str | Path | np.ndarray,
    sample_rate: int = 44100,
    method: str = "auto",
    threshold: float = 0.3,
    combine_ms: float = 20.0,
) -> list[DetectedOnset]:
    """Detect audio onsets using madmom (preferred) or librosa.

    Args:
        audio: File path (str or Path) or numpy array of audio samples.
        sample_rate: Sample rate of the audio (used only when *audio* is an array).
        method: Detection backend -- ``"auto"`` (madmom if available, else librosa),
            ``"madmom"``, or ``"librosa"``.
        threshold: Peak-picking / detection threshold.
        combine_ms: Merge onsets closer than this many milliseconds (keep stronger).

    Returns:
        Sorted list of :class:`DetectedOnset` instances.
    """
    path, y, sr, _ = _load_audio_path(audio, sample_rate)

    use_madmom = False
    if method == "auto":
        use_madmom = MADMOM_AVAILABLE
    elif method == "madmom":
        if not MADMOM_AVAILABLE:
            raise ImportError("madmom is not installed")
        use_madmom = True
    elif method == "librosa":
        use_madmom = False
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if use_madmom:
        return _detect_onsets_madmom(path, y, sr, threshold, combine_ms)
    return _detect_onsets_librosa(path, y, sr, threshold, combine_ms)


def get_onset_activation(
    audio: str | Path | np.ndarray,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, int]:
    """Return the raw onset activation function.

    Uses madmom if available (100 fps), otherwise librosa onset_strength
    (fps = sr // hop_length).

    Args:
        audio: File path (str or Path) or numpy array of audio samples.
        sample_rate: Sample rate (used only when *audio* is an array).

    Returns:
        Tuple of (activation_array, frames_per_second).
    """
    path, y, sr, _ = _load_audio_path(audio, sample_rate)

    if MADMOM_AVAILABLE:
        return _activation_madmom(path, y, sr)
    return _activation_librosa(path, y, sr)
