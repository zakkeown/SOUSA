# Audio Analysis Tooling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build CLI + importable module that produces high-resolution PNG visualizations of snare rudiment audio with madmom neural onset detection and MIDI ground-truth overlay.

**Architecture:** New `dataset_gen/audio_analysis/` module with three sub-modules (onsets, midi_alignment, views) plus a CLI script at `scripts/analyze_audio.py`. Each view renders a PNG via matplotlib.

**Tech Stack:** madmom (RNN onset detection), mido (MIDI parsing), matplotlib (rendering), librosa (fallback onset detection), soundfile (audio I/O)

---

### Task 1: Add `[analysis]` dependency group to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add the analysis extra**

In `pyproject.toml`, add a new entry under `[project.optional-dependencies]`:

```toml
analysis = [
    "madmom>=0.16.1",
    "matplotlib>=3.7.0",
]
```

Place it alphabetically (after `[cloud]`, before `[dev]`).

**Step 2: Install and verify**

Run:
```bash
pip install -e '.[analysis]'
```

If madmom fails to install (known Python 3.10+ compatibility issues with `collections.abc`), try:
```bash
pip install madmom --no-build-isolation
```

If that also fails, install from the dev branch:
```bash
pip install git+https://github.com/CPJKU/madmom.git
```

Verify:
```bash
python -c "import madmom; print(madmom.__version__)"
python -c "import matplotlib; print(matplotlib.__version__)"
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat(analysis): add [analysis] dependency group with madmom"
```

---

### Task 2: Create `dataset_gen/audio_analysis/` module with data types

**Files:**
- Create: `dataset_gen/audio_analysis/__init__.py`
- Create: `dataset_gen/audio_analysis/onsets.py`
- Create: `dataset_gen/audio_analysis/midi_alignment.py`
- Create: `dataset_gen/audio_analysis/views.py`
- Test: `tests/test_audio_analysis.py`

**Step 1: Write the failing test for data types**

Create `tests/test_audio_analysis.py`:

```python
"""Tests for audio analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from dataset_gen.audio_analysis.onsets import DetectedOnset
from dataset_gen.audio_analysis.midi_alignment import MidiOnset


class TestDataTypes:
    """Test data types used across the analysis module."""

    def test_detected_onset_creation(self):
        onset = DetectedOnset(time_sec=0.5, strength=0.8)
        assert onset.time_sec == 0.5
        assert onset.strength == 0.8

    def test_midi_onset_creation(self):
        onset = MidiOnset(time_sec=0.5, velocity=100, note=38)
        assert onset.time_sec == 0.5
        assert onset.velocity == 100
        assert onset.note == 38
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_analysis.py::TestDataTypes -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

Create `dataset_gen/audio_analysis/__init__.py`:

```python
"""Audio analysis module for high-resolution visualization of drum rudiments."""

from dataset_gen.audio_analysis.midi_alignment import MidiOnset
from dataset_gen.audio_analysis.onsets import DetectedOnset

__all__ = [
    "DetectedOnset",
    "MidiOnset",
]
```

Create `dataset_gen/audio_analysis/onsets.py`:

```python
"""Onset detection using madmom (neural) with librosa fallback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectedOnset:
    """A detected onset in the audio signal."""

    time_sec: float
    strength: float
```

Create `dataset_gen/audio_analysis/midi_alignment.py`:

```python
"""MIDI file parsing for ground-truth onset alignment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MidiOnset:
    """A note-on event extracted from a MIDI file."""

    time_sec: float
    velocity: int
    note: int
```

Create empty `dataset_gen/audio_analysis/views.py`:

```python
"""PNG rendering for audio analysis views."""

from __future__ import annotations
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_analysis.py::TestDataTypes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/audio_analysis/ tests/test_audio_analysis.py
git commit -m "feat(analysis): add audio_analysis module with data types"
```

---

### Task 3: Implement MIDI onset extraction

**Files:**
- Modify: `dataset_gen/audio_analysis/midi_alignment.py`
- Test: `tests/test_audio_analysis.py`

**Step 1: Write the failing test**

Add to `tests/test_audio_analysis.py`:

```python
import tempfile
from pathlib import Path
import mido

from dataset_gen.audio_analysis.midi_alignment import extract_midi_onsets


class TestMidiAlignment:
    """Test MIDI onset extraction."""

    @pytest.fixture
    def simple_midi_file(self, tmp_path):
        """Create a simple MIDI file with known note events."""
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo to 120 BPM (500000 microseconds per beat)
        track.append(mido.MetaMessage("set_tempo", tempo=500000))

        # Four quarter notes at 120 BPM = 500ms apart
        # note=38 (snare), velocities: 100, 60, 100, 60 (accent/tap pattern)
        track.append(mido.Message("note_on", note=38, velocity=100, time=0))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        track.append(mido.Message("note_on", note=38, velocity=60, time=240))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        track.append(mido.Message("note_on", note=38, velocity=100, time=240))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))
        track.append(mido.Message("note_on", note=38, velocity=60, time=240))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))

        midi_path = tmp_path / "test.mid"
        mid.save(str(midi_path))
        return midi_path

    @pytest.fixture
    def flam_midi_file(self, tmp_path):
        """Create a MIDI file with a flam (grace note ~30ms before main note)."""
        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        track.append(mido.MetaMessage("set_tempo", tempo=500000))

        # Grace note: low velocity, then main note 30ms later
        # 30ms at 120BPM with 480 ticks/beat = ~29 ticks
        track.append(mido.Message("note_on", note=38, velocity=40, time=0))
        track.append(mido.Message("note_off", note=38, velocity=0, time=14))
        track.append(mido.Message("note_on", note=38, velocity=110, time=15))
        track.append(mido.Message("note_off", note=38, velocity=0, time=240))

        midi_path = tmp_path / "flam.mid"
        mid.save(str(midi_path))
        return midi_path

    def test_extract_basic_onsets(self, simple_midi_file):
        onsets = extract_midi_onsets(simple_midi_file)
        assert len(onsets) == 4
        # All notes should be snare (note 38)
        assert all(o.note == 38 for o in onsets)
        # Velocities alternate accent/tap
        assert [o.velocity for o in onsets] == [100, 60, 100, 60]

    def test_extract_onset_timing(self, simple_midi_file):
        onsets = extract_midi_onsets(simple_midi_file)
        # At 120 BPM, quarter notes are 500ms apart
        for i in range(1, len(onsets)):
            gap_ms = (onsets[i].time_sec - onsets[i - 1].time_sec) * 1000
            assert abs(gap_ms - 500.0) < 5.0, f"Gap {i}: {gap_ms}ms, expected ~500ms"

    def test_extract_flam_timing(self, flam_midi_file):
        onsets = extract_midi_onsets(flam_midi_file)
        assert len(onsets) == 2
        gap_ms = (onsets[1].time_sec - onsets[0].time_sec) * 1000
        # Grace note should be ~30ms before main note
        assert 25.0 < gap_ms < 35.0, f"Flam gap: {gap_ms}ms, expected ~30ms"

    def test_extract_from_path_string(self, simple_midi_file):
        """Accept both Path objects and strings."""
        onsets = extract_midi_onsets(str(simple_midi_file))
        assert len(onsets) == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_analysis.py::TestMidiAlignment -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Update `dataset_gen/audio_analysis/midi_alignment.py`:

```python
"""MIDI file parsing for ground-truth onset alignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido


@dataclass
class MidiOnset:
    """A note-on event extracted from a MIDI file."""

    time_sec: float
    velocity: int
    note: int


def extract_midi_onsets(midi_path: str | Path) -> list[MidiOnset]:
    """Extract note-on events from a MIDI file with absolute timestamps.

    Iterates through all tracks and returns onsets sorted by time.
    Uses mido's built-in tempo-aware iteration (mid.__iter__) which
    converts tick-based deltas to seconds automatically.

    Args:
        midi_path: Path to a .mid file.

    Returns:
        List of MidiOnset sorted by time_sec.
    """
    mid = mido.MidiFile(str(midi_path))
    onsets: list[MidiOnset] = []
    current_time = 0.0

    for msg in mid:
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            onsets.append(
                MidiOnset(time_sec=current_time, velocity=msg.velocity, note=msg.note)
            )

    onsets.sort(key=lambda o: o.time_sec)
    return onsets
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_analysis.py::TestMidiAlignment -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/audio_analysis/midi_alignment.py tests/test_audio_analysis.py
git commit -m "feat(analysis): implement MIDI onset extraction"
```

---

### Task 4: Implement onset detection (madmom + librosa fallback)

**Files:**
- Modify: `dataset_gen/audio_analysis/onsets.py`
- Test: `tests/test_audio_analysis.py`

**Step 1: Write the failing test**

Add to `tests/test_audio_analysis.py`:

```python
import soundfile as sf

from dataset_gen.audio_analysis.onsets import detect_onsets, DetectedOnset


class TestOnsetDetection:
    """Test onset detection on synthetic audio."""

    @pytest.fixture
    def click_audio(self, tmp_path):
        """Generate audio with 4 clear clicks spaced 500ms apart."""
        sr = 44100
        duration_sec = 2.5
        n_samples = int(sr * duration_sec)
        audio = np.zeros(n_samples)

        # Place sharp transients (5ms clicks) at 0.25, 0.75, 1.25, 1.75 seconds
        click_times = [0.25, 0.75, 1.25, 1.75]
        click_len = int(0.005 * sr)  # 5ms
        for t in click_times:
            start = int(t * sr)
            # Sharp attack, fast decay
            click = np.exp(-np.linspace(0, 10, click_len)) * 0.9
            audio[start : start + click_len] += click

        audio_path = tmp_path / "clicks.wav"
        sf.write(str(audio_path), audio, sr)
        return audio_path, click_times

    def test_detect_onsets_returns_list(self, click_audio):
        audio_path, _ = click_audio
        onsets = detect_onsets(audio_path)
        assert isinstance(onsets, list)
        assert all(isinstance(o, DetectedOnset) for o in onsets)

    def test_detect_onsets_finds_clicks(self, click_audio):
        audio_path, expected_times = click_audio
        onsets = detect_onsets(audio_path)
        # Should find approximately 4 onsets
        assert 3 <= len(onsets) <= 6, f"Expected ~4 onsets, got {len(onsets)}"

    def test_detect_onsets_timing_accuracy(self, click_audio):
        audio_path, expected_times = click_audio
        onsets = detect_onsets(audio_path)
        detected_times = [o.time_sec for o in onsets]

        # Each expected click should have a detected onset within 50ms
        for expected in expected_times:
            closest = min(detected_times, key=lambda t: abs(t - expected))
            error_ms = abs(closest - expected) * 1000
            assert error_ms < 50, f"Onset at {expected}s: closest detection {closest}s ({error_ms}ms off)"

    def test_detect_onsets_from_array(self, click_audio):
        """Accept numpy array + sample rate instead of file path."""
        audio_path, _ = click_audio
        audio, sr = sf.read(str(audio_path))
        onsets = detect_onsets(audio, sample_rate=sr)
        assert len(onsets) >= 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_analysis.py::TestOnsetDetection -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Update `dataset_gen/audio_analysis/onsets.py`:

```python
"""Onset detection using madmom (neural) with librosa fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import madmom
    from madmom.features.onsets import (
        CNNOnsetProcessor,
        RNNOnsetProcessor,
        OnsetPeakPickingProcessor,
    )

    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False

import librosa


@dataclass
class DetectedOnset:
    """A detected onset in the audio signal."""

    time_sec: float
    strength: float


def detect_onsets(
    audio: str | Path | np.ndarray,
    sample_rate: int = 44100,
    method: str = "auto",
    threshold: float = 0.3,
    combine_ms: float = 20.0,
) -> list[DetectedOnset]:
    """Detect onsets in an audio signal.

    Args:
        audio: Path to audio file, or numpy array of audio samples.
        sample_rate: Sample rate (only used if audio is a numpy array).
        method: Detection method - "madmom", "librosa", or "auto" (madmom if available).
        threshold: Peak-picking threshold (0-1). Lower = more sensitive.
        combine_ms: Combine onsets within this window (ms). Set low for flams (~20ms).

    Returns:
        List of DetectedOnset sorted by time_sec.
    """
    if method == "auto":
        method = "madmom" if MADMOM_AVAILABLE else "librosa"

    if method == "madmom":
        return _detect_madmom(audio, sample_rate, threshold, combine_ms)
    else:
        return _detect_librosa(audio, sample_rate, threshold, combine_ms)


def _detect_madmom(
    audio: str | Path | np.ndarray,
    sample_rate: int,
    threshold: float,
    combine_ms: float,
) -> list[DetectedOnset]:
    """Detect onsets using madmom's RNN processor."""
    if not MADMOM_AVAILABLE:
        raise ImportError("madmom is required for RNN onset detection: pip install madmom")

    # madmom needs a file path; if we have an array, write to temp file
    if isinstance(audio, np.ndarray):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            audio_path = f.name
        cleanup = True
    else:
        audio_path = str(audio)
        cleanup = False

    try:
        # Get onset activation function from RNN (100 fps)
        proc = RNNOnsetProcessor()
        activation = proc(audio_path)

        # Peak-pick to get onset times
        picker = OnsetPeakPickingProcessor(
            threshold=threshold,
            combine=combine_ms / 1000.0,
            fps=100,
        )
        onset_times = picker(activation)

        # Build result with activation strength at each onset
        onsets = []
        for t in onset_times:
            frame_idx = int(t * 100)
            strength = float(activation[min(frame_idx, len(activation) - 1)])
            onsets.append(DetectedOnset(time_sec=float(t), strength=strength))

        return onsets
    finally:
        if cleanup:
            Path(audio_path).unlink(missing_ok=True)


def _detect_librosa(
    audio: str | Path | np.ndarray,
    sample_rate: int,
    threshold: float,
    combine_ms: float,
) -> list[DetectedOnset]:
    """Detect onsets using librosa with tight parameters for percussion."""
    if isinstance(audio, (str, Path)):
        y, sr = librosa.load(str(audio), sr=None, mono=True)
    else:
        y = audio
        sr = sample_rate
        if y.ndim > 1:
            y = np.mean(y, axis=1)

    # Use very small hop for sub-ms resolution on tight rudiments
    hop_length = 64  # ~1.45ms at 44.1kHz

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        onset_envelope=onset_env,
        delta=threshold,
        backtrack=False,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Combine onsets within the combine window
    combine_sec = combine_ms / 1000.0
    onsets: list[DetectedOnset] = []
    for i, t in enumerate(onset_times):
        strength = float(onset_env[onset_frames[i]])
        if onsets and (t - onsets[-1].time_sec) < combine_sec:
            # Keep the stronger onset
            if strength > onsets[-1].strength:
                onsets[-1] = DetectedOnset(time_sec=float(t), strength=strength)
        else:
            onsets.append(DetectedOnset(time_sec=float(t), strength=strength))

    return onsets


def get_onset_activation(
    audio: str | Path | np.ndarray,
    sample_rate: int = 44100,
) -> tuple[np.ndarray, int]:
    """Get the raw onset activation function (for plotting).

    Returns:
        Tuple of (activation_array, fps).
        If madmom: 100 fps activation from RNN.
        If librosa fallback: onset_strength envelope at hop_length=64.
    """
    if MADMOM_AVAILABLE:
        if isinstance(audio, np.ndarray):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sample_rate)
                audio_path = f.name
            cleanup = True
        else:
            audio_path = str(audio)
            cleanup = False

        try:
            proc = RNNOnsetProcessor()
            activation = proc(audio_path)
            return activation, 100
        finally:
            if cleanup:
                Path(audio_path).unlink(missing_ok=True)
    else:
        if isinstance(audio, (str, Path)):
            y, sr = librosa.load(str(audio), sr=None, mono=True)
        else:
            y = audio
            sr = sample_rate
            if y.ndim > 1:
                y = np.mean(y, axis=1)

        hop_length = 64
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        fps = sr // hop_length
        return onset_env, fps
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_analysis.py::TestOnsetDetection -v`
Expected: PASS (may use librosa fallback if madmom didn't install)

**Step 5: Commit**

```bash
git add dataset_gen/audio_analysis/onsets.py tests/test_audio_analysis.py
git commit -m "feat(analysis): implement onset detection with madmom + librosa fallback"
```

---

### Task 5: Implement waveform view

**Files:**
- Modify: `dataset_gen/audio_analysis/views.py`
- Test: `tests/test_audio_analysis.py`

**Step 1: Write the failing test**

Add to `tests/test_audio_analysis.py`:

```python
from dataset_gen.audio_analysis.views import render_waveform
from dataset_gen.audio_analysis.midi_alignment import MidiOnset
from dataset_gen.audio_analysis.onsets import DetectedOnset


class TestWaveformView:
    """Test waveform PNG rendering."""

    @pytest.fixture
    def sample_audio_data(self):
        """Generate 1 second of audio with 2 clicks."""
        sr = 44100
        audio = np.zeros(sr)
        # Click at 250ms and 750ms
        for t in [0.25, 0.75]:
            start = int(t * sr)
            click_len = int(0.005 * sr)
            audio[start : start + click_len] = 0.8 * np.exp(-np.linspace(0, 10, click_len))
        return audio, sr

    def test_render_waveform_creates_file(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        output = tmp_path / "waveform.png"
        render_waveform(audio, sr, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_render_waveform_with_midi_onsets(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        midi_onsets = [
            MidiOnset(time_sec=0.25, velocity=100, note=38),
            MidiOnset(time_sec=0.75, velocity=60, note=38),
        ]
        output = tmp_path / "waveform_midi.png"
        render_waveform(audio, sr, midi_onsets=midi_onsets, output_path=output)
        assert output.exists()

    def test_render_waveform_with_detected_onsets(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        detected = [
            DetectedOnset(time_sec=0.251, strength=0.9),
            DetectedOnset(time_sec=0.749, strength=0.7),
        ]
        output = tmp_path / "waveform_detected.png"
        render_waveform(audio, sr, detected_onsets=detected, output_path=output)
        assert output.exists()

    def test_render_waveform_with_zoom(self, sample_audio_data, tmp_path):
        audio, sr = sample_audio_data
        output = tmp_path / "waveform_zoom.png"
        render_waveform(audio, sr, start_ms=200, end_ms=350, output_path=output)
        assert output.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_analysis.py::TestWaveformView -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Update `dataset_gen/audio_analysis/views.py`:

```python
"""PNG rendering for audio analysis views."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for PNG output
import matplotlib.pyplot as plt
import numpy as np

from dataset_gen.audio_analysis.midi_alignment import MidiOnset
from dataset_gen.audio_analysis.onsets import DetectedOnset


def render_waveform(
    audio: np.ndarray,
    sample_rate: int,
    midi_onsets: list[MidiOnset] | None = None,
    detected_onsets: list[DetectedOnset] | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
    output_path: str | Path = "waveform.png",
    figsize: tuple[float, float] = (16, 4),
    dpi: int = 150,
) -> Path:
    """Render a high-resolution waveform with onset markers.

    Args:
        audio: Audio samples (mono or stereo, will be mixed to mono).
        sample_rate: Sample rate in Hz.
        midi_onsets: Optional MIDI ground-truth onsets (red vertical lines).
        detected_onsets: Optional detected onsets (blue dashed lines).
        start_ms: Zoom start in milliseconds (None = beginning).
        end_ms: Zoom end in milliseconds (None = end).
        output_path: Where to save the PNG.
        figsize: Figure size in inches.
        dpi: Resolution.

    Returns:
        Path to the saved PNG.
    """
    output_path = Path(output_path)

    # Mix to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Time axis in milliseconds
    times_ms = np.arange(len(audio)) / sample_rate * 1000.0

    # Apply zoom
    if start_ms is not None or end_ms is not None:
        s = start_ms or 0.0
        e = end_ms or times_ms[-1]
        mask = (times_ms >= s) & (times_ms <= e)
        audio = audio[mask]
        times_ms = times_ms[mask]

    fig, ax = plt.subplots(figsize=figsize)

    # Waveform
    ax.plot(times_ms, audio, color="#333333", linewidth=0.3, alpha=0.8)

    # MIDI onsets (red)
    if midi_onsets:
        for onset in midi_onsets:
            t_ms = onset.time_sec * 1000.0
            if times_ms[0] <= t_ms <= times_ms[-1]:
                height = onset.velocity / 127.0
                ax.axvline(
                    x=t_ms, color="red", linewidth=1.2, alpha=0.7,
                    ymin=0.5 - height * 0.5, ymax=0.5 + height * 0.5,
                    label="MIDI" if onset == midi_onsets[0] else None,
                )

    # Detected onsets (blue dashed)
    if detected_onsets:
        for onset in detected_onsets:
            t_ms = onset.time_sec * 1000.0
            if times_ms[0] <= t_ms <= times_ms[-1]:
                ax.axvline(
                    x=t_ms, color="blue", linewidth=0.8, linestyle="--", alpha=0.6,
                    label="Detected" if onset == detected_onsets[0] else None,
                )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Transient Waveform")

    if midi_onsets or detected_onsets:
        ax.legend(loc="upper right", fontsize=8)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)

    return output_path
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_analysis.py::TestWaveformView -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/audio_analysis/views.py tests/test_audio_analysis.py
git commit -m "feat(analysis): implement waveform view with onset markers"
```

---

### Task 6: Implement onset + velocity timeline view

**Files:**
- Modify: `dataset_gen/audio_analysis/views.py`
- Test: `tests/test_audio_analysis.py`

**Step 1: Write the failing test**

Add to `tests/test_audio_analysis.py`:

```python
from dataset_gen.audio_analysis.views import render_onset_timeline


class TestOnsetTimelineView:
    """Test onset + velocity timeline rendering."""

    def test_render_onset_timeline_creates_file(self, tmp_path):
        midi_onsets = [
            MidiOnset(time_sec=0.0, velocity=100, note=38),
            MidiOnset(time_sec=0.5, velocity=60, note=38),
            MidiOnset(time_sec=1.0, velocity=100, note=38),
            MidiOnset(time_sec=1.5, velocity=60, note=38),
        ]
        output = tmp_path / "onsets.png"
        render_onset_timeline(midi_onsets=midi_onsets, output_path=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_render_with_detected_onsets(self, tmp_path):
        midi_onsets = [
            MidiOnset(time_sec=0.0, velocity=100, note=38),
            MidiOnset(time_sec=0.5, velocity=60, note=38),
        ]
        detected = [
            DetectedOnset(time_sec=0.002, strength=0.9),
            DetectedOnset(time_sec=0.498, strength=0.6),
        ]
        output = tmp_path / "onsets_both.png"
        render_onset_timeline(
            midi_onsets=midi_onsets, detected_onsets=detected, output_path=output
        )
        assert output.exists()

    def test_render_detected_only(self, tmp_path):
        detected = [
            DetectedOnset(time_sec=0.1, strength=0.9),
            DetectedOnset(time_sec=0.6, strength=0.5),
        ]
        output = tmp_path / "onsets_detected.png"
        render_onset_timeline(detected_onsets=detected, output_path=output)
        assert output.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_analysis.py::TestOnsetTimelineView -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Add to `dataset_gen/audio_analysis/views.py`:

```python
def render_onset_timeline(
    midi_onsets: list[MidiOnset] | None = None,
    detected_onsets: list[DetectedOnset] | None = None,
    output_path: str | Path = "onsets.png",
    figsize: tuple[float, float] = (16, 5),
    dpi: int = 150,
) -> Path:
    """Render onset + velocity timeline showing hits as markers.

    Args:
        midi_onsets: MIDI ground-truth onsets (red circles).
        detected_onsets: Detected onsets (blue diamonds).
        output_path: Where to save the PNG.
        figsize: Figure size in inches.
        dpi: Resolution.

    Returns:
        Path to the saved PNG.
    """
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=figsize)

    # MIDI onsets (red circles with stems)
    if midi_onsets:
        times = [o.time_sec * 1000 for o in midi_onsets]
        vels = [o.velocity for o in midi_onsets]
        ax.vlines(times, 0, vels, colors="red", alpha=0.3, linewidth=0.8)
        ax.scatter(times, vels, color="red", s=50, zorder=5, label="MIDI", marker="o")

        # IOI labels between consecutive MIDI onsets
        for i in range(1, len(midi_onsets)):
            ioi_ms = (midi_onsets[i].time_sec - midi_onsets[i - 1].time_sec) * 1000
            mid_t = (times[i - 1] + times[i]) / 2
            mid_v = max(vels[i - 1], vels[i]) + 8
            ax.annotate(
                f"{ioi_ms:.0f}ms",
                (mid_t, mid_v),
                fontsize=6,
                ha="center",
                color="gray",
            )

    # Detected onsets (blue diamonds with stems)
    if detected_onsets:
        times = [o.time_sec * 1000 for o in detected_onsets]
        # Scale strength to 0-127 range for visual comparison
        max_strength = max(o.strength for o in detected_onsets) or 1.0
        vels = [o.strength / max_strength * 127 for o in detected_onsets]
        ax.vlines(times, 0, vels, colors="blue", alpha=0.2, linewidth=0.8)
        ax.scatter(
            times, vels, color="blue", s=40, zorder=5, label="Detected", marker="D"
        )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity / Strength")
    ax.set_ylim(0, 140)
    ax.set_title("Onset + Velocity Timeline")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)

    return output_path
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_analysis.py::TestOnsetTimelineView -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/audio_analysis/views.py tests/test_audio_analysis.py
git commit -m "feat(analysis): implement onset + velocity timeline view"
```

---

### Task 7: Implement multi-panel dashboard view

**Files:**
- Modify: `dataset_gen/audio_analysis/views.py`
- Test: `tests/test_audio_analysis.py`

**Step 1: Write the failing test**

Add to `tests/test_audio_analysis.py`:

```python
from dataset_gen.audio_analysis.views import render_dashboard


class TestDashboardView:
    """Test multi-panel dashboard rendering."""

    @pytest.fixture
    def dashboard_data(self):
        """Generate audio + onsets for dashboard test."""
        sr = 44100
        audio = np.zeros(sr * 2)
        click_times = [0.25, 0.75, 1.25, 1.75]
        click_len = int(0.005 * sr)
        for t in click_times:
            start = int(t * sr)
            audio[start : start + click_len] = 0.8 * np.exp(-np.linspace(0, 10, click_len))

        midi_onsets = [
            MidiOnset(time_sec=t, velocity=100 if i % 2 == 0 else 60, note=38)
            for i, t in enumerate(click_times)
        ]
        detected = [
            DetectedOnset(time_sec=t + 0.002, strength=0.9 if i % 2 == 0 else 0.5)
            for i, t in enumerate(click_times)
        ]
        return audio, sr, midi_onsets, detected

    def test_render_dashboard_creates_file(self, dashboard_data, tmp_path):
        audio, sr, midi_onsets, detected = dashboard_data
        output = tmp_path / "dashboard.png"
        render_dashboard(
            audio, sr, midi_onsets=midi_onsets, detected_onsets=detected, output_path=output
        )
        assert output.exists()
        assert output.stat().st_size > 1000  # Should be a substantial image

    def test_render_dashboard_audio_only(self, dashboard_data, tmp_path):
        audio, sr, _, _ = dashboard_data
        output = tmp_path / "dashboard_audio_only.png"
        render_dashboard(audio, sr, output_path=output)
        assert output.exists()

    def test_render_dashboard_with_activation(self, dashboard_data, tmp_path):
        audio, sr, midi_onsets, detected = dashboard_data
        # Simulate an activation function (100 fps for 2 seconds)
        activation = np.random.rand(200).astype(np.float32) * 0.3
        output = tmp_path / "dashboard_activation.png"
        render_dashboard(
            audio,
            sr,
            midi_onsets=midi_onsets,
            detected_onsets=detected,
            activation=activation,
            activation_fps=100,
            output_path=output,
        )
        assert output.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_analysis.py::TestDashboardView -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Add to `dataset_gen/audio_analysis/views.py`:

```python
def render_dashboard(
    audio: np.ndarray,
    sample_rate: int,
    midi_onsets: list[MidiOnset] | None = None,
    detected_onsets: list[DetectedOnset] | None = None,
    activation: np.ndarray | None = None,
    activation_fps: int = 100,
    output_path: str | Path = "dashboard.png",
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 150,
) -> Path:
    """Render a multi-panel dashboard with waveform, activation, IOI, and velocity.

    Four panels sharing X-axis:
    1. Waveform with onset markers
    2. Onset strength envelope / activation function
    3. Inter-onset intervals (stem plot)
    4. Velocity comparison (MIDI vs detected)

    Args:
        audio: Audio samples (mono or stereo).
        sample_rate: Sample rate in Hz.
        midi_onsets: Optional MIDI ground-truth onsets.
        detected_onsets: Optional detected onsets.
        activation: Optional raw activation function for panel 2.
        activation_fps: Frames per second of the activation function.
        output_path: Where to save the PNG.
        figsize: Figure size in inches.
        dpi: Resolution.

    Returns:
        Path to the saved PNG.
    """
    output_path = Path(output_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    times_ms = np.arange(len(audio)) / sample_rate * 1000.0

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    ax_wave, ax_act, ax_ioi, ax_vel = axes

    # --- Panel 1: Waveform ---
    ax_wave.plot(times_ms, audio, color="#333333", linewidth=0.3, alpha=0.8)
    if midi_onsets:
        for o in midi_onsets:
            ax_wave.axvline(x=o.time_sec * 1000, color="red", linewidth=0.8, alpha=0.5)
    if detected_onsets:
        for o in detected_onsets:
            ax_wave.axvline(
                x=o.time_sec * 1000, color="blue", linewidth=0.6, linestyle="--", alpha=0.4
            )
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform + Onsets")
    ax_wave.grid(True, alpha=0.2)

    # --- Panel 2: Onset activation ---
    if activation is not None:
        act_times = np.arange(len(activation)) / activation_fps * 1000.0
        ax_act.plot(act_times, activation, color="purple", linewidth=0.8)
        ax_act.fill_between(act_times, activation, alpha=0.2, color="purple")
    ax_act.set_ylabel("Activation")
    ax_act.set_title("Onset Strength Envelope")
    ax_act.grid(True, alpha=0.2)

    # --- Panel 3: Inter-onset intervals ---
    all_onsets_ms = sorted(
        [o.time_sec * 1000 for o in (detected_onsets or midi_onsets or [])]
    )
    if len(all_onsets_ms) > 1:
        ioi_times = all_onsets_ms[1:]
        ioi_values = [
            all_onsets_ms[i] - all_onsets_ms[i - 1] for i in range(1, len(all_onsets_ms))
        ]
        ax_ioi.stem(ioi_times, ioi_values, linefmt="g-", markerfmt="go", basefmt="gray")
        # Highlight sub-50ms gaps (likely flams/drags)
        for t, v in zip(ioi_times, ioi_values):
            if v < 50:
                ax_ioi.annotate(
                    f"{v:.0f}ms", (t, v + 2), fontsize=7, color="red", ha="center"
                )
    ax_ioi.set_ylabel("IOI (ms)")
    ax_ioi.set_title("Inter-Onset Intervals")
    ax_ioi.grid(True, alpha=0.2)

    # --- Panel 4: Velocity comparison ---
    if midi_onsets and detected_onsets:
        midi_t = [o.time_sec * 1000 for o in midi_onsets]
        midi_v = [o.velocity for o in midi_onsets]
        max_str = max(o.strength for o in detected_onsets) or 1.0
        det_t = [o.time_sec * 1000 for o in detected_onsets]
        det_v = [o.strength / max_str * 127 for o in detected_onsets]

        bar_w = 5.0  # 5ms bar width
        ax_vel.bar(
            midi_t, midi_v, width=bar_w, color="red", alpha=0.6, label="MIDI velocity"
        )
        ax_vel.bar(
            [t + bar_w for t in det_t],
            det_v,
            width=bar_w,
            color="blue",
            alpha=0.6,
            label="Detected strength",
        )
        ax_vel.legend(fontsize=8)
    elif midi_onsets:
        midi_t = [o.time_sec * 1000 for o in midi_onsets]
        midi_v = [o.velocity for o in midi_onsets]
        ax_vel.bar(midi_t, midi_v, width=5.0, color="red", alpha=0.6, label="MIDI velocity")
        ax_vel.legend(fontsize=8)
    ax_vel.set_ylabel("Velocity")
    ax_vel.set_xlabel("Time (ms)")
    ax_vel.set_title("Velocity Comparison")
    ax_vel.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)

    return output_path
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_analysis.py::TestDashboardView -v`
Expected: PASS

**Step 5: Commit**

```bash
git add dataset_gen/audio_analysis/views.py tests/test_audio_analysis.py
git commit -m "feat(analysis): implement multi-panel dashboard view"
```

---

### Task 8: Implement CLI script

**Files:**
- Create: `scripts/analyze_audio.py`
- Test: manual CLI invocation

**Step 1: Write the CLI script**

Create `scripts/analyze_audio.py`:

```python
#!/usr/bin/env python3
"""Analyze audio files and produce high-resolution visualizations.

Usage:
    python scripts/analyze_audio.py path/to/audio.flac --view all
    python scripts/analyze_audio.py path/to/audio.flac --midi path/to/midi.mid --view waveform
    python scripts/analyze_audio.py path/to/audio.flac --view dashboard --start 500 --end 1500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset_gen.audio_analysis.midi_alignment import extract_midi_onsets
from dataset_gen.audio_analysis.onsets import detect_onsets, get_onset_activation
from dataset_gen.audio_analysis.views import render_dashboard, render_onset_timeline, render_waveform


def main():
    parser = argparse.ArgumentParser(description="Analyze snare rudiment audio files.")
    parser.add_argument("audio", type=Path, help="Path to audio file (FLAC, WAV, etc.)")
    parser.add_argument("--midi", type=Path, default=None, help="Path to corresponding MIDI file")
    parser.add_argument(
        "--view",
        choices=["waveform", "onsets", "dashboard", "all"],
        default="all",
        help="Which view(s) to render (default: all)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory for PNGs")
    parser.add_argument("--start", type=float, default=None, help="Zoom start in ms")
    parser.add_argument("--end", type=float, default=None, help="Zoom end in ms")
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Onset detection threshold (default: 0.3)"
    )
    parser.add_argument(
        "--combine", type=float, default=20.0, help="Combine onsets within N ms (default: 20)"
    )
    parser.add_argument(
        "--method",
        choices=["auto", "madmom", "librosa"],
        default="auto",
        help="Onset detection method (default: auto)",
    )

    args = parser.parse_args()

    # Resolve output directory
    if args.output is None:
        args.output = args.audio.parent / "analysis"
    args.output.mkdir(parents=True, exist_ok=True)

    sample_id = args.audio.stem

    # Load audio
    print(f"Loading {args.audio}...")
    audio, sr = sf.read(str(args.audio))
    print(f"  Sample rate: {sr} Hz, duration: {len(audio) / sr:.2f}s, shape: {audio.shape}")

    # Extract MIDI onsets if provided
    midi_onsets = None
    if args.midi:
        print(f"Extracting MIDI onsets from {args.midi}...")
        midi_onsets = extract_midi_onsets(args.midi)
        print(f"  Found {len(midi_onsets)} MIDI note-on events")

    # Detect onsets in audio
    print(f"Detecting onsets (method={args.method}, threshold={args.threshold})...")
    detected = detect_onsets(
        audio,
        sample_rate=sr,
        method=args.method,
        threshold=args.threshold,
        combine_ms=args.combine,
    )
    print(f"  Detected {len(detected)} onsets")

    # Get activation function for dashboard
    activation, act_fps = get_onset_activation(audio, sample_rate=sr)

    # Mix to mono for views
    if audio.ndim > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio

    views = [args.view] if args.view != "all" else ["waveform", "onsets", "dashboard"]

    for view in views:
        output_file = args.output / f"{sample_id}_{view}.png"
        print(f"Rendering {view} -> {output_file}...")

        if view == "waveform":
            render_waveform(
                audio_mono,
                sr,
                midi_onsets=midi_onsets,
                detected_onsets=detected,
                start_ms=args.start,
                end_ms=args.end,
                output_path=output_file,
            )
        elif view == "onsets":
            render_onset_timeline(
                midi_onsets=midi_onsets,
                detected_onsets=detected,
                output_path=output_file,
            )
        elif view == "dashboard":
            render_dashboard(
                audio_mono,
                sr,
                midi_onsets=midi_onsets,
                detected_onsets=detected,
                activation=activation,
                activation_fps=act_fps,
                output_path=output_file,
            )

    print(f"Done. Output in {args.output}/")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test the CLI**

Find a sample audio file in the dataset and run:
```bash
# Find an audio file
AUDIO=$(ls output/dataset_multisf/audio/*.flac 2>/dev/null | head -1)
MIDI=$(echo "$AUDIO" | sed 's|audio/|midi/|;s|\.flac$|.mid|')

if [ -n "$AUDIO" ]; then
    python scripts/analyze_audio.py "$AUDIO" --midi "$MIDI" --view all --output /tmp/sousa_analysis/
else
    echo "No audio files found. Test with synthetic data:"
    python -c "
import soundfile as sf
import numpy as np
sr = 44100
audio = np.zeros(sr * 2)
for t in [0.25, 0.75, 1.25, 1.75]:
    s = int(t * sr)
    l = int(0.005 * sr)
    audio[s:s+l] = 0.8 * np.exp(-np.linspace(0, 10, l))
sf.write('/tmp/test_clicks.wav', audio, sr)
"
    python scripts/analyze_audio.py /tmp/test_clicks.wav --view all --output /tmp/sousa_analysis/
fi
```

Expected: PNG files created in output directory.

**Step 3: Commit**

```bash
git add scripts/analyze_audio.py
git commit -m "feat(analysis): add CLI script for audio analysis"
```

---

### Task 9: Update `__init__.py` exports and verify full integration

**Files:**
- Modify: `dataset_gen/audio_analysis/__init__.py`

**Step 1: Update exports**

Update `dataset_gen/audio_analysis/__init__.py`:

```python
"""Audio analysis module for high-resolution visualization of drum rudiments."""

from dataset_gen.audio_analysis.midi_alignment import MidiOnset, extract_midi_onsets
from dataset_gen.audio_analysis.onsets import (
    DetectedOnset,
    detect_onsets,
    get_onset_activation,
)
from dataset_gen.audio_analysis.views import (
    render_dashboard,
    render_onset_timeline,
    render_waveform,
)

__all__ = [
    "DetectedOnset",
    "MidiOnset",
    "detect_onsets",
    "extract_midi_onsets",
    "get_onset_activation",
    "render_dashboard",
    "render_onset_timeline",
    "render_waveform",
]
```

**Step 2: Run full test suite**

Run: `pytest tests/test_audio_analysis.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add dataset_gen/audio_analysis/__init__.py
git commit -m "feat(analysis): finalize module exports"
```
