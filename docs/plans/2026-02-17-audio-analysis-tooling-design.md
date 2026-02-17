# Audio Analysis Tooling Design

## Problem

Claude cannot hear audio. To verify that SOUSA's generated audio is correct — that flams have proper gaps, accents are louder than taps, diddles are distinct hits — we need visual representations with enough temporal resolution to distinguish hits that are only milliseconds apart. Standard mel-spectrograms smear snare rudiments into blobs because all hits are the same instrument and pitch.

## Approach

A CLI tool backed by an importable module. Takes an audio file (+ optional MIDI for ground truth overlay), runs neural onset detection via madmom, and produces high-resolution PNG visualizations.

## Architecture

```
dataset_gen/audio_analysis/
├── __init__.py
├── onsets.py              # madmom RNN onset detection, librosa fallback
├── midi_alignment.py      # Parse MIDI, extract note-on events with timing/velocity
└── views.py               # PNG rendering for each view type

scripts/analyze_audio.py   # CLI entry point
```

### onsets.py

Wraps madmom's `RNNOnsetProcessor` + `OnsetPeakPickingProcessor` for neural onset detection. Falls back to librosa `onset_detect` with tight parameters (hop_length=64, ~1.5ms resolution at 44.1kHz). Returns `DetectedOnset(time_sec, strength)` objects.

### midi_alignment.py

Reads a MIDI file via `mido`, extracts note-on events with times and velocities. Returns `MidiOnset(time_sec, velocity, note)` objects used as ground-truth markers overlaid on audio views.

### views.py

Three rendering functions, each producing a PNG via matplotlib.

## Views

### Transient Waveform (`--view waveform`)

High-resolution waveform plot for seeing individual hits.

- X-axis: time in milliseconds
- Y-axis: amplitude
- MIDI onsets: vertical red lines, height scaled to velocity
- Detected onsets: vertical blue dashed lines from madmom
- Supports `--start` and `--end` flags (in ms) to zoom into regions
- hop_length=64 samples (~1.45ms at 44.1kHz)
- Figure: 16x4 inches at 150 DPI

### Onset + Velocity Timeline (`--view onsets`)

Abstract timeline showing just the hits without waveform clutter.

- X-axis: time in ms
- Y-axis: MIDI velocity (0-127)
- MIDI onsets: red circles at (time, velocity)
- Detected onsets: blue diamonds at (time, strength scaled to 0-127)
- Gray vertical lines from each onset to x-axis
- IOI (inter-onset interval) labels in ms between consecutive hits
- Sticking annotations (R/L) from MIDI if available

### Multi-Panel Dashboard (`--view dashboard`)

Four stacked subplots sharing X-axis (time in ms):

1. Waveform with onset markers (compact version of waveform view)
2. Onset strength envelope (madmom activation function)
3. Inter-onset intervals (stem plot of gaps between hits)
4. Velocity comparison (MIDI velocity vs detected strength as paired bars)

Figure: 16x12 inches at 150 DPI.

## CLI Interface

```bash
# Full analysis with MIDI overlay
python scripts/analyze_audio.py audio.flac --midi audio.mid --view all --output /tmp/analysis/

# Zoomed waveform
python scripts/analyze_audio.py audio.flac --view waveform --start 500 --end 800

# Dashboard only, no MIDI
python scripts/analyze_audio.py audio.flac --view dashboard
```

Output files: `{sample_id}_{view}.png`

## Dependencies

New `[analysis]` extra in pyproject.toml:

- `madmom>=0.16.1` — RNN-based onset detection trained on percussion
- `matplotlib>=3.7.0` (already available under `[ml]`)
