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
from dataset_gen.audio_analysis.views import (
    render_cycle_zoom,
    render_dashboard,
    render_onset_timeline,
    render_waveform,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze snare rudiment audio files.")
    parser.add_argument("audio", type=Path, help="Path to audio file (FLAC, WAV, etc.)")
    parser.add_argument("--midi", type=Path, default=None, help="Path to corresponding MIDI file")
    parser.add_argument(
        "--view",
        choices=["waveform", "onsets", "dashboard", "cycles", "all"],
        default="all",
        help="Which view(s) to render (default: all)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory for PNGs")
    parser.add_argument("--start", type=float, default=None, help="Zoom start in ms")
    parser.add_argument("--end", type=float, default=None, help="Zoom end in ms")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Onset detection threshold (default: 0.1)"
    )
    parser.add_argument(
        "--combine", type=float, default=10.0, help="Combine onsets within N ms (default: 10)"
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

    # Mix to mono for views
    if audio.ndim > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio

    # Extract MIDI onsets if provided
    midi_onsets = None
    if args.midi:
        print(f"Extracting MIDI onsets from {args.midi}...")
        midi_onsets = extract_midi_onsets(args.midi)
        print(f"  Found {len(midi_onsets)} MIDI note-on events")

    # Detect onsets in audio
    print(f"Detecting onsets (method={args.method}, threshold={args.threshold})...")
    detected = detect_onsets(
        audio_mono,
        sample_rate=sr,
        method=args.method,
        threshold=args.threshold,
        combine_ms=args.combine,
    )
    print(f"  Detected {len(detected)} onsets")

    # Get activation function for dashboard
    activation, act_fps = get_onset_activation(audio_mono, sample_rate=sr)

    views = [args.view] if args.view != "all" else ["waveform", "onsets", "dashboard", "cycles"]
    # cycles view requires MIDI onsets
    if "cycles" in views and not midi_onsets:
        views = [v for v in views if v != "cycles"]
        print("  Skipping cycles view (requires --midi)")

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
        elif view == "cycles":
            render_cycle_zoom(
                audio_mono,
                sr,
                midi_onsets=midi_onsets,
                detected_onsets=detected,
                activation=activation,
                activation_fps=act_fps,
                threshold=args.threshold,
                output_path=output_file,
            )

    print(f"Done. Output in {args.output}/")


if __name__ == "__main__":
    main()
