"""Visualization utilities for audio analysis results.

Provides plotting and inspection tools for onset detection,
MIDI alignment, and scoring diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dataset_gen.audio_analysis.midi_alignment import MidiOnset  # noqa: E402
from dataset_gen.audio_analysis.onsets import DetectedOnset  # noqa: E402

__all__ = ["render_waveform", "render_onset_timeline", "render_dashboard"]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Mix stereo (or multi-channel) audio down to mono."""
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


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
    """Render a high-resolution waveform plot with optional onset markers.

    Args:
        audio: Audio samples as a numpy array.
        sample_rate: Audio sample rate in Hz.
        midi_onsets: Optional MIDI onset markers (red vertical lines).
        detected_onsets: Optional detected onset markers (blue dashed lines).
        start_ms: Start of zoom window in milliseconds.
        end_ms: End of zoom window in milliseconds.
        output_path: Path to save the output PNG.
        figsize: Figure size as (width, height) in inches.
        dpi: Resolution in dots per inch.

    Returns:
        Path to the saved PNG file.
    """
    output_path = Path(output_path)
    mono = _to_mono(audio)

    # Compute time axis in milliseconds
    time_ms = np.arange(len(mono)) / sample_rate * 1000.0

    # Apply zoom window by masking
    if start_ms is not None or end_ms is not None:
        lo = start_ms if start_ms is not None else 0.0
        hi = end_ms if end_ms is not None else time_ms[-1]
        mask = (time_ms >= lo) & (time_ms <= hi)
        time_ms = time_ms[mask]
        mono = mono[mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_ms, mono, linewidth=0.5, color="steelblue", alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.grid(True, alpha=0.2)

    has_legend = False

    # Draw MIDI onsets as red vertical lines with velocity-scaled height
    if midi_onsets:
        for i, onset in enumerate(midi_onsets):
            t_ms = onset.time_sec * 1000.0
            # Scale ymin/ymax by velocity (0-127)
            vel_frac = onset.velocity / 127.0
            label = "MIDI onset" if i == 0 else None
            ax.axvline(
                x=t_ms,
                ymin=0.5 - vel_frac * 0.5,
                ymax=0.5 + vel_frac * 0.5,
                color="red",
                linewidth=1.0,
                alpha=0.7,
                label=label,
            )
        has_legend = True

    # Draw detected onsets as blue dashed vertical lines
    if detected_onsets:
        for i, onset in enumerate(detected_onsets):
            t_ms = onset.time_sec * 1000.0
            label = "Detected onset" if i == 0 else None
            ax.axvline(
                x=t_ms,
                color="blue",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label=label,
            )
        has_legend = True

    if has_legend:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)
    return output_path


def render_onset_timeline(
    midi_onsets: list[MidiOnset] | None = None,
    detected_onsets: list[DetectedOnset] | None = None,
    output_path: str | Path = "onsets.png",
    figsize: tuple[float, float] = (16, 5),
    dpi: int = 150,
) -> Path:
    """Render an abstract onset timeline showing hits and inter-onset intervals.

    Args:
        midi_onsets: Optional MIDI onsets (red circles with stems).
        detected_onsets: Optional detected onsets (blue diamonds with stems).
        output_path: Path to save the output PNG.
        figsize: Figure size as (width, height) in inches.
        dpi: Resolution in dots per inch.

    Returns:
        Path to the saved PNG file.
    """
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=figsize)

    has_legend = False

    # MIDI onsets: red circles with stems, Y = velocity
    if midi_onsets:
        times_ms = [o.time_sec * 1000.0 for o in midi_onsets]
        velocities = [o.velocity for o in midi_onsets]
        ax.vlines(
            times_ms,
            ymin=0,
            ymax=velocities,
            colors="red",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.scatter(
            times_ms,
            velocities,
            color="red",
            marker="o",
            s=40,
            zorder=5,
            label="MIDI onset",
        )
        has_legend = True

        # IOI labels between consecutive MIDI onsets
        for i in range(1, len(midi_onsets)):
            ioi_ms = (midi_onsets[i].time_sec - midi_onsets[i - 1].time_sec) * 1000.0
            mid_x = (times_ms[i - 1] + times_ms[i]) / 2.0
            higher_y = max(velocities[i - 1], velocities[i])
            ax.annotate(
                f"{ioi_ms:.0f}ms",
                xy=(mid_x, higher_y + 3),
                fontsize=7,
                color="gray",
                ha="center",
                va="bottom",
            )

    # Detected onsets: blue diamonds with stems, Y = strength scaled to 0-127
    if detected_onsets:
        max_strength = max(o.strength for o in detected_onsets) if detected_onsets else 1.0
        if max_strength <= 0:
            max_strength = 1.0
        times_ms_det = [o.time_sec * 1000.0 for o in detected_onsets]
        scaled = [o.strength / max_strength * 127.0 for o in detected_onsets]
        ax.vlines(
            times_ms_det,
            ymin=0,
            ymax=scaled,
            colors="blue",
            linewidth=1.0,
            alpha=0.6,
        )
        ax.scatter(
            times_ms_det,
            scaled,
            color="blue",
            marker="D",
            s=40,
            zorder=5,
            label="Detected onset",
        )
        has_legend = True

    ax.set_ylim(0, 140)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity / Strength")
    ax.set_title("Onset Timeline")
    ax.grid(True, alpha=0.2)

    if has_legend:
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)
    return output_path


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
    """Render a four-panel dashboard with shared X-axis (time in ms).

    Panels:
        1. Waveform with onset markers
        2. Onset activation envelope
        3. Inter-onset intervals stem plot
        4. Velocity comparison paired bars

    Args:
        audio: Audio samples as a numpy array.
        sample_rate: Audio sample rate in Hz.
        midi_onsets: Optional MIDI onset markers.
        detected_onsets: Optional detected onset markers.
        activation: Optional onset activation envelope array.
        activation_fps: Frames per second of the activation array.
        output_path: Path to save the output PNG.
        figsize: Figure size as (width, height) in inches.
        dpi: Resolution in dots per inch.

    Returns:
        Path to the saved PNG file.
    """
    output_path = Path(output_path)
    mono = _to_mono(audio)
    time_ms = np.arange(len(mono)) / sample_rate * 1000.0

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # --- Panel 1: Waveform with onset markers ---
    ax1 = axes[0]
    ax1.plot(time_ms, mono, linewidth=0.4, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform")
    ax1.grid(True, alpha=0.2)

    if midi_onsets:
        for i, onset in enumerate(midi_onsets):
            t_ms = onset.time_sec * 1000.0
            vel_frac = onset.velocity / 127.0
            label = "MIDI onset" if i == 0 else None
            ax1.axvline(
                x=t_ms,
                ymin=0.5 - vel_frac * 0.5,
                ymax=0.5 + vel_frac * 0.5,
                color="red",
                linewidth=1.0,
                alpha=0.7,
                label=label,
            )

    if detected_onsets:
        for i, onset in enumerate(detected_onsets):
            t_ms = onset.time_sec * 1000.0
            label = "Detected onset" if i == 0 else None
            ax1.axvline(
                x=t_ms,
                color="blue",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label=label,
            )

    if midi_onsets or detected_onsets:
        ax1.legend(loc="upper right", fontsize=7)

    # --- Panel 2: Onset activation envelope ---
    ax2 = axes[1]
    if activation is not None and len(activation) > 0:
        act_time_ms = np.arange(len(activation)) / activation_fps * 1000.0
        ax2.plot(act_time_ms, activation, color="purple", linewidth=0.8)
        ax2.fill_between(act_time_ms, activation, alpha=0.3, color="purple")
    ax2.set_ylabel("Activation")
    ax2.set_title("Onset Activation Envelope")
    ax2.grid(True, alpha=0.2)

    # --- Panel 3: Inter-onset intervals stem plot ---
    ax3 = axes[2]
    # Use detected_onsets if available, else midi_onsets
    ioi_source = detected_onsets if detected_onsets else midi_onsets
    if ioi_source and len(ioi_source) >= 2:
        sorted_src = sorted(ioi_source, key=lambda o: o.time_sec)
        ioi_times_ms = []
        ioi_values = []
        for i in range(1, len(sorted_src)):
            gap_ms = (sorted_src[i].time_sec - sorted_src[i - 1].time_sec) * 1000.0
            mid_t = (sorted_src[i - 1].time_sec + sorted_src[i].time_sec) / 2.0 * 1000.0
            ioi_times_ms.append(mid_t)
            ioi_values.append(gap_ms)

        markerline, stemlines, baseline = ax3.stem(
            ioi_times_ms, ioi_values, linefmt="C0-", markerfmt="C0o", basefmt="k-"
        )
        plt.setp(stemlines, linewidth=0.8)
        plt.setp(markerline, markersize=4)

        # Highlight sub-50ms gaps with red annotations
        for t, v in zip(ioi_times_ms, ioi_values):
            if v < 50.0:
                ax3.annotate(
                    f"{v:.0f}ms",
                    xy=(t, v),
                    fontsize=7,
                    color="red",
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

    ax3.set_ylabel("IOI (ms)")
    ax3.set_title("Inter-Onset Intervals")
    ax3.grid(True, alpha=0.2)

    # --- Panel 4: Velocity comparison paired bars ---
    ax4 = axes[3]
    bar_width = 5.0  # 5ms bar width

    if midi_onsets:
        midi_times = [o.time_sec * 1000.0 for o in midi_onsets]
        midi_vels = [o.velocity for o in midi_onsets]
        ax4.bar(
            midi_times,
            midi_vels,
            width=bar_width,
            color="red",
            alpha=0.7,
            label="MIDI velocity",
        )

    if detected_onsets:
        max_str = max(o.strength for o in detected_onsets) if detected_onsets else 1.0
        if max_str <= 0:
            max_str = 1.0
        det_times = [o.time_sec * 1000.0 + bar_width for o in detected_onsets]
        det_scaled = [o.strength / max_str * 127.0 for o in detected_onsets]
        ax4.bar(
            det_times,
            det_scaled,
            width=bar_width,
            color="blue",
            alpha=0.7,
            label="Detected strength",
        )

    ax4.set_ylabel("Velocity / Strength (0-127)")
    ax4.set_xlabel("Time (ms)")
    ax4.set_title("Velocity Comparison")
    ax4.grid(True, alpha=0.2)

    if midi_onsets or detected_onsets:
        ax4.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)
    return output_path
