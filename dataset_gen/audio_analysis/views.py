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

__all__ = [
    "render_waveform",
    "render_onset_timeline",
    "render_dashboard",
    "render_cycle_zoom",
]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Mix stereo (or multi-channel) audio down to mono."""
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def _rms_envelope(
    audio: np.ndarray, sample_rate: int, frame_ms: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMS envelope for cleaner waveform display.

    Returns:
        (time_ms_array, rms_array) where rms is in dB-like scale.
    """
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    hop = frame_len // 2
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = audio[start : start + frame_len]
        rms[i] = np.sqrt(np.mean(frame**2))
    time_ms = np.arange(n_frames) * hop / sample_rate * 1000.0
    return time_ms, rms


def _find_cycle_boundaries(
    midi_onsets: list[MidiOnset],
    target_cycles: int | None = None,
) -> list[tuple[float, float]]:
    """Split MIDI onsets into equal cycles for visual inspection.

    Uses strict equal-count splitting: each cycle gets the same number of
    notes (±1). The only adjustment is avoiding splits inside flams/drags
    (IOI < 40ms) by shifting the split point by +1.

    Args:
        midi_onsets: List of MIDI onsets sorted by time.
        target_cycles: Number of cycles to create. If None, auto-selects
            based on total note count (aims for 6-10 notes per cycle).

    Returns:
        List of (start_ms, end_ms) for each cycle with padding.
    """
    n = len(midi_onsets)
    if n < 4:
        pad = 50.0
        return [
            (max(0, midi_onsets[0].time_sec * 1000 - pad), midi_onsets[-1].time_sec * 1000 + pad)
        ]

    iois = [(midi_onsets[i].time_sec - midi_onsets[i - 1].time_sec) * 1000 for i in range(1, n)]

    # Auto-select number of cycles: aim for 6-10 notes per cycle
    if target_cycles is None:
        target_cycles = max(2, min(6, n // 7))

    # Strict equal-count split indices
    split_indices: list[int] = []
    for k in range(1, target_cycles):
        idx = round(k * n / target_cycles)
        # Avoid splitting inside a flam/drag (IOI < 40ms): shift +1
        if 0 < idx < n and idx - 1 < len(iois) and iois[idx - 1] < 40:
            idx = min(idx + 1, n - 1)
        split_indices.append(idx)

    # Build cycle ranges
    all_starts = [0] + split_indices
    all_ends = [s - 1 for s in split_indices] + [n - 1]

    pad_ms = 50.0
    cycles = []
    for start_idx, end_idx in zip(all_starts, all_ends):
        start_ms = midi_onsets[start_idx].time_sec * 1000 - pad_ms
        end_ms = midi_onsets[end_idx].time_sec * 1000 + pad_ms
        cycles.append((max(0, start_ms), end_ms))

    return cycles


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
    """Render waveform with onset markers.

    At full scale uses RMS envelope (readable). When zoomed <500ms shows raw
    waveform (individual transients visible).
    """
    output_path = Path(output_path)
    mono = _to_mono(audio)
    total_duration_ms = len(mono) / sample_rate * 1000.0

    # Determine visible window
    lo = start_ms if start_ms is not None else 0.0
    hi = end_ms if end_ms is not None else total_duration_ms
    window_ms = hi - lo
    is_zoomed = window_ms < 500.0

    fig, ax = plt.subplots(figsize=figsize)

    if is_zoomed:
        # Raw waveform for zoomed view — can see individual transients
        time_ms = np.arange(len(mono)) / sample_rate * 1000.0
        mask = (time_ms >= lo) & (time_ms <= hi)
        ax.plot(time_ms[mask], mono[mask], linewidth=0.5, color="steelblue", alpha=0.9)
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Waveform ({lo:.0f}-{hi:.0f} ms)")
    else:
        # RMS envelope for full-scale view — much more readable
        env_time, env_rms = _rms_envelope(mono, sample_rate, frame_ms=3.0)
        # Mask to visible window
        mask = (env_time >= lo) & (env_time <= hi)
        t = env_time[mask]
        r = env_rms[mask]
        ax.fill_between(t, -r, r, color="steelblue", alpha=0.6)
        ax.plot(t, r, linewidth=0.4, color="steelblue", alpha=0.9)
        ax.plot(t, -r, linewidth=0.4, color="steelblue", alpha=0.9)
        ax.set_ylabel("RMS Amplitude")
        ax.set_title("Waveform (RMS Envelope)")

    ax.set_xlabel("Time (ms)")
    ax.grid(True, alpha=0.2)

    has_legend = False

    # MIDI onsets: solid red lines, height proportional to velocity
    if midi_onsets:
        for i, onset in enumerate(midi_onsets):
            t_ms = onset.time_sec * 1000.0
            if lo <= t_ms <= hi:
                vel_frac = onset.velocity / 127.0
                ax.axvline(
                    x=t_ms,
                    ymin=0.5 - vel_frac * 0.5,
                    ymax=0.5 + vel_frac * 0.5,
                    color="red",
                    linewidth=1.2,
                    alpha=0.7,
                    label="MIDI onset" if i == 0 else None,
                )
        has_legend = True

    # Detected onsets: blue dashed lines
    if detected_onsets:
        for i, onset in enumerate(detected_onsets):
            t_ms = onset.time_sec * 1000.0
            if lo <= t_ms <= hi:
                ax.axvline(
                    x=t_ms,
                    color="blue",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                    label="Detected onset" if i == 0 else None,
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

        # IOI labels — only for sub-50ms gaps (flams, drags, grace notes)
        for i in range(1, len(midi_onsets)):
            ioi_ms = (midi_onsets[i].time_sec - midi_onsets[i - 1].time_sec) * 1000.0
            if ioi_ms < 50.0:
                mid_x = (times_ms[i - 1] + times_ms[i]) / 2.0
                higher_y = max(velocities[i - 1], velocities[i])
                ax.annotate(
                    f"{ioi_ms:.0f}ms",
                    xy=(mid_x, higher_y + 5),
                    fontsize=7,
                    color="red",
                    fontweight="bold",
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

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # --- Panel 1: RMS envelope with onset markers ---
    ax1 = axes[0]
    env_time, env_rms = _rms_envelope(mono, sample_rate, frame_ms=3.0)
    ax1.fill_between(env_time, -env_rms, env_rms, color="steelblue", alpha=0.6)
    ax1.plot(env_time, env_rms, linewidth=0.4, color="steelblue", alpha=0.9)
    ax1.plot(env_time, -env_rms, linewidth=0.4, color="steelblue", alpha=0.9)
    ax1.set_ylabel("RMS Amplitude")
    ax1.set_title("Waveform (RMS Envelope)")
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


def render_cycle_zoom(
    audio: np.ndarray,
    sample_rate: int,
    midi_onsets: list[MidiOnset],
    detected_onsets: list[DetectedOnset] | None = None,
    activation: np.ndarray | None = None,
    activation_fps: int = 100,
    max_cycles: int = 4,
    output_path: str | Path = "cycles.png",
    dpi: int = 150,
) -> Path:
    """Render zoomed-in views of individual pattern cycles.

    Each cycle gets two rows:
    - Top: RMS envelope (shows amplitude peaks clearly despite drum resonance)
      with MIDI onset lines, velocity labels, and IOI annotations
    - Bottom: Onset activation function (neural net confidence) with detected
      onset markers

    This is the highest-resolution view for inspecting flams, drags, and diddles.
    """
    output_path = Path(output_path)
    mono = _to_mono(audio)

    # Precompute RMS envelope (3ms frames — smooth enough to read, fine enough for flams)
    env_time, env_rms = _rms_envelope(mono, sample_rate, frame_ms=3.0)

    cycles = _find_cycle_boundaries(midi_onsets)
    n_cycles = min(len(cycles), max_cycles)

    # Precompute global activation y-max for consistent scaling across cycles
    act_y_max = 1.0
    if activation is not None and len(activation) > 0:
        act_y_max = float(np.max(activation)) if np.max(activation) > 0 else 1.0

    # 2 rows per cycle: RMS envelope + activation
    n_rows = n_cycles * 2
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(16, 3.0 * n_rows),
        squeeze=False,
    )

    for cyc_idx, (cyc_start, cyc_end) in enumerate(cycles[:n_cycles]):
        ax_env = axes[cyc_idx * 2, 0]
        ax_act = axes[cyc_idx * 2 + 1, 0]

        # --- Top row: RMS envelope with MIDI markers ---
        mask = (env_time >= cyc_start) & (env_time <= cyc_end)
        t_env = env_time[mask]
        r_env = env_rms[mask]

        ax_env.fill_between(t_env, 0, r_env, color="steelblue", alpha=0.4)
        ax_env.plot(t_env, r_env, linewidth=0.8, color="steelblue")

        # MIDI onsets: red lines + velocity labels (staggered to avoid overlap)
        cycle_midi = [o for o in midi_onsets if cyc_start <= o.time_sec * 1000 <= cyc_end]
        rms_max = r_env.max() if len(r_env) > 0 and r_env.max() > 0 else 0.1

        # Pre-classify: grace notes (part of a flam, <50ms from neighbor) vs main strokes
        is_grace = [False] * len(cycle_midi)
        for i in range(len(cycle_midi)):
            if i > 0:
                gap_prev = (cycle_midi[i].time_sec - cycle_midi[i - 1].time_sec) * 1000
                if gap_prev < 50 and cycle_midi[i].velocity > cycle_midi[i - 1].velocity:
                    is_grace[i - 1] = True  # quieter note before loud = grace
            if i < len(cycle_midi) - 1:
                gap_next = (cycle_midi[i + 1].time_sec - cycle_midi[i].time_sec) * 1000
                if gap_next < 50 and cycle_midi[i].velocity < cycle_midi[i + 1].velocity:
                    is_grace[i] = True

        for i, onset in enumerate(cycle_midi):
            t = onset.time_sec * 1000
            vel_frac = onset.velocity / 127.0
            grace = is_grace[i]
            ax_env.axvline(
                x=t,
                color="red",
                linewidth=0.8 if grace else 1.5,
                alpha=0.5 if grace else 0.7,
                linestyle=":" if grace else "-",
                label="MIDI onset" if cyc_idx == 0 and i == 0 else None,
            )
            ax_env.plot(
                t,
                rms_max * 1.05,
                marker="v",
                color="red",
                markersize=4 + vel_frac * 6,
                alpha=0.9,
            )
            # Stagger velocity labels: grace notes go lower to avoid overlap
            label_y = rms_max * (1.0 if grace else 1.15)
            ax_env.annotate(
                f"v{onset.velocity}",
                xy=(t, label_y),
                fontsize=7 if grace else 9,
                color="darkred" if grace else "red",
                fontweight="normal" if grace else "bold",
                ha="center",
                va="bottom",
            )

        # IOI annotations — only sub-50ms gaps (flams/drags/grace notes)
        for i in range(1, len(cycle_midi)):
            ioi = (cycle_midi[i].time_sec - cycle_midi[i - 1].time_sec) * 1000
            if ioi >= 50:
                continue
            mid_t = (cycle_midi[i - 1].time_sec + cycle_midi[i].time_sec) / 2 * 1000
            ax_env.annotate(
                f"{ioi:.0f}ms",
                xy=(mid_t, -rms_max * 0.12),
                fontsize=9,
                color="red",
                fontweight="bold",
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            )

        n_midi = len(cycle_midi)
        ax_env.set_title(
            f"Cycle {cyc_idx + 1}  |  {cyc_start:.0f}\u2013{cyc_end:.0f} ms  |  "
            f"{n_midi} MIDI hits",
            fontsize=10,
            fontweight="bold",
        )
        ax_env.set_ylabel("RMS")
        ax_env.set_xlim(cyc_start, cyc_end)
        ax_env.set_ylim(-rms_max * 0.2, rms_max * 1.5)
        ax_env.grid(True, alpha=0.2)

        # --- Bottom row: activation + detected onsets ---
        if activation is not None and len(activation) > 0:
            act_time = np.arange(len(activation)) / activation_fps * 1000.0
            act_mask = (act_time >= cyc_start) & (act_time <= cyc_end)
            t_act = act_time[act_mask]
            a_act = activation[act_mask]
            ax_act.fill_between(t_act, 0, a_act, color="purple", alpha=0.3)
            ax_act.plot(
                t_act,
                a_act,
                linewidth=1.0,
                color="purple",
                label="Activation" if cyc_idx == 0 else None,
            )

        # MIDI onset reference lines (thin red)
        for onset in cycle_midi:
            ax_act.axvline(x=onset.time_sec * 1000, color="red", linewidth=0.5, alpha=0.3)

        # Detected onsets: blue markers
        if detected_onsets:
            cycle_det = [o for o in detected_onsets if cyc_start <= o.time_sec * 1000 <= cyc_end]
            for i, onset in enumerate(cycle_det):
                ax_act.axvline(
                    x=onset.time_sec * 1000,
                    color="blue",
                    linewidth=1.2,
                    alpha=0.7,
                    label="Detected onset" if cyc_idx == 0 and i == 0 else None,
                )
            n_det = len(cycle_det)
        else:
            n_det = 0

        ax_act.set_ylabel("Activation")
        ax_act.set_xlim(cyc_start, cyc_end)
        ax_act.set_ylim(0, act_y_max * 1.1)  # consistent across cycles
        ax_act.grid(True, alpha=0.2)
        ax_act.set_title(
            f"  Onset activation  |  {n_det} detected",
            fontsize=9,
            loc="left",
        )

    # Add legend to first row pair only
    axes[0, 0].legend(loc="upper right", fontsize=8)
    if n_rows >= 2:
        axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[-1, 0].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)
    return output_path
