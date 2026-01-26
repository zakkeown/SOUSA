"""
Hierarchical Label Access for SOUSA Dataset
============================================

Functions and utilities for accessing SOUSA's three-level label hierarchy:
- Exercise level: Overall performance scores
- Measure level: Per-measure aggregate statistics
- Stroke level: Individual stroke timing and velocity

This module shows how to:
1. Get all strokes for a sample
2. Get all measures for a sample
3. Align stroke timestamps with audio samples
4. Create stroke-level sequences for RNN/Transformer input
5. Visualize stroke timeline overlaid on waveform

Usage:
    from examples.hierarchical_labels import HierarchicalLabels

    labels = HierarchicalLabels("output/dataset")

    # Get strokes for a sample
    strokes = labels.get_strokes("adv042_single_paradiddle_100bpm_marching_practiceroom")

    # Align with audio
    aligned = labels.align_strokes_with_audio(sample_id, sample_rate=16000)

    # Create sequence for RNN
    sequence = labels.create_stroke_sequence(sample_id, max_strokes=128)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class HierarchicalLabels:
    """
    Access hierarchical labels (exercise -> measures -> strokes) for SOUSA samples.

    Args:
        data_dir: Path to SOUSA dataset directory

    Example:
        >>> labels = HierarchicalLabels("output/dataset")
        >>> strokes = labels.get_strokes("sample_id")
        >>> measures = labels.get_measures("sample_id")
        >>> exercise = labels.get_exercise("sample_id")
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._load_labels()

    def _load_labels(self):
        """Load all label parquet files."""
        labels_dir = self.data_dir / "labels"

        # Load each level
        self.samples_df = pd.read_parquet(labels_dir / "samples.parquet")
        self.exercises_df = pd.read_parquet(labels_dir / "exercises.parquet")
        self.measures_df = pd.read_parquet(labels_dir / "measures.parquet")
        self.strokes_df = pd.read_parquet(labels_dir / "strokes.parquet")

        # Index for fast lookup
        self.exercises_df.set_index("sample_id", inplace=True, drop=False)
        self._strokes_grouped = self.strokes_df.groupby("sample_id")
        self._measures_grouped = self.measures_df.groupby("sample_id")

    def get_sample_ids(self) -> list[str]:
        """Get all sample IDs in the dataset."""
        return self.samples_df["sample_id"].tolist()

    def get_sample_metadata(self, sample_id: str) -> dict:
        """Get metadata for a sample (rudiment, tempo, skill tier, etc.)."""
        row = self.samples_df[self.samples_df["sample_id"] == sample_id].iloc[0]
        return row.to_dict()

    def get_exercise(self, sample_id: str) -> dict:
        """
        Get exercise-level scores for a sample.

        Returns:
            Dictionary with all exercise scores (timing_accuracy, overall_score, etc.)
        """
        if sample_id not in self.exercises_df.index:
            raise KeyError(f"Sample not found: {sample_id}")
        return self.exercises_df.loc[sample_id].to_dict()

    def get_measures(self, sample_id: str) -> pd.DataFrame:
        """
        Get measure-level labels for a sample.

        Returns:
            DataFrame with one row per measure, including:
            - index: Measure number (0-based)
            - stroke_start, stroke_end: Stroke indices in this measure
            - timing_mean_error_ms, timing_std_ms: Timing statistics
            - velocity_mean, velocity_std: Velocity statistics
            - lr_velocity_ratio: Hand balance (if applicable)
        """
        try:
            return self._measures_grouped.get_group(sample_id).reset_index(drop=True)
        except KeyError:
            raise KeyError(f"Sample not found: {sample_id}")

    def get_strokes(self, sample_id: str) -> pd.DataFrame:
        """
        Get stroke-level labels for a sample.

        Returns:
            DataFrame with one row per stroke, including:
            - index: Stroke number (0-based)
            - hand: "L" or "R"
            - stroke_type: Type of stroke (tap, accent, grace, diddle)
            - intended_time_ms, actual_time_ms: Timing
            - timing_error_ms: actual - intended
            - intended_velocity, actual_velocity: Velocity
            - is_grace_note, is_accent: Articulation flags
        """
        try:
            return self._strokes_grouped.get_group(sample_id).reset_index(drop=True)
        except KeyError:
            raise KeyError(f"Sample not found: {sample_id}")

    def get_strokes_for_measure(
        self,
        sample_id: str,
        measure_index: int
    ) -> pd.DataFrame:
        """
        Get strokes within a specific measure.

        Args:
            sample_id: Sample identifier
            measure_index: 0-based measure index

        Returns:
            DataFrame of strokes in that measure
        """
        measures = self.get_measures(sample_id)
        measure = measures[measures["index"] == measure_index].iloc[0]

        strokes = self.get_strokes(sample_id)
        return strokes[
            (strokes["index"] >= measure["stroke_start"]) &
            (strokes["index"] < measure["stroke_end"])
        ].reset_index(drop=True)

    def align_strokes_with_audio(
        self,
        sample_id: str,
        sample_rate: int = 16000,
        original_sample_rate: int = 44100,
    ) -> pd.DataFrame:
        """
        Align stroke timestamps with audio sample indices.

        Adds columns for the audio sample indices where each stroke occurs,
        useful for creating aligned labels for sequence models.

        Args:
            sample_id: Sample identifier
            sample_rate: Target sample rate (after resampling)
            original_sample_rate: Original audio sample rate

        Returns:
            DataFrame with additional columns:
            - intended_sample_idx: Audio sample index for intended time
            - actual_sample_idx: Audio sample index for actual time
        """
        strokes = self.get_strokes(sample_id)

        # Convert milliseconds to sample indices
        # time_ms * (sample_rate / 1000) = sample_idx
        ms_to_samples = sample_rate / 1000.0

        strokes = strokes.copy()
        strokes["intended_sample_idx"] = (
            strokes["intended_time_ms"] * ms_to_samples
        ).astype(int)
        strokes["actual_sample_idx"] = (
            strokes["actual_time_ms"] * ms_to_samples
        ).astype(int)

        return strokes

    def create_stroke_sequence(
        self,
        sample_id: str,
        max_strokes: int = 128,
        features: list[str] | None = None,
        pad_value: float = 0.0,
    ) -> dict:
        """
        Create a fixed-length stroke sequence for RNN/Transformer input.

        Extracts stroke features into a padded sequence suitable for
        sequence modeling.

        Args:
            sample_id: Sample identifier
            max_strokes: Maximum sequence length (pad/truncate to this)
            features: List of stroke columns to include (default: timing and velocity)
            pad_value: Value to use for padding

        Returns:
            Dictionary with:
            - sequence: (max_strokes, num_features) numpy array
            - mask: (max_strokes,) boolean mask (True = valid stroke)
            - num_strokes: Original number of strokes
            - feature_names: List of feature names
        """
        if features is None:
            features = [
                "timing_error_ms",
                "actual_velocity",
                "is_accent",
                "is_grace_note",
            ]

        strokes = self.get_strokes(sample_id)
        num_strokes = len(strokes)

        # Extract features
        sequence = np.full((max_strokes, len(features)), pad_value, dtype=np.float32)
        mask = np.zeros(max_strokes, dtype=bool)

        # Fill in actual values
        n_fill = min(num_strokes, max_strokes)
        for i, feat in enumerate(features):
            values = strokes[feat].values[:n_fill]
            # Handle boolean columns
            if values.dtype == bool:
                values = values.astype(np.float32)
            sequence[:n_fill, i] = values

        mask[:n_fill] = True

        return {
            "sequence": sequence,
            "mask": mask,
            "num_strokes": num_strokes,
            "feature_names": features,
        }

    def create_stroke_sequence_torch(
        self,
        sample_id: str,
        max_strokes: int = 128,
        features: list[str] | None = None,
    ) -> dict:
        """
        Create stroke sequence as PyTorch tensors.

        Same as create_stroke_sequence but returns torch tensors.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        result = self.create_stroke_sequence(
            sample_id, max_strokes, features, pad_value=0.0
        )

        return {
            "sequence": torch.from_numpy(result["sequence"]),
            "mask": torch.from_numpy(result["mask"]),
            "num_strokes": result["num_strokes"],
            "feature_names": result["feature_names"],
        }

    def create_onset_targets(
        self,
        sample_id: str,
        duration_ms: float,
        resolution_ms: float = 10.0,
    ) -> np.ndarray:
        """
        Create frame-level onset detection targets.

        Creates a binary array indicating stroke onsets at each time frame.

        Args:
            sample_id: Sample identifier
            duration_ms: Total duration in milliseconds
            resolution_ms: Frame resolution in milliseconds

        Returns:
            Binary array of shape (num_frames,) where 1 = onset
        """
        strokes = self.get_strokes(sample_id)
        num_frames = int(duration_ms / resolution_ms)
        targets = np.zeros(num_frames, dtype=np.float32)

        for _, stroke in strokes.iterrows():
            frame_idx = int(stroke["actual_time_ms"] / resolution_ms)
            if 0 <= frame_idx < num_frames:
                targets[frame_idx] = 1.0

        return targets

    def create_hand_targets(
        self,
        sample_id: str,
        duration_ms: float,
        resolution_ms: float = 10.0,
    ) -> np.ndarray:
        """
        Create frame-level hand (L/R) targets.

        Creates an array indicating which hand is playing at each onset.

        Args:
            sample_id: Sample identifier
            duration_ms: Total duration in milliseconds
            resolution_ms: Frame resolution in milliseconds

        Returns:
            Array of shape (num_frames,) with values:
            - 0: No stroke
            - 1: Left hand
            - 2: Right hand
        """
        strokes = self.get_strokes(sample_id)
        num_frames = int(duration_ms / resolution_ms)
        targets = np.zeros(num_frames, dtype=np.int64)

        for _, stroke in strokes.iterrows():
            frame_idx = int(stroke["actual_time_ms"] / resolution_ms)
            if 0 <= frame_idx < num_frames:
                targets[frame_idx] = 1 if stroke["hand"] == "L" else 2

        return targets

    def get_timing_error_sequence(
        self,
        sample_id: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Get sequence of timing errors for all strokes.

        Useful for regression targets in sequence models.

        Args:
            sample_id: Sample identifier
            normalize: Normalize errors to roughly [-1, 1] range

        Returns:
            Array of timing errors (positive = late, negative = early)
        """
        strokes = self.get_strokes(sample_id)
        errors = strokes["timing_error_ms"].values.astype(np.float32)

        if normalize:
            # Normalize by typical range (~50ms max error)
            errors = errors / 50.0
            errors = np.clip(errors, -1.0, 1.0)

        return errors

    def visualize_strokes(
        self,
        sample_id: str,
        audio_array: np.ndarray | None = None,
        sample_rate: int = 16000,
        figsize: tuple[int, int] = (14, 8),
        save_path: str | None = None,
    ):
        """
        Visualize stroke timeline, optionally overlaid on waveform.

        Creates a multi-panel figure showing:
        - Waveform with stroke markers (if audio provided)
        - Timing errors over time
        - Velocity pattern
        - Hand alternation

        Args:
            sample_id: Sample identifier
            audio_array: Optional audio waveform array
            sample_rate: Audio sample rate
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required. Install with: pip install matplotlib")

        strokes = self.get_strokes(sample_id)
        metadata = self.get_sample_metadata(sample_id)

        # Create figure
        n_panels = 4 if audio_array is not None else 3
        fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)

        # Time axis in seconds
        times_sec = strokes["actual_time_ms"].values / 1000.0

        # Panel 1: Waveform with strokes (if audio provided)
        panel_idx = 0
        if audio_array is not None:
            ax = axes[panel_idx]
            time_axis = np.arange(len(audio_array)) / sample_rate

            ax.plot(time_axis, audio_array, color="steelblue", alpha=0.7, linewidth=0.5)

            # Mark strokes
            for _, stroke in strokes.iterrows():
                time_sec = stroke["actual_time_ms"] / 1000.0
                color = "red" if stroke["hand"] == "R" else "blue"
                ax.axvline(time_sec, color=color, alpha=0.3, linewidth=1)

            ax.set_ylabel("Amplitude")
            ax.set_title(f"Waveform with Strokes: {metadata['rudiment_slug']} @ {metadata['tempo_bpm']} BPM")
            ax.legend(["Audio", "R hand", "L hand"], loc="upper right")
            panel_idx += 1

        # Panel 2: Timing errors
        ax = axes[panel_idx]
        colors = ["red" if h == "R" else "blue" for h in strokes["hand"]]
        ax.bar(times_sec, strokes["timing_error_ms"], width=0.02, color=colors, alpha=0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
        ax.set_ylabel("Timing Error (ms)")
        ax.set_title("Timing Errors (+ = late, - = early)")
        panel_idx += 1

        # Panel 3: Velocity
        ax = axes[panel_idx]
        ax.scatter(times_sec, strokes["actual_velocity"], c=colors, alpha=0.7, s=30)
        ax.set_ylabel("Velocity (0-127)")
        ax.set_title("Stroke Velocities")
        panel_idx += 1

        # Panel 4: Hand pattern
        ax = axes[panel_idx]
        hand_numeric = [1 if h == "R" else 0 for h in strokes["hand"]]
        ax.step(times_sec, hand_numeric, where="mid", color="purple", linewidth=1.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["L", "R"])
        ax.set_ylabel("Hand")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Sticking Pattern")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved visualization to: {save_path}")

        return fig, axes

    def compute_measure_summary(self, sample_id: str) -> pd.DataFrame:
        """
        Compute summary statistics for each measure.

        Combines measure-level labels with additional computed metrics.

        Returns:
            DataFrame with one row per measure including computed metrics
        """
        measures = self.get_measures(sample_id)
        strokes = self.get_strokes(sample_id)

        summaries = []
        for _, measure in measures.iterrows():
            measure_strokes = strokes[
                (strokes["index"] >= measure["stroke_start"]) &
                (strokes["index"] < measure["stroke_end"])
            ]

            summary = measure.to_dict()
            summary["num_strokes"] = len(measure_strokes)
            summary["num_accents"] = measure_strokes["is_accent"].sum()
            summary["num_grace_notes"] = measure_strokes["is_grace_note"].sum()
            summary["pct_right_hand"] = (measure_strokes["hand"] == "R").mean()

            summaries.append(summary)

        return pd.DataFrame(summaries)


def load_hierarchical_labels(data_dir: str | Path) -> HierarchicalLabels:
    """Convenience function to load hierarchical labels."""
    return HierarchicalLabels(data_dir)


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "output/dataset"

    print(f"Loading hierarchical labels from: {data_dir}")

    try:
        labels = HierarchicalLabels(data_dir)

        # Get a sample ID
        sample_ids = labels.get_sample_ids()
        print(f"Found {len(sample_ids)} samples")

        if sample_ids:
            sample_id = sample_ids[0]
            print(f"\nExample sample: {sample_id}")

            # Get exercise scores
            exercise = labels.get_exercise(sample_id)
            print(f"Overall score: {exercise['overall_score']:.1f}")
            print(f"Timing accuracy: {exercise['timing_accuracy']:.1f}")

            # Get measures
            measures = labels.get_measures(sample_id)
            print(f"\nNumber of measures: {len(measures)}")
            print(f"Measure columns: {list(measures.columns)}")

            # Get strokes
            strokes = labels.get_strokes(sample_id)
            print(f"\nNumber of strokes: {len(strokes)}")
            print(f"Stroke columns: {list(strokes.columns)}")

            # Create stroke sequence
            sequence = labels.create_stroke_sequence(sample_id, max_strokes=64)
            print(f"\nStroke sequence shape: {sequence['sequence'].shape}")
            print(f"Sequence mask sum: {sequence['mask'].sum()}")

            # Align with audio
            aligned = labels.align_strokes_with_audio(sample_id, sample_rate=16000)
            print(f"\nFirst stroke sample idx: {aligned['actual_sample_idx'].iloc[0]}")

            # Compute measure summary
            summary = labels.compute_measure_summary(sample_id)
            print(f"\nMeasure summary columns: {list(summary.columns)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate a dataset first with: python scripts/generate_dataset.py --preset small")
