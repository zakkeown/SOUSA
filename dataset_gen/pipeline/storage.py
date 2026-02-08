"""
Storage utilities for dataset persistence.

This module handles writing samples to disk in efficient formats:
- Parquet for labels (hierarchical, compressed)
- FLAC for audio
- Standard MIDI files
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from dataset_gen.labels.schema import Sample


@dataclass
class StorageConfig:
    """Configuration for dataset storage."""

    output_dir: Path
    audio_format: Literal["flac", "wav"] = "flac"
    audio_subtype: str = "PCM_24"  # PCM_16, PCM_24
    sample_rate: int = 44100

    # Directory structure
    midi_subdir: str = "midi"
    audio_subdir: str = "audio"
    labels_subdir: str = "labels"

    # Parquet settings
    compression: str = "snappy"
    row_group_size: int = 1000

    # Whether to create index files
    create_index: bool = True

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


class DatasetWriter:
    """
    Write dataset samples to disk in organized format.

    Directory structure:
    output_dir/
    ├── midi/
    │   └── {sample_id}.mid
    ├── audio/
    │   └── {sample_id}.flac
    ├── labels/
    │   ├── strokes.parquet
    │   ├── measures.parquet
    │   ├── exercises.parquet
    │   └── samples.parquet
    └── index.json
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize dataset writer.

        Args:
            config: Storage configuration
        """
        self.config = config
        self._setup_directories()

        # Accumulators for batch writing
        self._stroke_records: list[dict] = []
        self._measure_records: list[dict] = []
        self._exercise_records: list[dict] = []
        self._sample_records: list[dict] = []

        # Index tracking
        self._sample_ids: list[str] = []

    def _setup_directories(self) -> None:
        """Create output directory structure."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / self.config.midi_subdir).mkdir(exist_ok=True)
        (self.config.output_dir / self.config.audio_subdir).mkdir(exist_ok=True)
        (self.config.output_dir / self.config.labels_subdir).mkdir(exist_ok=True)

    def write_sample(
        self,
        sample: Sample,
        midi_data: bytes | None = None,
        audio_data: np.ndarray | None = None,
    ) -> dict[str, Path]:
        """
        Write a single sample to disk.

        Args:
            sample: Sample with labels
            midi_data: Raw MIDI bytes (optional)
            audio_data: Audio samples as numpy array (optional)

        Returns:
            Dictionary of written file paths
        """
        paths = {}

        # Write MIDI
        if midi_data is not None:
            midi_path = self._write_midi(sample.sample_id, midi_data)
            paths["midi"] = midi_path
            sample.midi_path = str(midi_path.relative_to(self.config.output_dir))

        # Write audio
        if audio_data is not None:
            audio_path = self._write_audio(sample.sample_id, audio_data)
            paths["audio"] = audio_path
            sample.audio_path = str(audio_path.relative_to(self.config.output_dir))

        # Accumulate label records
        self._accumulate_labels(sample)
        self._sample_ids.append(sample.sample_id)

        return paths

    def _write_midi(self, sample_id: str, midi_data: bytes) -> Path:
        """Write MIDI data to file."""
        midi_path = self.config.output_dir / self.config.midi_subdir / f"{sample_id}.mid"
        midi_path.write_bytes(midi_data)
        return midi_path

    def _write_audio(self, sample_id: str, audio_data: np.ndarray) -> Path:
        """Write audio data to file."""
        ext = self.config.audio_format
        audio_path = self.config.output_dir / self.config.audio_subdir / f"{sample_id}.{ext}"

        sf.write(
            str(audio_path),
            audio_data,
            self.config.sample_rate,
            format=self.config.audio_format.upper(),
            subtype=self.config.audio_subtype,
        )
        return audio_path

    def _accumulate_labels(self, sample: Sample) -> None:
        """Accumulate label records for batch writing."""
        sample_id = sample.sample_id

        # Stroke-level records
        for stroke in sample.strokes:
            record = stroke.model_dump()
            record["sample_id"] = sample_id
            self._stroke_records.append(record)

        # Measure-level records
        for measure in sample.measures:
            record = measure.model_dump()
            record["sample_id"] = sample_id
            self._measure_records.append(record)

        # Exercise-level record
        exercise_record = sample.exercise_scores.model_dump()
        exercise_record["sample_id"] = sample_id
        self._exercise_records.append(exercise_record)

        # Sample-level record (metadata)
        sample_record = {
            "sample_id": sample_id,
            "profile_id": sample.profile_id,
            "rudiment_slug": sample.rudiment_slug,
            "tempo_bpm": sample.tempo_bpm,
            "duration_sec": sample.duration_sec,
            "num_cycles": sample.num_cycles,
            "skill_tier": sample.skill_tier,
            "skill_tier_binary": sample.skill_tier_binary,
            "dominant_hand": sample.dominant_hand,
            "midi_path": sample.midi_path,
            "audio_path": sample.audio_path,
            "num_strokes": len(sample.strokes),
            "num_measures": len(sample.measures),
            # Augmentation tracking (always present for ML filtering)
            "soundfont": sample.soundfont,
            "augmentation_preset": sample.augmentation_preset,
            "augmentation_group_id": sample.augmentation_group_id,
        }

        # Add detailed audio augmentation params if present
        if sample.audio_augmentation:
            aug = sample.audio_augmentation.model_dump()
            for key, value in aug.items():
                sample_record[f"aug_{key}"] = value

        self._sample_records.append(sample_record)

    def flush(self) -> dict[str, Path]:
        """
        Write accumulated labels to Parquet files.

        Returns:
            Dictionary of written Parquet file paths
        """
        paths = {}
        labels_dir = self.config.output_dir / self.config.labels_subdir

        # Write strokes
        if self._stroke_records:
            strokes_path = labels_dir / "strokes.parquet"
            self._write_parquet(self._stroke_records, strokes_path)
            paths["strokes"] = strokes_path
            self._stroke_records = []

        # Write measures
        if self._measure_records:
            measures_path = labels_dir / "measures.parquet"
            self._write_parquet(self._measure_records, measures_path)
            paths["measures"] = measures_path
            self._measure_records = []

        # Write exercises
        if self._exercise_records:
            exercises_path = labels_dir / "exercises.parquet"
            self._write_parquet(self._exercise_records, exercises_path)
            paths["exercises"] = exercises_path
            self._exercise_records = []

        # Write samples
        if self._sample_records:
            samples_path = labels_dir / "samples.parquet"
            self._write_parquet(self._sample_records, samples_path)
            paths["samples"] = samples_path
            self._sample_records = []

        # Write index
        if self.config.create_index:
            index_path = self._write_index()
            paths["index"] = index_path

        return paths

    def _write_parquet(self, records: list[dict], path: Path) -> None:
        """Write records to Parquet file, appending to existing if present.

        Uses atomic writes (write to temp, then rename) to prevent corruption
        if the process is interrupted mid-write.
        """
        import tempfile
        import os

        new_df = pd.DataFrame(records)

        # Append to existing file if it exists
        if path.exists():
            try:
                existing_df = pd.read_parquet(path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Deduplicate based on sample_id (and index for strokes)
                if "sample_id" in combined_df.columns:
                    if "index" in combined_df.columns:
                        # Strokes/measures: dedupe on (sample_id, index)
                        combined_df = combined_df.drop_duplicates(
                            subset=["sample_id", "index"], keep="last"
                        )
                    else:
                        # Samples/exercises: dedupe on sample_id
                        combined_df = combined_df.drop_duplicates(subset=["sample_id"], keep="last")
                table = pa.Table.from_pandas(combined_df)
            except Exception as e:
                # If existing file is corrupted, log warning and use only new data
                import logging

                logging.getLogger(__name__).warning(
                    f"Could not read existing parquet file {path}, starting fresh: {e}"
                )
                table = pa.Table.from_pandas(new_df)
        else:
            table = pa.Table.from_pandas(new_df)

        # Write to temp file first, then atomically rename
        # This prevents corruption if the process is interrupted mid-write
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".parquet", dir=path.parent, prefix=f".{path.stem}_"
        )
        os.close(temp_fd)

        try:
            pq.write_table(
                table,
                temp_path,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
            )
            # Atomic rename (on POSIX systems)
            os.replace(temp_path, path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _write_index(self) -> Path:
        """Write dataset index file."""
        index_path = self.config.output_dir / "index.json"

        index_data = {
            "num_samples": len(self._sample_ids),
            "sample_ids": self._sample_ids,
            "config": {
                "audio_format": self.config.audio_format,
                "sample_rate": self.config.sample_rate,
            },
        }

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        return index_path

    def get_stats(self) -> dict:
        """Get statistics about accumulated data."""
        return {
            "samples_pending": len(self._sample_records),
            "strokes_pending": len(self._stroke_records),
            "measures_pending": len(self._measure_records),
            "exercises_pending": len(self._exercise_records),
            "total_samples_written": len(self._sample_ids),
        }


class ParquetReader:
    """Read dataset from Parquet files."""

    def __init__(self, dataset_dir: Path | str):
        """
        Initialize reader.

        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.labels_dir = self.dataset_dir / "labels"

    def load_samples(self) -> pd.DataFrame:
        """Load sample metadata."""
        return pd.read_parquet(self.labels_dir / "samples.parquet")

    def load_strokes(self, sample_ids: list[str] | None = None) -> pd.DataFrame:
        """Load stroke-level labels, optionally filtered by sample IDs."""
        df = pd.read_parquet(self.labels_dir / "strokes.parquet")
        if sample_ids is not None:
            df = df[df["sample_id"].isin(sample_ids)]
        return df

    def load_measures(self, sample_ids: list[str] | None = None) -> pd.DataFrame:
        """Load measure-level labels, optionally filtered by sample IDs."""
        df = pd.read_parquet(self.labels_dir / "measures.parquet")
        if sample_ids is not None:
            df = df[df["sample_id"].isin(sample_ids)]
        return df

    def load_exercises(self, sample_ids: list[str] | None = None) -> pd.DataFrame:
        """Load exercise-level scores, optionally filtered by sample IDs."""
        df = pd.read_parquet(self.labels_dir / "exercises.parquet")
        if sample_ids is not None:
            df = df[df["sample_id"].isin(sample_ids)]
        return df

    def load_index(self) -> dict:
        """Load dataset index."""
        with open(self.dataset_dir / "index.json") as f:
            return json.load(f)

    def get_sample(self, sample_id: str) -> dict:
        """
        Load all data for a single sample.

        Returns:
            Dictionary with sample metadata, strokes, measures, and scores
        """
        samples = self.load_samples()
        sample_row = samples[samples["sample_id"] == sample_id].iloc[0]

        return {
            "metadata": sample_row.to_dict(),
            "strokes": self.load_strokes([sample_id]).to_dict(orient="records"),
            "measures": self.load_measures([sample_id]).to_dict(orient="records"),
            "scores": self.load_exercises([sample_id]).iloc[0].to_dict(),
        }


def write_sample(
    sample: Sample,
    output_dir: Path | str,
    midi_data: bytes | None = None,
    audio_data: np.ndarray | None = None,
    audio_format: str = "flac",
) -> dict[str, Path]:
    """
    Convenience function to write a single sample.

    Args:
        sample: Sample with labels
        output_dir: Output directory
        midi_data: Raw MIDI bytes
        audio_data: Audio samples
        audio_format: Output audio format

    Returns:
        Dictionary of written file paths
    """
    config = StorageConfig(
        output_dir=Path(output_dir),
        audio_format=audio_format,
    )
    writer = DatasetWriter(config)
    paths = writer.write_sample(sample, midi_data, audio_data)
    writer.flush()
    return paths


def write_batch(
    samples: list[Sample],
    output_dir: Path | str,
    midi_data_list: list[bytes | None] | None = None,
    audio_data_list: list[np.ndarray | None] | None = None,
    audio_format: str = "flac",
) -> dict[str, Path]:
    """
    Convenience function to write a batch of samples.

    Args:
        samples: List of samples with labels
        output_dir: Output directory
        midi_data_list: List of MIDI bytes (parallel to samples)
        audio_data_list: List of audio arrays (parallel to samples)
        audio_format: Output audio format

    Returns:
        Dictionary of written Parquet file paths
    """
    config = StorageConfig(
        output_dir=Path(output_dir),
        audio_format=audio_format,
    )
    writer = DatasetWriter(config)

    if midi_data_list is None:
        midi_data_list = [None] * len(samples)
    if audio_data_list is None:
        audio_data_list = [None] * len(samples)

    for sample, midi, audio in zip(samples, midi_data_list, audio_data_list):
        writer.write_sample(sample, midi, audio)

    return writer.flush()
