"""
PyTorch DataLoader for SOUSA Dataset
====================================

Production-ready PyTorch Dataset and DataLoader implementations for SOUSA.

Features:
- Lazy audio loading with torchaudio
- Configurable resampling (44.1kHz -> 16kHz)
- Custom collate function for variable-length batching
- Support for regression and classification targets
- Memory-efficient streaming option

Usage:
    from examples.pytorch_dataloader import SOUSADataset, create_dataloader

    # Create dataset
    dataset = SOUSADataset(
        data_dir="output/dataset",
        split="train",
        target="skill_tier",  # or "overall_score", "rudiment_slug"
        resample_rate=16000,
    )

    # Create dataloader with custom collate
    dataloader = create_dataloader(dataset, batch_size=16)

    for batch in dataloader:
        waveforms = batch["waveforms"]  # (B, max_length)
        lengths = batch["lengths"]      # (B,)
        labels = batch["labels"]        # (B,) or (B, num_classes)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("Warning: torchaudio not available. Install with: pip install torchaudio")


# Label mappings
SKILL_TIER_LABELS = ["beginner", "intermediate", "advanced", "professional"]
SKILL_TIER_TO_ID = {label: i for i, label in enumerate(SKILL_TIER_LABELS)}
ID_TO_SKILL_TIER = {i: label for label, i in SKILL_TIER_TO_ID.items()}


class SOUSADataset(Dataset):
    """
    PyTorch Dataset for SOUSA drum rudiment samples.

    Args:
        data_dir: Path to SOUSA dataset directory (containing audio/, labels/)
        split: One of "train", "validation", "test", or None for all data
        target: Target variable - "skill_tier", "overall_score", "rudiment_slug",
                or a list of score columns for multi-task learning
        resample_rate: Target sample rate (None to keep original 44.1kHz)
        max_length_sec: Maximum audio length in seconds (None for no limit)
        transform: Optional audio transform function
        normalize_audio: Whether to normalize audio to [-1, 1]

    Example:
        >>> dataset = SOUSADataset("output/dataset", split="train", target="skill_tier")
        >>> sample = dataset[0]
        >>> print(sample["waveform"].shape, sample["label"])
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "validation", "test"] | None = None,
        target: str | list[str] = "overall_score",
        resample_rate: int | None = 16000,
        max_length_sec: float | None = None,
        transform: Callable | None = None,
        normalize_audio: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target = target
        self.resample_rate = resample_rate
        self.max_length_sec = max_length_sec
        self.transform = transform
        self.normalize_audio = normalize_audio

        # Load metadata
        self.samples_df = self._load_samples()
        self.exercises_df = self._load_exercises()

        # Merge for easy access
        self.data = self.samples_df.merge(
            self.exercises_df,
            on="sample_id",
            how="left"
        )

        # Filter by split if specified
        if split is not None:
            self.data = self._filter_by_split(self.data, split)

        # Build rudiment label mapping
        self.rudiments = sorted(self.data["rudiment_slug"].unique())
        self.rudiment_to_id = {r: i for i, r in enumerate(self.rudiments)}
        self.id_to_rudiment = {i: r for r, i in self.rudiment_to_id.items()}

        # Setup resampler
        self.resampler = None
        if resample_rate is not None and TORCHAUDIO_AVAILABLE:
            self.resampler = T.Resample(
                orig_freq=44100,
                new_freq=resample_rate
            )

        # Calculate max length in samples
        self.max_length_samples = None
        if max_length_sec is not None:
            target_sr = resample_rate or 44100
            self.max_length_samples = int(max_length_sec * target_sr)

    def _load_samples(self) -> pd.DataFrame:
        """Load samples.parquet metadata."""
        samples_path = self.data_dir / "labels" / "samples.parquet"
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_path}")
        return pd.read_parquet(samples_path)

    def _load_exercises(self) -> pd.DataFrame:
        """Load exercises.parquet scores."""
        exercises_path = self.data_dir / "labels" / "exercises.parquet"
        if not exercises_path.exists():
            raise FileNotFoundError(f"Exercises file not found: {exercises_path}")
        return pd.read_parquet(exercises_path)

    def _filter_by_split(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Filter dataframe by train/val/test split."""
        splits_path = self.data_dir / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"Splits file not found: {splits_path}. "
                "Generate dataset with splits or use split=None."
            )

        with open(splits_path) as f:
            splits = json.load(f)

        # Handle both "validation" and "val" naming conventions
        if split == "validation":
            split_key = "val_profile_ids" if "val_profile_ids" in splits else "validation_profile_ids"
        else:
            split_key = f"{split}_profile_ids"

        if split_key not in splits:
            raise ValueError(f"Invalid split: {split}. Available keys in splits.json: {list(splits.keys())}")

        profile_ids = set(splits[split_key])
        return df[df["profile_id"].isin(profile_ids)].reset_index(drop=True)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and return waveform tensor."""
        full_path = self.data_dir / audio_path

        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio required for audio loading")

        waveform, sample_rate = torchaudio.load(full_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if self.resampler is not None:
            waveform = self.resampler(waveform)

        # Remove channel dimension
        waveform = waveform.squeeze(0)

        # Normalize
        if self.normalize_audio:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val

        # Truncate if needed
        if self.max_length_samples is not None:
            if len(waveform) > self.max_length_samples:
                waveform = waveform[:self.max_length_samples]

        return waveform

    def _get_label(self, row: pd.Series) -> torch.Tensor | int:
        """Extract label from row based on target specification."""
        if isinstance(self.target, list):
            # Multi-task: return tensor of multiple scores
            values = [row[col] for col in self.target]
            return torch.tensor(values, dtype=torch.float32)

        elif self.target == "skill_tier":
            # Classification: return integer label
            return SKILL_TIER_TO_ID[row["skill_tier"]]

        elif self.target == "rudiment_slug":
            # Classification: return integer label
            return self.rudiment_to_id[row["rudiment_slug"]]

        elif self.target == "overall_score":
            # Regression: return normalized score
            return torch.tensor(row["overall_score"] / 100.0, dtype=torch.float32)

        else:
            # Assume it's a score column for regression
            value = row[self.target]
            if pd.isna(value):
                return torch.tensor(float("nan"), dtype=torch.float32)
            return torch.tensor(value / 100.0, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]

        # Load audio
        waveform = self._load_audio(row["audio_path"])

        # Apply transform if provided
        if self.transform is not None:
            waveform = self.transform(waveform)

        # Get label
        label = self._get_label(row)

        return {
            "waveform": waveform,
            "label": label,
            "sample_id": row["sample_id"],
            "skill_tier": row["skill_tier"],
            "rudiment_slug": row["rudiment_slug"],
            "tempo_bpm": row["tempo_bpm"],
        }

    @property
    def num_classes(self) -> int | None:
        """Return number of classes for classification targets."""
        if self.target == "skill_tier":
            return len(SKILL_TIER_LABELS)
        elif self.target == "rudiment_slug":
            return len(self.rudiments)
        return None  # Regression task


def collate_variable_length(
    batch: list[dict],
    pad_value: float = 0.0,
) -> dict:
    """
    Collate function for variable-length audio.

    Pads waveforms to the maximum length in the batch.

    Args:
        batch: List of sample dicts from SOUSADataset
        pad_value: Value to use for padding

    Returns:
        Dictionary with batched tensors:
        - waveforms: (B, max_length) padded waveforms
        - lengths: (B,) original lengths before padding
        - labels: (B,) or (B, num_targets) labels
        - attention_mask: (B, max_length) mask where 1 = valid, 0 = padding
        - sample_ids: list of sample IDs
        - metadata: dict of other metadata
    """
    waveforms = [sample["waveform"] for sample in batch]
    labels = [sample["label"] for sample in batch]

    # Record original lengths
    lengths = torch.tensor([len(w) for w in waveforms])

    # Pad waveforms
    padded_waveforms = pad_sequence(
        waveforms,
        batch_first=True,
        padding_value=pad_value
    )

    # Create attention mask
    max_len = padded_waveforms.shape[1]
    attention_mask = torch.arange(max_len).expand(len(batch), -1) < lengths.unsqueeze(1)

    # Stack labels
    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)
    else:
        labels = torch.tensor(labels)

    return {
        "waveforms": padded_waveforms,
        "lengths": lengths,
        "labels": labels,
        "attention_mask": attention_mask.float(),
        "sample_ids": [s["sample_id"] for s in batch],
        "metadata": {
            "skill_tiers": [s["skill_tier"] for s in batch],
            "rudiment_slugs": [s["rudiment_slug"] for s in batch],
            "tempo_bpms": [s["tempo_bpm"] for s in batch],
        },
    }


def collate_fixed_length(
    batch: list[dict],
    target_length: int,
    pad_value: float = 0.0,
) -> dict:
    """
    Collate function that pads/truncates to a fixed length.

    More efficient for training as all batches have same shape.

    Args:
        batch: List of sample dicts
        target_length: Fixed length to pad/truncate to
        pad_value: Value for padding

    Returns:
        Dictionary with batched tensors (same as collate_variable_length)
    """
    waveforms = []
    lengths = []

    for sample in batch:
        waveform = sample["waveform"]
        orig_len = len(waveform)
        lengths.append(min(orig_len, target_length))

        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        elif len(waveform) < target_length:
            padding = target_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, padding), value=pad_value)

        waveforms.append(waveform)

    labels = [sample["label"] for sample in batch]

    waveforms = torch.stack(waveforms)
    lengths = torch.tensor(lengths)
    attention_mask = torch.arange(target_length).expand(len(batch), -1) < lengths.unsqueeze(1)

    if isinstance(labels[0], torch.Tensor):
        labels = torch.stack(labels)
    else:
        labels = torch.tensor(labels)

    return {
        "waveforms": waveforms,
        "lengths": lengths,
        "labels": labels,
        "attention_mask": attention_mask.float(),
        "sample_ids": [s["sample_id"] for s in batch],
        "metadata": {
            "skill_tiers": [s["skill_tier"] for s in batch],
            "rudiment_slugs": [s["rudiment_slug"] for s in batch],
            "tempo_bpms": [s["tempo_bpm"] for s in batch],
        },
    }


def create_dataloader(
    dataset: SOUSADataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    fixed_length: int | None = None,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader with appropriate collate function.

    Args:
        dataset: SOUSADataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        fixed_length: If provided, use fixed-length collation (more efficient)
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader
    """
    if fixed_length is not None:
        collate_fn = lambda batch: collate_fixed_length(batch, fixed_length)
    else:
        collate_fn = collate_variable_length

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **kwargs,
    )


class StratifiedBatchSampler:
    """
    Batch sampler that ensures each batch has balanced skill tier representation.

    Useful for classification tasks where class imbalance is a concern.
    """

    def __init__(
        self,
        dataset: SOUSADataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by skill tier
        self.tier_indices = {}
        for tier in SKILL_TIER_LABELS:
            mask = dataset.data["skill_tier"] == tier
            self.tier_indices[tier] = dataset.data[mask].index.tolist()

        # Samples per tier per batch
        self.samples_per_tier = batch_size // len(SKILL_TIER_LABELS)

    def __iter__(self):
        # Shuffle indices within each tier
        shuffled = {
            tier: np.random.permutation(indices).tolist()
            for tier, indices in self.tier_indices.items()
        }

        # Track position in each tier
        positions = {tier: 0 for tier in SKILL_TIER_LABELS}

        while True:
            batch = []

            for tier in SKILL_TIER_LABELS:
                start = positions[tier]
                end = start + self.samples_per_tier

                if end > len(shuffled[tier]):
                    # Reshuffle and reset if exhausted
                    shuffled[tier] = np.random.permutation(
                        self.tier_indices[tier]
                    ).tolist()
                    positions[tier] = 0
                    start = 0
                    end = self.samples_per_tier

                batch.extend(shuffled[tier][start:end])
                positions[tier] = end

            # Check if we've gone through enough data
            total_used = sum(positions.values())
            total_available = sum(len(v) for v in self.tier_indices.values())

            if total_used >= total_available:
                if not self.drop_last or len(batch) == self.batch_size:
                    yield batch
                break

            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.tier_indices.values())
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


# Example usage
if __name__ == "__main__":
    import sys

    # Check for dataset path argument
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "output/dataset"

    print(f"Loading SOUSA dataset from: {data_dir}")

    # Create dataset
    try:
        dataset = SOUSADataset(
            data_dir=data_dir,
            split="train",
            target="skill_tier",
            resample_rate=16000,
            max_length_sec=5.0,
        )
        print(f"Dataset size: {len(dataset)} samples")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Rudiments: {len(dataset.rudiments)}")

        # Get a sample
        sample = dataset[0]
        print(f"\nSample waveform shape: {sample['waveform'].shape}")
        print(f"Sample label: {sample['label']} ({ID_TO_SKILL_TIER[sample['label']]})")
        print(f"Sample ID: {sample['sample_id']}")

        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging
            fixed_length=80000,  # 5 seconds at 16kHz
        )

        # Get a batch
        batch = next(iter(dataloader))
        print(f"\nBatch waveforms shape: {batch['waveforms'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate a dataset first with: python scripts/generate_dataset.py --preset small")
