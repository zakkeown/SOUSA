#!/usr/bin/env python3
"""
Augmentation Impact Ablation Study
===================================

Research question: Do augmentations help or hurt? Does the model learn acoustic artifacts?

Experiment design:
1. Train on clean only → test on clean
2. Train on clean only → test on augmented
3. Train on augmented → test on clean
4. Train on augmented → test on augmented
5. Train on mixed → test on both

Note: This experiment requires a dataset generated with both clean and augmented samples.
The SOUSA dataset should have an 'augmentation_preset' column to distinguish them.

Output:
    experiments/results/augmentation_ablation.json

Usage:
    python -m experiments.augmentation_ablation --data-dir output/dataset --preset fast_debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.pytorch_dataloader import SOUSADataset, collate_fixed_length
from examples.baselines.config import get_preset
from examples.baselines.models import Wav2Vec2Classifier
from examples.baselines.training import seed_everything


class FilteredDataset(SOUSADataset):
    """Dataset filtered by augmentation condition."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        augmentation_filter: str | None = None,  # "clean", "augmented", or None for all
        **kwargs,
    ):
        super().__init__(data_dir, split=split, **kwargs)

        # Filter by augmentation if column exists
        if "augmentation_preset" in self.data.columns and augmentation_filter:
            if augmentation_filter == "clean":
                mask = self.data["augmentation_preset"].isna() | (self.data["augmentation_preset"] == "none")
            else:  # "augmented"
                mask = self.data["augmentation_preset"].notna() & (self.data["augmentation_preset"] != "none")

            self.data = self.data[mask].reset_index(drop=True)


def check_augmentation_support(data_dir: Path) -> tuple[bool, dict]:
    """Check if dataset has augmentation metadata."""
    samples_path = data_dir / "labels" / "samples.parquet"
    df = pd.read_parquet(samples_path)

    has_augmentation_column = "augmentation_preset" in df.columns

    if has_augmentation_column:
        presets = df["augmentation_preset"].value_counts(dropna=False).to_dict()
        n_clean = df["augmentation_preset"].isna().sum()
        n_augmented = (~df["augmentation_preset"].isna()).sum()
        return True, {
            "presets": presets,
            "n_clean": n_clean,
            "n_augmented": n_augmented,
        }

    return False, {}


def quick_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 1e-4,
) -> torch.nn.Module:
    """Quick training loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model = model.to(device)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            waveforms = batch["waveforms"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(waveforms, attention_mask, labels=labels)
            outputs["loss"].backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                waveforms = batch["waveforms"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                outputs = model(waveforms, attention_mask)
                preds = outputs["logits"].argmax(dim=-1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveforms"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(waveforms, attention_mask)
            preds = outputs["logits"].argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def run_ablation(data_dir: Path, config, seed: int) -> dict:
    """Run full ablation study."""
    seed_everything(seed)
    device = torch.device(config.device)
    fixed_length = config.data.max_length_samples
    collate_fn = lambda batch: collate_fixed_length(batch, fixed_length)

    results = {}

    # Create datasets
    datasets = {}
    for split in ["train", "validation", "test"]:
        for aug_filter in ["clean", "augmented", None]:
            key = f"{split}_{aug_filter or 'all'}"
            try:
                ds = FilteredDataset(
                    data_dir=data_dir,
                    split=split,
                    augmentation_filter=aug_filter,
                    target="skill_tier",
                    resample_rate=config.data.sample_rate,
                    max_length_sec=config.data.max_length_sec,
                )
                if len(ds) > 0:
                    datasets[key] = ds
            except Exception as e:
                print(f"  Warning: Could not create {key} dataset: {e}")

    print(f"  Dataset sizes:")
    for key, ds in datasets.items():
        print(f"    {key}: {len(ds)} samples")

    # Check if we have clean/augmented split
    has_clean = "train_clean" in datasets and len(datasets.get("train_clean", [])) > 0
    has_aug = "train_augmented" in datasets and len(datasets.get("train_augmented", [])) > 0

    if not has_clean or not has_aug:
        print("  Warning: Dataset doesn't have separate clean/augmented samples.")
        print("  Running simplified ablation on full dataset only.")

        # Simplified: just train on all, test on all
        train_loader = DataLoader(
            datasets["train_all"],
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            datasets["validation_all"],
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            datasets["test_all"],
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        model = Wav2Vec2Classifier(config.model, num_classes=4)
        model = quick_train(model, train_loader, val_loader, device, config.num_epochs, config.learning_rate)
        test_acc = evaluate(model, test_loader, device)

        return {
            "full_dataset_only": True,
            "train_all_test_all": test_acc,
        }

    # Full ablation study
    conditions = [
        ("train_clean", "test_clean", "clean→clean"),
        ("train_clean", "test_augmented", "clean→augmented"),
        ("train_augmented", "test_clean", "augmented→clean"),
        ("train_augmented", "test_augmented", "augmented→augmented"),
        ("train_all", "test_clean", "mixed→clean"),
        ("train_all", "test_augmented", "mixed→augmented"),
        ("train_all", "test_all", "mixed→all"),
    ]

    for train_key, test_key, name in conditions:
        if train_key not in datasets or test_key not in datasets:
            print(f"  Skipping {name}: missing dataset")
            continue

        print(f"  Running: {name}...")

        train_loader = DataLoader(
            datasets[train_key],
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Use corresponding validation set
        val_key = train_key.replace("train", "validation")
        if val_key not in datasets:
            val_key = "validation_all"
        val_loader = DataLoader(
            datasets[val_key],
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            datasets[test_key],
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        model = Wav2Vec2Classifier(config.model, num_classes=4)
        model = quick_train(model, train_loader, val_loader, device, config.num_epochs, config.learning_rate)
        test_acc = evaluate(model, test_loader, device)

        results[name] = {
            "train_samples": len(datasets[train_key]),
            "test_samples": len(datasets[test_key]),
            "accuracy": test_acc,
        }

        print(f"    {name}: {test_acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Augmentation ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="output/dataset")
    parser.add_argument("--preset", type=str, default="fast_debug")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="experiments/results")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_preset(args.preset)

    print("=" * 60)
    print("Augmentation Ablation Study")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Preset: {args.preset}")
    print()

    # Check augmentation support
    has_aug, aug_info = check_augmentation_support(data_dir)
    if has_aug:
        print("Augmentation metadata found:")
        print(f"  Clean samples: {aug_info['n_clean']}")
        print(f"  Augmented samples: {aug_info['n_augmented']}")
        print(f"  Presets: {aug_info['presets']}")
    else:
        print("Warning: No augmentation metadata in dataset.")
        print("This experiment requires 'augmentation_preset' column in samples.parquet")
        print("Continuing with simplified analysis...")
    print()

    # Run experiments
    all_results = []
    for seed in range(args.num_seeds):
        print(f"\n--- Seed {seed} ---")
        result = run_ablation(data_dir, config, seed=42 + seed)
        all_results.append(result)

    # Aggregate if multiple seeds
    if args.num_seeds > 1 and not all_results[0].get("full_dataset_only"):
        aggregated = {}
        for key in all_results[0].keys():
            if isinstance(all_results[0][key], dict) and "accuracy" in all_results[0][key]:
                accs = [r[key]["accuracy"] for r in all_results]
                aggregated[key] = {
                    "mean_accuracy": np.mean(accs),
                    "std_accuracy": np.std(accs),
                }
    else:
        aggregated = all_results[0]

    # Analysis
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    if "full_dataset_only" in aggregated:
        print(f"Full dataset accuracy: {aggregated['train_all_test_all']:.4f}")
    else:
        print("\nAccuracy Matrix (train condition → test condition):")
        print("-" * 50)
        for name, data in aggregated.items():
            if isinstance(data, dict):
                acc = data.get("mean_accuracy", data.get("accuracy", 0))
                print(f"  {name:<30}: {acc:.4f}")

    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    print("""
Based on the results:

1. If augmented→augmented >> clean→augmented:
   Augmentations help the model generalize to diverse acoustic conditions.
   RECOMMENDATION: Train with augmentations for robust models.

2. If clean→clean >> augmented→clean:
   Model trained on augmented data may learn acoustic artifacts.
   RECOMMENDATION: Use clean data for evaluation, augmented for training.

3. If mixed→all performs best:
   Combining clean and augmented data provides best generalization.
   RECOMMENDATION: Use full augmented dataset for training.
""")

    # Save results
    final_results = {
        "experiment": "augmentation_ablation",
        "has_augmentation_metadata": has_aug,
        "augmentation_info": aug_info if has_aug else None,
        "config": {"preset": args.preset, "num_seeds": args.num_seeds},
        "per_seed_results": all_results,
        "aggregated": aggregated,
    }

    results_path = output_dir / "augmentation_ablation.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
