#!/usr/bin/env python3
"""
Split Methodology Validation Experiment
========================================

Research question: Does profile-based splitting matter vs. random splitting?

Hypothesis: If profile-based splits yield worse test accuracy than random splits,
models are learning "player fingerprints" rather than generalizing to new players.

Experiment:
1. Train skill classifier with profile-based splits (current SOUSA method)
2. Train skill classifier with random sample splits (same 70/15/15 ratio)
3. Compare test accuracy
4. Measure within-profile vs cross-profile accuracy (for random splits)

Output:
    experiments/results/split_validation.json

Usage:
    python -m experiments.split_validation --data-dir output/dataset --preset fast_debug
    python -m experiments.split_validation --data-dir output/dataset --num-seeds 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.pytorch_dataloader import (
    SOUSADataset,
    collate_fixed_length,
    SKILL_TIER_TO_ID,
)
from examples.baselines.config import get_preset
from examples.baselines.models import Wav2Vec2Classifier
from examples.baselines.training import seed_everything


class RandomSplitDataset(SOUSADataset):
    """Dataset with random sample splits instead of profile-based."""

    def __init__(
        self,
        data_dir: str | Path,
        indices: list[int],
        **kwargs,
    ):
        # Initialize without split filtering
        kwargs["split"] = None
        super().__init__(data_dir, **kwargs)

        # Filter to specified indices
        self.data = self.data.iloc[indices].reset_index(drop=True)


def create_random_splits(
    data_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Create random sample-based splits."""
    # Load all samples
    samples_df = pd.read_parquet(data_dir / "labels" / "samples.parquet")
    n_samples = len(samples_df)

    # Random split
    indices = list(range(n_samples))
    train_idx, test_val_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=samples_df["skill_tier"],
    )
    val_idx, test_idx = train_test_split(
        test_val_idx,
        train_size=val_ratio / (1 - train_ratio),
        random_state=seed,
        stratify=samples_df.iloc[test_val_idx]["skill_tier"],
    )

    return train_idx, val_idx, test_idx


def quick_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5,
    lr: float = 1e-4,
) -> dict:
    """Quick training loop for experiment."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model = model.to(device)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Train
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            waveforms = batch["waveforms"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(waveforms, attention_mask, labels=labels)
            outputs["loss"].backward()
            optimizer.step()

        # Validate
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
        best_val_acc = max(best_val_acc, val_acc)

    return {"best_val_accuracy": best_val_acc}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_profile_ids = []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveforms"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(waveforms, attention_mask)
            preds = outputs["logits"].argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()

    # Per-class accuracy
    per_class = {}
    for cls in range(4):
        mask = all_labels == cls
        if mask.sum() > 0:
            per_class[cls] = float((all_preds[mask] == cls).mean())

    return {
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class,
    }


def run_experiment_single_seed(
    data_dir: Path,
    config,
    seed: int,
) -> dict:
    """Run one iteration of the experiment."""
    seed_everything(seed)
    device = torch.device(config.device)

    fixed_length = config.data.max_length_samples

    # === Profile-based splits (SOUSA default) ===
    print(f"\n  [Seed {seed}] Training with profile-based splits...")

    train_dataset_profile = SOUSADataset(
        data_dir=data_dir,
        split="train",
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    val_dataset_profile = SOUSADataset(
        data_dir=data_dir,
        split="validation",
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    test_dataset_profile = SOUSADataset(
        data_dir=data_dir,
        split="test",
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )

    collate_fn = lambda batch: collate_fixed_length(batch, fixed_length)

    train_loader_profile = DataLoader(
        train_dataset_profile,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader_profile = DataLoader(
        val_dataset_profile,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader_profile = DataLoader(
        test_dataset_profile,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model_profile = Wav2Vec2Classifier(config.model, num_classes=4)
    train_result_profile = quick_train(
        model_profile,
        train_loader_profile,
        val_loader_profile,
        device,
        num_epochs=config.num_epochs,
        lr=config.learning_rate,
    )
    test_result_profile = evaluate(model_profile, test_loader_profile, device)

    # === Random sample splits ===
    print(f"  [Seed {seed}] Training with random splits...")

    train_idx, val_idx, test_idx = create_random_splits(data_dir, seed=seed)

    train_dataset_random = RandomSplitDataset(
        data_dir=data_dir,
        indices=train_idx,
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    val_dataset_random = RandomSplitDataset(
        data_dir=data_dir,
        indices=val_idx,
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )
    test_dataset_random = RandomSplitDataset(
        data_dir=data_dir,
        indices=test_idx,
        target="skill_tier",
        resample_rate=config.data.sample_rate,
        max_length_sec=config.data.max_length_sec,
    )

    train_loader_random = DataLoader(
        train_dataset_random,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader_random = DataLoader(
        val_dataset_random,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader_random = DataLoader(
        test_dataset_random,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model_random = Wav2Vec2Classifier(config.model, num_classes=4)
    train_result_random = quick_train(
        model_random,
        train_loader_random,
        val_loader_random,
        device,
        num_epochs=config.num_epochs,
        lr=config.learning_rate,
    )
    test_result_random = evaluate(model_random, test_loader_random, device)

    return {
        "seed": seed,
        "profile_based": {
            "train_samples": len(train_dataset_profile),
            "val_samples": len(val_dataset_profile),
            "test_samples": len(test_dataset_profile),
            "best_val_accuracy": train_result_profile["best_val_accuracy"],
            "test_accuracy": test_result_profile["accuracy"],
            "test_per_class": test_result_profile["per_class_accuracy"],
        },
        "random_split": {
            "train_samples": len(train_dataset_random),
            "val_samples": len(val_dataset_random),
            "test_samples": len(test_dataset_random),
            "best_val_accuracy": train_result_random["best_val_accuracy"],
            "test_accuracy": test_result_random["accuracy"],
            "test_per_class": test_result_random["per_class_accuracy"],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate split methodology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/dataset",
        help="Path to SOUSA dataset",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast_debug", "baseline", "full"],
        default="fast_debug",
        help="Config preset",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Number of random seeds to average over",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_preset(args.preset)

    print("=" * 60)
    print("Split Methodology Validation Experiment")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Preset: {args.preset}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Epochs per run: {config.num_epochs}")
    print()

    # Run experiments
    results = []
    seeds = [42 + i * 7 for i in range(args.num_seeds)]

    for seed in seeds:
        result = run_experiment_single_seed(data_dir, config, seed)
        results.append(result)

        print(f"\n  Seed {seed} results:")
        print(f"    Profile-based test acc: {result['profile_based']['test_accuracy']:.4f}")
        print(f"    Random split test acc:  {result['random_split']['test_accuracy']:.4f}")

    # Aggregate results
    profile_accs = [r["profile_based"]["test_accuracy"] for r in results]
    random_accs = [r["random_split"]["test_accuracy"] for r in results]

    profile_mean = np.mean(profile_accs)
    profile_std = np.std(profile_accs)
    random_mean = np.mean(random_accs)
    random_std = np.std(random_accs)

    # Statistical test
    try:
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(profile_accs, random_accs)
    except ImportError:
        t_stat, p_value = None, None

    print("\n" + "=" * 60)
    print("Aggregated Results")
    print("=" * 60)
    print(f"Profile-based splits: {profile_mean:.4f} ± {profile_std:.4f}")
    print(f"Random sample splits: {random_mean:.4f} ± {random_std:.4f}")
    print(f"Difference: {random_mean - profile_mean:+.4f}")

    if p_value is not None:
        print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
        if p_value < 0.05:
            print("Result: Statistically significant difference (p < 0.05)")
        else:
            print("Result: No statistically significant difference (p >= 0.05)")

    # Interpretation
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)

    if random_mean > profile_mean + 0.02:
        print("""
FINDING: Random splits yield higher accuracy than profile-based splits.

This suggests models may be learning "player fingerprints" - characteristics
unique to each player's style that help identify them, rather than learning
generalizable skill assessment features.

RECOMMENDATION: Use profile-based splits for realistic evaluation. The
profile-based accuracy is a better estimate of real-world performance
when assessing new players the model hasn't seen before.
""")
    elif profile_mean > random_mean + 0.02:
        print("""
FINDING: Profile-based splits yield higher accuracy than random splits.

This is unexpected and may indicate issues with the random split stratification
or small sample effects. Profile-based splits should generally be harder.

RECOMMENDATION: Investigate further with more seeds and larger dataset.
""")
    else:
        print("""
FINDING: No significant difference between split methods.

This suggests the model is learning general skill assessment features
rather than player-specific characteristics. Both splitting methods
appear valid for evaluation.

RECOMMENDATION: Profile-based splits are still preferred for consistency
with realistic deployment scenarios.
""")

    # Save results
    final_results = {
        "experiment": "split_validation",
        "config": {
            "preset": args.preset,
            "num_seeds": args.num_seeds,
            "epochs": config.num_epochs,
        },
        "per_seed_results": results,
        "summary": {
            "profile_based": {
                "mean_accuracy": profile_mean,
                "std_accuracy": profile_std,
                "all_accuracies": profile_accs,
            },
            "random_split": {
                "mean_accuracy": random_mean,
                "std_accuracy": random_std,
                "all_accuracies": random_accs,
            },
            "difference": random_mean - profile_mean,
            "t_statistic": t_stat,
            "p_value": p_value,
        },
    }

    results_path = output_dir / "split_validation.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
