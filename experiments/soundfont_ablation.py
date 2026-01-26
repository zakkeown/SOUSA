#!/usr/bin/env python3
"""
Soundfont Generalization Ablation Study
========================================

Research question: Should models train on all soundfonts or specialize?

Experiment design:
1. Train on single soundfont → test on same soundfont
2. Train on single soundfont → test on different soundfonts
3. Train on all soundfonts → test on each soundfont
4. Leave-one-out: train on N-1, test on held-out 1

Output:
    experiments/results/soundfont_ablation.json

Usage:
    python -m experiments.soundfont_ablation --data-dir output/dataset --preset fast_debug
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


class SoundfontFilteredDataset(SOUSADataset):
    """Dataset filtered by soundfont."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        soundfonts: list[str] | None = None,  # None for all
        exclude_soundfonts: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(data_dir, split=split, **kwargs)

        if "soundfont" in self.data.columns:
            if soundfonts:
                mask = self.data["soundfont"].isin(soundfonts)
                self.data = self.data[mask].reset_index(drop=True)
            elif exclude_soundfonts:
                mask = ~self.data["soundfont"].isin(exclude_soundfonts)
                self.data = self.data[mask].reset_index(drop=True)


def get_soundfonts(data_dir: Path) -> list[str]:
    """Get list of soundfonts in dataset."""
    samples_path = data_dir / "labels" / "samples.parquet"
    df = pd.read_parquet(samples_path)

    if "soundfont" not in df.columns:
        return []

    return sorted(df["soundfont"].unique().tolist())


def quick_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
) -> torch.nn.Module:
    """Quick training loop."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model = model.to(device)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
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

        val_acc = correct / total if total > 0 else 0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate and return accuracy."""
    if len(loader.dataset) == 0:
        return 0.0

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


def run_ablation(data_dir: Path, config, soundfonts: list[str], seed: int) -> dict:
    """Run full soundfont ablation."""
    seed_everything(seed)
    device = torch.device(config.device)
    fixed_length = config.data.max_length_samples
    collate_fn = lambda batch: collate_fixed_length(batch, fixed_length)

    results = {
        "single_to_single": {},  # Train on A, test on A
        "single_to_other": {},  # Train on A, test on B (averaged)
        "all_to_single": {},  # Train on all, test on each
        "leave_one_out": {},  # Train on all except A, test on A
    }

    # Dataset cache
    def make_loader(soundfont_list, split, shuffle):
        ds = SoundfontFilteredDataset(
            data_dir=data_dir,
            split=split,
            soundfonts=soundfont_list,
            target="skill_tier",
            resample_rate=config.data.sample_rate,
            max_length_sec=config.data.max_length_sec,
        )
        if len(ds) == 0:
            return None
        return DataLoader(
            ds,
            batch_size=config.data.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn,
        )

    # Train on all soundfonts once
    print("  Training on all soundfonts...")
    all_train_loader = make_loader(None, "train", shuffle=True)
    all_val_loader = make_loader(None, "validation", shuffle=False)

    if all_train_loader is None:
        print("  Error: No training data available")
        return results

    model_all = Wav2Vec2Classifier(config.model, num_classes=4)
    model_all = quick_train(
        model_all, all_train_loader, all_val_loader, device,
        config.num_epochs, config.learning_rate
    )

    # Test on each soundfont
    for sf in soundfonts:
        test_loader = make_loader([sf], "test", shuffle=False)
        if test_loader:
            acc = evaluate(model_all, test_loader, device)
            results["all_to_single"][sf] = acc
            print(f"    all→{sf}: {acc:.4f}")

    # Single soundfont experiments
    for train_sf in soundfonts:
        print(f"  Training on {train_sf}...")

        train_loader = make_loader([train_sf], "train", shuffle=True)
        val_loader = make_loader([train_sf], "validation", shuffle=False)

        if train_loader is None or len(train_loader.dataset) < config.data.batch_size:
            print(f"    Skipping {train_sf}: insufficient data")
            continue

        model_single = Wav2Vec2Classifier(config.model, num_classes=4)
        model_single = quick_train(
            model_single, train_loader, val_loader, device,
            config.num_epochs, config.learning_rate
        )

        # Test on same soundfont
        test_loader_same = make_loader([train_sf], "test", shuffle=False)
        if test_loader_same:
            acc_same = evaluate(model_single, test_loader_same, device)
            results["single_to_single"][train_sf] = acc_same
            print(f"    {train_sf}→{train_sf}: {acc_same:.4f}")

        # Test on other soundfonts
        other_accs = []
        for test_sf in soundfonts:
            if test_sf == train_sf:
                continue
            test_loader_other = make_loader([test_sf], "test", shuffle=False)
            if test_loader_other:
                acc_other = evaluate(model_single, test_loader_other, device)
                other_accs.append(acc_other)

        if other_accs:
            results["single_to_other"][train_sf] = {
                "mean": np.mean(other_accs),
                "std": np.std(other_accs),
                "accuracies": other_accs,
            }
            print(f"    {train_sf}→others: {np.mean(other_accs):.4f} ± {np.std(other_accs):.4f}")

    # Leave-one-out experiments
    for held_out_sf in soundfonts:
        print(f"  Leave-one-out: holding out {held_out_sf}...")

        other_sfs = [sf for sf in soundfonts if sf != held_out_sf]

        train_loader = make_loader(other_sfs, "train", shuffle=True)
        val_loader = make_loader(other_sfs, "validation", shuffle=False)

        if train_loader is None or len(train_loader.dataset) < config.data.batch_size:
            print(f"    Skipping: insufficient data without {held_out_sf}")
            continue

        model_loo = Wav2Vec2Classifier(config.model, num_classes=4)
        model_loo = quick_train(
            model_loo, train_loader, val_loader, device,
            config.num_epochs, config.learning_rate
        )

        test_loader = make_loader([held_out_sf], "test", shuffle=False)
        if test_loader:
            acc = evaluate(model_loo, test_loader, device)
            results["leave_one_out"][held_out_sf] = acc
            print(f"    without {held_out_sf}→{held_out_sf}: {acc:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Soundfont generalization ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="output/dataset")
    parser.add_argument("--preset", type=str, default="fast_debug")
    parser.add_argument("--output-dir", type=str, default="experiments/results")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_preset(args.preset)

    print("=" * 60)
    print("Soundfont Generalization Ablation Study")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Preset: {args.preset}")
    print()

    # Get soundfonts
    soundfonts = get_soundfonts(data_dir)

    if not soundfonts:
        print("Error: No soundfont column found in dataset.")
        print("This experiment requires 'soundfont' column in samples.parquet")
        return

    print(f"Found {len(soundfonts)} soundfonts: {soundfonts}")
    print()

    # Run ablation
    results = run_ablation(data_dir, config, soundfonts, seed=42)

    # Analysis
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    print("\n1. Train on single soundfont, test on same:")
    for sf, acc in results["single_to_single"].items():
        print(f"   {sf}: {acc:.4f}")

    print("\n2. Train on single soundfont, test on others (generalization gap):")
    for sf, data in results["single_to_other"].items():
        same_acc = results["single_to_single"].get(sf, 0)
        gap = same_acc - data["mean"]
        print(f"   {sf}: {data['mean']:.4f} (gap: {gap:+.4f})")

    print("\n3. Train on all soundfonts, test on each:")
    for sf, acc in results["all_to_single"].items():
        print(f"   {sf}: {acc:.4f}")

    print("\n4. Leave-one-out (test on held-out soundfont):")
    for sf, acc in results["leave_one_out"].items():
        print(f"   {sf}: {acc:.4f}")

    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)

    # Calculate average generalization gap
    gaps = []
    for sf in results["single_to_single"]:
        if sf in results["single_to_other"]:
            same = results["single_to_single"][sf]
            other = results["single_to_other"][sf]["mean"]
            gaps.append(same - other)

    avg_gap = np.mean(gaps) if gaps else 0

    if avg_gap > 0.1:
        print(f"""
Large generalization gap detected (avg {avg_gap:.2f}).

Models trained on a single soundfont do not generalize well to others.
RECOMMENDATION: Train on all soundfonts for robust cross-soundfont performance.
""")
    elif avg_gap > 0.05:
        print(f"""
Moderate generalization gap detected (avg {avg_gap:.2f}).

Some soundfont-specific characteristics are learned.
RECOMMENDATION: Use diverse soundfont training for best results.
""")
    else:
        print(f"""
Small generalization gap (avg {avg_gap:.2f}).

Models generalize reasonably well across soundfonts.
RECOMMENDATION: Training on any soundfont should work, but diverse training
is still preferred for safety.
""")

    # Save results
    final_results = {
        "experiment": "soundfont_ablation",
        "soundfonts": soundfonts,
        "config": {"preset": args.preset},
        "results": results,
        "analysis": {
            "average_generalization_gap": avg_gap if gaps else None,
            "per_soundfont_gaps": {
                sf: results["single_to_single"].get(sf, 0) - results["single_to_other"].get(sf, {}).get("mean", 0)
                for sf in results["single_to_single"]
            },
        },
    }

    results_path = output_dir / "soundfont_ablation.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
