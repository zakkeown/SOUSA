#!/usr/bin/env python3
"""
Generate visualization plots for SOUSA dataset.

Creates publication-ready plots showing dataset distributions and quality metrics.
Output images can be embedded in the HuggingFace dataset card.

Usage:
    python scripts/generate_plots.py                     # Generate all plots
    python scripts/generate_plots.py output/dataset      # Specify dataset directory
    python scripts/generate_plots.py --output-dir plots  # Custom output directory

Examples:
    # Generate plots for dataset card
    python scripts/generate_plots.py --output-dir output/dataset/plots
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd

from dataset_gen.pipeline.storage import ParquetReader

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


TIER_ORDER = ["beginner", "intermediate", "advanced", "professional"]
TIER_COLORS = {
    "beginner": "#e74c3c",
    "intermediate": "#f39c12",
    "advanced": "#3498db",
    "professional": "#27ae60",
}


def load_data(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load samples, exercises, and strokes data."""
    reader = ParquetReader(dataset_dir)
    samples = reader.load_samples()
    exercises = reader.load_exercises()
    strokes = reader.load_strokes()
    return samples, exercises, strokes


def plot_score_distributions(
    samples: pd.DataFrame, exercises: pd.DataFrame, output_path: Path
) -> None:
    """Create box plot of overall scores by skill tier."""
    # Merge exercises with samples for skill tier
    merged = exercises.merge(samples[["sample_id", "skill_tier"]], on="sample_id")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare data for box plot
    data = [merged[merged["skill_tier"] == tier]["overall_score"] for tier in TIER_ORDER]
    colors = [TIER_COLORS[tier] for tier in TIER_ORDER]

    bp = ax.boxplot(
        data,
        tick_labels=[t.capitalize() for t in TIER_ORDER],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Skill Tier")
    ax.set_ylabel("Overall Score (0-100)")
    ax.set_title("Performance Score Distribution by Skill Tier")

    # Add sample counts
    for i, tier in enumerate(TIER_ORDER):
        n = len(merged[merged["skill_tier"] == tier])
        ax.text(i + 1, ax.get_ylim()[0] - 5, f"n={n:,}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_timing_accuracy(samples: pd.DataFrame, exercises: pd.DataFrame, output_path: Path) -> None:
    """Create violin plot of timing accuracy by skill tier."""
    merged = exercises.merge(samples[["sample_id", "skill_tier"]], on="sample_id")

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [merged[merged["skill_tier"] == tier]["timing_accuracy"] for tier in TIER_ORDER]
    colors = [TIER_COLORS[tier] for tier in TIER_ORDER]

    parts = ax.violinplot(data, positions=range(1, 5), showmedians=True, showextrema=False)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([t.capitalize() for t in TIER_ORDER])
    ax.set_xlabel("Skill Tier")
    ax.set_ylabel("Timing Accuracy Score (0-100)")
    ax.set_title("Timing Accuracy Distribution by Skill Tier")

    # Add mean annotations
    for i, tier in enumerate(TIER_ORDER):
        tier_data = merged[merged["skill_tier"] == tier]["timing_accuracy"]
        mean_val = tier_data.mean()
        ax.scatter([i + 1], [mean_val], color="white", edgecolor="black", s=50, zorder=5)
        ax.annotate(
            f"{mean_val:.1f}",
            (i + 1, mean_val),
            textcoords="offset points",
            xytext=(15, 0),
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_rudiment_distribution(samples: pd.DataFrame, output_path: Path) -> None:
    """Create horizontal bar chart of samples per rudiment."""
    rudiment_counts = samples["rudiment_slug"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(10, 12))

    # Create color map based on rudiment category
    def get_category_color(slug: str) -> str:
        if "roll" in slug or "stroke" in slug:
            return "#3498db"  # blue - rolls
        elif "paradiddle" in slug or "diddle" in slug:
            return "#27ae60"  # green - diddles
        elif "flam" in slug:
            return "#e74c3c"  # red - flams
        elif "drag" in slug or "ratamacue" in slug:
            return "#9b59b6"  # purple - drags
        return "#7f8c8d"  # gray - other

    colors = [get_category_color(slug) for slug in rudiment_counts.index]

    bars = ax.barh(range(len(rudiment_counts)), rudiment_counts.values, color=colors, alpha=0.8)

    ax.set_yticks(range(len(rudiment_counts)))
    ax.set_yticklabels([s.replace("_", " ").title() for s in rudiment_counts.index], fontsize=8)
    ax.set_xlabel("Number of Samples")
    ax.set_title("Samples per Rudiment")

    # Add count labels
    for bar, count in zip(bars, rudiment_counts.values):
        ax.text(
            count + 50, bar.get_y() + bar.get_height() / 2, f"{count:,}", va="center", fontsize=7
        )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.8, label="Roll Rudiments"),
        Patch(facecolor="#27ae60", alpha=0.8, label="Diddle Rudiments"),
        Patch(facecolor="#e74c3c", alpha=0.8, label="Flam Rudiments"),
        Patch(facecolor="#9b59b6", alpha=0.8, label="Drag Rudiments"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_skill_tier_pie(samples: pd.DataFrame, output_path: Path) -> None:
    """Create pie chart of skill tier distribution."""
    tier_counts = samples["skill_tier"].value_counts()
    tier_counts = tier_counts.reindex(TIER_ORDER)

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = [TIER_COLORS[tier] for tier in TIER_ORDER]

    wedges, texts, autotexts = ax.pie(
        tier_counts.values,
        labels=[t.capitalize() for t in TIER_ORDER],
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*sum(tier_counts)):,})",
        startangle=90,
        explode=[0.02] * 4,
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)

    ax.set_title("Skill Tier Distribution")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tempo_distribution(samples: pd.DataFrame, output_path: Path) -> None:
    """Create histogram of tempo distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tempos = samples["tempo_bpm"]
    bins = sorted(tempos.unique())

    # Color by skill tier
    for tier in TIER_ORDER:
        tier_tempos = samples[samples["skill_tier"] == tier]["tempo_bpm"]
        ax.hist(
            tier_tempos,
            bins=len(bins),
            alpha=0.5,
            label=tier.capitalize(),
            color=TIER_COLORS[tier],
        )

    ax.set_xlabel("Tempo (BPM)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Tempo Distribution by Skill Tier")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_timing_vs_velocity(
    samples: pd.DataFrame, exercises: pd.DataFrame, output_path: Path
) -> None:
    """Create scatter plot of timing accuracy vs velocity control."""
    merged = exercises.merge(samples[["sample_id", "skill_tier"]], on="sample_id")

    fig, ax = plt.subplots(figsize=(8, 6))

    for tier in TIER_ORDER:
        tier_data = merged[merged["skill_tier"] == tier]
        ax.scatter(
            tier_data["timing_accuracy"],
            tier_data["velocity_control"],
            c=TIER_COLORS[tier],
            label=tier.capitalize(),
            alpha=0.3,
            s=10,
        )

    ax.set_xlabel("Timing Accuracy Score")
    ax.set_ylabel("Velocity Control Score")
    ax.set_title("Timing Accuracy vs Velocity Control by Skill Tier")
    ax.legend()

    # Add correlation line
    correlation = merged["timing_accuracy"].corr(merged["velocity_control"])
    ax.annotate(f"r = {correlation:.3f}", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots for SOUSA dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default="output/dataset",
        help="Path to dataset directory (default: output/dataset)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots (default: {dataset_dir}/plots)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {dataset_dir}...")
    samples, exercises, strokes = load_data(dataset_dir)
    print(f"  Loaded {len(samples):,} samples, {len(exercises):,} exercises")

    print(f"\nGenerating plots to {output_dir}...")

    plot_score_distributions(samples, exercises, output_dir / "score_distribution.png")
    plot_timing_accuracy(samples, exercises, output_dir / "timing_accuracy.png")
    plot_rudiment_distribution(samples, output_dir / "rudiment_distribution.png")
    plot_skill_tier_pie(samples, output_dir / "skill_tier_distribution.png")
    plot_tempo_distribution(samples, output_dir / "tempo_distribution.png")
    plot_timing_vs_velocity(samples, exercises, output_dir / "timing_vs_velocity.png")

    print(f"\nDone! Generated 6 plots in {output_dir}")


if __name__ == "__main__":
    main()
