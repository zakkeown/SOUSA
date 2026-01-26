#!/usr/bin/env python3
"""
Score Correlation Analysis
===========================

Analyze correlations between the 17 score columns to identify:
- Highly correlated score pairs (redundancy)
- Score clusters
- Minimal orthogonal score set for training

This is an analysis-only script (no model training).

Output:
    experiments/results/score_correlations.json
    experiments/results/score_correlation_matrix.png

Usage:
    python -m experiments.score_analysis --data-dir output/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")


# Score columns in the SOUSA dataset
SCORE_COLUMNS = [
    # Exercise-level scores
    "overall_score",
    "timing_score",
    "dynamics_score",
    "consistency_score",
    # Timing subscores
    "timing_accuracy",
    "timing_consistency",
    "tempo_stability",
    # Dynamics subscores
    "velocity_accuracy",
    "velocity_consistency",
    "accent_contrast",
    "dynamic_range",
    # Consistency subscores
    "evenness",
    "hand_balance",
    # Groove metrics
    "groove_quality",
    "swing_factor",
    "pocket_tightness",
    "micro_timing_variance",
]


def load_scores(data_dir: Path) -> pd.DataFrame:
    """Load exercise scores from dataset."""
    exercises_path = data_dir / "labels" / "exercises.parquet"
    if not exercises_path.exists():
        raise FileNotFoundError(f"Exercises file not found: {exercises_path}")

    df = pd.read_parquet(exercises_path)

    # Filter to only score columns that exist
    available_scores = [col for col in SCORE_COLUMNS if col in df.columns]
    missing = set(SCORE_COLUMNS) - set(available_scores)
    if missing:
        print(f"Note: {len(missing)} score columns not found: {missing}")

    return df[available_scores]


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation matrix."""
    return df.corr()


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list[dict]:
    """Find pairs of scores with correlation above threshold."""
    pairs = []
    cols = corr_matrix.columns

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i < j:  # Upper triangle only
                r = corr_matrix.loc[col1, col2]
                if abs(r) >= threshold:
                    pairs.append({
                        "score1": col1,
                        "score2": col2,
                        "correlation": float(r),
                    })

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs


def cluster_scores(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list[list[str]]:
    """Group highly correlated scores into clusters using simple greedy approach."""
    cols = list(corr_matrix.columns)
    assigned = set()
    clusters = []

    for col in cols:
        if col in assigned:
            continue

        # Start new cluster
        cluster = [col]
        assigned.add(col)

        # Add correlated columns
        for other in cols:
            if other in assigned:
                continue
            # Check if correlated with all cluster members
            all_correlated = all(
                abs(corr_matrix.loc[member, other]) >= threshold
                for member in cluster
            )
            if all_correlated:
                cluster.append(other)
                assigned.add(other)

        clusters.append(cluster)

    # Sort by size
    clusters.sort(key=len, reverse=True)
    return clusters


def compute_pca_analysis(df: pd.DataFrame) -> dict:
    """Run PCA to identify principal components."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Warning: scikit-learn not available. Skipping PCA.")
        return {}

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.dropna())

    # PCA
    pca = PCA()
    pca.fit(scaled)

    # Variance explained
    explained_variance = pca.explained_variance_ratio_

    # Find number of components for 90% and 95% variance
    cumulative = np.cumsum(explained_variance)
    n_90 = int(np.argmax(cumulative >= 0.90)) + 1
    n_95 = int(np.argmax(cumulative >= 0.95)) + 1

    # Top loadings for first 3 components
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=df.columns,
        columns=["PC1", "PC2", "PC3"],
    )

    top_loadings = {}
    for pc in ["PC1", "PC2", "PC3"]:
        sorted_loadings = loadings[pc].abs().sort_values(ascending=False)
        top_loadings[pc] = [
            {"score": idx, "loading": float(loadings.loc[idx, pc])}
            for idx in sorted_loadings.head(5).index
        ]

    return {
        "explained_variance": explained_variance.tolist(),
        "cumulative_variance": cumulative.tolist(),
        "components_for_90_percent": n_90,
        "components_for_95_percent": n_95,
        "top_loadings": top_loadings,
    }


def recommend_score_set(
    corr_matrix: pd.DataFrame,
    clusters: list[list[str]],
) -> list[str]:
    """Recommend minimal orthogonal score set."""
    # Strategy: pick one representative from each cluster
    # Prefer commonly used scores (overall_score, timing_score, etc.)

    priority = [
        "overall_score",
        "timing_score",
        "dynamics_score",
        "consistency_score",
        "groove_quality",
        "timing_accuracy",
        "velocity_accuracy",
        "hand_balance",
    ]

    recommended = []
    for cluster in clusters:
        # Find highest priority score in cluster
        for prio in priority:
            if prio in cluster:
                recommended.append(prio)
                break
        else:
            # No priority score, just take first
            recommended.append(cluster[0])

    return recommended


def plot_correlation_matrix(corr_matrix: pd.DataFrame, output_path: Path):
    """Generate heatmap of correlation matrix."""
    if not PLOTTING_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title("SOUSA Score Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze score correlations in SOUSA dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/dataset",
        help="Path to SOUSA dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.8,
        help="Threshold for flagging high correlations",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.7,
        help="Threshold for score clustering",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SOUSA Score Correlation Analysis")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    print("Loading exercise scores...")
    scores_df = load_scores(data_dir)
    print(f"Loaded {len(scores_df)} samples with {len(scores_df.columns)} score columns")
    print(f"Score columns: {list(scores_df.columns)}")
    print()

    # Basic statistics
    print("Score Statistics:")
    stats = scores_df.describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())
    print()

    # Correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(scores_df)

    # High correlations
    high_corr = find_high_correlations(corr_matrix, args.correlation_threshold)
    print(f"\nHigh correlations (|r| >= {args.correlation_threshold}):")
    for pair in high_corr[:15]:
        print(f"  {pair['score1']:<25} ↔ {pair['score2']:<25}: r = {pair['correlation']:+.3f}")

    # Score clusters
    clusters = cluster_scores(corr_matrix, args.cluster_threshold)
    print(f"\nScore clusters (threshold = {args.cluster_threshold}):")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i+1}: {', '.join(cluster)}")

    # PCA analysis
    print("\nPCA Analysis:")
    pca_results = compute_pca_analysis(scores_df)
    if pca_results:
        print(f"  Components for 90% variance: {pca_results['components_for_90_percent']}")
        print(f"  Components for 95% variance: {pca_results['components_for_95_percent']}")
        print("  Variance explained by first 5 components:", end=" ")
        print([f"{v:.1%}" for v in pca_results['explained_variance'][:5]])

    # Recommendations
    recommended = recommend_score_set(corr_matrix, clusters)
    print(f"\nRecommended minimal score set ({len(recommended)} scores):")
    for score in recommended:
        print(f"  - {score}")

    # Training recommendations
    print("\n" + "=" * 60)
    print("Recommendations for ML Training")
    print("=" * 60)
    print("""
1. PRIMARY TARGET: Use 'overall_score' as the main regression target.
   It captures the overall performance quality.

2. AVOID REDUNDANCY: These pairs are highly correlated (r > 0.8):""")
    for pair in high_corr[:5]:
        print(f"   - {pair['score1']} ↔ {pair['score2']} (r = {pair['correlation']:.2f})")
    print("""
3. MULTI-TASK LEARNING: If predicting multiple scores, use one per cluster:""")
    for i, cluster in enumerate(clusters[:4]):
        print(f"   Cluster {i+1}: prefer '{cluster[0]}' (drop: {cluster[1:]})")
    print("""
4. ORTHOGONAL TARGETS: For independent signals, consider PCA components
   or the recommended minimal set above.
""")

    # Save results
    results = {
        "num_samples": len(scores_df),
        "score_columns": list(scores_df.columns),
        "correlation_matrix": corr_matrix.to_dict(),
        "high_correlations": high_corr,
        "clusters": clusters,
        "recommended_scores": recommended,
        "pca": pca_results,
        "statistics": scores_df.describe().to_dict(),
    }

    results_path = output_dir / "score_correlations.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Plot
    plot_path = output_dir / "score_correlation_matrix.png"
    plot_correlation_matrix(corr_matrix, plot_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
