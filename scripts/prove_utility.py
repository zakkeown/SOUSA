#!/usr/bin/env python3
"""
Utility proofs for the SOUSA dataset.

This script trains baseline models and analyzes learnability to prove
the dataset is useful for machine learning.

Proofs generated:
1. Baseline Models: Train simple models and report metrics
2. Learnability Analysis: Learning curves and distribution checks
3. Interpretability: Feature importance showing models learn the right things

Usage:
    python scripts/prove_utility.py output/dataset/ [--output-dir output/proofs/]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Metrics for a baseline model."""

    task: str
    model_name: str
    accuracy: float | None = None
    f1_macro: float | None = None
    f1_weighted: float | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    confusion_matrix: list | None = None
    class_names: list[str] | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LearnabilityResult:
    """Results from learnability analysis."""

    task: str
    train_sizes: list[float]
    train_scores: list[float]
    val_scores: list[float]
    learning_rate_positive: bool  # Does performance improve with more data?
    convergence_score: float  # How stable is final performance?
    distribution_similarity: dict[str, float]  # Train/val/test distribution similarity

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeatureImportance:
    """Feature importance for interpretability."""

    task: str
    model_name: str
    feature_importances: dict[str, float]
    top_features: list[str]
    musically_meaningful: bool  # Do top features make musical sense?
    analysis: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UtilityReport:
    """Complete utility proof report."""

    dataset_path: str
    generated_at: str
    baseline_metrics: list[BaselineMetrics] = field(default_factory=list)
    learnability: list[LearnabilityResult] = field(default_factory=list)
    interpretability: list[FeatureImportance] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "generated_at": self.generated_at,
            "baseline_metrics": [m.to_dict() for m in self.baseline_metrics],
            "learnability": [l.to_dict() for l in self.learnability],
            "interpretability": [i.to_dict() for i in self.interpretability],
            "summary": self.summary,
        }

    def save(self, output_path: Path | str) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Report saved to {output_path}")


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the dataset parquet files."""
    labels_dir = dataset_dir / "labels"

    samples = pd.read_parquet(labels_dir / "samples.parquet")
    exercises = pd.read_parquet(labels_dir / "exercises.parquet")
    strokes = pd.read_parquet(labels_dir / "strokes.parquet")

    return samples, exercises, strokes


def load_splits(dataset_dir: Path) -> dict | None:
    """Load train/val/test splits if available."""
    splits_path = dataset_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            return json.load(f)
    return None


def prepare_classification_data(
    samples: pd.DataFrame,
    exercises: pd.DataFrame,
    target: str = "skill_tier",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Prepare features and labels for classification."""
    # Merge samples with exercises
    merged = exercises.merge(samples[["sample_id", "skill_tier", "tempo_bpm"]], on="sample_id")

    # Feature columns from exercise scores
    feature_cols = [
        "timing_accuracy",
        "timing_consistency",
        "tempo_stability",
        "subdivision_evenness",
        "velocity_control",
        "accent_differentiation",
        "accent_accuracy",
        "hand_balance",
    ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in merged.columns]

    # Add tempo as feature
    if "tempo_bpm" in merged.columns:
        feature_cols.append("tempo_bpm")

    X = merged[feature_cols].values
    y = merged[target].values

    return X, y, feature_cols, list(merged[target].unique())


def prepare_regression_data(
    samples: pd.DataFrame,
    exercises: pd.DataFrame,
    strokes: pd.DataFrame,
    target: str = "overall_score",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features for regression from stroke-level statistics."""
    # Compute per-sample statistics from strokes
    stroke_stats = (
        strokes.groupby("sample_id")
        .agg(
            {
                "timing_error_ms": ["mean", "std", "max"],
                "actual_velocity": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    stroke_stats.columns = [
        "sample_id",
        "timing_error_mean",
        "timing_error_std",
        "timing_error_max",
        "velocity_mean",
        "velocity_std",
    ]

    # Merge with samples
    merged = samples[["sample_id", "tempo_bpm"]].merge(stroke_stats, on="sample_id")
    merged = merged.merge(exercises[["sample_id", target]], on="sample_id")

    feature_cols = [
        "tempo_bpm",
        "timing_error_mean",
        "timing_error_std",
        "timing_error_max",
        "velocity_mean",
        "velocity_std",
    ]

    X = merged[feature_cols].values
    y = merged[target].values

    return X, y, feature_cols


def train_baseline_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    task_name: str,
) -> list[BaselineMetrics]:
    """Train baseline classifiers and compute metrics."""
    results = []

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (80/20)
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Model 1: Logistic Regression
    logger.info(f"Training Logistic Regression for {task_name}...")
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    results.append(
        BaselineMetrics(
            task=task_name,
            model_name="LogisticRegression",
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_macro=float(f1_score(y_test, y_pred, average="macro")),
            f1_weighted=float(f1_score(y_test, y_pred, average="weighted")),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            class_names=le.classes_.tolist(),
        )
    )

    # Model 2: Random Forest
    logger.info(f"Training Random Forest for {task_name}...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    results.append(
        BaselineMetrics(
            task=task_name,
            model_name="RandomForest",
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_macro=float(f1_score(y_test, y_pred, average="macro")),
            f1_weighted=float(f1_score(y_test, y_pred, average="weighted")),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            class_names=le.classes_.tolist(),
        )
    )

    return results


def train_baseline_regressors(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task_name: str,
) -> list[BaselineMetrics]:
    """Train baseline regressors and compute metrics."""
    results = []

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Model 1: Ridge Regression
    logger.info(f"Training Ridge Regression for {task_name}...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    results.append(
        BaselineMetrics(
            task=task_name,
            model_name="RidgeRegression",
            mae=float(mean_absolute_error(y_test, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            r2=float(r2_score(y_test, y_pred)),
        )
    )

    # Model 2: Random Forest Regressor
    logger.info(f"Training Random Forest Regressor for {task_name}...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    results.append(
        BaselineMetrics(
            task=task_name,
            model_name="RandomForest",
            mae=float(mean_absolute_error(y_test, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            r2=float(r2_score(y_test, y_pred)),
        )
    )

    return results


def analyze_learnability(
    X: np.ndarray,
    y: np.ndarray,
    task_name: str,
    is_classification: bool = True,
) -> LearnabilityResult:
    """Analyze learning curves to prove the data is learnable."""
    logger.info(f"Analyzing learnability for {task_name}...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if is_classification:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        y_encoded = y
        model = RandomForestRegressor(n_estimators=50, random_state=42)

    # Compute learning curve
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model,
        X_scaled,
        y_encoded,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="accuracy" if is_classification else "r2",
        n_jobs=-1,
    )

    train_sizes = (train_sizes_abs / len(X)).tolist()
    train_means = train_scores.mean(axis=1).tolist()
    val_means = val_scores.mean(axis=1).tolist()

    # Check if learning rate is positive (performance improves with data)
    # Compare first half to second half
    first_half_val = np.mean(val_means[: len(val_means) // 2])
    second_half_val = np.mean(val_means[len(val_means) // 2 :])
    learning_rate_positive = second_half_val > first_half_val

    # Convergence score: stability of final performance
    final_scores = val_means[-3:]
    convergence_score = 1.0 - (np.std(final_scores) / (np.mean(final_scores) + 1e-10))

    # Check distribution similarity using cross-validation variance
    cv_scores = cross_val_score(
        model,
        X_scaled,
        y_encoded,
        cv=5,
        scoring="accuracy" if is_classification else "r2",
    )
    distribution_similarity = {
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_min": float(np.min(cv_scores)),
        "cv_max": float(np.max(cv_scores)),
    }

    return LearnabilityResult(
        task=task_name,
        train_sizes=train_sizes,
        train_scores=train_means,
        val_scores=val_means,
        learning_rate_positive=learning_rate_positive,
        convergence_score=float(convergence_score),
        distribution_similarity=distribution_similarity,
    )


def analyze_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task_name: str,
    is_classification: bool = True,
) -> FeatureImportance:
    """Analyze feature importance for interpretability."""
    logger.info(f"Analyzing feature importance for {task_name}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if is_classification:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        y_encoded = y
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_scaled, y_encoded)

    # Get feature importances
    importances = model.feature_importances_
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}

    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:5]]

    # Check if top features are musically meaningful
    # Timing and dynamics features should be important for skill assessment
    musically_important_features = {
        "timing_accuracy",
        "timing_consistency",
        "timing_error_mean",
        "timing_error_std",
        "velocity_control",
        "hand_balance",
        "accent_differentiation",
        "subdivision_evenness",
    }

    top_set = set(top_features[:3])
    musically_meaningful = len(top_set & musically_important_features) >= 2

    if musically_meaningful:
        analysis = (
            f"Top features ({', '.join(top_features[:3])}) are musically meaningful. "
            "The model is learning timing and dynamics patterns as expected for skill assessment."
        )
    else:
        analysis = (
            f"Top features ({', '.join(top_features[:3])}) may not be the most musically relevant. "
            "Consider investigating if the model is learning appropriate patterns."
        )

    return FeatureImportance(
        task=task_name,
        model_name="RandomForest",
        feature_importances=importance_dict,
        top_features=top_features,
        musically_meaningful=musically_meaningful,
        analysis=analysis,
    )


def generate_utility_report(dataset_dir: Path, output_dir: Path | None = None) -> UtilityReport:
    """Generate complete utility proof report."""
    logger.info(f"Generating utility proofs for {dataset_dir}")

    # Load data
    samples, exercises, strokes = load_dataset(dataset_dir)
    logger.info(f"Loaded {len(samples)} samples")

    report = UtilityReport(
        dataset_path=str(dataset_dir),
        generated_at=datetime.now().isoformat(),
    )

    # Task 1: Skill Tier Classification
    logger.info("\n=== Task 1: Skill Tier Classification ===")
    X_cls, y_cls, feat_cls, classes = prepare_classification_data(samples, exercises, "skill_tier")
    logger.info(f"  Features: {feat_cls}")
    logger.info(f"  Classes: {classes}")
    logger.info(f"  Samples: {len(X_cls)}")

    report.baseline_metrics.extend(
        train_baseline_classifiers(X_cls, y_cls, feat_cls, classes, "skill_tier_classification")
    )
    report.learnability.append(
        analyze_learnability(X_cls, y_cls, "skill_tier_classification", is_classification=True)
    )
    report.interpretability.append(
        analyze_feature_importance(
            X_cls, y_cls, feat_cls, "skill_tier_classification", is_classification=True
        )
    )

    # Task 2: Overall Score Regression
    logger.info("\n=== Task 2: Overall Score Regression ===")
    X_reg, y_reg, feat_reg = prepare_regression_data(samples, exercises, strokes, "overall_score")
    logger.info(f"  Features: {feat_reg}")
    logger.info(f"  Samples: {len(X_reg)}")

    report.baseline_metrics.extend(
        train_baseline_regressors(X_reg, y_reg, feat_reg, "overall_score_regression")
    )
    report.learnability.append(
        analyze_learnability(X_reg, y_reg, "overall_score_regression", is_classification=False)
    )
    report.interpretability.append(
        analyze_feature_importance(
            X_reg, y_reg, feat_reg, "overall_score_regression", is_classification=False
        )
    )

    # Generate summary
    cls_metrics = [m for m in report.baseline_metrics if "classification" in m.task]
    reg_metrics = [m for m in report.baseline_metrics if "regression" in m.task]

    best_cls_acc = max((m.accuracy or 0) for m in cls_metrics)
    best_reg_r2 = max((m.r2 or 0) for m in reg_metrics)

    report.summary = {
        "best_classification_accuracy": best_cls_acc,
        "best_regression_r2": best_reg_r2,
        "all_learnability_positive": all(l.learning_rate_positive for l in report.learnability),
        "all_features_meaningful": all(i.musically_meaningful for i in report.interpretability),
        "verdict": (
            "PASSED: Dataset demonstrates strong ML utility"
            if best_cls_acc > 0.7 and best_reg_r2 > 0.5
            else (
                "PARTIAL: Some tasks show learning but metrics could improve"
                if best_cls_acc > 0.5 or best_reg_r2 > 0.3
                else "NEEDS REVIEW: Low baseline metrics suggest data quality issues"
            )
        ),
    }

    # Save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report.save(output_dir / "utility_report.json")

    return report


def print_report_summary(report: UtilityReport) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("UTILITY PROOF REPORT")
    print("=" * 60)

    print("\n--- Baseline Model Performance ---")
    for m in report.baseline_metrics:
        print(f"\n  {m.task} ({m.model_name}):")
        if m.accuracy is not None:
            print(f"    Accuracy: {m.accuracy:.3f}")
            print(f"    F1 (macro): {m.f1_macro:.3f}")
        if m.r2 is not None:
            print(f"    R2: {m.r2:.3f}")
            print(f"    MAE: {m.mae:.3f}")
            print(f"    RMSE: {m.rmse:.3f}")

    print("\n--- Learnability Analysis ---")
    for l in report.learnability:
        status = "✓" if l.learning_rate_positive else "✗"
        print(f"\n  {l.task}:")
        print(f"    {status} Learning rate positive: {l.learning_rate_positive}")
        print(f"    Convergence score: {l.convergence_score:.3f}")
        print(
            f"    CV mean: {l.distribution_similarity['cv_mean']:.3f} ± {l.distribution_similarity['cv_std']:.3f}"
        )

    print("\n--- Feature Interpretability ---")
    for i in report.interpretability:
        status = "✓" if i.musically_meaningful else "✗"
        print(f"\n  {i.task}:")
        print(f"    Top features: {', '.join(i.top_features[:5])}")
        print(f"    {status} Musically meaningful: {i.musically_meaningful}")
        print(f"    Analysis: {i.analysis}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Best classification accuracy: {report.summary['best_classification_accuracy']:.3f}")
    print(f"  Best regression R2: {report.summary['best_regression_r2']:.3f}")
    print(f"  All learnability positive: {report.summary['all_learnability_positive']}")
    print(f"  All features meaningful: {report.summary['all_features_meaningful']}")
    print(f"\n  Verdict: {report.summary['verdict']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate utility proofs for the SOUSA dataset")
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save report (default: dataset_dir/proofs/)",
    )

    args = parser.parse_args()

    if not args.dataset_dir.exists():
        logger.error(f"Dataset directory not found: {args.dataset_dir}")
        return 1

    output_dir = args.output_dir or args.dataset_dir / "proofs"

    try:
        report = generate_utility_report(args.dataset_dir, output_dir)
        print_report_summary(report)
        return 0
    except Exception as e:
        logger.error(f"Failed to generate utility report: {e}")
        raise


if __name__ == "__main__":
    exit(main())
