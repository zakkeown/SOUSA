#!/usr/bin/env python3
"""
Experiment Runner
=================

CLI to run all or specific SOUSA experiments with automatic result aggregation.

Experiments:
- score_analysis: Score correlation analysis (no training)
- split_validation: Compare profile-based vs random splits
- augmentation_ablation: Test augmentation impact
- soundfont_ablation: Test soundfont generalization

Usage:
    # Run all experiments
    python scripts/run_experiments.py --data-dir output/dataset

    # Run specific experiments
    python scripts/run_experiments.py --experiments score_analysis split_validation

    # Quick mode (fast_debug preset)
    python scripts/run_experiments.py --quick

    # Generate summary report
    python scripts/run_experiments.py --report-only

Output:
    experiments/results/
    ├── score_correlations.json
    ├── split_validation.json
    ├── augmentation_ablation.json
    ├── soundfont_ablation.json
    └── experiment_summary.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENTS = {
    "score_analysis": {
        "module": "experiments.score_analysis",
        "description": "Score correlation analysis",
        "requires_training": False,
    },
    "split_validation": {
        "module": "experiments.split_validation",
        "description": "Profile-based vs random split comparison",
        "requires_training": True,
    },
    "augmentation_ablation": {
        "module": "experiments.augmentation_ablation",
        "description": "Augmentation impact study",
        "requires_training": True,
    },
    "soundfont_ablation": {
        "module": "experiments.soundfont_ablation",
        "description": "Soundfont generalization study",
        "requires_training": True,
    },
}


def run_experiment(
    name: str,
    data_dir: Path,
    output_dir: Path,
    preset: str = "fast_debug",
    extra_args: list[str] | None = None,
) -> tuple[bool, str]:
    """Run a single experiment and return (success, message)."""
    exp_config = EXPERIMENTS.get(name)
    if not exp_config:
        return False, f"Unknown experiment: {name}"

    cmd = [
        sys.executable, "-m", exp_config["module"],
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
    ]

    # Add preset for training experiments
    if exp_config["requires_training"]:
        cmd.extend(["--preset", preset])

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Exit code: {result.returncode}"

    except Exception as e:
        return False, str(e)


def generate_summary_report(results_dir: Path, output_path: Path):
    """Generate markdown summary of all experiment results."""
    report = []
    report.append("# SOUSA Experiment Results Summary")
    report.append(f"\nGenerated: {datetime.now().isoformat()}\n")

    # Score analysis
    score_path = results_dir / "score_correlations.json"
    if score_path.exists():
        with open(score_path) as f:
            data = json.load(f)

        report.append("## Score Correlation Analysis\n")
        report.append(f"- Samples analyzed: {data.get('num_samples', 'N/A')}")
        report.append(f"- Score columns: {len(data.get('score_columns', []))}")

        if data.get("high_correlations"):
            report.append("\n**High correlations (|r| > 0.8):**")
            for pair in data["high_correlations"][:5]:
                report.append(f"- {pair['score1']} ↔ {pair['score2']}: r = {pair['correlation']:.3f}")

        if data.get("recommended_scores"):
            report.append(f"\n**Recommended minimal score set:** {data['recommended_scores']}")

        report.append("")

    # Split validation
    split_path = results_dir / "split_validation.json"
    if split_path.exists():
        with open(split_path) as f:
            data = json.load(f)

        summary = data.get("summary", {})
        report.append("## Split Validation\n")
        report.append("| Split Method | Mean Accuracy | Std |")
        report.append("|--------------|---------------|-----|")

        profile = summary.get("profile_based", {})
        report.append(f"| Profile-based | {profile.get('mean_accuracy', 0):.4f} | {profile.get('std_accuracy', 0):.4f} |")

        random = summary.get("random_split", {})
        report.append(f"| Random | {random.get('mean_accuracy', 0):.4f} | {random.get('std_accuracy', 0):.4f} |")

        if summary.get("p_value"):
            report.append(f"\n**Statistical test:** p = {summary['p_value']:.4f}")

        report.append("")

    # Augmentation ablation
    aug_path = results_dir / "augmentation_ablation.json"
    if aug_path.exists():
        with open(aug_path) as f:
            data = json.load(f)

        report.append("## Augmentation Ablation\n")

        if data.get("has_augmentation_metadata"):
            info = data.get("augmentation_info", {})
            report.append(f"- Clean samples: {info.get('n_clean', 'N/A')}")
            report.append(f"- Augmented samples: {info.get('n_augmented', 'N/A')}")

            aggregated = data.get("aggregated", {})
            if aggregated and not aggregated.get("full_dataset_only"):
                report.append("\n| Condition | Accuracy |")
                report.append("|-----------|----------|")
                for condition, result in aggregated.items():
                    if isinstance(result, dict):
                        acc = result.get("mean_accuracy", result.get("accuracy", "N/A"))
                        report.append(f"| {condition} | {acc:.4f} |")
        else:
            report.append("*Dataset does not have augmentation metadata.*")

        report.append("")

    # Soundfont ablation
    sf_path = results_dir / "soundfont_ablation.json"
    if sf_path.exists():
        with open(sf_path) as f:
            data = json.load(f)

        report.append("## Soundfont Ablation\n")

        soundfonts = data.get("soundfonts", [])
        report.append(f"- Soundfonts tested: {soundfonts}")

        results = data.get("results", {})

        if results.get("all_to_single"):
            report.append("\n**Train on all → Test on each:**")
            report.append("| Soundfont | Accuracy |")
            report.append("|-----------|----------|")
            for sf, acc in results["all_to_single"].items():
                report.append(f"| {sf} | {acc:.4f} |")

        analysis = data.get("analysis", {})
        if analysis.get("average_generalization_gap") is not None:
            report.append(f"\n**Average generalization gap:** {analysis['average_generalization_gap']:.4f}")

        report.append("")

    # Write report
    report_text = "\n".join(report)
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"\nSummary report saved to: {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Run SOUSA experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output/dataset",
        help="Path to SOUSA dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(EXPERIMENTS.keys()),
        default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast_debug", "baseline", "full"],
        default="fast_debug",
        help="Config preset for training experiments",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fast_debug preset, analysis only)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate summary report from existing results",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip experiments that require training",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode
    if args.quick:
        args.preset = "fast_debug"
        args.experiments = ["score_analysis"]

    # Report only
    if args.report_only:
        report_path = output_dir / "experiment_summary.md"
        generate_summary_report(output_dir, report_path)
        return

    # Determine experiments to run
    experiments = args.experiments or list(EXPERIMENTS.keys())

    if args.skip_training:
        experiments = [e for e in experiments if not EXPERIMENTS[e]["requires_training"]]

    print("=" * 60)
    print("SOUSA Experiment Runner")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Preset: {args.preset}")
    print(f"Experiments: {experiments}")
    print()

    # Check data directory
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Generate a dataset first with: python scripts/generate_dataset.py --preset small")
        sys.exit(1)

    # Run experiments
    results = {}
    for exp_name in experiments:
        success, message = run_experiment(
            exp_name,
            data_dir,
            output_dir,
            preset=args.preset,
        )
        results[exp_name] = {"success": success, "message": message}

    # Summary
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)

    for exp_name, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {exp_name}: {result['message']}")

    # Generate report
    report_path = output_dir / "experiment_summary.md"
    generate_summary_report(output_dir, report_path)

    # Exit code
    all_success = all(r["success"] for r in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
