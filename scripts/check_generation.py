#!/usr/bin/env python3
"""
Quick validation check during dataset generation.

Run periodically to monitor generation health:
    python scripts/check_generation.py output/dataset

Returns exit code 0 if OK, 1 if issues found.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_gen.validation.report import generate_report


def main():
    if len(sys.argv) < 2:
        dataset_dir = Path("output/dataset")
    else:
        dataset_dir = Path(sys.argv[1])

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    labels_dir = dataset_dir / "labels"
    if not labels_dir.exists() or not any(labels_dir.glob("*.parquet")):
        print("No labels generated yet. Wait for more samples.")
        sys.exit(0)

    print(f"Checking {dataset_dir}...")
    print()

    try:
        report = generate_report(
            str(dataset_dir),
            include_realism=True,
            include_midi_checks=True,
        )
    except Exception as e:
        print(f"ERROR: Failed to generate report: {e}")
        sys.exit(1)

    # Count samples
    n_samples = report.stats.num_samples if report.stats else 0
    print(f"Samples: {n_samples}")

    # Tier distribution
    if report.stats and report.stats.skill_tier_counts:
        tiers = report.stats.skill_tier_counts
        print(f"Tiers: {', '.join(f'{k}={v}' for k, v in tiers.items())}")
    print()

    # Check for critical issues
    issues = []
    warnings = []

    # Verification checks
    if report.verification:
        for check in report.verification.checks:
            if not check.passed:
                if check.name == "label_midi_alignment":
                    # Get alignment rate
                    rate = check.details.get("alignment_rate", 0)
                    if rate < 95:
                        issues.append(f"MIDI alignment: {rate:.1f}% (need >95%)")
                    else:
                        warnings.append(f"MIDI alignment: {rate:.1f}%")
                elif check.name in ("velocity_range", "timing_range", "score_ranges"):
                    issues.append(f"{check.name}: {check.message}")
                else:
                    warnings.append(f"{check.name}: {check.message}")

    # Realism checks
    if report.realism:
        for comp in report.realism.literature_comparisons:
            if not comp.within_range:
                msg = f"{comp.skill_tier} {comp.metric}: {comp.dataset_value:.1f} (expected {comp.expected_range})"
                # Timing and velocity are critical
                if comp.metric in ("timing_error_sd_ms", "velocity_cv"):
                    issues.append(msg)
                else:
                    warnings.append(msg)

    # Print status
    if issues:
        print("=" * 50)
        print("STOP GENERATION - ISSUES FOUND")
        print("=" * 50)
        for issue in issues:
            print(f"  [X] {issue}")
        print()
        print("Consult Claude before continuing.")
        sys.exit(1)

    elif warnings:
        print("STATUS: OK (with warnings)")
        print("-" * 30)
        for warn in warnings:
            print(f"  [!] {warn}")
        print()
        print("Generation can continue. Monitor these.")
        sys.exit(0)

    else:
        print("STATUS: ALL CHECKS PASS")
        print("-" * 30)

        # Show key metrics
        if report.realism:
            for comp in report.realism.literature_comparisons:
                status = "OK" if comp.within_range else "!!"
                print(f"  [{status}] {comp.skill_tier} {comp.metric}: {comp.dataset_value:.1f}")

        print()
        print("Generation looks healthy.")
        sys.exit(0)


if __name__ == "__main__":
    main()
