"""
Validation report generation.

This module generates comprehensive validation reports
combining statistical analysis and verification results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import logging

from dataset_gen.validation.analysis import DatasetAnalyzer, DatasetStats
from dataset_gen.validation.verify import LabelVerifier, VerificationResult
from dataset_gen.validation.realism import RealismValidator, RealismReport

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Complete validation report."""

    dataset_path: str
    generated_at: str
    stats: DatasetStats
    verification: VerificationResult
    skill_tier_ordering: dict[str, Any]
    realism: RealismReport | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "dataset_path": self.dataset_path,
            "generated_at": self.generated_at,
            "stats": self.stats.to_dict(),
            "verification": self.verification.to_dict(),
            "skill_tier_ordering": self.skill_tier_ordering,
        }
        if self.realism:
            result["realism"] = self.realism.to_dict()
        return result

    def save(self, output_path: Path | str) -> None:
        """Save report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Report saved to {output_path}")

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "DATASET VALIDATION REPORT",
            "=" * 60,
            f"Dataset: {self.dataset_path}",
            f"Generated: {self.generated_at}",
            "",
            "--- Dataset Overview ---",
            f"Samples:    {self.stats.num_samples:,}",
            f"Profiles:   {self.stats.num_profiles:,}",
            f"Rudiments:  {self.stats.num_rudiments:,}",
            "",
            "--- Skill Tier Distribution ---",
        ]

        for tier, count in sorted(self.stats.skill_tier_counts.items()):
            pct = 100 * count / max(self.stats.num_samples, 1)
            lines.append(f"  {tier}: {count:,} ({pct:.1f}%)")

        lines.extend(
            [
                "",
                "--- Split Distribution ---",
            ]
        )

        for split, count in sorted(self.stats.split_counts.items()):
            pct = 100 * count / max(self.stats.num_samples, 1)
            lines.append(f"  {split}: {count:,} ({pct:.1f}%)")

        lines.extend(
            [
                "",
                "--- Key Statistics ---",
                f"Tempo:           {self.stats.tempo_stats.mean:.1f} ± {self.stats.tempo_stats.std:.1f} BPM",
                f"Duration:        {self.stats.duration_stats.mean:.2f} ± {self.stats.duration_stats.std:.2f} sec",
                f"Strokes/sample:  {self.stats.num_strokes_stats.mean:.1f} ± {self.stats.num_strokes_stats.std:.1f}",
                "",
                "--- Timing Error by Skill Tier ---",
            ]
        )

        for tier, tier_stats in sorted(
            self.stats.timing_error_stats.by_group.items(),
            key=lambda x: x[0],
        ):
            lines.append(f"  {tier}: {tier_stats.mean:.2f} ± {tier_stats.std:.2f} ms")

        lines.extend(
            [
                "",
                "--- Exercise Scores by Skill Tier ---",
                "Timing Accuracy:",
            ]
        )

        for tier, tier_stats in sorted(
            self.stats.exercise_timing_accuracy.by_group.items(),
            key=lambda x: x[0],
        ):
            lines.append(f"  {tier}: {tier_stats.mean:.1f} ± {tier_stats.std:.1f}")

        lines.append("Hand Balance:")
        for tier, tier_stats in sorted(
            self.stats.exercise_hand_balance.by_group.items(),
            key=lambda x: x[0],
        ):
            lines.append(f"  {tier}: {tier_stats.mean:.1f} ± {tier_stats.std:.1f}")

        lines.extend(
            [
                "",
                "--- Skill Tier Ordering ---",
            ]
        )

        for key, value in self.skill_tier_ordering.items():
            if isinstance(value, bool):
                status = "✓" if value else "✗"
                lines.append(f"  {status} {key}")

        lines.extend(
            [
                "",
                "--- Verification Results ---",
                f"Passed: {self.verification.num_passed}/{len(self.verification.checks)}",
                "",
            ]
        )

        for check in self.verification.checks:
            status = "✓" if check.passed else "✗"
            lines.append(f"  {status} {check.name}: {check.message}")

        # Realism validation results
        if self.realism:
            lines.extend(
                [
                    "",
                    "--- Realism Validation ---",
                    f"Literature comparison: {self.realism.literature_pass_rate:.1f}% pass",
                    f"Correlation structure: {self.realism.correlation_pass_rate:.1f}% pass",
                    "",
                ]
            )

            # Literature comparisons by tier
            for comp in self.realism.literature_comparisons:
                status = "✓" if comp.within_range else "✗"
                lines.append(
                    f"  {status} {comp.skill_tier} {comp.metric}: "
                    f"{comp.dataset_value:.2f} (expected {comp.expected_range})"
                )

        lines.extend(
            [
                "",
                "=" * 60,
                f"Overall: {'PASSED' if self.verification.all_passed else 'FAILED'}",
                "=" * 60,
            ]
        )

        return "\n".join(lines)


def generate_report(
    dataset_dir: Path | str,
    output_path: Path | str | None = None,
    include_realism: bool = True,
    include_midi_checks: bool = True,
) -> ValidationReport:
    """
    Generate comprehensive validation report for a dataset.

    Args:
        dataset_dir: Path to dataset directory
        output_path: Optional path to save report JSON
        include_realism: Whether to run realism validation (literature comparison)
        include_midi_checks: Whether to run MIDI alignment checks

    Returns:
        ValidationReport with stats and verification results
    """
    dataset_dir = Path(dataset_dir)
    logger.info(f"Generating validation report for {dataset_dir}")

    # Run analysis
    analyzer = DatasetAnalyzer(dataset_dir)
    stats = analyzer.compute_stats()
    skill_ordering = analyzer.check_skill_tier_ordering()

    # Run verification
    verifier = LabelVerifier(dataset_dir)
    verification = verifier.verify_all(include_midi_checks=include_midi_checks)

    # Run realism validation
    realism = None
    if include_realism:
        logger.info("Running realism validation...")
        realism_validator = RealismValidator(dataset_dir)
        realism = realism_validator.validate_all()

    # Create report
    report = ValidationReport(
        dataset_path=str(dataset_dir),
        generated_at=datetime.now().isoformat(),
        stats=stats,
        verification=verification,
        skill_tier_ordering=skill_ordering,
        realism=realism,
    )

    # Save if requested
    if output_path:
        report.save(output_path)

    return report


def quick_validate(dataset_dir: Path | str) -> bool:
    """
    Quick validation check - returns True if dataset is valid.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        True if all verification checks pass
    """
    verifier = LabelVerifier(dataset_dir)
    result = verifier.verify_all()
    return result.all_passed


def generate_html_report(
    report: ValidationReport,
    output_path: Path | str,
    template_path: Path | str | None = None,
) -> None:
    """
    Generate an HTML validation report with visualizations.

    Args:
        report: ValidationReport to render
        output_path: Path to save HTML file
        template_path: Optional path to custom template
    """
    import re

    # Find template
    if template_path is None:
        # Look in scripts/templates relative to this file
        module_dir = Path(__file__).parent.parent.parent
        template_path = module_dir / "scripts" / "templates" / "validation_report.html"

    template_path = Path(template_path)
    if not template_path.exists():
        logger.warning(f"HTML template not found at {template_path}")
        # Fall back to generating a simple HTML report
        _generate_simple_html(report, output_path)
        return

    # Read template
    with open(template_path) as f:
        html = f.read()

    # Embed report data as JSON
    report_json = json.dumps(report.to_dict(), indent=2, default=str)

    # Replace placeholder with actual data
    html = re.sub(
        r"/\* REPORT_DATA_PLACEHOLDER \*/ \{\}",
        report_json,
        html,
    )

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"HTML report saved to {output_path}")


def _generate_simple_html(report: ValidationReport, output_path: Path | str) -> None:
    """Generate a simple HTML report without the template."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>SOUSA Validation Report</title>
    <style>
        body {{ font-family: sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>SOUSA Dataset Validation Report</h1>
    <p>Generated: {report.generated_at}</p>
    <p>Dataset: {report.dataset_path}</p>

    <h2>Verification Results</h2>
    <p class="{'passed' if report.verification.all_passed else 'failed'}">
        {report.verification.num_passed}/{len(report.verification.checks)} checks passed
    </p>

    <h2>Full Report Data</h2>
    <pre>{json.dumps(report.to_dict(), indent=2, default=str)}</pre>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"Simple HTML report saved to {output_path}")
