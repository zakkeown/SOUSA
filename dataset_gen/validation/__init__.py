"""Dataset validation and statistical analysis."""

from dataset_gen.validation.analysis import (
    DatasetAnalyzer,
    DistributionStats,
    analyze_dataset,
)
from dataset_gen.validation.verify import (
    LabelVerifier,
    VerificationResult,
    verify_labels,
)
from dataset_gen.validation.report import (
    ValidationReport,
    generate_report,
)

__all__ = [
    # Analysis
    "DatasetAnalyzer",
    "DistributionStats",
    "analyze_dataset",
    # Verification
    "LabelVerifier",
    "VerificationResult",
    "verify_labels",
    # Reports
    "ValidationReport",
    "generate_report",
]
