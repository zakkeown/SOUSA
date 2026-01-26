# Validation Module

The validation module provides tools for verifying dataset correctness and analyzing statistical properties. It includes:

- **Verification**: Data integrity checks, range validation, skill tier ordering
- **Analysis**: Distribution statistics across all hierarchical levels
- **Reporting**: Comprehensive validation reports with pass/fail summaries

## Validation Report

Generates comprehensive validation reports combining statistical analysis and verification results.

::: dataset_gen.validation.report
    options:
      show_root_heading: false
      members:
        - ValidationReport
        - generate_report
        - quick_validate
        - generate_html_report

## Label Verification

Verifies label correctness and data integrity through a series of automated checks.

::: dataset_gen.validation.verify
    options:
      show_root_heading: false
      members:
        - VerificationCheck
        - VerificationResult
        - LabelVerifier
        - verify_labels

## Statistical Analysis

Analyzes generated dataset distributions across samples, strokes, measures, and exercises.

::: dataset_gen.validation.analysis
    options:
      show_root_heading: false
      members:
        - DistributionStats
        - DatasetStats
        - DatasetAnalyzer
        - analyze_dataset
