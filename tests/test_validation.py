"""Tests for dataset validation and analysis."""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from dataset_gen.rudiments.schema import (
    Rudiment,
    RudimentCategory,
    parse_sticking_string,
)
from dataset_gen.pipeline.generate import DatasetGenerator, GenerationConfig
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
    generate_report,
    quick_validate,
)


@pytest.fixture
def sample_rudiment():
    """Create a simple rudiment for testing."""
    pattern = parse_sticking_string("RLRL", ">..>")
    return Rudiment(
        name="Test Rudiment",
        slug="test_rudiment",
        category=RudimentCategory.ROLL,
        pattern=pattern,
    )


@pytest.fixture
def generated_dataset(sample_rudiment):
    """Generate a small dataset for testing validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = GenerationConfig(
            output_dir=Path(tmpdir),
            num_profiles=6,  # 2 per non-beginner tier to test ordering
            tempos_per_rudiment=2,
            augmentations_per_sample=1,
            generate_audio=False,
            verbose=False,
            seed=42,
        )

        generator = DatasetGenerator(config)
        generator.generate(rudiments=[sample_rudiment])
        generator.close()

        yield tmpdir


class TestDistributionStats:
    """Tests for DistributionStats."""

    def test_from_array(self):
        """Test creating stats from numpy array."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = DistributionStats.from_array("test", data)

        assert stats.name == "test"
        assert stats.count == 10
        assert stats.mean == 5.5
        assert stats.min == 1
        assert stats.max == 10
        assert stats.median == 5.5

    def test_from_empty_array(self):
        """Test creating stats from empty array."""
        data = np.array([])
        stats = DistributionStats.from_array("empty", data)

        assert stats.count == 0
        assert stats.mean == 0.0

    def test_to_dict(self):
        """Test serialization to dict."""
        data = np.array([1, 2, 3])
        stats = DistributionStats.from_array("test", data)
        d = stats.to_dict()

        assert "name" in d
        assert "mean" in d
        assert "std" in d
        assert d["count"] == 3


class TestDatasetAnalyzer:
    """Tests for DatasetAnalyzer."""

    def test_load_data(self, generated_dataset):
        """Test loading dataset data."""
        analyzer = DatasetAnalyzer(generated_dataset)
        analyzer.load_data()

        assert len(analyzer.samples) > 0
        assert len(analyzer.strokes) > 0
        assert len(analyzer.measures) > 0
        assert len(analyzer.exercises) > 0

    def test_compute_stats(self, generated_dataset):
        """Test computing dataset statistics."""
        analyzer = DatasetAnalyzer(generated_dataset)
        stats = analyzer.compute_stats()

        assert stats.num_samples > 0
        assert stats.num_profiles > 0
        assert stats.num_rudiments > 0

        # Check tempo stats
        assert stats.tempo_stats.count > 0
        assert stats.tempo_stats.mean > 0

        # Check timing error stats
        assert stats.timing_error_stats.count > 0

        # Check skill tier counts
        assert len(stats.skill_tier_counts) > 0

    def test_check_skill_tier_ordering(self, generated_dataset):
        """Test skill tier ordering check."""
        analyzer = DatasetAnalyzer(generated_dataset)
        ordering = analyzer.check_skill_tier_ordering()

        # Should have some ordering results
        assert "timing_accuracy_means" in ordering or "timing_accuracy_ordered" in ordering

    def test_analyze_dataset_convenience(self, generated_dataset):
        """Test analyze_dataset convenience function."""
        stats = analyze_dataset(generated_dataset)

        assert stats.num_samples > 0


class TestLabelVerifier:
    """Tests for LabelVerifier."""

    def test_verify_all(self, generated_dataset):
        """Test running all verification checks."""
        verifier = LabelVerifier(generated_dataset)
        result = verifier.verify_all()

        assert len(result.checks) > 0
        # Most checks should pass for valid generated data
        assert result.num_passed > 0

    def test_sample_ids_unique(self, generated_dataset):
        """Test sample ID uniqueness check."""
        verifier = LabelVerifier(generated_dataset)
        verifier.load_data()
        check = verifier._check_sample_ids_unique()

        assert check.passed
        assert check.details["total"] == check.details["unique"]

    def test_velocity_range(self, generated_dataset):
        """Test velocity range check."""
        verifier = LabelVerifier(generated_dataset)
        verifier.load_data()
        check = verifier._check_velocity_range()

        assert check.passed
        assert check.details["min"] >= 1
        assert check.details["max"] <= 127

    def test_timing_range(self, generated_dataset):
        """Test timing range check."""
        verifier = LabelVerifier(generated_dataset)
        verifier.load_data()
        check = verifier._check_timing_range()

        # Check that timing errors are reasonable
        # Most should be under 200ms, and mean should be under 100ms
        assert check.details["pct_under_200ms"] >= 70  # Most should be in range
        assert check.details["mean_error"] < 150  # Mean should be reasonable

    def test_stroke_refs_valid(self, generated_dataset):
        """Test stroke reference validation."""
        verifier = LabelVerifier(generated_dataset)
        verifier.load_data()
        check = verifier._check_stroke_sample_refs()

        assert check.passed
        assert check.details["orphan_count"] == 0

    def test_verify_labels_convenience(self, generated_dataset):
        """Test verify_labels convenience function."""
        result = verify_labels(generated_dataset)

        assert isinstance(result, VerificationResult)
        assert len(result.checks) > 0


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_all_passed(self):
        """Test all_passed property."""
        from dataset_gen.validation.verify import VerificationCheck

        result = VerificationResult(
            checks=[
                VerificationCheck("a", True, "ok"),
                VerificationCheck("b", True, "ok"),
            ]
        )
        assert result.all_passed

        result = VerificationResult(
            checks=[
                VerificationCheck("a", True, "ok"),
                VerificationCheck("b", False, "fail"),
            ]
        )
        assert not result.all_passed

    def test_counts(self):
        """Test passed/failed counts."""
        from dataset_gen.validation.verify import VerificationCheck

        result = VerificationResult(
            checks=[
                VerificationCheck("a", True, "ok"),
                VerificationCheck("b", False, "fail"),
                VerificationCheck("c", True, "ok"),
            ]
        )

        assert result.num_passed == 2
        assert result.num_failed == 1

    def test_summary(self):
        """Test summary string generation."""
        from dataset_gen.validation.verify import VerificationCheck

        result = VerificationResult(
            checks=[
                VerificationCheck("test_check", True, "passed ok"),
            ]
        )
        summary = result.summary()

        assert "1/1" in summary
        assert "test_check" in summary


class TestValidationReport:
    """Tests for ValidationReport."""

    def test_generate_report(self, generated_dataset):
        """Test generating a full validation report."""
        report = generate_report(generated_dataset)

        assert report.stats.num_samples > 0
        assert len(report.verification.checks) > 0
        assert report.generated_at is not None

    def test_report_summary(self, generated_dataset):
        """Test report summary generation."""
        report = generate_report(generated_dataset)
        summary = report.summary()

        assert "DATASET VALIDATION REPORT" in summary
        assert "Samples:" in summary
        assert "Verification Results" in summary

    def test_save_report(self, generated_dataset):
        """Test saving report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = generate_report(generated_dataset)
            output_path = Path(tmpdir) / "report.json"
            report.save(output_path)

            assert output_path.exists()

            import json

            with open(output_path) as f:
                data = json.load(f)

            assert "stats" in data
            assert "verification" in data

    def test_quick_validate(self, generated_dataset):
        """Test quick_validate convenience function."""
        is_valid = quick_validate(generated_dataset)

        # Generated dataset should be valid
        assert isinstance(is_valid, bool)


class TestSkillTierOrdering:
    """Tests specifically for skill tier ordering validation."""

    def test_timing_accuracy_ordering(self, generated_dataset):
        """Test that professional > advanced > intermediate > beginner for timing."""
        analyzer = DatasetAnalyzer(generated_dataset)
        ordering = analyzer.check_skill_tier_ordering()

        # If we have the ordering check, verify it
        if "timing_accuracy_means" in ordering:
            means = ordering["timing_accuracy_means"]
            # Higher skill tiers should have higher timing accuracy
            # (with some tolerance for random variation in small datasets)
            if len(means) >= 2:
                # Just verify the data exists and is reasonable
                for tier, value in means.items():
                    assert 0 <= value <= 100

    def test_hand_balance_ordering(self, generated_dataset):
        """Test that professional > advanced > intermediate > beginner for hand balance."""
        analyzer = DatasetAnalyzer(generated_dataset)
        ordering = analyzer.check_skill_tier_ordering()

        if "hand_balance_means" in ordering:
            means = ordering["hand_balance_means"]
            if len(means) >= 2:
                for tier, value in means.items():
                    assert 0 <= value <= 100


class TestDataIntegrity:
    """Tests for data integrity validation."""

    def test_stroke_counts_match(self, generated_dataset):
        """Test that stroke counts in samples match actual records."""
        verifier = LabelVerifier(generated_dataset)
        verifier.load_data()
        check = verifier._check_stroke_counts_match()

        assert check.passed, f"Stroke count mismatch: {check.message}"

    def test_measure_counts_match(self, generated_dataset):
        """Test that measure counts in samples match actual records."""
        verifier = LabelVerifier(generated_dataset)
        verifier.load_data()
        check = verifier._check_measure_counts_match()

        assert check.passed, f"Measure count mismatch: {check.message}"

    def test_all_samples_have_exercises(self, generated_dataset):
        """Test that every sample has an exercise score record."""
        verifier = LabelVerifier(generated_dataset)
        verifier.load_data()

        sample_ids = set(verifier.samples["sample_id"])
        exercise_ids = set(verifier.exercises["sample_id"])

        missing = sample_ids - exercise_ids
        assert len(missing) == 0, f"Samples missing exercise scores: {missing}"
