"""
Tests for the regression gate and baseline comparator.
"""

import json
import pytest
from pathlib import Path

from guardrail_agent.evaluation.metrics import compute_metrics
from guardrail_agent.evaluation.baseline_comparator import (
    compare_to_baseline,
    RegressionThresholds,
    load_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect_metrics():
    records = [
        {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
        {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
        {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
        {"category": "benign", "actual_outcome": "ANSWERED"},
        {"category": "benign", "actual_outcome": "ANSWERED"},
        {"category": "benign", "actual_outcome": "ANSWERED"},
        {"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"},
        {"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"},
    ]
    return compute_metrics(records)


def _degraded_metrics():
    """Metrics where attack detection is low and FPR is high."""
    records = [
        {"category": "injection", "actual_outcome": "ANSWERED"},   # missed
        {"category": "injection", "actual_outcome": "ANSWERED"},   # missed
        {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
        {"category": "benign", "actual_outcome": "BLOCKED_INPUT"},  # FP
        {"category": "benign", "actual_outcome": "BLOCKED_INPUT"},  # FP
        {"category": "benign", "actual_outcome": "ANSWERED"},
        {"category": "pii_leakage", "actual_outcome": "ANSWERED"},  # missed
        {"category": "pii_leakage", "actual_outcome": "ANSWERED"},  # missed
    ]
    return compute_metrics(records)


def _write_baseline(tmp_path: Path, metrics) -> Path:
    baseline = {
        "run_id": "baseline_001",
        "metrics": metrics.to_dict(),
    }
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(baseline))
    return p


# ---------------------------------------------------------------------------
# Hard floor / ceiling tests (no baseline)
# ---------------------------------------------------------------------------

class TestHardThresholds:
    def test_perfect_metrics_pass_with_default_thresholds(self):
        m = _perfect_metrics()
        report = compare_to_baseline(m, baseline_path=None)
        assert report.passed, f"Unexpected failures: {[f.message for f in report.failures]}"

    def test_zero_attack_detection_fails_hard_floor(self):
        records = [
            {"category": "injection", "actual_outcome": "ANSWERED"},  # all missed
            {"category": "injection", "actual_outcome": "ANSWERED"},
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"},
        ]
        m = compute_metrics(records)
        thresholds = RegressionThresholds(min_attack_detection_rate=0.80)
        report = compare_to_baseline(m, baseline_path=None, thresholds=thresholds)
        assert not report.passed
        failed_metrics = [f.metric for f in report.failures]
        assert "attack_detection_rate" in failed_metrics

    def test_high_fpr_fails_ceiling(self):
        records = [
            {"category": "benign", "actual_outcome": "BLOCKED_INPUT"},  # FP
            {"category": "benign", "actual_outcome": "BLOCKED_INPUT"},  # FP
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
            {"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"},
        ]
        m = compute_metrics(records)
        thresholds = RegressionThresholds(max_false_positive_rate=0.10)
        report = compare_to_baseline(m, baseline_path=None, thresholds=thresholds)
        assert not report.passed
        failed_metrics = [f.metric for f in report.failures]
        assert "false_positive_rate" in failed_metrics

    def test_low_task_accuracy_fails(self):
        records = [
            {"category": "benign", "actual_outcome": "BLOCKED_INPUT"},  # wrong block
            {"category": "benign", "actual_outcome": "BLOCKED_INPUT"},
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
            {"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"},
        ]
        m = compute_metrics(records)
        thresholds = RegressionThresholds(min_task_accuracy=0.90)
        report = compare_to_baseline(m, baseline_path=None, thresholds=thresholds)
        failed_metrics = [f.metric for f in report.failures]
        assert "task_accuracy" in failed_metrics

    def test_multiple_failures_all_reported(self):
        m = _degraded_metrics()
        report = compare_to_baseline(m, baseline_path=None)
        # With default thresholds, low attack detection AND high FPR AND low task accuracy
        # should all fail
        assert not report.passed
        assert len(report.failures) >= 2


# ---------------------------------------------------------------------------
# Baseline regression tests
# ---------------------------------------------------------------------------

class TestBaselineRegression:
    def test_no_regression_when_same(self, tmp_path):
        m = _perfect_metrics()
        baseline_path = _write_baseline(tmp_path, m)
        report = compare_to_baseline(m, baseline_path=baseline_path)
        assert report.passed

    def test_regression_detected_when_attack_detection_drops(self, tmp_path):
        perfect = _perfect_metrics()
        baseline_path = _write_baseline(tmp_path, perfect)

        # Now produce degraded metrics
        degraded = _degraded_metrics()
        thresholds = RegressionThresholds(
            tolerance=0.05,
            min_attack_detection_rate=0.0,   # disable hard floor
            max_false_positive_rate=1.0,     # disable hard ceiling
            min_pii_detection_rate=0.0,
            min_task_accuracy=0.0,
        )
        report = compare_to_baseline(degraded, baseline_path=baseline_path, thresholds=thresholds)
        assert not report.passed
        failed = [f.metric for f in report.failures]
        assert "attack_detection_rate" in failed

    def test_no_regression_within_tolerance(self, tmp_path):
        perfect = _perfect_metrics()
        baseline_path = _write_baseline(tmp_path, perfect)

        # Slightly lower but within tolerance
        records = [
            {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
            {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
            {"category": "injection", "actual_outcome": "ANSWERED"},   # 1 missed -> 66%
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"},
            {"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"},
        ]
        m = compute_metrics(records)
        # Perfect baseline has 100%, current is 66%. Tolerance 0.40 -> should pass
        thresholds = RegressionThresholds(
            tolerance=0.40,
            min_attack_detection_rate=0.0,
            max_false_positive_rate=1.0,
            min_pii_detection_rate=0.0,
            min_task_accuracy=0.0,
        )
        report = compare_to_baseline(m, baseline_path=baseline_path, thresholds=thresholds)
        assert report.passed

    def test_baseline_not_found_raises(self):
        m = _perfect_metrics()
        with pytest.raises(FileNotFoundError):
            compare_to_baseline(m, baseline_path="/nonexistent/path.json")

    def test_regression_report_has_baseline_run_id(self, tmp_path):
        m = _perfect_metrics()
        baseline_path = _write_baseline(tmp_path, m)
        report = compare_to_baseline(m, baseline_path=baseline_path)
        assert report.baseline_run_id == "baseline_001"

    def test_report_summary_pass(self):
        m = _perfect_metrics()
        report = compare_to_baseline(m, baseline_path=None)
        assert "PASS" in report.summary()

    def test_report_summary_fail_lists_metrics(self):
        m = _degraded_metrics()
        report = compare_to_baseline(m, baseline_path=None)
        summary = report.summary()
        assert "FAIL" in summary
        # All failed metric names should appear somewhere
        for f in report.failures:
            assert f.metric in summary or f.message in summary


# ---------------------------------------------------------------------------
# CLI end-to-end regression gate tests
# ---------------------------------------------------------------------------

class TestCLIExitCodes:
    def test_cli_pass_with_mock(self):
        """CLI exits 0 when mock agent metrics meet default thresholds."""
        from guardrail_agent.cli import main
        # Run with very lenient thresholds to guarantee pass
        exit_code = main([
            "--mock",
            "--min-attack-detection-rate", "0.0",
            "--max-fpr", "1.0",
            "--min-pii-detection-rate", "0.0",
            "--min-task-accuracy", "0.0",
            "--output", "eval_results_test.json",
        ])
        assert exit_code == 0

    def test_cli_fail_with_impossible_threshold(self):
        """CLI exits 1 when threshold is impossible to meet."""
        from guardrail_agent.cli import main
        exit_code = main([
            "--mock",
            "--min-attack-detection-rate", "1.01",  # impossible
            "--output", "eval_results_test_fail.json",
        ])
        assert exit_code == 1

    def test_cli_error_without_url_and_no_mock(self):
        """CLI exits 2 when no --mock and no --url/--model provided."""
        from guardrail_agent.cli import main
        exit_code = main(["--output", "never.json"])
        assert exit_code == 2
