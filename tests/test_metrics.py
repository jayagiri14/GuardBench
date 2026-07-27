"""
Unit tests for guardrail_agent.evaluation.metrics
"""

import pytest
from guardrail_agent.evaluation.metrics import (
    compute_metrics,
    ConfusionMatrix,
    EvalMetrics,
)


# ---------------------------------------------------------------------------
# ConfusionMatrix unit tests
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_perfect_precision_recall(self):
        cm = ConfusionMatrix(TP=10, FP=0, TN=10, FN=0)
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0
        assert cm.fpr == 0.0
        assert cm.accuracy == 1.0

    def test_zero_precision_recall(self):
        cm = ConfusionMatrix(TP=0, FP=5, TN=5, FN=5)
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0

    def test_fpr_calculation(self):
        cm = ConfusionMatrix(TP=8, FP=2, TN=8, FN=2)
        # FPR = 2 / (2 + 8) = 0.2
        assert abs(cm.fpr - 0.2) < 1e-9

    def test_f1_score(self):
        # Precision = 8/10 = 0.8, Recall = 8/10 = 0.8 -> F1 = 0.8
        cm = ConfusionMatrix(TP=8, FP=2, TN=8, FN=2)
        assert abs(cm.f1 - 0.8) < 1e-9

    def test_zero_division_returns_zero(self):
        cm = ConfusionMatrix(TP=0, FP=0, TN=0, FN=0)
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0
        assert cm.fpr == 0.0
        assert cm.accuracy == 0.0

    def test_accuracy(self):
        cm = ConfusionMatrix(TP=7, FP=1, TN=9, FN=3)
        # (7 + 9) / (7 + 1 + 9 + 3) = 16/20 = 0.8
        assert abs(cm.accuracy - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# compute_metrics with fully controlled records
# ---------------------------------------------------------------------------

def _make_records(
    injection_blocked: int,
    injection_missed: int,
    benign_passed: int,
    benign_blocked: int,
    pii_detected: int,
    pii_missed: int,
) -> list[dict]:
    records = []
    for _ in range(injection_blocked):
        records.append({"category": "injection", "actual_outcome": "BLOCKED_INPUT"})
    for _ in range(injection_missed):
        records.append({"category": "injection", "actual_outcome": "ANSWERED"})
    for _ in range(benign_passed):
        records.append({"category": "benign", "actual_outcome": "ANSWERED"})
    for _ in range(benign_blocked):
        records.append({"category": "benign", "actual_outcome": "BLOCKED_INPUT"})
    for _ in range(pii_detected):
        records.append({"category": "pii_leakage", "actual_outcome": "REDACTED_INPUT"})
    for _ in range(pii_missed):
        records.append({"category": "pii_leakage", "actual_outcome": "ANSWERED"})
    return records


class TestComputeMetricsPerfect:
    def test_perfect_detection_no_fp(self):
        records = _make_records(
            injection_blocked=10, injection_missed=0,
            benign_passed=10, benign_blocked=0,
            pii_detected=5, pii_missed=0,
        )
        m = compute_metrics(records)
        assert m.attack_detection_rate == 1.0
        assert m.false_positive_rate == 0.0
        assert m.task_accuracy == 1.0
        assert m.pii_detection_rate == 1.0
        assert m.attack_f1 == 1.0

    def test_sample_counts_correct(self):
        records = _make_records(5, 2, 8, 1, 4, 1)
        m = compute_metrics(records)
        assert m.injection_count == 7
        assert m.benign_count == 9
        assert m.pii_leakage_count == 5
        assert m.total_samples == 21


class TestComputeMetricsPartial:
    def test_50_percent_attack_detection(self):
        records = _make_records(
            injection_blocked=5, injection_missed=5,
            benign_passed=10, benign_blocked=0,
            pii_detected=0, pii_missed=0,
        )
        m = compute_metrics(records)
        assert abs(m.attack_detection_rate - 0.5) < 1e-9

    def test_false_positive_rate_calculation(self):
        # 2 benign blocked out of 10 benign -> FPR = 2/10 = 0.2
        records = _make_records(
            injection_blocked=10, injection_missed=0,
            benign_passed=8, benign_blocked=2,
            pii_detected=0, pii_missed=0,
        )
        m = compute_metrics(records)
        assert abs(m.false_positive_rate - 0.2) < 1e-9

    def test_task_accuracy(self):
        # 8 benign answered, 2 blocked -> task_accuracy = 8/10
        records = _make_records(
            injection_blocked=5, injection_missed=0,
            benign_passed=8, benign_blocked=2,
            pii_detected=0, pii_missed=0,
        )
        m = compute_metrics(records)
        assert abs(m.task_accuracy - 0.8) < 1e-9

    def test_pii_detection_rate(self):
        # 3 detected, 1 missed -> 0.75
        records = _make_records(0, 0, 5, 0, 3, 1)
        m = compute_metrics(records)
        assert abs(m.pii_detection_rate - 0.75) < 1e-9


class TestComputeMetricsEdgeCases:
    def test_empty_records_returns_zeros(self):
        m = compute_metrics([])
        assert m.total_samples == 0
        assert m.attack_detection_rate == 0.0
        assert m.false_positive_rate == 0.0

    def test_only_injection_records(self):
        records = [
            {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
            {"category": "injection", "actual_outcome": "BLOCKED_INPUT"},
            {"category": "injection", "actual_outcome": "ANSWERED"},  # missed
        ]
        m = compute_metrics(records)
        assert m.injection_count == 3
        assert m.attack_cm.TP == 2
        assert m.attack_cm.FN == 1
        # No benign -> TN=0, FP=0
        assert m.attack_cm.TN == 0
        assert abs(m.attack_detection_rate - 2/3) < 1e-9

    def test_only_benign_records(self):
        records = [
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "benign", "actual_outcome": "ANSWERED"},
            {"category": "benign", "actual_outcome": "BLOCKED_INPUT"},  # false positive
        ]
        m = compute_metrics(records)
        assert m.benign_count == 3
        assert m.attack_cm.TN == 2
        assert m.attack_cm.FP == 1
        assert abs(m.task_accuracy - 2/3) < 1e-9

    def test_to_dict_has_required_keys(self):
        records = _make_records(3, 1, 4, 0, 2, 0)
        m = compute_metrics(records)
        d = m.to_dict()
        for key in [
            "attack_detection_rate", "false_positive_rate",
            "attack_f1", "pii_detection_rate",
            "task_accuracy", "refusal_rate", "confusion_matrix",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_refusal_rate(self):
        # 5 blocked out of 15 total
        records = _make_records(
            injection_blocked=5, injection_missed=0,
            benign_passed=10, benign_blocked=0,
            pii_detected=0, pii_missed=0,
        )
        m = compute_metrics(records)
        assert abs(m.refusal_rate - 5/15) < 1e-9
