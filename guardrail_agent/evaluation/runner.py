"""
Evaluation runner: runs the GuardrailAgent over every dataset sample and
produces structured JSON + markdown results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from guardrail_agent.agent.guardrail_agent import GuardrailAgent
from guardrail_agent.evaluation.dataset import DatasetSample, load_dataset
from guardrail_agent.evaluation.metrics import compute_metrics, EvalMetrics


@dataclass
class SampleResult:
    sample_id: str
    category: str
    text: str
    expected_outcome: str
    actual_outcome: str
    reply_preview: str          # first 150 chars
    canary_leaked: bool
    injection_detected: bool
    pii_in_input: bool
    pii_in_output: bool
    correct: bool               # actual == expected
    latency_s: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "category": self.category,
            "text": self.text[:120] + "..." if len(self.text) > 120 else self.text,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "correct": self.correct,
            "canary_leaked": self.canary_leaked,
            "injection_detected": self.injection_detected,
            "pii_in_input": self.pii_in_input,
            "pii_in_output": self.pii_in_output,
            "reply_preview": self.reply_preview,
            "latency_s": self.latency_s,
        }


@dataclass
class EvalRunResult:
    """Complete result of one evaluation run."""
    run_id: str
    timestamp: str
    dataset_path: str
    total_samples: int
    sample_results: list[SampleResult] = field(default_factory=list)
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    wall_clock_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "metrics": self.metrics.to_dict(),
            "sample_results": [r.to_dict() for r in self.sample_results],
        }

    def to_markdown(self) -> str:
        m = self.metrics
        lines = [
            "# Guardrail Agent Evaluation Report",
            "",
            f"**Run ID**: `{self.run_id}`  ",
            f"**Timestamp**: {self.timestamp}  ",
            f"**Dataset**: `{self.dataset_path}`  ",
            f"**Samples**: {self.total_samples} "
            f"(benign={m.benign_count}, injection={m.injection_count}, "
            f"pii_leakage={m.pii_leakage_count})  ",
            f"**Wall clock**: {self.wall_clock_s:.1f}s  ",
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| **Attack Detection Rate** (Recall) | {m.attack_detection_rate:.1%} |",
            f"| **False Positive Rate** | {m.false_positive_rate:.1%} |",
            f"| **Attack Precision** | {m.attack_cm.precision:.1%} |",
            f"| **Attack F1** | {m.attack_f1:.1%} |",
            f"| **PII Detection Rate** | {m.pii_detection_rate:.1%} |",
            f"| **Task Accuracy** (benign) | {m.task_accuracy:.1%} |",
            f"| **Overall Refusal Rate** | {m.refusal_rate:.1%} |",
            "",
            "## Confusion Matrix (Attack Detection)",
            "",
            "| | Predicted Threat | Predicted Safe |",
            "|---|---|---|",
            f"| **Actual Attack** | TP={m.attack_cm.TP} | FN={m.attack_cm.FN} |",
            f"| **Actual Benign** | FP={m.attack_cm.FP} | TN={m.attack_cm.TN} |",
            "",
            "## Outcome Distribution",
            "",
        ]
        for outcome, count in sorted(m.outcome_counts.items()):
            lines.append(f"- `{outcome}`: {count}")

        lines += [
            "",
            "## Sample-Level Results",
            "",
            "| ID | Category | Expected | Actual | Correct |",
            "|---|---|---|---|---|",
        ]
        for r in self.sample_results:
            tick = "✅" if r.correct else "❌"
            lines.append(
                f"| {r.sample_id} | {r.category} "
                f"| {r.expected_outcome} | {r.actual_outcome} | {tick} |"
            )

        return "\n".join(lines)


def run_evaluation(
    agent: GuardrailAgent,
    dataset_path: str | Path,
    run_id: str | None = None,
    output_path: str | Path | None = None,
) -> EvalRunResult:
    """
    Run the full evaluation benchmark.

    Parameters
    ----------
    agent:
        GuardrailAgent instance to evaluate.
    dataset_path:
        Path to the labeled JSON evaluation dataset.
    run_id:
        Optional identifier for this run. Auto-generated if None.
    output_path:
        If provided, write JSON results here.

    Returns
    -------
    EvalRunResult
    """
    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = run_id or f"eval_{int(time.time())}"

    samples: list[DatasetSample] = load_dataset(dataset_path)
    sample_results: list[SampleResult] = []
    t_start = time.time()

    for sample in samples:
        t0 = time.time()
        agent_result = agent.process(sample.text)
        latency = round(time.time() - t0, 3)

        actual_outcome = agent_result.outcome.value
        correct = (actual_outcome == sample.expected_outcome)

        sample_results.append(SampleResult(
            sample_id=sample.id,
            category=sample.category,
            text=sample.text,
            expected_outcome=sample.expected_outcome,
            actual_outcome=actual_outcome,
            reply_preview=agent_result.reply[:150],
            canary_leaked=agent_result.canary_leaked,
            injection_detected=agent_result.injection_detected,
            pii_in_input=agent_result.pii_in_input,
            pii_in_output=agent_result.pii_in_output,
            correct=correct,
            latency_s=latency,
            notes=sample.notes,
        ))

    wall = round(time.time() - t_start, 2)

    # Build records for metrics
    metric_records = [
        {
            "category": r.category,
            "actual_outcome": r.actual_outcome,
            "expected_outcome": r.expected_outcome,
        }
        for r in sample_results
    ]
    metrics = compute_metrics(metric_records)

    run_result = EvalRunResult(
        run_id=run_id,
        timestamp=timestamp,
        dataset_path=str(dataset_path),
        total_samples=len(samples),
        sample_results=sample_results,
        metrics=metrics,
        wall_clock_s=wall,
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(run_result.to_dict(), f, indent=2)

    return run_result
