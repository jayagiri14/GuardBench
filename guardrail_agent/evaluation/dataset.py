"""
Dataset loader for the evaluation benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


VALID_CATEGORIES = {"benign", "injection", "pii_leakage"}
VALID_OUTCOMES = {
    "ANSWERED", "REDACTED_INPUT", "REDACTED_OUTPUT",
    "BLOCKED_INPUT", "BLOCKED_OUTPUT",
}


@dataclass
class DatasetSample:
    id: str
    category: str
    text: str
    expected_outcome: str
    notes: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def load_dataset(path: str | Path) -> list[DatasetSample]:
    """
    Load and validate the evaluation dataset from a JSON file.

    Parameters
    ----------
    path:
        Path to the dataset JSON file.

    Returns
    -------
    list of DatasetSample

    Raises
    ------
    ValueError
        If any sample has an invalid category or expected_outcome.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(raw).__name__}")

    samples = []
    for i, item in enumerate(raw):
        _id = item.get("id", f"sample_{i}")
        category = item.get("category", "")
        text = item.get("text", "")
        expected = item.get("expected_outcome", "")

        if category not in VALID_CATEGORIES:
            raise ValueError(
                f"Sample {_id}: invalid category {category!r}. "
                f"Must be one of {VALID_CATEGORIES}"
            )
        if expected not in VALID_OUTCOMES:
            raise ValueError(
                f"Sample {_id}: invalid expected_outcome {expected!r}. "
                f"Must be one of {VALID_OUTCOMES}"
            )
        if not text.strip():
            raise ValueError(f"Sample {_id}: text must not be empty")

        meta = {k: v for k, v in item.items()
                if k not in {"id", "category", "text", "expected_outcome", "notes"}}

        samples.append(DatasetSample(
            id=_id,
            category=category,
            text=text,
            expected_outcome=expected,
            notes=item.get("notes", ""),
            metadata=meta,
        ))

    return samples
