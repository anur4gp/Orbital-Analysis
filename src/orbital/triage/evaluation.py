"""Triage evaluation at full recall.

Reports how much of the catalog can be discarded while keeping every positive;
accuracy and AUC are uninformative at ~0.1% positives.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class TriageResult:
    """Screening performance at full recall of the positive class."""

    name: str
    kept_fraction: float
    recall: float
    precision: float
    n_kept: int
    n_positive: int

    @property
    def reduction(self) -> float:
        """Fraction of the catalog eliminated from expensive analysis."""
        return 1.0 - self.kept_fraction


def full_recall_operating_point(scores: np.ndarray, labels: np.ndarray,
                                name: str) -> TriageResult:
    """Threshold at the lowest-scoring positive (higher score = riskier)."""
    positives = labels.astype(bool)
    n_pos = int(positives.sum())
    if n_pos == 0:
        return TriageResult(name, 1.0, float("nan"), float("nan"), len(labels), 0)

    cutoff = scores[positives].min()
    kept = scores >= cutoff
    return TriageResult(
        name=name,
        kept_fraction=float(kept.mean()),
        recall=float((kept & positives).sum() / n_pos),
        precision=float((kept & positives).sum() / max(kept.sum(), 1)),
        n_kept=int(kept.sum()),
        n_positive=n_pos,
    )


def grouped_split(
    groups: np.ndarray, test_groups: Iterable[object]
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test masks holding out ``test_groups``; random splits leak shared pairs."""
    test = np.isin(groups, np.asarray(list(test_groups), dtype=groups.dtype))
    return ~test, test
