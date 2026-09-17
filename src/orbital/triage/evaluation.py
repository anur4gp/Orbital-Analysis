"""Phase 4: risk-triage classifier.

Decides which conjunctions deserve expensive treatment, using only features
available BEFORE any Pc computation. The operational asymmetry dominates the
design: missing a genuinely high-risk conjunction is unacceptable, while
passing a harmless one through costs only compute. So the model is evaluated
at **full recall** -- how much of the catalog can be discarded while still
catching every positive -- rather than by accuracy or even AUC, both of which
are close to meaningless at 0.1% positive rate.

The baseline to beat is not "random". It is a **miss-distance cut**, which is
what an analyst would do without any model. A classifier that cannot beat one
threshold on one feature is not worth having, and this module reports that
comparison directly.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class TriageResult:
    """Screening performance at full recall of the positive class."""

    name: str
    kept_fraction: float     # fraction of conjunctions passed on for full analysis
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
    """Smallest keep-set that still contains every positive.

    The threshold is placed just below the lowest-scoring true positive,
    which is the best any threshold on this score can do at 100% recall.

    Parameters
    ----------
    scores
        Risk scores, higher meaning riskier, shape (n,).
    labels
        1 for a positive (high-risk) conjunction, 0 otherwise, shape (n,).
    name
        Label for the method, carried into the result.

    Returns
    -------
    TriageResult
        Kept fraction, recall and precision at that threshold. With no
        positives, everything is kept and recall is NaN.
    """
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
    """Split by catalog/day group rather than at random.

    Conjunctions from the same debris family within the same window share
    objects and geometry, so a random split leaks: the same pair can appear
    in both halves at slightly different times. Splitting by group is the
    honest test of whether the model generalizes.

    Parameters
    ----------
    groups
        Group label per row, shape (n,).
    test_groups
        Labels held out for testing.

    Returns
    -------
    train, test : numpy.ndarray
        Complementary boolean masks, shape (n,).
    """
    test = np.isin(groups, np.asarray(list(test_groups), dtype=groups.dtype))
    return ~test, test
