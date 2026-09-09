"""Phase 4: train and evaluate the risk-triage classifier.

The operating question is not "how accurate is the model" -- at a 0.1%
positive rate a constant "no" scores 99.9%. It is: **how much of the catalog
can be discarded while still catching every high-risk conjunction?**

Thresholds are chosen from OUT-OF-FOLD predictions on the training set, then
applied unchanged to held-out data. Two traps this avoids:

  * Choosing the threshold from test labels reports the oracle, not a
    deployable system. The oracle is shown separately, as an upper bound.
  * Choosing it from in-sample training scores is worse than useless for a
    model that can fit its training set. Gradient boosting scores every
    training positive highly, so the implied threshold does not transfer at
    all, and the operating point collapses to "keep everything".

Thresholds are quoted at a target recall slightly below 1.0 as well, because
requiring literally every training positive makes the operating point
hostage to a single hardest example.

Run: ./venv/bin/python src/run_triage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import FEATURES

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "triage_dataset.csv"
THRESHOLD = -10.0
TEST_DAYS = (5, 6)          # zero-indexed: last two days of each window
SEED = 7


def load():
    raw = np.genfromtxt(DATA, delimiter=",", names=True, dtype=None, encoding="utf-8")
    x = np.column_stack([raw[f] for f in FEATURES])
    y = raw["log10_pc"]
    day = raw["day"].astype(int)
    group = raw["group"].astype(str)
    norad = np.column_stack([raw["norad_i"].astype(int), raw["norad_j"].astype(int)])
    return x, y, day, group, norad


def out_of_fold_scores(model, x, y, n_splits: int = 5) -> np.ndarray:
    """Cross-validated scores, so the threshold is picked on unseen data."""
    scores = np.empty(len(y))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for train_idx, val_idx in folds.split(x, y):
        m = clone(model)
        m.fit(x[train_idx], y[train_idx])
        scores[val_idx] = m.decision_function(x[val_idx])
    return scores


def threshold_at_recall(scores: np.ndarray, y: np.ndarray, recall: float) -> float:
    """Lowest threshold retaining `recall` of the positives."""
    pos = np.sort(scores[y.astype(bool)])
    if pos.size == 0:
        return -np.inf
    # Keep the top `recall` fraction of positives.
    idx = int(np.floor((1.0 - recall) * pos.size))
    return float(pos[min(idx, pos.size - 1)])


def evaluate(name, model, xtr, ytr, xte, yte, target_recall: float):
    """Fit, pick a threshold out-of-fold, then measure on held-out data."""
    if model is None:                     # raw-feature baseline
        oof, s_te = xtr, xte
    else:
        oof = out_of_fold_scores(model, xtr, ytr)
        model.fit(xtr, ytr)
        s_te = model.decision_function(xte)

    cut = threshold_at_recall(oof, ytr, target_recall)
    kept = s_te >= cut
    pos_te = yte.astype(bool)
    oracle_cut = s_te[pos_te].min() if pos_te.any() else -np.inf
    return {
        "name": name,
        "kept": float(kept.mean()),
        "recall": float((kept & pos_te).sum() / max(pos_te.sum(), 1)),
        "missed": int((~kept & pos_te).sum()),
        "oracle_kept": float((s_te >= oracle_cut).mean()),
    }


def main() -> int:
    x, y_raw, day, group, norad = load()
    y = (y_raw > THRESHOLD).astype(int)
    print(f"{len(y):,} conjunctions, {y.sum()} positives "
          f"({100*y.mean():.3f}%) at log10(Pc) > {THRESHOLD:g}\n")

    for split_name, test_mask in (
        ("day split (train days 1-5, test days 6-7)", np.isin(day, TEST_DAYS)),
        ("object split (test pairs share no object with train)", None),
    ):
        if test_mask is None:
            rng = np.random.default_rng(SEED)
            objects = np.unique(norad)
            held = set(rng.choice(objects, size=int(0.35 * len(objects)), replace=False).tolist())
            in_held = np.array([[a in held, b in held] for a, b in norad])
            test_mask = in_held.all(axis=1)
            train_mask = ~in_held.any(axis=1)   # drop cross pairs entirely
        else:
            train_mask = ~test_mask

        xtr, ytr = x[train_mask], y[train_mask]
        xte, yte = x[test_mask], y[test_mask]
        print(f"--- {split_name}")
        print(f"    train {len(ytr):,} ({ytr.sum()} pos)   "
              f"test {len(yte):,} ({yte.sum()} pos)")
        if ytr.sum() == 0 or yte.sum() == 0:
            print("    insufficient positives, skipped\n")
            continue

        miss_i = FEATURES.index("miss_km")
        scaled_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"))
        gb = HistGradientBoostingClassifier(max_iter=300, random_state=SEED,
                                            class_weight="balanced")

        for target in (1.00, 0.99):
            print(f"    threshold set for {target:.0%} out-of-fold recall on train:")
            print(f"    {'model':>30}  {'kept':>7}  {'recall':>7}  {'missed':>7}  {'oracle':>7}")
            rows = [
                evaluate("miss-distance cut (baseline)", None,
                         -xtr[:, miss_i], ytr, -xte[:, miss_i], yte, target),
                evaluate("logistic regression", scaled_lr, xtr, ytr, xte, yte, target),
                evaluate("gradient boosting", gb, xtr, ytr, xte, yte, target),
            ]
            for r in rows:
                print(f"    {r['name']:>30}  {r['kept']:>6.2%}  {r['recall']:>6.1%}  "
                      f"{r['missed']:>7}  {r['oracle_kept']:>6.2%}")
            print()

    print("kept   = fraction of conjunctions passed on for expensive analysis")
    print("recall = fraction of true high-risk events retained (must be 100%)")
    print("oracle = best possible at full test recall, i.e. an upper bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
