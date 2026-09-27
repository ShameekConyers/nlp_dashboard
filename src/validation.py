"""Classification metrics and paired bootstrap for sentiment scorer validation.

Pure functions only. The Decision 11 runner in ``src/sentiment_transformer.py``
feeds them the recommend flag as ground truth and each scorer's binary
prediction and continuous score.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata

from src.stats import balanced_accuracy, wilson_interval

METRIC_NAMES: tuple[str, ...] = (
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "auc",
    "false_positive_rate",
)
"""Metrics reported per scorer. ``false_positive_rate`` is the share of
not-recommended reviews scored positive (the strategy misread)."""

VADER_THRESHOLD_GRID: tuple[float, ...] = tuple(round(-0.95 + 0.05 * i, 2) for i in range(39))
"""Compound thresholds searched on the dev split: -0.95 to 0.95 in steps of 0.05."""


@dataclass(frozen=True)
class Predictor:
    """One scorer's output on a set of reviews.

    Attributes:
        pred: Binary prediction, True for positive.
        score: Continuous score where higher means more positive (for AUC).
    """

    pred: np.ndarray
    score: np.ndarray


def macro_f1(y_true: Sequence[bool] | np.ndarray, y_pred: Sequence[bool] | np.ndarray) -> float:
    """Unweighted mean of the positive-class and negative-class F1.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Macro-F1; a class with no true or predicted members scores F1 = 0.
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    scores = []
    for cls_t, cls_p in ((t, p), (~t, ~p)):
        tp = float((cls_t & cls_p).sum())
        denom = float(cls_t.sum() + cls_p.sum())
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def macro_f1_multiclass(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> float:
    """Unweighted mean F1 over the given classes.

    Args:
        y_true: Ground-truth class names.
        y_pred: Predicted class names.
        classes: Classes to average over.

    Returns:
        Macro-F1; a class with no true or predicted members scores F1 = 0.
    """
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    scores = []
    for c in classes:
        tp = float(((t == c) & (p == c)).sum())
        denom = float((t == c).sum() + (p == c).sum())
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def roc_auc(y_true: Sequence[bool] | np.ndarray, score: Sequence[float] | np.ndarray) -> float:
    """ROC AUC by the rank-sum formula, ties at average rank.

    Args:
        y_true: Ground-truth labels.
        score: Continuous scores, higher meaning more likely positive.

    Returns:
        AUC, or nan if either class is absent.
    """
    t = np.asarray(y_true, dtype=bool)
    n_pos = int(t.sum())
    n_neg = int(t.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = rankdata(np.asarray(score, dtype=float))
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y: np.ndarray, p: Predictor) -> dict[str, float]:
    """Compute every metric in ``METRIC_NAMES`` for one scorer.

    Args:
        y: Ground-truth labels (recommended).
        p: The scorer's predictions and scores.

    Returns:
        ``{metric: value}``.
    """
    not_rec = ~y
    return {
        "balanced_accuracy": balanced_accuracy(y, p.pred),
        "accuracy": float((y == p.pred).mean()) if y.size else math.nan,
        "macro_f1": macro_f1(y, p.pred),
        "auc": roc_auc(y, p.score),
        "false_positive_rate": float(p.pred[not_rec].mean()) if not_rec.any() else math.nan,
    }


def tune_threshold(
    y: np.ndarray, score: np.ndarray, grid: Sequence[float] = VADER_THRESHOLD_GRID,
    default: float = 0.05,
) -> tuple[float, float]:
    """Pick the score threshold with the highest balanced accuracy.

    Ties go to the threshold closest to ``default``.

    Args:
        y: Ground-truth labels.
        score: Continuous scores; ``score >= threshold`` predicts positive.
        grid: Thresholds to try.
        default: Tie-break anchor.

    Returns:
        ``(threshold, balanced_accuracy)``.
    """
    best_t, best_ba = default, -1.0
    for t in grid:
        ba = balanced_accuracy(y, score >= t)
        if ba > best_ba + 1e-12 or (
            abs(ba - best_ba) <= 1e-12 and abs(t - default) < abs(best_t - default)
        ):
            best_t, best_ba = t, ba
    return float(best_t), float(best_ba)


def _ci(samples: Sequence[float]) -> list[float | None]:
    """Return the 2.5th and 97.5th percentiles, ignoring nan.

    Args:
        samples: Bootstrap replicates.

    Returns:
        ``[low, high]``, or ``[None, None]`` if every replicate is nan.
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [None, None]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def bootstrap_validation(
    y: np.ndarray,
    predictors: Mapping[str, Predictor],
    pairs: Sequence[tuple[str, str]],
    n_boot: int,
    seed: int,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """Point estimates and paired percentile intervals for several scorers.

    Every replicate resamples the same review indices for all scorers, so the
    interval of a difference is paired. With ``clusters`` the resample is two
    stage: clusters (games) with replacement, then reviews within each drawn
    cluster.

    Args:
        y: Ground-truth labels.
        predictors: ``{name: Predictor}`` aligned with ``y``.
        pairs: ``(a, b)`` names whose balanced-accuracy difference ``a - b``
            is reported.
        n_boot: Resamples.
        seed: Random seed.
        clusters: Optional cluster id per review.

    Returns:
        ``{"n", "n_not_recommended", "scorers": {name: {metric: {value, ci}}},
        "differences": {"a_minus_b": {value, ci}}}``. The false-positive rate
        also carries a Wilson interval.
    """
    rng = np.random.default_rng(seed)
    point = {name: classification_metrics(y, p) for name, p in predictors.items()}
    reps: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in METRIC_NAMES} for name in predictors
    }
    diff_reps: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
    groups = None
    if clusters is not None:
        uniq = np.unique(clusters)
        groups = [np.flatnonzero(clusters == c) for c in uniq]
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, y.size, size=y.size)
        else:
            picks = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate(
                [groups[g][rng.integers(0, groups[g].size, size=groups[g].size)] for g in picks]
            )
        yb = y[idx]
        ba_b: dict[str, float] = {}
        for name, p in predictors.items():
            m = classification_metrics(yb, Predictor(p.pred[idx], p.score[idx]))
            for k, v in m.items():
                reps[name][k].append(v)
            ba_b[name] = m["balanced_accuracy"]
        for a, b in pairs:
            diff_reps[(a, b)].append(ba_b[a] - ba_b[b])
    not_rec = ~y
    scorers: dict[str, Any] = {}
    for name, p in predictors.items():
        entry = {m: {"value": point[name][m], "ci": _ci(reps[name][m])} for m in METRIC_NAMES}
        k = int(p.pred[not_rec].sum())
        entry["false_positive_rate"]["wilson_ci"] = list(wilson_interval(k, int(not_rec.sum())))
        entry["false_positive_rate"]["k"] = k
        scorers[name] = entry
    differences = {
        f"{a}_minus_{b}": {
            "value": point[a]["balanced_accuracy"] - point[b]["balanced_accuracy"],
            "ci": _ci(diff_reps[(a, b)]),
        }
        for a, b in pairs
    }
    return {
        "n": int(y.size),
        "n_not_recommended": int(not_rec.sum()),
        "scorers": scorers,
        "differences": differences,
    }
