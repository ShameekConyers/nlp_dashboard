"""Inter-rater agreement: Cohen's kappa and Gwet's AC1 with bootstrap intervals.

Both statistics compare two raters who labelled the same items with the
same categories. Kappa corrects observed agreement for the agreement
expected from each rater's own label shares. AC1 (Gwet 2008) uses the
average share per category instead, which keeps it stable when one
category dominates, the case where kappa can fall near zero despite high
raw agreement. Intervals are percentile bootstraps over items.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

N_BOOT: int = 2000
"""Bootstrap resamples for agreement intervals."""

BOOT_SEED: int = 42
"""Seed for agreement bootstraps."""


def _codes(a: Sequence[str], b: Sequence[str], categories: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map two label sequences to integer codes over ``categories``.

    Args:
        a: First rater's labels.
        b: Second rater's labels.
        categories: Allowed labels.

    Returns:
        ``(codes_a, codes_b)``.

    Raises:
        ValueError: If lengths differ, no items are given, or a label is not a category.
    """
    if len(a) != len(b):
        raise ValueError(f"Raters labelled {len(a)} and {len(b)} items.")
    if not a:
        raise ValueError("No items.")
    index = {c: i for i, c in enumerate(categories)}
    unknown = sorted({x for x in (*a, *b) if x not in index})
    if unknown:
        raise ValueError(f"Labels outside the categories: {unknown}")
    return np.array([index[x] for x in a]), np.array([index[x] for x in b])


def _kappa_codes(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Cohen's kappa from integer codes.

    Args:
        x: First rater's codes.
        y: Second rater's codes.
        k: Number of categories.

    Returns:
        Kappa, or nan when expected agreement is 1.
    """
    po = float((x == y).mean())
    px = np.bincount(x, minlength=k) / x.size
    py = np.bincount(y, minlength=k) / y.size
    pe = float(px @ py)
    return math.nan if pe >= 1.0 else (po - pe) / (1.0 - pe)


def _ac1_codes(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Gwet's AC1 from integer codes.

    Args:
        x: First rater's codes.
        y: Second rater's codes.
        k: Number of categories (at least 2).

    Returns:
        AC1, or nan when chance agreement is 1.
    """
    po = float((x == y).mean())
    pi = (np.bincount(x, minlength=k) + np.bincount(y, minlength=k)) / (2.0 * x.size)
    pe = float((pi * (1.0 - pi)).sum() / (k - 1))
    return math.nan if pe >= 1.0 else (po - pe) / (1.0 - pe)


def cohen_kappa(a: Sequence[str], b: Sequence[str], categories: Sequence[str]) -> float:
    """Cohen's kappa for two raters.

    Args:
        a: First rater's labels.
        b: Second rater's labels.
        categories: Allowed labels.

    Returns:
        Kappa.
    """
    x, y = _codes(a, b, categories)
    return _kappa_codes(x, y, len(categories))


def gwet_ac1(a: Sequence[str], b: Sequence[str], categories: Sequence[str]) -> float:
    """Gwet's AC1 for two raters.

    Args:
        a: First rater's labels.
        b: Second rater's labels.
        categories: Allowed labels (at least two).

    Returns:
        AC1.

    Raises:
        ValueError: If fewer than two categories are given.
    """
    if len(categories) < 2:
        raise ValueError("AC1 needs at least two categories.")
    x, y = _codes(a, b, categories)
    return _ac1_codes(x, y, len(categories))


def bootstrap_interval(
    stat: Callable[[np.ndarray], float], n_items: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float]:
    """Percentile bootstrap interval of a statistic over items.

    Args:
        stat: Maps an array of item indices to the statistic.
        n_items: Number of items.
        n_boot: Resamples.
        seed: RNG seed.

    Returns:
        ``(low, high)`` 2.5th and 97.5th percentiles over finite resamples.
    """
    rng = np.random.default_rng(seed)
    values = np.array([stat(rng.integers(0, n_items, size=n_items)) for _ in range(n_boot)])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def agreement_report(
    a: Sequence[str],
    b: Sequence[str],
    categories: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Observed agreement, kappa and AC1 with bootstrap intervals.

    Args:
        a: First rater's labels.
        b: Second rater's labels.
        categories: Allowed labels.
        n_boot: Resamples.
        seed: RNG seed (the same item resamples serve both statistics).

    Returns:
        ``{n, percent_agreement, kappa, kappa_ci, ac1, ac1_ci, confusion}``;
        ``confusion[x][y]`` counts items the first rater labelled x and the
        second y.
    """
    x, y = _codes(a, b, categories)
    k = len(categories)
    kappa_ci = bootstrap_interval(lambda i: _kappa_codes(x[i], y[i], k), x.size, n_boot, seed)
    ac1_ci = bootstrap_interval(lambda i: _ac1_codes(x[i], y[i], k), x.size, n_boot, seed)
    return {
        "n": int(x.size),
        "percent_agreement": float((x == y).mean()),
        "kappa": _kappa_codes(x, y, k),
        "kappa_ci": list(kappa_ci),
        "ac1": _ac1_codes(x, y, k),
        "ac1_ci": list(ac1_ci),
        "confusion": {
            ca: {cb: int(((x == i) & (y == j)).sum()) for j, cb in enumerate(categories)}
            for i, ca in enumerate(categories)
        },
    }
