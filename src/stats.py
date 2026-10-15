"""Statistics for a natural-sampling study: rates, intervals, themes, findings.

Reads a study seed database after preprocessing, VADER, BERTopic and aspect
tagging, and writes one ``analysis_runs`` row plus ``game_stats``,
``cohort_stats`` and ``theme_stats`` rows keyed by that run.

Themes come from two lenses. ``topic`` (primary) is the BERTopic assignment
of each kept English review. ``aspect`` (supplementary) is the phrase
lexicon; an aspect supports claims only after its precision audit passes.
For every theme and scope the run records the theme share, the complaint
share (share of not-recommended reviews in the theme), the praise share
(same for recommended reviews), their ratio (complaint lift), and VADER
sentiment of the theme's reviews.

Uncertainty: Wilson intervals for per-game rates, review-level bootstrap for
per-game theme metrics, and a two-stage cluster bootstrap (games, then
reviews within games) for cohort macro averages. ``FINDING_RULE`` was fixed
before results were read, with two post-hoc changes made after the first
market run and disclosed in the README: the minimum-support condition
(``MIN_SIDE_MENTIONS``) and the correction of the leave-one-game-out rank
check to rank within each lens among precision-passing themes, as the plan
specifies.
"""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.study import UNFREEZE_FLAG, UNFREEZE_HELP, Study, guard_writable, load_study
from src.utils import (
    MIN_TOKEN_COUNT,
    PROJECT_ROOT,
    TRANSFORMER_POSITIVE_CUTOFF,
    VADER_POSITIVE_THRESHOLD,
    get_logger,
)

logger = get_logger(__name__)

N_BOOT: int = 2000
"""Bootstrap resamples for every interval."""

BOOT_SEED: int = 42
"""Seed for every bootstrap."""

MIN_KEPT_FOR_ELIGIBLE: int = 100
"""A game needs this many kept English reviews in the seed to be eligible."""

MIN_NOT_REC_FOR_ELIGIBLE: int = 30
"""A game also needs this many not-recommended kept English reviews."""

MIN_GAMES_FOR_FINDING: int = 3
"""A cohort needs this many eligible games before any finding."""

MIN_AUDIT_N: int = 30
"""Minimum labelled sentences for an aspect's precision to count."""

MIN_PRECISION: float = 0.80
"""Minimum audited precision for an aspect to support claims."""

MIN_SIDE_MENTIONS: int = 5
"""Minimum theme mentions among not-recommended and among recommended reviews.

Added after the first market run (2026-09-26): a theme with zero mentions on
one side has a lift of 0 or infinity whose percentile bootstrap interval
collapses to a point, which the lift rule then read as a finding. Below this
support the theme stays descriptive.
"""

FINDING_RULE: str = (
    "A theme is a cohort finding only if (1) for aspects, its audited precision is at "
    f"least {MIN_PRECISION:.2f} on {MIN_AUDIT_N}+ test-split sentences; (2) the cohort "
    f"has {MIN_GAMES_FOR_FINDING}+ eligible games; (3) the 95% cluster-bootstrap interval "
    "of the complaint lift lies entirely above 1 (complaint finding) or below 1 (praise "
    "finding); (4) it is leave-one-game-out stable (the lift stays on the same side of 1 "
    "and the theme's complaint-share rank within its lens, among themes that passed the "
    "precision gate, moves by at most one place when any one game is dropped); (5) it has at least "
    f"{MIN_SIDE_MENTIONS} mentions among both recommended and not-recommended reviews in "
    "the scope's eligible games. A ranking finding (most-cited complaint theme in a lens) "
    "needs conditions (1), (2) and (4) and a complaint-share interval that does not overlap "
    "the second theme's. Praise rankings and the all-games scope are always descriptive. "
    "Everything else is descriptive."
)

SCOPE_GAME: str = "game"
SCOPE_COHORT: str = "cohort"
SCOPE_SUBGROUP: str = "subgroup"
SCOPE_SUBGROUP_MATCHED: str = "subgroup_matched"
SCOPE_COHORT_SUBGROUP: str = "cohort_subgroup"
"""One subgroup inside one cohort (scope id ``cohort/subgroup``). Emitted only
for a subgroup name that appears in more than one cohort, where the plain
subgroup scope pools cohorts."""
SCOPE_ALL: str = "all"


# ---------------------------------------------------------------------------
# Pure statistics
# ---------------------------------------------------------------------------


def dev_test_split(steam_review_id: str) -> str:
    """Assign a review to the dev or test split by hashing its Steam ID.

    Args:
        steam_review_id: Steam recommendation ID.

    Returns:
        ``"dev"`` if the SHA-256 of the ID is even, otherwise ``"test"``.
    """
    digest = int(hashlib.sha256(steam_review_id.encode("utf-8")).hexdigest(), 16)
    return "dev" if digest % 2 == 0 else "test"


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    Args:
        k: Successes.
        n: Trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        ``(low, high)``; ``(nan, nan)`` when ``n`` is 0.
    """
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def balanced_accuracy(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float:
    """Mean of the true-positive rate and the true-negative rate.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Balanced accuracy; a class absent from ``y_true`` is left out of the mean.
    """
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    rates = []
    if t.any():
        rates.append(float((p & t).sum() / t.sum()))
    if (~t).any():
        rates.append(float((~p & ~t).sum() / (~t).sum()))
    return float(np.mean(rates)) if rates else math.nan


@dataclass
class GameData:
    """Per-review arrays for one game.

    Attributes:
        app_id: Steam application ID.
        rec: Recommend flag per kept review (bool).
        compound: VADER compound per kept review.
        themes: ``(n_reviews, n_themes)`` boolean membership matrix.
        model_pos: 1.0 where the adopted transformer scores the review
            positive, else 0.0; None when no transformer was adopted.
    """

    app_id: int
    rec: np.ndarray
    compound: np.ndarray
    themes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    model_pos: np.ndarray | None = None


@dataclass
class ThemeMetrics:
    """Theme metrics for one scope (arrays over themes).

    Attributes:
        share: Share of reviews in each theme.
        complaint: Share of not-recommended reviews in each theme.
        praise: Share of recommended reviews in each theme.
        lift: ``complaint / praise``.
    """

    share: np.ndarray
    complaint: np.ndarray
    praise: np.ndarray
    lift: np.ndarray


def _weighted_game_metrics(g: GameData, w: np.ndarray) -> tuple[np.ndarray, ...]:
    """Theme metrics for one game under review weights (resample counts).

    Args:
        g: The game.
        w: Non-negative weight per review.

    Returns:
        ``(share, complaint, praise, rec_share)``; entries are nan where a
        denominator is zero.
    """
    m = g.themes.astype(float)
    rec = g.rec.astype(float)
    n = w.sum()
    n_rec = (w * rec).sum()
    n_not = (w * (1 - rec)).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        share = (w @ m) / n
        complaint = ((w * (1 - rec)) @ m) / n_not
        praise = ((w * rec) @ m) / n_rec
        rec_share = n_rec / n
    return share, complaint, praise, rec_share


def game_theme_metrics(g: GameData) -> ThemeMetrics:
    """Point estimates of theme metrics for one game.

    Args:
        g: The game.

    Returns:
        ThemeMetrics.
    """
    share, complaint, praise, _ = _weighted_game_metrics(g, np.ones(g.rec.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        lift = complaint / praise
    return ThemeMetrics(share, complaint, praise, lift)


def macro_theme_metrics(games: Sequence[GameData]) -> ThemeMetrics:
    """Macro (game-weighted) theme metrics; lift is the ratio of macro shares.

    Args:
        games: Games in the scope.

    Returns:
        ThemeMetrics.
    """
    per = [_weighted_game_metrics(g, np.ones(g.rec.size)) for g in games]
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        share = np.nanmean([p[0] for p in per], axis=0)
        complaint = np.nanmean([p[1] for p in per], axis=0)
        praise = np.nanmean([p[2] for p in per], axis=0)
        lift = complaint / praise
    return ThemeMetrics(share, complaint, praise, lift)


def _percentiles(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2.5th and 97.5th percentiles over axis 0, ignoring nan and inf.

    Args:
        samples: ``(n_boot, n_themes)`` array.

    Returns:
        ``(low, high)`` arrays.
    """
    clean = np.where(np.isfinite(samples), samples, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(clean, 2.5, axis=0), np.nanpercentile(clean, 97.5, axis=0)


def bootstrap_game_themes(
    g: GameData, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Review-level percentile bootstrap of one game's theme metrics.

    Args:
        g: The game.
        n_boot: Resamples.
        seed: RNG seed.

    Returns:
        ``{metric: (low, high)}`` for ``share``, ``complaint``, ``praise``, ``lift``.
    """
    rng = np.random.default_rng(seed)
    n = g.rec.size
    out: dict[str, list[np.ndarray]] = {"share": [], "complaint": [], "praise": [], "lift": []}
    weights = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
    for w in weights:
        share, complaint, praise, _ = _weighted_game_metrics(g, w.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            lift = complaint / praise
        out["share"].append(share)
        out["complaint"].append(complaint)
        out["praise"].append(praise)
        out["lift"].append(lift)
    return {k: _percentiles(np.array(v)) for k, v in out.items()}


def cluster_bootstrap_macro(
    games: Sequence[GameData], n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Two-stage cluster bootstrap of macro theme metrics and recommend share.

    Each resample draws games with replacement, then reviews with
    replacement within each drawn game, and recomputes the game-weighted
    (macro) metrics. Callers pass eligible games only.

    Args:
        games: Games in the scope.
        n_boot: Resamples.
        seed: RNG seed.

    Returns:
        ``{metric: (low, high)}`` for ``share``, ``complaint``, ``praise``,
        ``lift`` (arrays over themes), and ``rec_share``, ``mean_compound``
        and ``model_pos`` (macro recommend share, macro mean VADER compound and
        macro transformer-positive share, scalars in arrays).
    """
    rng = np.random.default_rng(seed)
    k = len(games)
    out: dict[str, list[np.ndarray]] = {
        "share": [], "complaint": [], "praise": [], "lift": [], "rec_share": [],
        "mean_compound": [], "model_pos": [],
    }
    for _ in range(n_boot):
        picks = rng.integers(0, k, size=k)
        per = []
        for gi in picks:
            g = games[gi]
            n = g.rec.size
            w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
            pos = float(w @ g.model_pos / w.sum()) if g.model_pos is not None else math.nan
            per.append((*_weighted_game_metrics(g, w), float(w @ g.compound / w.sum()), pos))
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            share = np.nanmean([p[0] for p in per], axis=0)
            complaint = np.nanmean([p[1] for p in per], axis=0)
            praise = np.nanmean([p[2] for p in per], axis=0)
            rec_share = np.nanmean([p[3] for p in per])
            mean_compound = np.nanmean([p[4] for p in per])
            model_pos = np.nanmean([p[5] for p in per])
            lift = complaint / praise
        out["share"].append(share)
        out["complaint"].append(complaint)
        out["praise"].append(praise)
        out["lift"].append(lift)
        out["rec_share"].append(np.array([rec_share]))
        out["mean_compound"].append(np.array([mean_compound]))
        out["model_pos"].append(np.array([model_pos]))
    return {key: _percentiles(np.array(v)) for key, v in out.items()}


def leave_one_game_out(
    games: Sequence[GameData],
    estimator: Callable[[Sequence[GameData]], ThemeMetrics] = macro_theme_metrics,
    groups: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Flag themes whose cohort estimate survives dropping any one game.

    A theme is stable if, for every left-out game, its lift stays on the same
    side of 1 as the full estimate and its complaint-share rank within its
    group moves by at most one place. Groups are the ranking pools: the
    runner passes one pool per lens holding only precision-passing themes,
    so aspects rank among passing aspects and topics among topics. Themes
    in no group are returned as not stable.

    Args:
        games: Eligible games in the cohort (at least two).
        estimator: Function returning ThemeMetrics for a set of games.
        groups: Column index pools for the rank check. None ranks all
            themes in one pool.

    Returns:
        Boolean array over themes.
    """
    full = estimator(games)
    n = full.lift.size
    pools = [list(g) for g in groups] if groups is not None else [list(range(n))]
    in_pool = np.zeros(n, dtype=bool)
    for pool in pools:
        in_pool[pool] = True

    def ranks(values: np.ndarray) -> np.ndarray:
        """Within-pool descending ranks; columns outside every pool get -1.

        Args:
            values: Complaint shares over all themes.

        Returns:
            Integer ranks.
        """
        out = np.full(n, -1)
        for pool in pools:
            if pool:
                out[pool] = _rank_desc(values[pool])
        return out

    full_side = full.lift > 1
    full_rank = ranks(full.complaint)
    stable = in_pool & np.isfinite(full.lift) & (full.lift != 1)
    for i in range(len(games)):
        sub = estimator([g for j, g in enumerate(games) if j != i])
        same_side = (sub.lift > 1) == full_side
        rank_ok = np.abs(ranks(sub.complaint) - full_rank) <= 1
        stable &= same_side & rank_ok & np.isfinite(sub.lift)
    return stable


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Rank values in descending order (0 = largest), nan last.

    Args:
        values: Values to rank.

    Returns:
        Integer ranks.
    """
    filled = np.where(np.isfinite(values), values, -np.inf)
    order = np.argsort(-filled, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    return ranks


def theme_sentiment(games: Sequence[GameData], j: int) -> tuple[float, float, float, float]:
    """Macro sentiment inside a theme: recommend share, VADER and transformer.

    Args:
        games: Games in the scope.
        j: Theme column.

    Returns:
        ``(rec_share_in_theme, mean_compound, vader_pos_share, model_pos_share)``;
        nan if no game has a review in the theme (``model_pos_share`` is also
        nan when no transformer was adopted).
    """
    recs, comps, poss, models = [], [], [], []
    for g in games:
        sel = g.themes[:, j]
        if sel.any():
            recs.append(g.rec[sel].mean())
            comps.append(g.compound[sel].mean())
            poss.append((g.compound[sel] >= VADER_POSITIVE_THRESHOLD).mean())
            if g.model_pos is not None:
                models.append(g.model_pos[sel].mean())
    if not recs:
        return math.nan, math.nan, math.nan, math.nan
    model = float(np.mean(models)) if models else math.nan
    return float(np.mean(recs)), float(np.mean(comps)), float(np.mean(poss)), model


def classify_finding(
    lift_low: float,
    lift_high: float,
    n_eligible: int,
    logo_stable: bool,
    precision_ok: bool,
    k_complaint: int = MIN_SIDE_MENTIONS,
    k_praise: int = MIN_SIDE_MENTIONS,
) -> str | None:
    """Apply ``FINDING_RULE`` to one theme in one cohort scope.

    Args:
        lift_low: Lower bound of the lift interval.
        lift_high: Upper bound of the lift interval.
        n_eligible: Eligible games in the scope.
        logo_stable: Leave-one-game-out stability.
        precision_ok: Whether the theme passed its precision audit (always
            True for topics).
        k_complaint: Theme mentions among not-recommended reviews.
        k_praise: Theme mentions among recommended reviews.

    Returns:
        ``"complaint"``, ``"praise"`` or None (descriptive).
    """
    if not precision_ok or n_eligible < MIN_GAMES_FOR_FINDING or not logo_stable:
        return None
    if k_complaint < MIN_SIDE_MENTIONS or k_praise < MIN_SIDE_MENTIONS:
        return None
    if math.isfinite(lift_low) and lift_low > 1:
        return "complaint"
    if math.isfinite(lift_high) and lift_high < 1:
        return "praise"
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class ReviewRow:
    """One kept English review with everything the stats need.

    Attributes:
        document_id: Row ID in documents.
        steam_review_id: Steam recommendation ID.
        app_id: Steam application ID.
        recommended: Recommend flag.
        compound: VADER compound score.
        hours: Playtime at review in hours (None if missing).
        topic_id: BERTopic topic (-1 outlier).
    """

    document_id: int
    steam_review_id: str
    app_id: int
    recommended: bool
    compound: float
    hours: float | None
    topic_id: int


_IN_SPLIT_WHERE: str = "WHERE d.id IN (SELECT document_id FROM temp.stats_split_ids)"
"""Filter to the documents ``_split_ids`` stored for the study's stats split."""


def load_reviews(conn: sqlite3.Connection, in_split_only: bool = False) -> list[ReviewRow]:
    """Load kept English reviews with sentiment and topic.

    Args:
        conn: Study seed connection.
        in_split_only: Load only documents in ``temp.stats_split_ids`` (filled
            by ``_split_ids``), so rows of the other split are never read.

    Returns:
        Rows ordered by document id.
    """
    rows = conn.execute(
        """
        SELECT d.id, d.steam_review_id, d.app_id, d.recommended, n.sentiment_compound,
               d.playtime_at_review, COALESCE(n.topic_id, -1)
        FROM documents d
        JOIN processed_documents p ON p.document_id = d.id
        JOIN nlp_results n ON n.document_id = d.id
        WHERE p.token_count >= ?
        """ + ("AND d.id IN (SELECT document_id FROM temp.stats_split_ids)" if in_split_only else "")
        + """
        ORDER BY d.id
        """,
        (MIN_TOKEN_COUNT,),
    ).fetchall()
    return [
        ReviewRow(r[0], r[1], r[2], bool(r[3]), float(r[4]),
                  None if r[5] is None else r[5] / 60.0, int(r[6]))
        for r in rows
    ]


def load_aspect_sets(conn: sqlite3.Connection, version: str) -> dict[int, set[str]]:
    """Map document id to the set of aspects tagged in it.

    Args:
        conn: Study seed connection.
        version: Lexicon version.

    Returns:
        ``{document_id: {aspect, ...}}``.
    """
    out: dict[int, set[str]] = {}
    for doc_id, aspect in conn.execute(
        "SELECT document_id, aspect FROM aspect_mentions WHERE lexicon_version = ?", (version,)
    ):
        out.setdefault(doc_id, set()).add(aspect)
    return out


def load_audit_precision(label_path: Path) -> dict[str, dict[str, Any]]:
    """Compute per-aspect precision from the committed audit labels.

    Args:
        label_path: ``data/labels/aspect_audit_<version>.csv``.

    Returns:
        ``{aspect: {n, relevant, precision, ci_low, ci_high, passes}}``;
        empty if the file does not exist yet.
    """
    if not label_path.exists():
        return {}
    counts: dict[str, list[int]] = {}
    with open(label_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            c = counts.setdefault(row["aspect"], [0, 0])
            c[1] += 1
            c[0] += row["label"] == "relevant"
    out: dict[str, dict[str, Any]] = {}
    for aspect, (rel, n) in counts.items():
        lo, hi = wilson_interval(rel, n)
        precision = rel / n if n else math.nan
        out[aspect] = {
            "n": n,
            "relevant": rel,
            "precision": precision,
            "ci_low": lo,
            "ci_high": hi,
            "passes": n >= MIN_AUDIT_N and precision >= MIN_PRECISION,
        }
    return out


def study_precision(study: Study, lexicon: Any, required: bool = False) -> dict[str, dict[str, Any]]:
    """Return each aspect's precision from the label file that governs it.

    A study with ``[aspect_labels]`` takes each aspect's precision from the
    audit labels of the lexicon version named for it there. After an audit
    retry, that is the retry's labels for the revised aspects and the earlier
    audit's labels for the unchanged ones. Rows of other aspects in a
    governing file are ignored, so a retried aspect's earlier labels never
    count. A study without ``[aspect_labels]`` reads only its lexicon's own
    label file, as before.

    Args:
        study: The study.
        lexicon: The study's ``aspects.Lexicon``.
        required: Every lexicon aspect must get a precision. Use it once the
            labels are expected to exist, for example before a power run on
            testable candidates or a confirmatory read.

    Returns:
        ``{aspect: {n, relevant, precision, ci_low, ci_high, passes,
        label_version}}`` in lexicon order. Without ``required``, aspects whose
        label file does not exist yet are left out.

    Raises:
        ValueError: If ``[aspect_labels]`` leaves a lexicon aspect without a
            governing version, names an aspect the lexicon does not define,
            or governs an aspect the lexicon revised by another version than
            the lexicon's own. With ``required``, also if a governed aspect has
            no rows in its label file.
        FileNotFoundError: With ``required``, if a governing label file is
            missing.
    """
    from src.aspects import audit_paths, label_path

    names = [a.name for a in lexicon.aspects]
    if not study.aspect_labels:
        path = audit_paths(study, lexicon)[1]
        if required and not path.exists():
            raise FileNotFoundError(f"Label file {path} is missing.")
        legacy = load_audit_precision(path)
        unlabeled = [a for a in names if a not in legacy]
        if required and unlabeled:
            raise ValueError(f"Study {study.name}: aspects {unlabeled} have no rows in {path}.")
        return legacy
    governs = study.label_versions()
    ungoverned = [a for a in names if a not in governs]
    if ungoverned:
        raise ValueError(f"Study {study.name}: aspects {ungoverned} have no governing label "
                         "file in [aspect_labels].")
    unknown = sorted(set(governs) - set(names))
    if unknown:
        raise ValueError(f"Study {study.name}: [aspect_labels] names aspects {unknown} that "
                         f"lexicon {lexicon.version} does not define.")
    stale = [a for a in lexicon.revised if governs[a] != lexicon.version]
    if stale:
        raise ValueError(f"Study {study.name}: aspects {stale} were revised in "
                         f"{lexicon.version} but [aspect_labels] governs them by another "
                         "version's labels.")
    found: dict[str, dict[str, Any]] = {}
    for version in dict.fromkeys(governs[a] for a in names):
        path = label_path(version)
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Label file {path} is missing (it governs "
                                        f"{[a for a in names if governs[a] == version]}).")
            continue
        for aspect, p in load_audit_precision(path).items():
            if governs.get(aspect) == version:
                found[aspect] = {**p, "label_version": version}
    if required:
        unlabeled = [a for a in names if a not in found]
        if unlabeled:
            raise ValueError(f"Study {study.name}: aspects {unlabeled} have no rows in their "
                             "governing label file.")
    return {a: found[a] for a in names if a in found}


def _git_sha() -> str:
    """Return the current commit SHA with a ``-dirty`` suffix if the tree changed.

    Returns:
        SHA string, or ``unknown`` outside a Git checkout.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _num(x: float) -> float | None:
    """Convert nan or inf to None for storage.

    Args:
        x: A float.

    Returns:
        The float, or None if not finite.
    """
    return float(x) if x is not None and math.isfinite(x) else None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def vader_validation(rows: Sequence[ReviewRow], split: str = "test") -> dict[str, Any]:
    """VADER against the recommend flag on one split of a natural sample.

    Args:
        rows: Kept English reviews.
        split: ``test`` (default) or ``dev``. A study whose statistics are
            restricted to the dev split validates on the dev split, so no
            recommend flag of a test review is read.

    Returns:
        Accuracy, balanced accuracy with a bootstrap interval, the strategy
        false-positive rate (not-recommended reviews scored positive) and n.
    """
    test = [r for r in rows if dev_test_split(r.steam_review_id) == split]
    y = np.array([r.recommended for r in test])
    pred = np.array([r.compound >= VADER_POSITIVE_THRESHOLD for r in test])
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, y.size, size=y.size)
        boots.append(balanced_accuracy(y[idx], pred[idx]))
    not_rec = ~y
    return {
        "split": split,
        "threshold": VADER_POSITIVE_THRESHOLD,
        "n": int(y.size),
        "n_not_recommended": int(not_rec.sum()),
        "accuracy": float((y == pred).mean()),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "balanced_accuracy_ci": [float(np.percentile(boots, 2.5)),
                                 float(np.percentile(boots, 97.5))],
        "false_positive_rate_not_recommended": float(pred[not_rec].mean()),
        "recommended_share": float(y.mean()),
    }


def load_adopted_model(study: Study) -> tuple[dict[str, str], dict[int, float]] | None:
    """Return the adopted transformer and its document scores, if one was adopted.

    Reads the newest ``sentiment_validation`` run in the study's scores
    database (Decision 11). When VADER was not beaten, returns None and the
    sentiment views stay on VADER alone.

    Args:
        study: The study.

    Returns:
        ``({"model_name", "revision"}, {document_id: positive_prob})`` or None.
    """
    if not study.scores_path.exists():
        return None
    conn = sqlite3.connect(study.scores_path)
    try:
        has = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sentiment_scores'"
        ).fetchone()
        row = conn.execute(
            "SELECT metrics FROM analysis_runs WHERE stage = 'sentiment_validation' "
            "ORDER BY run_id DESC LIMIT 1"
        ).fetchone() if has else None
        if row is None:
            return None
        v = json.loads(row[0])
        if not v.get("adopted"):
            return None
        model = v["models"][v["selected"]]
        probs = dict(conn.execute(
            """
            SELECT s.document_id, s.positive_prob
            FROM sentiment_scores s
            JOIN sentiment_models m ON m.model_id = s.model_id
            WHERE m.model_name = ? AND m.model_revision = ? AND s.unit = 'doc'
            """,
            (model["model_name"], model["revision"]),
        ).fetchall())
        return {"model_name": model["model_name"], "revision": model["revision"]}, probs
    finally:
        conn.close()


def _scopes(study: Study, eligible: set[int], kept_ok: set[int]) -> list[tuple[str, str, list[int], list[int]]]:
    """Enumerate cohort-level scopes.

    Args:
        study: The study.
        eligible: App IDs eligible for theme estimates.
        kept_ok: App IDs with enough kept reviews for recommend-share macros.

    Returns:
        ``(scope_type, scope_id, all_app_ids, eligible_app_ids)`` tuples. A
        subgroup name shared by several cohorts also yields one
        ``cohort_subgroup`` scope per cohort.
    """
    scopes: list[tuple[str, str, list[int], list[int]]] = []
    all_ids = [a.app_id for a in study.apps]
    scopes.append((SCOPE_ALL, "all", all_ids, [a for a in all_ids if a in eligible]))
    for cohort in dict.fromkeys(a.cohort for a in study.apps):
        ids = [a.app_id for a in study.apps if a.cohort == cohort]
        scopes.append((SCOPE_COHORT, cohort, ids, [a for a in ids if a in eligible]))
    for sub in dict.fromkeys(a.subgroup for a in study.apps if a.subgroup):
        ids = [a.app_id for a in study.apps if a.subgroup == sub]
        scopes.append((SCOPE_SUBGROUP, sub, ids, [a for a in ids if a in eligible]))
        matched = [a.app_id for a in study.apps if a.subgroup == sub and "matched_basket" in a.tags]
        scopes.append((SCOPE_SUBGROUP_MATCHED, sub, matched, [a for a in matched if a in eligible]))
    for cohort, sub in dict.fromkeys((a.cohort, a.subgroup) for a in study.apps if a.subgroup):
        if len({a.cohort for a in study.apps if a.subgroup == sub}) < 2:
            continue
        ids = [a.app_id for a in study.apps if a.cohort == cohort and a.subgroup == sub]
        scopes.append((SCOPE_COHORT_SUBGROUP, f"{cohort}/{sub}", ids, [a for a in ids if a in eligible]))
    return scopes


def _split_ids(conn: sqlite3.Connection, split: str | None) -> int | None:
    """Fill ``temp.stats_split_ids`` with the seed documents in one split.

    Args:
        conn: Study seed connection.
        split: ``dev``, ``test`` or None (no restriction, nothing written).

    Returns:
        Number of documents in the split, or None when ``split`` is None.
    """
    if split is None:
        return None
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS stats_split_ids (document_id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM temp.stats_split_ids")
    ids = [
        (doc_id,) for doc_id, rid in conn.execute("SELECT id, steam_review_id FROM documents")
        if dev_test_split(rid) == split
    ]
    conn.executemany("INSERT INTO temp.stats_split_ids VALUES (?)", ids)
    return len(ids)


def run(
    study: Study, n_boot: int = N_BOOT, unfreeze: bool = False, assume_precision: bool = False
) -> int:
    """Compute and store all statistics for a study seed.

    With ``study.stats_split`` set, every statistic (eligibility, rates,
    theme metrics, findings, the VADER validation and the seed and drop
    counts in ``game_stats``) uses only the seed reviews in that split, and
    no recommend flag of a review in the other split is read.

    Args:
        study: A natural-sampling study whose NLP stages have run.
        n_boot: Bootstrap resamples.
        unfreeze: Allow writing a frozen study's seed database.
        assume_precision: Treat every aspect as if it had passed its precision
            audit. Only for ``candidate_run``, which runs on a copy of the
            seed, so a stored ``stats`` run never rests on an assumed audit.

    Returns:
        The new ``analysis_runs.run_id``.

    Raises:
        FrozenStudyError: If a target path belongs to a frozen study and ``unfreeze`` is False.
    """
    guard_writable([study.seed_db], unfreeze)
    from src.aspects import ANY_TURN_STRUCTURE, TURN_STRUCTURE_ASPECTS, load_lexicon
    from src.db_setup import create_study_tables

    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    precision = study_precision(study, lexicon)
    if assume_precision:
        precision = {
            a.name: {"n": 0, "relevant": 0, "precision": None, "ci_low": None, "ci_high": None,
                     "passes": True, "assumed": True}
            for a in lexicon.aspects
        }

    conn = sqlite3.connect(study.seed_db)
    try:
        create_study_tables(conn)
        split = study.stats_split
        in_split = _split_ids(conn, split)
        rows = load_reviews(conn, in_split_only=split is not None)
        aspect_sets = load_aspect_sets(conn, lexicon.version)
        topics = [
            (tid, label) for tid, label in conn.execute(
                "SELECT topic_id, label FROM topics WHERE topic_id != -1 ORDER BY topic_id"
            )
        ]

        lexicon_names = [a.name for a in lexicon.aspects]
        # The derived turn-structure theme exists only for lexicons that define
        # every turn-structure aspect (the strategy lexicon), not the generic one.
        derived = [ANY_TURN_STRUCTURE] if set(TURN_STRUCTURE_ASPECTS) <= set(lexicon_names) else []
        aspect_names = lexicon_names + derived
        theme_keys: list[tuple[str, str]] = [("aspect", a) for a in aspect_names] + [
            ("topic", str(t)) for t, _ in topics
        ]
        topic_index = {t: i for i, (t, _) in enumerate(topics)}

        by_app: dict[int, list[ReviewRow]] = {}
        for r in rows:
            by_app.setdefault(r.app_id, []).append(r)
        adopted = load_adopted_model(study)
        model_prob = adopted[1] if adopted else {}

        def model_pos(rs: Sequence[ReviewRow]) -> np.ndarray | None:
            """Transformer-positive flags (as floats) for reviews, or None if none adopted."""
            if not adopted:
                return None
            return np.array(
                [model_prob[r.document_id] >= TRANSFORMER_POSITIVE_CUTOFF for r in rs], dtype=float
            )

        games: dict[int, GameData] = {}
        for app_id, rs in by_app.items():
            m = np.zeros((len(rs), len(theme_keys)), dtype=bool)
            for i, r in enumerate(rs):
                tags = aspect_sets.get(r.document_id, set())
                for j, name in enumerate(lexicon_names):
                    m[i, j] = name in tags
                if derived:
                    m[i, len(aspect_names) - 1] = bool(tags & set(TURN_STRUCTURE_ASPECTS))
                if r.topic_id in topic_index:
                    m[i, len(aspect_names) + topic_index[r.topic_id]] = True
            games[app_id] = GameData(
                app_id,
                np.array([r.recommended for r in rs]),
                np.array([r.compound for r in rs]),
                m,
                model_pos(rs),
            )

        # ---- per-game base stats ------------------------------------------
        pulls = {
            r[0]: r for r in conn.execute(
                """
                SELECT app_id, reviews_collected, total_positive, total_reviews,
                       baseline_total_positive, baseline_total_reviews,
                       effective_window_start, effective_window_end, cap_hit
                FROM pull_runs
                WHERE run_id IN (SELECT MAX(run_id) FROM pull_runs GROUP BY app_id)
                """
            )
        }
        drops = {
            r[0]: r[1:] for r in conn.execute(
                """
                SELECT d.app_id,
                       COUNT(*),
                       SUM(p.char_count < ?),
                       SUM(p.char_count >= ? AND p.token_count < ? AND l.detected_lang IS NULL),
                       SUM(p.token_count < ? AND l.detected_lang IS NOT NULL),
                       AVG(d.early_access)
                FROM documents d
                JOIN processed_documents p ON p.document_id = d.id
                LEFT JOIN review_language l ON l.document_id = d.id
                """ + (_IN_SPLIT_WHERE if split is not None else "") + """
                GROUP BY d.app_id
                """,
                (10, 10, MIN_TOKEN_COUNT, MIN_TOKEN_COUNT),
            )
        }

        seed_counts = {
            r[0]: (r[1], r[2]) for r in conn.execute(
                "SELECT app_id, COUNT(*), SUM(recommended) FROM documents d "
                + (_IN_SPLIT_WHERE if split is not None else "") + " GROUP BY app_id"
            )
        }
        eligible: set[int] = set()
        kept_ok: set[int] = set()
        game_rows: list[tuple] = []
        for app in study.apps:
            rs = by_app.get(app.app_id, [])
            n_kept = len(rs)
            n_rec = sum(r.recommended for r in rs)
            n_not = n_kept - n_rec
            if n_kept >= MIN_KEPT_FOR_ELIGIBLE:
                kept_ok.add(app.app_id)
                if n_not >= MIN_NOT_REC_FOR_ELIGIBLE:
                    eligible.add(app.app_id)
            lo, hi = wilson_interval(n_rec, n_kept)
            seed_n, seed_rec = seed_counts.get(app.app_id, (0, 0))
            seed_lo, seed_hi = wilson_interval(seed_rec or 0, seed_n)
            pull = pulls.get(app.app_id)
            drop = drops.get(app.app_id, (0, 0, 0, 0, None))
            hours_rec = [r.hours for r in rs if r.recommended and r.hours is not None]
            hours_not = [r.hours for r in rs if not r.recommended and r.hours is not None]
            mp = model_pos(rs)
            k_mp = int(mp.sum()) if mp is not None else 0
            mp_lo, mp_hi = wilson_interval(k_mp, n_kept) if mp is not None else (math.nan, math.nan)
            game_rows.append((
                app.app_id,
                pull[1] if pull else None,
                drop[0], drop[1] or 0, drop[2] or 0, drop[3] or 0,
                n_kept, n_rec, n_not,
                _num(seed_rec / seed_n) if seed_n else None, _num(seed_lo), _num(seed_hi),
                _num(n_rec / n_kept) if n_kept else None, _num(lo), _num(hi),
                _num(pull[2] / pull[3]) if pull and pull[3] else None,
                _num(pull[4] / pull[5]) if pull and pull[5] else None,
                pull[6] if pull else None, pull[7] if pull else None,
                bool(pull[8]) if pull else None,
                median(hours_rec) if hours_rec else None,
                median(hours_not) if hours_not else None,
                drop[4],
                _num(float(np.mean([r.compound for r in rs]))) if rs else None,
                _num(float(np.mean([r.compound >= VADER_POSITIVE_THRESHOLD for r in rs])))
                if rs else None,
                _num(k_mp / n_kept) if mp is not None and n_kept else None,
                _num(mp_lo), _num(mp_hi),
                app.app_id in eligible,
            ))

        validation = vader_validation(rows, split="dev" if split == "dev" else "test")
        params = {
            "study": study.name,
            "stats_split": split,
            "lexicon_version": lexicon.version,
            "n_boot": n_boot,
            "boot_seed": BOOT_SEED,
            "min_kept_for_eligible": MIN_KEPT_FOR_ELIGIBLE,
            "min_not_rec_for_eligible": MIN_NOT_REC_FOR_ELIGIBLE,
            "vader_threshold": VADER_POSITIVE_THRESHOLD,
            "sentiment_model": (
                {**adopted[0], "positive_cutoff": TRANSFORMER_POSITIVE_CUTOFF} if adopted else None
            ),
        }
        cur = conn.execute(
            "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
            ("stats", json.dumps(params), _git_sha(), "{}"),
        )
        run_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO game_stats VALUES (" + ", ".join("?" * 30) + ")",
            [(run_id, *row) for row in game_rows],
        )

        # ---- per-game theme stats -----------------------------------------
        theme_rows: list[tuple] = []
        precision_ok = {
            (lens, name): lens == "topic" or bool(precision.get(name, {}).get("passes"))
            for lens, name in theme_keys
        }
        if derived:
            precision_ok[("aspect", ANY_TURN_STRUCTURE)] = all(
                precision.get(a, {}).get("passes") for a in TURN_STRUCTURE_ASPECTS
            )

        for app_id, g in games.items():
            point = game_theme_metrics(g)
            ci = bootstrap_game_themes(g, n_boot)
            for j, (lens, theme) in enumerate(theme_keys):
                rec_in, comp_in, pos_in, model_in = theme_sentiment([g], j)
                theme_rows.append((
                    lens, theme, SCOPE_GAME, str(app_id), int(g.rec.size),
                    int(g.themes[:, j].sum()),
                    _num(point.share[j]), _num(ci["share"][0][j]), _num(ci["share"][1][j]),
                    _num(point.complaint[j]), _num(ci["complaint"][0][j]),
                    _num(ci["complaint"][1][j]),
                    _num(point.praise[j]), _num(ci["praise"][0][j]), _num(ci["praise"][1][j]),
                    _num(point.lift[j]), _num(ci["lift"][0][j]), _num(ci["lift"][1][j]),
                    _num(rec_in), _num(comp_in), _num(pos_in), _num(model_in), None, None,
                ))

        # ---- cohort-level stats -------------------------------------------
        cohort_rows: list[tuple] = []
        comparisons = 0
        ranking_comparisons = 0
        findings: list[dict[str, Any]] = []
        ranking_findings: list[dict[str, Any]] = []
        rankings: dict[str, Any] = {}
        rank_pools = [
            [j for j, key in enumerate(theme_keys) if key[0] == lens and precision_ok[key]]
            for lens in ("aspect", "topic")
        ]
        for scope_type, scope_id, ids, elig in _scopes(study, eligible, kept_ok):
            present = [a for a in ids if a in games]
            rec_games = [games[a] for a in present if a in kept_ok]
            macro_rec = float(np.mean([g.rec.mean() for g in rec_games])) if rec_games else math.nan
            nan_ci = (np.array([math.nan]), np.array([math.nan]))
            rec_boot = cluster_bootstrap_macro(rec_games, n_boot) if len(rec_games) >= 2 else None
            rec_ci = rec_boot["rec_share"] if rec_boot else nan_ci
            comp_ci = rec_boot["mean_compound"] if rec_boot else nan_ci
            mp_ci = rec_boot["model_pos"] if rec_boot and adopted else nan_ci
            pooled = [r for a in present for r in by_app[a]]
            h_rec = [r.hours for r in pooled if r.recommended and r.hours is not None]
            h_not = [r.hours for r in pooled if not r.recommended and r.hours is not None]
            cohort_rows.append((
                scope_type, scope_id, len(ids), len(elig), len(pooled),
                _num(macro_rec), _num(rec_ci[0][0]), _num(rec_ci[1][0]),
                _num(float(np.mean([r.recommended for r in pooled]))) if pooled else None,
                median(h_rec) if h_rec else None, median(h_not) if h_not else None,
                _num(float(np.mean([g.compound.mean() for g in rec_games])))
                if rec_games else None,
                _num(comp_ci[0][0]), _num(comp_ci[1][0]),
                _num(float(np.mean([g.model_pos.mean() for g in rec_games])))
                if rec_games and adopted else None,
                _num(mp_ci[0][0]), _num(mp_ci[1][0]),
            ))

            elig_games = [games[a] for a in elig if a in games]
            if not elig_games:
                continue
            point = macro_theme_metrics(elig_games)
            ci = (
                cluster_bootstrap_macro(elig_games, n_boot)
                if len(elig_games) >= 2 else None
            )
            logo = (
                leave_one_game_out(elig_games, groups=rank_pools)
                if len(elig_games) >= 2 else np.zeros(len(theme_keys), dtype=bool)
            )
            for j, (lens, theme) in enumerate(theme_keys):
                lo = {k: (ci[k][0][j] if ci else math.nan) for k in ("share", "complaint", "praise", "lift")}
                hi = {k: (ci[k][1][j] if ci else math.nan) for k in ("share", "complaint", "praise", "lift")}
                finding = None
                if scope_type != SCOPE_ALL and precision_ok[(lens, theme)]:
                    comparisons += 1
                    k_not = int(sum(g.themes[~g.rec, j].sum() for g in elig_games))
                    k_rec = int(sum(g.themes[g.rec, j].sum() for g in elig_games))
                    finding = classify_finding(
                        lo["lift"], hi["lift"], len(elig_games), bool(logo[j]), True,
                        k_not, k_rec,
                    )
                    if finding:
                        findings.append({
                            "lens": lens, "theme": theme, "scope_type": scope_type,
                            "scope_id": scope_id, "finding": finding,
                            "lift": _num(point.lift[j]), "lift_ci": [_num(lo["lift"]), _num(hi["lift"])],
                        })
                rec_in, comp_in, pos_in, model_in = theme_sentiment(elig_games, j)
                theme_rows.append((
                    lens, theme, scope_type, scope_id,
                    int(sum(g.rec.size for g in elig_games)),
                    int(sum(g.themes[:, j].sum() for g in elig_games)),
                    _num(point.share[j]), _num(lo["share"]), _num(hi["share"]),
                    _num(point.complaint[j]), _num(lo["complaint"]), _num(hi["complaint"]),
                    _num(point.praise[j]), _num(lo["praise"]), _num(hi["praise"]),
                    _num(point.lift[j]), _num(lo["lift"]), _num(hi["lift"]),
                    _num(rec_in), _num(comp_in), _num(pos_in), _num(model_in),
                    bool(logo[j]) if precision_ok[(lens, theme)] else None, finding,
                ))
            top, n_rank_tests = _top_themes(
                theme_keys, point, lo_hi=ci, precision_ok=precision_ok, logo=logo,
                n_eligible=len(elig_games), is_cohort_scope=scope_type != SCOPE_ALL,
            )
            rankings[f"{scope_type}:{scope_id}"] = top
            ranking_comparisons += n_rank_tests
            for lens_side, items in top.items():
                if items and items[0]["status"] == "finding":
                    ranking_findings.append({
                        "lens": lens_side.split("_")[0], "theme": items[0]["theme"],
                        "scope_type": scope_type, "scope_id": scope_id,
                        "share": items[0]["share"], "ci": items[0]["ci"],
                    })

        conn.executemany(
            "INSERT INTO cohort_stats VALUES (" + ", ".join("?" * 18) + ")",
            [(run_id, *row) for row in cohort_rows],
        )
        conn.executemany(
            "INSERT INTO theme_stats VALUES (" + ", ".join("?" * 25) + ")",
            [(run_id, *row) for row in theme_rows],
        )
        metrics = {
            "finding_rule": FINDING_RULE,
            "stats_split": split,
            "n_split_documents": in_split,
            "n_games": len(study.apps),
            "n_kept": len(rows),
            "eligible_app_ids": sorted(eligible),
            "n_comparisons": comparisons + ranking_comparisons,
            "n_lift_comparisons": comparisons,
            "n_ranking_comparisons": ranking_comparisons,
            "n_findings": len(findings),
            "findings": findings,
            "n_ranking_findings": len(ranking_findings),
            "ranking_findings": ranking_findings,
            "top_themes": rankings,
            "vader_validation": validation,
            "sentiment_model": params["sentiment_model"],
            "aspect_precision": precision,
            "aspect_audit_status": (
                "assumed passing (candidates only)" if assume_precision
                else "partial" if study.aspect_labels and precision
                and len(precision) < len(lexicon.aspects)
                else "complete" if precision else "labels pending"
            ),
            "precision_assumed": assume_precision,
            "topic_labels": {str(t): label for t, label in topics},
            "lexicon_version": lexicon.version,
            "aspect_labels": {a.name: a.label for a in lexicon.aspects}
            | ({ANY_TURN_STRUCTURE: "Any turn-structure aspect"} if derived else {}),
            "aspect_kinds": {a.name: a.kind for a in lexicon.aspects}
            | ({ANY_TURN_STRUCTURE: "design"} if derived else {}),
        }
        conn.execute(
            "UPDATE analysis_runs SET metrics = ? WHERE run_id = ?",
            (json.dumps(metrics), run_id),
        )
        conn.commit()
        logger.info(
            "Stats run %d: %d games, %d eligible, %d comparisons, %d findings.",
            run_id, len(study.apps), len(eligible), comparisons, len(findings),
        )
        return run_id
    finally:
        conn.close()


def _top_themes(
    theme_keys: Sequence[tuple[str, str]],
    point: ThemeMetrics,
    lo_hi: dict[str, tuple[np.ndarray, np.ndarray]] | None,
    precision_ok: dict[tuple[str, str], bool],
    logo: np.ndarray,
    n_eligible: int,
    is_cohort_scope: bool,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Top three complaint and praise themes per lens by share, with ranking tests.

    Only the complaint side can hold a ranking finding ("most-cited
    complaint"): the top complaint theme is a finding if the scope is a
    cohort-level scope with enough eligible games, the theme is LOGO-stable
    (within-lens rank among precision-passing themes), and its complaint-share
    interval lies entirely above the second theme's. Praise rankings are
    always descriptive.

    Args:
        theme_keys: ``(lens, theme)`` per column.
        point: Macro point estimates.
        lo_hi: Cluster-bootstrap intervals, or None.
        precision_ok: Precision gate per theme.
        logo: LOGO stability per theme.
        n_eligible: Eligible games in the scope.
        is_cohort_scope: False for the all-games scope, which never yields findings.

    Returns:
        ``({"<lens>_complaint": [...], "<lens>_praise": [...]}, n_ranking_tests)``;
        each item has theme, share, interval and ``status`` (``finding`` or
        ``descriptive``). A ranking test is counted for each lens whose top
        complaint theme can be compared with a second theme under intervals.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    n_tests = 0
    for lens in ("aspect", "topic"):
        cols = [j for j, key in enumerate(theme_keys) if key[0] == lens and precision_ok[key]]
        for side, arr in (("complaint", point.complaint), ("praise", point.praise)):
            ranked = sorted(cols, key=lambda j: -(arr[j] if math.isfinite(arr[j]) else -1))[:3]
            items = []
            for pos, j in enumerate(ranked):
                lo = lo_hi[side][0][j] if lo_hi else math.nan
                hi = lo_hi[side][1][j] if lo_hi else math.nan
                status = "descriptive"
                if side == "complaint" and pos == 0 and is_cohort_scope and lo_hi and len(ranked) > 1:
                    n_tests += 1
                    nxt_hi = lo_hi[side][1][ranked[1]]
                    if n_eligible >= MIN_GAMES_FOR_FINDING and logo[j] and lo > nxt_hi:
                        status = "finding"
                items.append({"theme": theme_keys[j][1], "share": _num(arr[j]),
                              "ci": [_num(lo), _num(hi)], "status": status})
            out[f"{lens}_{side}"] = items
    return out, n_tests


# ---------------------------------------------------------------------------
# Reference checks against the prior hand-coded research (private input)
# ---------------------------------------------------------------------------


def reference_checks(study: Study, reference_path: Path, unfreeze: bool = False) -> int:
    """Compare the pull with a prior research table and store derived numbers only.

    The reference file is read at run time and never copied. Stored numbers:
    per-app gap between the Steam all-language positive share and the
    reference share, recomputed category medians, the direction check of
    measured turn-structure complaint versus praise against hand-coded tallies,
    and the lexicon recall proxy on hand-coded mentions for study apps.

    Args:
        study: The study.
        reference_path: Path to the research JSON.
        unfreeze: Allow writing a frozen study's seed database.

    Returns:
        The new ``analysis_runs.run_id``.

    Raises:
        FrozenStudyError: If a target path belongs to a frozen study and ``unfreeze`` is False.
    """
    guard_writable([study.seed_db], unfreeze)
    from src.aspects import TURN_STRUCTURE_ASPECTS, load_lexicon, tagged_share
    from src.db_setup import create_study_tables

    with open(reference_path, encoding="utf-8") as f:
        ref = json.load(f)["result"]
    table = ref["table"]
    study_ids = {a.app_id for a in study.apps}
    subgroup = {a.app_id: a.subgroup for a in study.apps}
    matched = {a.app_id for a in study.apps if "matched_basket" in a.tags}

    conn = sqlite3.connect(study.seed_db)
    try:
        create_study_tables(conn)
        base = {
            r[0]: (r[1], r[2]) for r in conn.execute(
                """
                SELECT app_id, baseline_total_positive, baseline_total_reviews FROM pull_runs
                WHERE run_id IN (SELECT MAX(run_id) FROM pull_runs GROUP BY app_id)
                """
            )
        }
        gaps = []
        ours_by_cat: dict[str, list[float]] = {}
        ours_matched: dict[str, list[float]] = {}
        tallies: dict[str, dict[str, int]] = {}
        mention_texts: list[str] = []
        for row in table:
            app_raw = (row.get("metrics") or {}).get("steamAppId")
            if not app_raw or int(app_raw) not in study_ids:
                continue
            app_id = int(app_raw)
            ref_pct = row["metrics"].get("reviewPercentPositive")
            pos, tot = base.get(app_id, (None, None))
            ours = 100.0 * pos / tot if tot else None
            gap = ours - ref_pct if ours is not None and ref_pct is not None else None
            gaps.append({"app_id": app_id, "subgroup": subgroup[app_id],
                         "reference_pct": ref_pct, "measured_pct": _num(ours) if ours else None,
                         "gap_pp": _num(gap) if gap is not None else None,
                         "within_1pp": gap is not None and abs(gap) <= 1.0})
            if ours is not None:
                ours_by_cat.setdefault(subgroup[app_id], []).append(ours)
                if app_id in matched:
                    ours_matched.setdefault(subgroup[app_id], []).append(ours)
            t = tallies.setdefault(subgroup[app_id], {"praise": 0, "complaint": 0})
            for m in row.get("mentions", []):
                if m.get("sentiment") in ("praise", "complaint"):
                    t[m["sentiment"]] += 1
                mention_texts.append(m.get("quote", ""))
        cat_key = {"igo_ugo": "igougo", "alternating": "alternating", "wego": "wego"}
        medians = {
            sub: {"measured": _num(float(np.median(v))),
                  "reference": ref["stats"].get(cat_key.get(sub, sub), {}).get("steamReviewPctMedian"),
                  "n": len(v)}
            for sub, v in ours_by_cat.items()
        }
        matched_medians = {sub: {"measured": _num(float(np.median(v))), "n": len(v)}
                           for sub, v in ours_matched.items()}

        stats_run = conn.execute(
            "SELECT MAX(run_id) FROM analysis_runs WHERE stage = 'stats'"
        ).fetchone()[0]
        direction = {}
        for sub, t in tallies.items():
            row = conn.execute(
                """
                SELECT complaint_share, praise_share, lift, lift_ci_low, lift_ci_high
                FROM theme_stats WHERE run_id = ? AND lens = 'aspect'
                  AND theme = 'turn_structure_any' AND scope_type = 'subgroup' AND scope_id = ?
                """,
                (stats_run, sub),
            ).fetchone()
            ref_dir = "complaint" if t["complaint"] > t["praise"] else "praise"
            measured_dir = None
            if row and row[2] is not None:
                measured_dir = "complaint" if row[2] > 1 else "praise"
            direction[sub] = {"reference_tally": t, "reference_direction": ref_dir,
                              "measured_lift": row[2] if row else None,
                              "measured_lift_ci": [row[3], row[4]] if row else None,
                              "measured_direction": measured_dir,
                              "agrees": measured_dir == ref_dir if measured_dir else None}

        assert study.aspect_lexicon is not None
        lexicon = load_lexicon(study.aspect_lexicon)
        hits, total = tagged_share(
            [t for t in mention_texts if t], lexicon, set(TURN_STRUCTURE_ASPECTS) | {"pacing"}
        )
        metrics = {
            "reproduction": gaps,
            "n_outside_1pp": sum(1 for g in gaps if not g["within_1pp"]),
            "subgroup_medians": medians,
            "matched_basket_medians": matched_medians,
            "direction_check": direction,
            "recall_proxy": {"hits": hits, "total": total,
                             "share": _num(hits / total) if total else None,
                             "aspects": sorted(set(TURN_STRUCTURE_ASPECTS) | {"pacing"}),
                             "lexicon_version": lexicon.version},
            "stats_run_id": stats_run,
        }
        cur = conn.execute(
            "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
            ("reference_checks", json.dumps({"study": study.name}), _git_sha(),
             json.dumps(metrics)),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


LARGE_UNMATCHED_TOPIC_SHARE: float = 0.05
"""A non-outlier topic holding at least this share of unmatched not-recommended
reviews is reported as a possible missing theme (report only, no lexicon change)."""


def coverage_proxy(study: Study, unfreeze: bool = False) -> int:
    """Measure how many not-recommended reviews the lexicon misses, and where they go.

    For each scope (all games, each cohort, each subgroup), the share of
    not-recommended kept English reviews with no aspect match, pooled, with a
    Wilson interval, for any v1 aspect and for precision-passing aspects only.
    The unmatched reviews (any aspect) are then broken down by BERTopic topic.

    Args:
        study: The study.
        unfreeze: Allow writing a frozen study's seed database.

    Returns:
        The new ``analysis_runs.run_id`` (stage ``coverage_proxy``).

    Raises:
        FrozenStudyError: If a target path belongs to a frozen study and ``unfreeze`` is False.
    """
    guard_writable([study.seed_db], unfreeze)
    from src.aspects import load_lexicon
    from src.db_setup import create_study_tables

    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    precision = study_precision(study, lexicon)
    passing = {a for a, p in precision.items() if p["passes"]}
    conn = sqlite3.connect(study.seed_db)
    try:
        create_study_tables(conn)
        rows = [r for r in load_reviews(conn) if not r.recommended]
        aspect_sets = load_aspect_sets(conn, lexicon.version)
        labels = dict(conn.execute("SELECT topic_id, label FROM topics"))
        scopes: list[tuple[str, str, set[int]]] = [("all", "all", {a.app_id for a in study.apps})]
        for cohort in dict.fromkeys(a.cohort for a in study.apps):
            scopes.append((SCOPE_COHORT, cohort, {a.app_id for a in study.apps if a.cohort == cohort}))
        for sub in dict.fromkeys(a.subgroup for a in study.apps if a.subgroup):
            scopes.append((SCOPE_SUBGROUP, sub, {a.app_id for a in study.apps if a.subgroup == sub}))
        out = []
        for scope_type, scope_id, apps in scopes:
            rs = [r for r in rows if r.app_id in apps]
            unmatched = [r for r in rs if not aspect_sets.get(r.document_id)]
            k_pass = sum(1 for r in rs if not (aspect_sets.get(r.document_id, set()) & passing))
            n = len(rs)
            topic_counts: dict[int, int] = {}
            for r in unmatched:
                topic_counts[r.topic_id] = topic_counts.get(r.topic_id, 0) + 1
            nr_topic: dict[int, int] = {}
            for r in rs:
                nr_topic[r.topic_id] = nr_topic.get(r.topic_id, 0) + 1
            top = sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            out.append({
                "scope_type": scope_type, "scope_id": scope_id, "n_not_recommended": n,
                "unmatched_any": len(unmatched),
                "unmatched_any_share": _num(len(unmatched) / n) if n else None,
                "unmatched_any_ci": list(wilson_interval(len(unmatched), n)) if n else None,
                "unmatched_passing": k_pass,
                "unmatched_passing_share": _num(k_pass / n) if n else None,
                "unmatched_passing_ci": list(wilson_interval(k_pass, n)) if n else None,
                "top_unmatched_topics": [
                    {"topic_id": t, "label": labels.get(t, ""), "count": c,
                     "share_of_unmatched": _num(c / len(unmatched)),
                     "unmatched_rate_in_topic": _num(c / nr_topic[t]),
                     "large": t != -1 and c / len(unmatched) >= LARGE_UNMATCHED_TOPIC_SHARE}
                    for t, c in top
                ],
            })
        metrics = {"lexicon_version": lexicon.version, "passing_aspects": sorted(passing),
                   "large_topic_share": LARGE_UNMATCHED_TOPIC_SHARE, "scopes": out}
        cur = conn.execute(
            "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
            ("coverage_proxy", json.dumps({"study": study.name}), _git_sha(), json.dumps(metrics)),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def candidate_run(study: Study, out_dir: Path, n_boot: int = N_BOOT) -> Path:
    """Compute exploratory candidates with every aspect assumed to pass its audit.

    Before the precision audit is labeled, no aspect can meet the finding
    rule, so the stats stage reports aspects as descriptive only. This run
    copies the seed into ``out_dir``, runs ``run`` on the copy with
    ``assume_precision=True`` and exports it there. The study's own seed is
    only read. Every finding in the export is a candidate, conditional on its
    aspect passing the audit later.

    Args:
        study: The study (its aspect stage must have run).
        out_dir: Directory for the seed copy and the export.
        n_boot: Bootstrap resamples.

    Returns:
        Path of the exported JSON file.
    """
    from dataclasses import replace

    out_dir.mkdir(parents=True, exist_ok=True)
    copy = out_dir / "candidate_seed.db"
    src = sqlite3.connect(f"{study.seed_db.resolve().as_uri()}?mode=ro", uri=True)
    dst = sqlite3.connect(copy)
    try:
        src.backup(dst)  # replaces any earlier copy's content
    finally:
        src.close()
        dst.close()
    shadow = replace(study, seed_db=copy, scores_db=study.scores_path)
    run_id = run(shadow, n_boot=n_boot, assume_precision=True)
    return export_run(shadow, out_dir, run_id, stem="candidates")


EXPORT_TABLES: tuple[str, ...] = ("game_stats", "cohort_stats", "theme_stats")
"""Per-run tables written by ``export_run``."""


def export_run(study: Study, out_dir: Path, run_id: int | None = None, stem: str = "stats") -> Path:
    """Write one stats run to files: metrics as JSON, each per-run table as CSV.

    Reads the seed only (read-only connection), so it never changes a study.

    Args:
        study: The study.
        out_dir: Directory for ``stats_run_<id>.json`` and
            ``<table>_run_<id>.csv``.
        run_id: The ``analysis_runs`` row (default: the newest ``stats`` run).
        stem: File name prefix (``stats`` or ``candidates``).

    Returns:
        Path of the JSON file.

    Raises:
        ValueError: If the seed has no such stats run.
    """
    conn = sqlite3.connect(f"{study.seed_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        if run_id is None:
            run_id = conn.execute(
                "SELECT MAX(run_id) FROM analysis_runs WHERE stage = 'stats'"
            ).fetchone()[0]
        row = conn.execute(
            "SELECT params, git_sha, created_at, metrics FROM analysis_runs "
            "WHERE run_id = ? AND stage = 'stats'",
            (run_id,),
        ).fetchone() if run_id is not None else None
        if row is None:
            raise ValueError(f"No stats run {run_id} in {study.seed_db}.")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}_run_{run_id}.json"
        json_path.write_text(json.dumps({
            "run_id": run_id, "params": json.loads(row[0]), "git_sha": row[1],
            "created_at": row[2], "metrics": json.loads(row[3]),
        }, indent=2), encoding="utf-8")
        queries = {
            "game_stats": "SELECT * FROM game_stats WHERE run_id = ?",
            "cohort_stats": "SELECT * FROM cohort_stats WHERE run_id = ?",
            "theme_stats": "SELECT * FROM theme_stats WHERE run_id = ?",
        }
        for table in EXPORT_TABLES:
            cur = conn.execute(queries[table], (run_id,))
            name = f"{table}_run_{run_id}.csv" if stem == "stats" else f"{stem}_{table}_run_{run_id}.csv"
            with open(out_dir / name, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([d[0] for d in cur.description])
                writer.writerows(cur.fetchall())
    finally:
        conn.close()
    logger.info("Exported stats run %d to %s", run_id, out_dir)
    return json_path


def main() -> None:
    """CLI entry point: compute stats, optionally run reference checks."""
    parser = argparse.ArgumentParser(description="Compute study statistics.")
    parser.add_argument("--study", type=Path, required=True, help="Study TOML file.")
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="Bootstrap resamples.")
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Private research JSON for reference checks (read only, never copied).",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Store the lexicon coverage proxy (unmatched not-recommended reviews by topic).",
    )
    parser.add_argument(
        "--export", type=Path, default=None,
        help="Write the newest stats run (metrics JSON, per-run tables CSV) to this directory "
             "and exit; reads the analysis database only.",
    )
    parser.add_argument(
        "--candidates", type=Path, default=None,
        help="Run the statistics on a copy of the analysis database with every aspect assumed "
             "to pass its precision audit and export it to this directory (exploratory "
             "candidates only).",
    )
    parser.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    args = parser.parse_args()
    # The seed, or the stats_split half of the full database for stats_source = "full".
    study = load_study(args.study).analysis_view()
    try:
        if args.export is not None:
            export_run(study, args.export)
        elif args.candidates is not None:
            candidate_run(study, args.candidates, n_boot=args.n_boot)
        elif args.coverage:
            coverage_proxy(study, unfreeze=args.unfreeze)
        elif args.reference is not None:
            reference_checks(study, args.reference, unfreeze=args.unfreeze)
        else:
            run(study, n_boot=args.n_boot, unfreeze=args.unfreeze)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
