"""Phase B confirmatory read of the rtwp_turn_2026 study.

The protocol is ``docs/rtwp_turn_protocol.md``. It is committed and tagged
``PROTOCOL_TAG`` before the one confirmatory read. This module implements it:

- ``frame`` writes the phase B frame: the sorted ``app_id<TAB>steam_review_id``
  lines of every test-half review of the full database outside both aspect
  audit sets (``FRAME_PATH``, committed). It reads review IDs only.
- ``agreement`` measures, on the development half, how often the
  development-fit BERTopic model's ``transform`` returns a review's fitted
  topic, the way the market protocol measured it.
- ``gate`` prints whether the read may run.
- ``dry-run`` runs the whole read path on a small development slice, with
  lowered eligibility, and never reads a test-half review.
- ``read`` is the confirmatory read. It refuses unless the working tree is
  clean, HEAD carries ``PROTOCOL_TAG``, no results file exists and every
  pinned file matches its SHA-256. Only then does it read the frame reviews'
  text and recommend flags from the full database and run every test-half
  instrument: cleaning, the language and token filters, VADER, the
  development-fit topic model's ``transform`` and aspect tagging with the
  frozen lexicon.

Every write goes under ``outputs/rtwp_turn_2026/`` (ignored by Git) except
the frame file. The full database, the analysis database, the lexicon and the
saved model are opened read-only and never changed.
"""

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.replication import _git, file_sha256, meets_criterion
from src.rtwp_power import CANDIDATES, STUDY_PATH, Finding
from src.stats import (
    BOOT_SEED,
    MIN_GAMES_FOR_FINDING,
    MIN_KEPT_FOR_ELIGIBLE,
    MIN_NOT_REC_FOR_ELIGIBLE,
    N_BOOT,
    SCOPE_ALL,
    GameData,
    _scopes,
    _top_themes,
    classify_finding,
    cluster_bootstrap_macro,
    dev_test_split,
    leave_one_game_out,
    macro_theme_metrics,
    study_precision,
)
from src.study import Study, load_study
from src.utils import DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT, MIN_TOKEN_COUNT, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants (pinned by docs/rtwp_turn_protocol.md)
# ---------------------------------------------------------------------------

PROTOCOL_TAG: str = "rtwp-turn-protocol-v1"
"""Git tag that must point at HEAD before the confirmatory read."""

PROTOCOL_PATH: Path = PROJECT_ROOT / "docs" / "rtwp_turn_protocol.md"
"""The committed protocol."""

STUDY_DIR: Path = OUTPUTS_DIR / "rtwp_turn_2026"
"""Every output of this module lives under this directory."""

CONFIRMATORY_DIR: Path = STUDY_DIR / "confirmatory"
"""Results of the one confirmatory read."""

DRY_RUN_DIR: Path = STUDY_DIR / "confirmatory_dry_run"
"""Results of a dry run on development reviews."""

FRAME_PATH: Path = DATA_DIR / "frames" / "rtwp_turn_2026_phase_b_frame.tsv"
"""The phase B frame: sorted ``app_id<TAB>steam_review_id`` lines, IDs only."""

CANDIDATES_PATH: Path = STUDY_DIR / "exploratory" / "candidates_run_4.json"
"""Phase A candidate export (run 4, lexicon rtwp_turn_v3)."""

MODEL_PATH: Path = STUDY_DIR / "bertopic_model_full_dev" / "model.pkl"
"""BERTopic model fitted on the kept development-half reviews."""

POWER_PATH: Path = STUDY_DIR / "exploratory" / "power_rtwp_turn_v3_cap0_testable.json"
"""Power simulation for the whole frame that sets the pass line."""

LEXICON_PATH: Path = PROJECT_ROOT / "studies" / "aspects_rtwp_turn_v3.toml"
"""The frozen lexicon."""

LABEL_PATHS: tuple[Path, ...] = (
    DATA_DIR / "labels" / "aspect_audit_rtwp_turn_v2.csv",
    DATA_DIR / "labels" / "aspect_audit_rtwp_turn_v3.csv",
)
"""The two audits' label files (``[aspect_labels]`` in the study file)."""

PINS: dict[str, tuple[Path, str | None]] = {
    "candidates": (CANDIDATES_PATH, "57c26c8245527dab90852f3cdc2b43590fdc927ff653e8869f7b9856db0295f2"),
    "lexicon": (LEXICON_PATH, "0b6907a312628913308c5a50dd05a9ca87ea1a92ef55e78c5622fc96c1037c8f"),
    "labels v2": (LABEL_PATHS[0], "d5262bb4bffa3a9eb3fbc5219f9445760dbbdcc5644417167dda80588900101f"),
    "labels v3": (LABEL_PATHS[1], "5cb56d300bfbd96f3587079a16df91b354d07f4460bcf7ca353926cc185b18e3"),
    "topic model": (MODEL_PATH, "dd037032288fa2df3c26df517c1fe0ea1517ffd5f173c6a2a11fd3ce9999e9d0"),
    "power": (POWER_PATH, "521cd42c18ffe1bbb23f6fcb8442e08858fb664ed21b9dc47da6a274ecd61274"),
    "frame": (FRAME_PATH, "4cf9fa3a44107836c96571096fd3ce9831dbc45071459b500e8456a42826965d"),
}
"""Pinned files and their SHA-256 (protocol section 10)."""

FRAME_TOTAL: int = 119485
"""Reviews in the frame."""

COUNT_THRESHOLD: int = 43
"""Pass line: the 5th percentile of the half log lift scenario (``POWER_PATH``)."""

EXPECTED_COUNT_PLUG_IN: float = 60.0
"""Mean replicated count, plug-in scenario."""

EXPECTED_COUNT_HALF: float = 46.7
"""Mean replicated count, half log lift scenario."""

COMBAT_ASPECTS: tuple[str, ...] = (
    "mode_turn_based", "mode_rtwp", "pause_micro", "realtime_chaos", "waiting_turns",
    "combat_pacing",
)
"""The combat-model aspects. Their candidates get their own count, with no verdict."""

COMBAT_EXPECTED_PLUG_IN: float = 9.8
"""Expected replicated combat-model candidates, plug-in scenario."""

COMBAT_EXPECTED_HALF: float = 2.5
"""Expected replicated combat-model candidates, half log lift scenario."""

DEV_RANKING_FINDINGS: tuple[tuple[str, str, str], ...] = (
    ("bugs_performance", "cohort", "crpg"),
    ("bugs_performance", "cohort_subgroup", "crpg/rtwp"),
    ("bugs_performance", "subgroup", "turn_based"),
)
"""Ranking findings of candidate run 4: reported without a verdict."""

TOPIC_FAMILIES: dict[str, str] = {
    "1": "Topic 1, DLC", "3": "Topic 3, party and enemies", "11": "Topic 11, crashes",
    "15": "Topic 15, AI and armies", "19": "Topic 19, multiplayer and desync",
    "38": "Topic 38, Paradox DLC",
}
"""Labels of the topic families."""

DRY_RUN_PER_GAME: int = 150
DRY_RUN_MIN_KEPT: int = 40
DRY_RUN_MIN_NOT_REC: int = 8
DRY_RUN_BOOT: int = 200


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def _read_only(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite file read-only.

    Args:
        path: Database file.

    Returns:
        A read-only connection.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def build_frame(study: Study, excluded: set[str]) -> list[tuple[int, str]]:
    """List the frame from review IDs alone.

    Args:
        study: The study.
        excluded: Review IDs of the audit sets.

    Returns:
        Sorted ``(app_id, steam_review_id)`` of every test-half review of the
        full database not in ``excluded``.
    """
    conn = _read_only(study.full_db)
    try:
        rows = [
            (int(app_id), rid)
            for app_id, rid in conn.execute("SELECT app_id, steam_review_id FROM documents")
            if dev_test_split(rid) == "test" and rid not in excluded
        ]
    finally:
        conn.close()
    return sorted(rows)


def frame_bytes(frame: Sequence[tuple[int, str]]) -> bytes:
    """Serialise a frame as ``app_id<TAB>steam_review_id`` lines.

    Args:
        frame: Sorted ``(app_id, steam_review_id)``.

    Returns:
        UTF-8 bytes, one line per review, newline-terminated.
    """
    return "".join(f"{a}\t{r}\n" for a, r in frame).encode("utf-8")


def write_frame(study: Study, path: Path = FRAME_PATH) -> dict[str, Any]:
    """Write the frame file from review IDs and the audit exports' ID columns.

    Args:
        study: The study.
        path: Frame file.

    Returns:
        ``{"path", "reviews", "bytes", "sha256", "excluded_audit_sets"}``.
    """
    from src.aspects import load_lexicon
    from src.rtwp_power import frame_exclusions

    assert study.aspect_lexicon is not None
    sets = frame_exclusions(study, load_lexicon(study.aspect_lexicon))
    frame = build_frame(study, set().union(*sets.values()))
    data = frame_bytes(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT)
            else str(path), "reviews": len(frame), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "excluded_audit_sets": {k: len(v) for k, v in sets.items()}}


def load_frame(path: Path = FRAME_PATH) -> list[tuple[int, str]]:
    """Read the frame file.

    Args:
        path: Frame file.

    Returns:
        ``(app_id, steam_review_id)`` in file order.
    """
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        app, rid = line.split("\t")
        out.append((int(app), rid))
    return out


# ---------------------------------------------------------------------------
# Topic transform agreement (development half only)
# ---------------------------------------------------------------------------


def topic_transform_agreement(study: Study, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Compare the dev-fit model's ``transform`` with its fitted topics.

    Reads the kept development-half reviews of the analysis database. As in
    the market protocol, passing the whole training set at once is only a
    reproduction check (umap-learn recognises its training input). Passing
    it in two halves (even and odd positions) forces the path new reviews
    take, which is the number the protocol records.

    Args:
        study: The study (read through its analysis view).
        model_path: Saved BERTopic model.

    Returns:
        Agreement, outlier shares and per-topic agreement for the candidate
        topics, for the whole set and for two halves.
    """
    from bertopic import BERTopic

    conn = _read_only(study.analysis_view().seed_db)
    try:
        rows = conn.execute(
            """
            SELECT p.cleaned_text, COALESCE(n.topic_id, -1)
            FROM processed_documents p JOIN nlp_results n ON n.document_id = p.document_id
            WHERE p.token_count >= ? ORDER BY p.document_id
            """,
            (MIN_TOKEN_COUNT,),
        ).fetchall()
    finally:
        conn.close()
    model = BERTopic.load(str(model_path))
    texts = [r[0] for r in rows]
    fitted = np.array([r[1] for r in rows])
    whole = np.array(model.transform(texts)[0])
    halves = np.empty_like(fitted)
    for start in (0, 1):
        halves[start::2] = model.transform(texts[start::2])[0]
    topics = sorted({int(f.theme) for f in CANDIDATES if f.lens == "topic"})

    def compare(transformed: np.ndarray) -> dict[str, Any]:
        """Agreement of one transform with the fitted assignments."""
        res: dict[str, Any] = {
            "agreement": float((fitted == transformed).mean()),
            "outlier_share_fit": float((fitted == -1).mean()),
            "outlier_share_transform": float((transformed == -1).mean()),
            "topics": {},
        }
        for topic in topics:
            in_fit = fitted == topic
            in_tr = transformed == topic
            res["topics"][str(topic)] = {
                "n_fit": int(in_fit.sum()),
                "n_transform": int(in_tr.sum()),
                "fit_kept_by_transform": float((in_fit & in_tr).sum() / max(in_fit.sum(), 1)),
                "transform_in_fit": float((in_fit & in_tr).sum() / max(in_tr.sum(), 1)),
            }
        return res

    return {"n": int(fitted.size), "model_sha256": file_sha256(model_path),
            "whole_set": compare(whole), "two_halves": compare(halves)}


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameReview:
    """One frame review after the instruments.

    Attributes:
        steam_review_id: Steam recommendation ID.
        app_id: Steam application ID.
        created: ``timestamp_created`` in Unix seconds, or None.
        kept: Passed the length, token and language rules.
        aspects: Aspects tagged in any sentence (kept reviews only).
        topic_id: Topic from the dev-fit model's ``transform`` (-1 outlier or dropped).
        compound: VADER compound of the cleaned text (kept reviews only).
    """

    steam_review_id: str
    app_id: int
    created: int | None
    kept: bool
    aspects: frozenset[str]
    topic_id: int
    compound: float


Instruments = tuple[
    Callable[[list[str]], list[dict]],
    Callable[[list[str]], list[set[str]]],
    Callable[[list[str]], list[int]],
    Callable[[list[str]], list[float]],
]
"""``(clean_and_filter, tag, assign_topics, vader)``."""


def frozen_instruments(study: Study) -> Instruments:
    """Build the real instruments: preprocessing, lexicon v3, topic transform, VADER.

    Args:
        study: The study.

    Returns:
        ``(clean_and_filter, tag, assign_topics, vader)``.
    """
    from bertopic import BERTopic

    from src.aspects import SentenceSplitter, load_lexicon, tag_sentences
    from src.preprocessing import LanguageFilter, Preprocessor, apply_language_filter
    from src.sentiment import SentimentScorer

    assert study.aspect_lexicon is not None
    pre = Preprocessor()
    lang = LanguageFilter() if study.language_filter else None
    lexicon = load_lexicon(study.aspect_lexicon)
    splitter = SentenceSplitter()
    model = BERTopic.load(str(MODEL_PATH))
    scorer = SentimentScorer()

    def clean_and_filter(raws: list[str]) -> list[dict]:
        """Clean, tokenize and language-filter raw texts."""
        results = pre.process_batch(raws)
        if lang is not None:
            apply_language_filter(results, lang)
        return results

    def tag(texts: list[str]) -> list[set[str]]:
        """Tag each cleaned text with the aspects of the frozen lexicon."""
        return [
            {m.aspect for m in tag_sentences([s[2] for s in spans], lexicon)}
            for spans in splitter.split(texts)
        ]

    def assign_topics(texts: list[str]) -> list[int]:
        """Assign topics with the dev-fit model's transform (never refit)."""
        return [int(t) for t in model.transform(texts)[0]]

    def vader(texts: list[str]) -> list[float]:
        """VADER compound of each cleaned text."""
        return [s["compound"] for s in scorer.score_batch(texts)]

    return clean_and_filter, tag, assign_topics, vader


def process_reviews(
    rows: Sequence[tuple[str, int, int | None, str]], instruments: Instruments
) -> list[FrameReview]:
    """Apply the frozen instruments to raw reviews. Reads no recommend flag.

    Args:
        rows: ``(steam_review_id, app_id, created, raw_text)``.
        instruments: ``(clean_and_filter, tag, assign_topics, vader)``.

    Returns:
        One FrameReview per input row, in input order.
    """
    clean_and_filter, tag, assign_topics, vader = instruments
    results = clean_and_filter([r[3] for r in rows])
    kept_idx = [i for i, res in enumerate(results) if not res["dropped"]]
    texts = [results[i]["cleaned_text"] for i in kept_idx]
    aspect_sets = tag(texts) if texts else []
    topic_ids = assign_topics(texts) if texts else []
    compounds = vader(texts) if texts else []
    out = [FrameReview(r[0], r[1], r[2], False, frozenset(), -1, math.nan) for r in rows]
    for i, aspects, topic, comp in zip(kept_idx, aspect_sets, topic_ids, compounds):
        r = rows[i]
        out[i] = FrameReview(r[0], r[1], r[2], True, frozenset(aspects), int(topic), float(comp))
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def theme_keys(lexicon: Any, topic_ids: Sequence[int]) -> list[tuple[str, str]]:
    """Theme columns as in the development stats run.

    Every aspect, the derived turn-structure theme when the lexicon defines
    every turn-structure aspect (as ``stats.run`` does), then every topic.

    Args:
        lexicon: The frozen lexicon.
        topic_ids: Non-outlier topic IDs of the dev-fit model.

    Returns:
        ``(lens, theme)`` per column.
    """
    from src.aspects import ANY_TURN_STRUCTURE, TURN_STRUCTURE_ASPECTS

    names = [a.name for a in lexicon.aspects]
    derived = [ANY_TURN_STRUCTURE] if set(TURN_STRUCTURE_ASPECTS) <= set(names) else []
    return [("aspect", a) for a in names + derived] + [("topic", str(t)) for t in topic_ids]


def _member(review: "FrameReview", lens: str, theme: str) -> bool:
    """Whether a kept review belongs to a theme column."""
    from src.aspects import ANY_TURN_STRUCTURE, TURN_STRUCTURE_ASPECTS

    if lens == "topic":
        return str(review.topic_id) == theme
    if theme == ANY_TURN_STRUCTURE:
        return bool(review.aspects & set(TURN_STRUCTURE_ASPECTS))
    return theme in review.aspects


def build_games(
    reviews: Sequence[FrameReview], recommended: dict[str, bool], keys: Sequence[tuple[str, str]]
) -> dict[int, GameData]:
    """Per-game arrays of kept reviews.

    Args:
        reviews: Processed frame reviews.
        recommended: ``{steam_review_id: recommend flag}``.
        keys: Theme columns.

    Returns:
        ``{app_id: GameData}``.
    """
    by_app: dict[int, list[FrameReview]] = {}
    for r in reviews:
        if r.kept:
            by_app.setdefault(r.app_id, []).append(r)
    games: dict[int, GameData] = {}
    for app_id, rs in by_app.items():
        m = np.zeros((len(rs), len(keys)), dtype=bool)
        for i, r in enumerate(rs):
            for j, (lens, theme) in enumerate(keys):
                m[i, j] = _member(r, lens, theme)
        games[app_id] = GameData(app_id, np.array([recommended[r.steam_review_id] for r in rs],
                                                  dtype=bool),
                                 np.array([r.compound for r in rs]), m)
    return games


def _num(x: float) -> float | None:
    """Convert nan or inf to None for JSON."""
    return float(x) if x is not None and math.isfinite(x) else None


def evaluate(
    study: Study,
    games: dict[int, GameData],
    keys: Sequence[tuple[str, str]],
    passing: set[str],
    development: dict[str, dict[str, Any]],
    dev_eligible: set[int],
    candidates: Sequence[Finding] = CANDIDATES,
    n_boot: int = N_BOOT,
    min_kept: int = MIN_KEPT_FOR_ELIGIBLE,
    min_not: int = MIN_NOT_REC_FOR_ELIGIBLE,
) -> dict[str, Any]:
    """Check every candidate and compute the descriptive rows with the frozen statistics.

    Eligibility is computed on the phase B sample itself. Each scope's games
    are its study games that are eligible there. Point lifts come from
    ``stats.macro_theme_metrics``, intervals from ``stats.cluster_bootstrap_macro``
    with ``N_BOOT`` resamples and ``BOOT_SEED``, stability from
    ``stats.leave_one_game_out`` with the development rank pools, and the full
    rule from ``stats.classify_finding``, as in the development run.

    Args:
        study: The study.
        games: Phase B data per game.
        keys: Theme columns of ``games``.
        passing: Aspects that pass their audit (merged precision).
        development: ``{candidate key: {"lift", "lift_ci"}}`` from phase A.
        dev_eligible: Games eligible in phase A.
        candidates: Candidates to check.
        n_boot: Bootstrap resamples.
        min_kept: Eligibility, kept reviews.
        min_not: Eligibility, not-recommended reviews.

    Returns:
        ``{"eligible", "candidates", "combat_rows", "ranking", "scopes"}``.
    """
    col = {k: j for j, k in enumerate(keys)}
    kept_ok = {a for a, g in games.items() if g.rec.size >= min_kept}
    eligible = {a for a in kept_ok if (~games[a].rec).sum() >= min_not}
    from src.aspects import ANY_TURN_STRUCTURE, TURN_STRUCTURE_ASPECTS

    passing_all = set(passing) | ({ANY_TURN_STRUCTURE} if set(TURN_STRUCTURE_ASPECTS) <= passing
                                  else set())
    precision_ok = {k: k[0] == "topic" or k[1] in passing_all for k in keys}
    pools = [[j for j, k in enumerate(keys) if k[0] == lens and precision_ok[k]]
             for lens in ("aspect", "topic")]
    scope_rows: dict[tuple[str, str], dict[str, Any]] = {}
    ranking: list[dict[str, Any]] = []
    for scope_type, scope_id, ids, elig in _scopes(study, eligible, kept_ok):
        sel = [games[a] for a in elig if a in games]
        entry: dict[str, Any] = {"ids": list(ids), "eligible": [g.app_id for g in sel]}
        scope_rows[(scope_type, scope_id)] = entry
        if not sel:
            continue
        point = macro_theme_metrics(sel)
        ci = cluster_bootstrap_macro(sel, n_boot, BOOT_SEED) if len(sel) >= 2 else None
        logo = (leave_one_game_out(sel, groups=pools) if len(sel) >= 2
                else np.zeros(len(keys), dtype=bool))
        entry.update(point=point, ci=ci, logo=logo, games=sel)
        top, _ = _top_themes(keys, point, lo_hi=ci, precision_ok=precision_ok, logo=logo,
                             n_eligible=len(sel), is_cohort_scope=scope_type != SCOPE_ALL)
        first = top.get("aspect_complaint", [])
        if first:
            ranking.append({"scope_type": scope_type, "scope_id": scope_id, **first[0],
                            "second": first[1] if len(first) > 1 else None})

    def row(lens: str, theme: str, scope_type: str, scope_id: str) -> dict[str, Any]:
        """Phase B statistics of one theme in one scope."""
        e = scope_rows.get((scope_type, scope_id), {"ids": [], "eligible": []})
        sel = e.get("games", [])
        out: dict[str, Any] = {"games_in_scope": len(e["ids"]), "phase_b_games": e["eligible"],
                               "dev_eligible_games": sum(a in dev_eligible for a in e["ids"])}
        if len(sel) < MIN_GAMES_FOR_FINDING:
            return {**out, "testable_scope": False}
        j = col[(lens, theme)]
        point, ci, logo = e["point"], e["ci"], e["logo"]
        lo = float(ci["lift"][0][j]) if ci else math.nan
        hi = float(ci["lift"][1][j]) if ci else math.nan
        k_not = int(sum(g.themes[~g.rec, j].sum() for g in sel))
        k_rec = int(sum(g.themes[g.rec, j].sum() for g in sel))
        return {**out, "testable_scope": True, "lift": float(point.lift[j]), "ci": [lo, hi],
                "complaint_share": float(point.complaint[j]),
                "praise_share": float(point.praise[j]), "k_not_recommended": k_not,
                "k_recommended": k_rec, "tagged": k_not + k_rec,
                "n_kept": int(sum(g.rec.size for g in sel)), "logo_stable": bool(logo[j]),
                "full_rule": classify_finding(lo, hi, len(sel), bool(logo[j]), True, k_not, k_rec)}

    results = []
    for f in candidates:
        r = row(f.lens, f.theme, f.scope_type, f.scope_id)
        dev = development[f.key]
        base = {"key": f.key, "family": f.family, "lens": f.lens, "theme": f.theme,
                "scope_type": f.scope_type, "scope_id": f.scope_id, "direction": f.direction,
                "combat": f.lens == "aspect" and f.theme in COMBAT_ASPECTS,
                "dev_lift": dev["lift"], "dev_ci": dev["lift_ci"],
                "games_in_scope": r["games_in_scope"], "phase_b_games": r["phase_b_games"],
                "dev_eligible_games": r["dev_eligible_games"]}
        if f.lens == "aspect" and f.theme not in passing:
            results.append({**base, "status": "not_testable", "reason": "aspect failed its audit"})
            continue
        if not r["testable_scope"]:
            results.append({**base, "status": "not_testable",
                            "reason": f"fewer than {MIN_GAMES_FOR_FINDING} eligible games"})
            continue
        lift, (lo, hi) = r["lift"], r["ci"]
        d_lo, d_hi = dev["lift_ci"]
        results.append({
            **base,
            "status": "replicated" if meets_criterion(f.direction, lift, lo, hi)
            else "not_replicated",
            "lift": _num(lift), "ci": [_num(lo), _num(hi)],
            "complaint_share": _num(r["complaint_share"]), "praise_share": _num(r["praise_share"]),
            "k_not_recommended": r["k_not_recommended"], "k_recommended": r["k_recommended"],
            "n_kept": r["n_kept"], "logo_stable": r["logo_stable"],
            "full_rule_same_direction": r["full_rule"] == f.direction,
            "dev_in_phase_b_ci": bool(math.isfinite(lo) and math.isfinite(hi)
                                      and lo <= dev["lift"] <= hi),
            "phase_b_in_dev_ci": bool(math.isfinite(lift) and d_lo <= lift <= d_hi),
            "log_lift_ratio": _num(math.log(lift) / math.log(dev["lift"]))
            if math.isfinite(lift) and lift > 0 else None,
        })

    cand_keys = {f.key for f in candidates}
    combat_rows = []
    for (scope_type, scope_id) in scope_rows:
        if scope_type == SCOPE_ALL:
            continue
        for aspect in COMBAT_ASPECTS:
            key = f"aspect:{aspect}@{scope_type}:{scope_id}"
            if aspect not in passing or key in cand_keys:
                continue
            r = row("aspect", aspect, scope_type, scope_id)
            if not r["testable_scope"]:
                continue
            combat_rows.append({
                "key": key, "tagged": r["tagged"], "complaint_share": _num(r["complaint_share"]),
                "praise_share": _num(r["praise_share"]), "lift": _num(r["lift"]),
                "ci": [_num(x) for x in r["ci"]], "logo_stable": r["logo_stable"],
            })
    dev_rank = {f"{t}:{s}": theme for theme, t, s in DEV_RANKING_FINDINGS}
    ranking_out = [
        {**{k: v for k, v in r.items() if k != "second"},
         "development_ranking_finding": f"{r['scope_type']}:{r['scope_id']}" in dev_rank,
         "second_theme": r["second"]["theme"] if r["second"] else None,
         "second_ci": r["second"]["ci"] if r["second"] else None}
        for r in ranking
        if f"{r['scope_type']}:{r['scope_id']}" in dev_rank or r["status"] == "finding"
    ]
    return {"eligible": sorted(eligible), "candidates": results, "combat_rows": combat_rows,
            "ranking": ranking_out}


def summarize(results: Sequence[dict[str, Any]],
              count_threshold: int | None = COUNT_THRESHOLD) -> dict[str, Any]:
    """Counts, family verdicts, the combat count and the study verdict.

    A family replicates when more than half of its testable candidates
    replicate. The study verdict compares the replicated count of all
    candidates with ``count_threshold``. The combat-model count is reported
    beside it with no verdict.

    Args:
        results: ``evaluate(...)["candidates"]``.
        count_threshold: Pass line, or None for no verdict (dry run).

    Returns:
        The summary dictionary.
    """
    def count(status: str, rows: Sequence[dict[str, Any]]) -> int:
        """Rows with the given status."""
        return sum(r["status"] == status for r in rows)

    testable = [r for r in results if r["status"] != "not_testable"]
    families: dict[str, Any] = {}
    for fam in dict.fromkeys(r["family"] for r in results):
        rows = [r for r in results if r["family"] == fam]
        t = [r for r in rows if r["status"] != "not_testable"]
        rep = count("replicated", rows)
        families[fam] = {
            "label": TOPIC_FAMILIES.get(rows[0]["theme"], fam) if rows[0]["lens"] == "topic"
            else fam,
            "candidates": len(rows), "testable": len(t), "replicated": rep,
            "not_replicated": count("not_replicated", rows),
            "not_testable": count("not_testable", rows),
            "verdict": "not_testable" if not t
            else ("replicated" if rep * 2 > len(t) else "not_replicated"),
        }
    combat = [r for r in results if r["combat"]]
    n_rep = count("replicated", results)
    verdict = None
    if count_threshold is not None:
        verdict = "consistent_with_power" if n_rep >= count_threshold else "below_power"
    return {
        "n_candidates": len(results), "n_testable": len(testable), "n_replicated": n_rep,
        "n_not_replicated": count("not_replicated", results),
        "n_not_testable": count("not_testable", results),
        "n_full_rule_same_direction": sum(bool(r.get("full_rule_same_direction")) for r in testable),
        "n_dev_in_phase_b_ci": sum(bool(r.get("dev_in_phase_b_ci")) for r in testable),
        "n_phase_b_in_dev_ci": sum(bool(r.get("phase_b_in_dev_ci")) for r in testable),
        "families": families,
        "families_replicated": sum(v["verdict"] == "replicated" for v in families.values()),
        "families_not_replicated": sum(v["verdict"] == "not_replicated" for v in families.values()),
        "families_not_testable": sum(v["verdict"] == "not_testable" for v in families.values()),
        "count_threshold": count_threshold,
        "expected_count_plug_in": EXPECTED_COUNT_PLUG_IN,
        "expected_count_half_log_lift": EXPECTED_COUNT_HALF,
        "study_verdict": verdict,
        "combat": {"candidates": len(combat), "replicated": count("replicated", combat),
                   "not_testable": count("not_testable", combat),
                   "expected_plug_in": COMBAT_EXPECTED_PLUG_IN,
                   "expected_half_log_lift": COMBAT_EXPECTED_HALF, "verdict": None},
    }


def render_report(payload: dict[str, Any]) -> str:
    """Fill the protocol's reporting template (section 9).

    Args:
        payload: ``{"header", "summary", "evaluation", "dual_mode"}``.

    Returns:
        Markdown text.
    """
    h, s, ev = payload["header"], payload["summary"], payload["evaluation"]

    def fmt(x: float | None, nd: int = 2) -> str:
        """Format a number or n/a."""
        return "n/a" if x is None else f"{x:.{nd}f}"

    lines = [f"# Phase B report, rtwp_turn_2026 ({h['mode']})", ""]
    if h["mode"] != "confirmatory":
        lines += ["DRY RUN on development reviews. These are not phase B results.", ""]
    lines += [
        f"- Commit: {h.get('commit')}, required tag: {h.get('tag')}",
        f"- Frame SHA-256: {h.get('frame_sha256')}",
        f"- Reviews read: {h.get('n_reviews')}, kept: {h.get('n_kept')}",
        f"- Games eligible in phase B: {len(ev['eligible'])}",
        f"- Audit precision: " + ", ".join(
            f"{a} {p['relevant']}/{p['n']} {fmt(p['precision'], 3)} "
            f"({fmt(p['ci_low'], 3)} to {fmt(p['ci_high'], 3)}, {p['label_version']}, "
            f"{'passes' if p['passes'] else 'fails'})"
            for a, p in h.get("aspect_precision", {}).items()),
        f"- Labelers: {h.get('labelers')}",
        "",
        "## Counts",
        "",
        f"- Candidates: {s['n_candidates']}",
        f"- Testable: {s['n_testable']}",
        f"- Replicated (primary criterion): {s['n_replicated']}",
        f"- Not replicated: {s['n_not_replicated']}",
        f"- Not testable: {s['n_not_testable']}",
        f"- Full finding rule met in the same direction: {s['n_full_rule_same_direction']}",
        f"- Development lift inside the phase B interval: {s['n_dev_in_phase_b_ci']}",
        f"- Families replicated / not replicated / not testable: {s['families_replicated']} / "
        f"{s['families_not_replicated']} / {s['families_not_testable']}",
        f"- Expected replicated count: {fmt(s['expected_count_plug_in'], 1)} (plug-in), "
        f"{fmt(s['expected_count_half_log_lift'], 1)} (half log lift)",
        f"- Pass line: {s['count_threshold']}, study verdict: {s['study_verdict']}",
        f"- Combat-model candidates replicated: {s['combat']['replicated']} of "
        f"{s['combat']['candidates']} (expected {fmt(s['combat']['expected_plug_in'], 1)} "
        f"plug-in, {fmt(s['combat']['expected_half_log_lift'], 1)} half log lift, no verdict)",
        "",
        "## Families",
        "",
        "| Family | Candidates | Testable | Replicated | Not replicated | Not testable | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in s["families"].values():
        lines.append(f"| {v['label']} | {v['candidates']} | {v['testable']} | {v['replicated']} | "
                     f"{v['not_replicated']} | {v['not_testable']} | {v['verdict']} |")
    lines += [
        "", "## Candidates", "",
        "| Candidate | Side | Development lift (95% CI) | Games: scope / dev eligible / phase B | "
        "Phase B lift (95% CI) | Status | Full rule | Stable |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ev["candidates"]:
        d_lo, d_hi = r["dev_ci"]
        pb = "n/a"
        if r["status"] != "not_testable":
            pb = f"{fmt(r['lift'])} ({fmt(r['ci'][0])} to {fmt(r['ci'][1])})"
        lines.append(
            f"| {r['key']} | {r['direction']} | {r['dev_lift']:.2f} ({d_lo:.2f} to {d_hi:.2f}) | "
            f"{r['games_in_scope']} / {r['dev_eligible_games']} / {len(r['phase_b_games'])} | "
            f"{pb} | {r['status']} | {r.get('full_rule_same_direction', 'n/a')} | "
            f"{r.get('logo_stable', 'n/a')} |")
    lines += ["", "## Ranking findings (no verdict)", "",
              "| Scope | Top complaint aspect | Share (95% CI) | Second | Second CI | "
              "Rule met | Development ranking finding |", "|---|---|---|---|---|---|---|"]
    for r in ev["ranking"]:
        sc = r["second_ci"] or [None, None]
        lines.append(f"| {r['scope_type']}:{r['scope_id']} | {r['theme']} | {fmt(r['share'], 3)} "
                     f"({fmt(r['ci'][0], 3)} to {fmt(r['ci'][1], 3)}) | {r['second_theme']} | "
                     f"{fmt(sc[0], 3)} to {fmt(sc[1], 3)} | {r['status'] == 'finding'} | "
                     f"{r['development_ranking_finding']} |")
    lines += ["", "## Combat-model rows that are not candidates (descriptive)", "",
              "| Row | Tagged | Complaint share | Praise share | Lift (95% CI) | Stable |",
              "|---|---|---|---|---|---|"]
    for r in ev["combat_rows"]:
        lines.append(f"| {r['key']} | {r['tagged']} | {fmt(r['complaint_share'], 3)} | "
                     f"{fmt(r['praise_share'], 3)} | {fmt(r['lift'])} ({fmt(r['ci'][0])} to "
                     f"{fmt(r['ci'][1])}) | {r['logo_stable']} |")
    lines += ["", "## Dual-mode views (descriptive)", ""]
    for app_id, g in (payload.get("dual_mode") or {}).items():
        mm = g["mode_mentions"]
        cells = []
        for grp in ("tb_only", "rt_only", "both", "neither"):
            rate = mm[grp]["recommend"]
            cells.append("n/a" if rate["share"] is None
                         else f"{100 * rate['share']:.1f}% ({rate['k']}/{rate['n']})")
        lines.append(f"- {g['name']}: TB only {cells[0]}, RTwP only {cells[1]}, both {cells[2]}, "
                     f"neither {cells[3]}")
    lines += ["", "## Deviations", "", f"- {h.get('deviations', 'none')}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def gate_problems(repo: Path = PROJECT_ROOT, tag: str = PROTOCOL_TAG,
                  protocol: Path = PROTOCOL_PATH,
                  results: Path = CONFIRMATORY_DIR / "results.json") -> list[str]:
    """List the git and read-once reasons the confirmatory read may not run.

    Args:
        repo: Repository root.
        tag: Required tag at HEAD.
        protocol: Protocol file that must be tracked.
        results: Results file that must not exist.

    Returns:
        Problems; empty when the read may run.
    """
    problems: list[str] = []
    status = _git(["status", "--porcelain"], repo)
    if status.returncode != 0:
        return [f"git status failed: {status.stderr.strip()}"]
    if status.stdout.strip():
        problems.append("working tree is not clean (git status --porcelain is not empty)")
    if tag not in _git(["tag", "--points-at", "HEAD"], repo).stdout.split():
        problems.append(f"HEAD does not carry the tag {tag}")
    rel = protocol.resolve().relative_to(repo.resolve()) if protocol.resolve().is_relative_to(
        repo.resolve()) else protocol
    if _git(["ls-files", "--error-unmatch", str(rel)], repo).returncode != 0:
        problems.append(f"{rel} is not tracked")
    if results.exists():
        problems.append(f"{results} exists: the confirmatory read already ran")
    return problems


def pin_problems(pins: dict[str, tuple[Path, str | None]] | None = None) -> list[str]:
    """List the pinned files whose SHA-256 does not match.

    Args:
        pins: ``{name: (path, expected sha256)}`` (default ``PINS``).

    Returns:
        Problems; empty when every pin matches.
    """
    problems = []
    for name, (path, want) in (PINS if pins is None else pins).items():
        if want is None:
            problems.append(f"{name} hash is not pinned in src/rtwp_read.py")
        elif not path.exists():
            problems.append(f"{name} file {path} is missing")
        elif file_sha256(path) != want:
            problems.append(f"{name} file {path} does not match its pinned SHA-256")
    return problems


def _head() -> str:
    """Return the HEAD commit SHA, or ``unknown``."""
    r = _git(["rev-parse", "HEAD"], PROJECT_ROOT)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _development(path: Path = CANDIDATES_PATH) -> tuple[dict[str, dict[str, Any]], set[int]]:
    """Development lifts of the candidates and the development-eligible games.

    Args:
        path: Candidate export.

    Returns:
        ``({candidate key: finding}, eligible app IDs)``.
    """
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    found = {f"{f['lens']}:{f['theme']}@{f['scope_type']}:{f['scope_id']}": f
             for f in metrics["findings"]}
    return found, set(metrics["eligible_app_ids"])


def _topic_ids(study: Study) -> list[int]:
    """Non-outlier topic IDs of the dev-fit model, from the analysis database."""
    conn = _read_only(study.analysis_view().seed_db)
    try:
        return [int(r[0]) for r in conn.execute(
            "SELECT topic_id FROM topics WHERE topic_id != -1 ORDER BY topic_id")]
    finally:
        conn.close()


def _run(
    study: Study,
    rows: Sequence[tuple[str, int, int | None, str]],
    recommended: dict[str, bool],
    out_dir: Path,
    header: dict[str, Any],
    instruments: Instruments,
    n_boot: int,
    min_kept: int,
    min_not: int,
    count_threshold: int | None,
) -> dict[str, Any]:
    """Process reviews, evaluate, write results.json and report.md.

    Args:
        study: The study.
        rows: ``(steam_review_id, app_id, created, raw_text)``.
        recommended: Recommend flag per review ID.
        out_dir: Output directory under ``STUDY_DIR``.
        header: Run facts.
        instruments: ``(clean_and_filter, tag, assign_topics, vader)``.
        n_boot: Bootstrap resamples.
        min_kept: Eligibility, kept reviews.
        min_not: Eligibility, not-recommended reviews.
        count_threshold: Pass line, or None for no verdict.

    Returns:
        The payload written to ``results.json``.
    """
    from src.aspects import load_lexicon
    from src.dual_mode import DUAL_MODE_TAG, ModeReview, compute, load_events

    if not out_dir.resolve().is_relative_to(STUDY_DIR.resolve()):
        raise ValueError(f"Refusing to write {out_dir}: outside {STUDY_DIR}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    precision = study_precision(study, lexicon, required=True)
    passing = {a for a, p in precision.items() if p["passes"]}
    processed = process_reviews(rows, instruments)
    keys = theme_keys(lexicon, _topic_ids(study))
    games = build_games(processed, recommended, keys)
    development, dev_eligible = _development()
    evaluation = evaluate(study, games, keys, passing, development, dev_eligible,
                          n_boot=n_boot, min_kept=min_kept, min_not=min_not)
    summary = summarize(evaluation["candidates"], count_threshold)
    dual_ids = {a.app_id for a in study.apps if DUAL_MODE_TAG in a.tags}
    mode_reviews = [ModeReview(p.app_id, recommended[p.steam_review_id], p.created, p.aspects)
                    for p in processed if p.kept and p.app_id in dual_ids]
    dual = compute(study, mode_reviews, load_events(STUDY_PATH), [a.name for a in lexicon.aspects])
    header = {**header, "n_reviews": len(rows), "n_kept": sum(p.kept for p in processed),
              "n_boot": n_boot, "boot_seed": BOOT_SEED,
              "eligibility": {"min_kept": min_kept, "min_not_rec": min_not},
              "dev_eligible": sorted(dev_eligible), "aspect_precision": precision,
              "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    payload = {"header": header, "summary": summary, "evaluation": evaluation, "dual_mode": dual}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, default=_num))
    (out_dir / "report.md").write_text(render_report(payload))
    return payload


def fetch_frame_reviews(
    study: Study, frame: Sequence[tuple[int, str]]
) -> tuple[list[tuple[str, int, int | None, str]], dict[str, bool]]:
    """Read the frame reviews' text, creation time and recommend flag. After the gate only.

    Args:
        study: The study.
        frame: ``(app_id, steam_review_id)`` of the frame.

    Returns:
        ``(rows, recommended)`` for the frame reviews only.

    Raises:
        RuntimeError: If a frame review is missing or sits in another game.
    """
    want = {rid: app for app, rid in frame}
    rows: list[tuple[str, int, int | None, str]] = []
    recommended: dict[str, bool] = {}
    ids = list(want)
    conn = _read_only(study.full_db)
    try:
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for rid, app, created, rec, text in conn.execute(
                "SELECT steam_review_id, app_id, timestamp_created, recommended, review_text "
                f"FROM documents WHERE steam_review_id IN ({marks})", chunk,
            ):
                if int(app) != want[rid]:
                    raise RuntimeError(f"Frame review {rid} belongs to app {app}, not {want[rid]}.")
                rows.append((rid, int(app), created, text))
                recommended[rid] = bool(rec)
    finally:
        conn.close()
    if len(rows) != len(want):
        raise RuntimeError(f"{len(want) - len(rows)} frame reviews are missing from the full database.")
    return rows, recommended


def confirmatory_read(study: Study, out_dir: Path = CONFIRMATORY_DIR,
                      instruments: Instruments | None = None) -> dict[str, Any]:
    """The single confirmatory read. Refuses unless every gate passes.

    Args:
        study: The study.
        out_dir: Output directory.
        instruments: Replacement instruments (tests only).

    Returns:
        The written payload.

    Raises:
        RuntimeError: If the gate, a pin or the frame check fails.
    """
    problems = gate_problems(results=out_dir / "results.json") + pin_problems()
    if problems:
        raise RuntimeError("Confirmatory read refused: " + "; ".join(problems))
    frame = load_frame(FRAME_PATH)
    if len(frame) != FRAME_TOTAL:
        raise RuntimeError(f"The frame has {len(frame)} reviews, not {FRAME_TOTAL}.")
    if any(dev_test_split(rid) != "test" for _, rid in frame):
        raise RuntimeError("The frame holds a review outside the test half.")
    rows, recommended = fetch_frame_reviews(study, frame)
    header = {"mode": "confirmatory", "commit": _head(), "tag": PROTOCOL_TAG,
              "frame_sha256": PINS["frame"][1], "labelers": "v2 and v3 audits: two separate "
              "fresh model labeler instances", "deviations": "none"}
    return _run(study, rows, recommended, out_dir, header, instruments or frozen_instruments(study),
                N_BOOT, MIN_KEPT_FOR_ELIGIBLE, MIN_NOT_REC_FOR_ELIGIBLE, COUNT_THRESHOLD)


def dry_run(study: Study, out_dir: Path = DRY_RUN_DIR, per_game: int = DRY_RUN_PER_GAME,
            n_boot: int = DRY_RUN_BOOT, instruments: Instruments | None = None) -> dict[str, Any]:
    """Run the read path on development-half reviews of the full database.

    Takes up to ``per_game`` development-half reviews per game, which phase A
    already read, with lowered eligibility so every step runs. Reports the
    gate and pin status without enforcing them. Never reads a test-half review.

    Args:
        study: The study.
        out_dir: Output directory.
        per_game: Development reviews per game.
        n_boot: Bootstrap resamples.
        instruments: Replacement instruments (tests).

    Returns:
        The written payload.
    """
    conn = _read_only(study.full_db)
    try:
        ids = sorted((int(a), rid) for a, rid in
                     conn.execute("SELECT app_id, steam_review_id FROM documents"))
    finally:
        conn.close()
    taken: dict[int, int] = {}
    frame = []
    for app, rid in ids:
        if dev_test_split(rid) == "dev" and taken.get(app, 0) < per_game:
            taken[app] = taken.get(app, 0) + 1
            frame.append((app, rid))
    rows, recommended = fetch_frame_reviews(study, frame)
    header = {"mode": "dry_run", "commit": _head(), "tag": PROTOCOL_TAG, "frame_sha256": None,
              "per_game": per_game, "gate_problems_now": gate_problems(),
              "pin_problems_now": pin_problems(), "labelers": "n/a", "deviations": "dry run"}
    return _run(study, rows, recommended, out_dir, header, instruments or frozen_instruments(study),
                n_boot, DRY_RUN_MIN_KEPT, DRY_RUN_MIN_NOT_REC, None)


def main() -> None:
    """CLI entry point: ``frame``, ``agreement``, ``gate``, ``dry-run`` or ``read``."""
    parser = argparse.ArgumentParser(description="Phase B confirmatory read of rtwp_turn_2026.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("frame", help="Write the frame file (review IDs only) and print its hash.")
    sub.add_parser("agreement", help="Measure topic transform agreement on the development half.")
    sub.add_parser("gate", help="Print whether the confirmatory read may run.")
    dry = sub.add_parser("dry-run", help="Run the read path on development reviews.")
    dry.add_argument("--per-game", type=int, default=DRY_RUN_PER_GAME)
    dry.add_argument("--n-boot", type=int, default=DRY_RUN_BOOT)
    sub.add_parser("read", help="The single confirmatory read (gated).")
    args = parser.parse_args()
    study = load_study(STUDY_PATH)
    try:
        if args.command == "frame":
            print(json.dumps(write_frame(study), indent=2))
        elif args.command == "agreement":
            res = topic_transform_agreement(study)
            out = STUDY_DIR / "exploratory" / "topic_transform_agreement.json"
            out.write_text(json.dumps(res, indent=2))
            print(json.dumps({k: res[k]["agreement"] for k in ("whole_set", "two_halves")}))
        elif args.command == "gate":
            problems = gate_problems() + pin_problems()
            print("gate open" if not problems else "gate closed:\n- " + "\n- ".join(problems))
        elif args.command == "dry-run":
            p = dry_run(study, per_game=args.per_game, n_boot=args.n_boot)
            s = p["summary"]
            print(json.dumps({k: s[k] for k in ("n_candidates", "n_testable", "n_replicated",
                                                "n_not_replicated", "n_not_testable")}))
        elif args.command == "read":
            confirmatory_read(study)
            print((CONFIRMATORY_DIR / "report.md").read_text())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
