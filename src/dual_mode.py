"""Descriptive views for games that ship two combat modes.

A study marks such games with the ``dual_mode`` tag and lists, in
``[[mode_events]]``, the date a second mode reached the default Steam branch
for games that added it after launch. Two views are computed per game, on one
half of the hash split only (``stats.dev_test_split``), so an exploratory run
never reads a recommend flag of the held-out half:

1. **Mode mentions.** Each kept English review is put in one group by the
   mode aspects its sentences carry: the turn-based mode only, the
   real-time mode only, both, or neither. Each group gets its size and its
   recommend share with a Wilson interval. This is the within-game comparison.
2. **Before and after.** For a game with a mode event, reviews created before
   the event date against reviews created on or after it: recommend share,
   the share of reviews in each mode group, and each aspect's share among
   recommended and not-recommended reviews. Time is confounded with every
   other change to the game and its audience, so this view is descriptive.

Reviews come from the seed (``--source seed``: the stored preprocessing and
aspect tags) or from the full database (``--source full``: every review of the
dual-mode games, run through the seed's cleaning, length and token checks,
language filter, sentencizer and lexicon). Results go to
``outputs/<study>/exploratory/`` as JSON and Markdown; nothing is written to
the study databases.
"""

import argparse
import json
import math
import sqlite3
import sys
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.stats import dev_test_split, wilson_interval
from src.study import STATS_SPLITS, Study, load_study, parse_timestamp
from src.utils import MIN_TOKEN_COUNT, OUTPUTS_DIR, get_logger

logger = get_logger(__name__)

DUAL_MODE_TAG: str = "dual_mode"
"""App tag that marks a game shipping both combat modes."""

TB_ASPECT: str = "mode_turn_based"
"""Lexicon aspect for explicit talk about turn-based combat or a turn-based mode."""

RT_ASPECT: str = "mode_rtwp"
"""Lexicon aspect for explicit talk about real-time (with pause) combat."""

GROUPS: tuple[str, ...] = ("tb_only", "rt_only", "both", "neither")
"""Mode-mention groups in report order."""


@dataclass(frozen=True)
class ModeEvent:
    """The date a second combat mode reached a game's default Steam branch.

    Attributes:
        app_id: Steam application ID.
        timestamp: Event time in Unix seconds (UTC midnight of the date).
        day: The event date as written in the study file.
        source: First-party URL that states the date.
    """

    app_id: int
    timestamp: int
    day: str
    source: str


@dataclass(frozen=True)
class ModeReview:
    """One kept English review of a dual-mode game.

    Attributes:
        app_id: Steam application ID.
        recommended: Recommend flag.
        created: ``timestamp_created`` in Unix seconds, or None.
        aspects: Aspects tagged in any of its sentences.
    """

    app_id: int
    recommended: bool
    created: int | None
    aspects: frozenset[str]


def load_events(study_path: Path) -> dict[int, ModeEvent]:
    """Read ``[[mode_events]]`` from a study file.

    Args:
        study_path: Study TOML file.

    Returns:
        ``{app_id: ModeEvent}``; empty when the file lists none.

    Raises:
        ValueError: If an event names an app twice or has no date.
    """
    with open(study_path, "rb") as f:
        raw = tomllib.load(f)
    events: dict[int, ModeEvent] = {}
    for item in raw.get("mode_events", []):
        app_id = int(item["app_id"])
        if app_id in events:
            raise ValueError(f"mode_events lists app {app_id} twice.")
        ts = parse_timestamp(item.get("date"))
        if ts is None:
            raise ValueError(f"mode_events entry for app {app_id} has no date.")
        events[app_id] = ModeEvent(app_id, ts, str(item["date"]), str(item.get("source", "")))
    return events


def mode_group(aspects: Iterable[str], tb: str = TB_ASPECT, rt: str = RT_ASPECT) -> str:
    """Return the mode-mention group of a review.

    Args:
        aspects: Aspects tagged in the review.
        tb: Turn-based mode aspect name.
        rt: Real-time mode aspect name.

    Returns:
        One of ``GROUPS``.
    """
    found = set(aspects)
    has_tb, has_rt = tb in found, rt in found
    if has_tb and has_rt:
        return "both"
    if has_tb:
        return "tb_only"
    if has_rt:
        return "rt_only"
    return "neither"


def _num(x: float) -> float | None:
    """Return a finite float or None.

    Args:
        x: A float.

    Returns:
        ``x``, or None for nan and inf.
    """
    return float(x) if x is not None and math.isfinite(x) else None


def _rate(k: int, n: int) -> dict[str, Any]:
    """A proportion with its Wilson interval.

    Args:
        k: Successes.
        n: Trials.

    Returns:
        ``{k, n, share, ci}``; share and interval are None when ``n`` is 0.
    """
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "share": _num(k / n) if n else None, "ci": [_num(lo), _num(hi)]}


def mode_mention_table(reviews: Sequence[ModeReview]) -> dict[str, dict[str, Any]]:
    """Recommend share by mode-mention group, for one game's reviews.

    Args:
        reviews: Kept reviews of one game.

    Returns:
        ``{group: {"reviews": n, "recommend": rate}}`` for every group in
        ``GROUPS``, where ``rate`` is ``_rate(recommended, n)``.
    """
    out: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        members = [r for r in reviews if mode_group(r.aspects) == group]
        out[group] = {
            "reviews": len(members),
            "recommend": _rate(sum(r.recommended for r in members), len(members)),
        }
    return out


def period_summary(reviews: Sequence[ModeReview], aspects: Sequence[str]) -> dict[str, Any]:
    """Recommend share, mode-group shares and aspect shares for one period.

    Args:
        reviews: Kept reviews of one game in the period.
        aspects: Aspect names to report.

    Returns:
        ``{reviews, first_created, last_created, recommend, mode_groups,
        aspects}``. ``mode_groups`` maps each group to its share of the
        period's reviews. ``aspects`` maps each aspect to its share of all
        reviews (``share``), of not-recommended reviews (``complaint``) and of
        recommended reviews (``praise``), each with a Wilson interval.
    """
    n = len(reviews)
    rec = [r for r in reviews if r.recommended]
    notrec = [r for r in reviews if not r.recommended]
    stamps = [r.created for r in reviews if r.created is not None]

    def day(ts: int | None) -> str | None:
        """Format a Unix time as a UTC date."""
        return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).date().isoformat()

    return {
        "reviews": n,
        "first_created": day(min(stamps)) if stamps else None,
        "last_created": day(max(stamps)) if stamps else None,
        "recommend": _rate(len(rec), n),
        "mode_groups": {
            g: _rate(sum(mode_group(r.aspects) == g for r in reviews), n) for g in GROUPS
        },
        "aspects": {
            a: {
                "share": _rate(sum(a in r.aspects for r in reviews), n),
                "complaint": _rate(sum(a in r.aspects for r in notrec), len(notrec)),
                "praise": _rate(sum(a in r.aspects for r in rec), len(rec)),
            }
            for a in aspects
        },
    }


def before_after(
    reviews: Sequence[ModeReview], event: ModeEvent, aspects: Sequence[str]
) -> dict[str, Any]:
    """Compare one game's reviews created before a mode event with those after.

    A review created at or after the event timestamp counts as after. Reviews
    without a creation time are left out and counted.

    Args:
        reviews: Kept reviews of one game.
        event: The game's mode event.
        aspects: Aspect names to report.

    Returns:
        ``{event_date, source, undated, before, after}`` with ``period_summary``
        for each period. The view is descriptive (time is confounded).
    """
    dated = [r for r in reviews if r.created is not None]
    return {
        "event_date": event.day,
        "source": event.source,
        "status": "descriptive",
        "undated": len(reviews) - len(dated),
        "before": period_summary([r for r in dated if r.created < event.timestamp], aspects),
        "after": period_summary([r for r in dated if r.created >= event.timestamp], aspects),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _split_rows(
    conn: sqlite3.Connection, app_ids: Sequence[int], split: str
) -> list[tuple[int, str, int, int | None]]:
    """Return ``(document_id, steam_review_id, app_id, created)`` for one split.

    The recommend flag is not read here, so the caller reads it only for
    documents in the split.

    Args:
        conn: Seed or full database connection.
        app_ids: Apps to include.
        split: ``dev`` or ``test``.

    Returns:
        Rows in the split, ordered by document id.
    """
    rows: list[tuple[int, str, int, int | None]] = []
    for app_id in app_ids:
        rows += conn.execute(
            "SELECT id, steam_review_id, app_id, timestamp_created FROM documents WHERE app_id = ?",
            (app_id,),
        ).fetchall()
    return sorted(r for r in rows if dev_test_split(r[1]) == split)


def _flags(conn: sqlite3.Connection, doc_ids: Sequence[int]) -> dict[int, bool]:
    """Read the recommend flag of the given documents only.

    Args:
        conn: Database connection.
        doc_ids: Document ids already restricted to one split.

    Returns:
        ``{document_id: recommended}``.
    """
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS dual_mode_ids (id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM temp.dual_mode_ids")
    conn.executemany("INSERT INTO temp.dual_mode_ids VALUES (?)", [(i,) for i in doc_ids])
    return {
        doc_id: bool(rec) for doc_id, rec in conn.execute(
            "SELECT id, recommended FROM documents WHERE id IN (SELECT id FROM temp.dual_mode_ids)"
        )
    }


def seed_reviews(study: Study, app_ids: Sequence[int], split: str, version: str) -> list[ModeReview]:
    """Kept English reviews of the given apps in one split of the seed.

    Args:
        study: The study (its aspect stage must have run).
        app_ids: Apps to include.
        split: ``dev`` or ``test``.
        version: Lexicon version of the stored aspect tags.

    Returns:
        Reviews with their aspect sets.
    """
    conn = sqlite3.connect(f"{study.seed_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        kept = {
            r[0] for r in conn.execute(
                "SELECT document_id FROM processed_documents WHERE token_count >= ?",
                (MIN_TOKEN_COUNT,),
            )
        }
        rows = [r for r in _split_rows(conn, app_ids, split) if r[0] in kept]
        flags = _flags(conn, [r[0] for r in rows])
        tags: dict[int, set[str]] = {}
        for doc_id, aspect in conn.execute(
            "SELECT document_id, aspect FROM aspect_mentions WHERE lexicon_version = ?", (version,)
        ):
            tags.setdefault(doc_id, set()).add(aspect)
    finally:
        conn.close()
    return [ModeReview(app, flags[doc], ts, frozenset(tags.get(doc, set()))) for doc, _, app, ts in rows]


def full_reviews(study: Study, app_ids: Sequence[int], split: str) -> list[ModeReview]:
    """Kept English reviews of the given apps in one split of the full database.

    Applies the seed's rules: cleaning, length and token checks, the language
    filter when the study uses it, the sentencizer and the study lexicon.

    Args:
        study: The study (its ``full_db`` must exist).
        app_ids: Apps to include.
        split: ``dev`` or ``test``.

    Returns:
        Reviews with their aspect sets.
    """
    from src.aspects import SentenceSplitter, load_lexicon, tag_sentences
    from src.preprocessing import LanguageFilter, Preprocessor, apply_language_filter

    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    conn = sqlite3.connect(f"{study.full_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = _split_rows(conn, app_ids, split)
        flags = _flags(conn, [r[0] for r in rows])
        texts = dict(conn.execute(
            "SELECT id, review_text FROM documents WHERE id IN (SELECT id FROM temp.dual_mode_ids)"
        ).fetchall())
    finally:
        conn.close()
    pre = Preprocessor()
    results = pre.process_batch([texts[r[0]] for r in rows])
    if study.language_filter:
        apply_language_filter(results, LanguageFilter())
    kept = [(row, res["cleaned_text"]) for row, res in zip(rows, results) if not res["dropped"]]
    out: list[ModeReview] = []
    splitter = SentenceSplitter()
    for ((doc, _rid, app, ts), _text), spans in zip(kept, splitter.split([t for _, t in kept])):
        found = frozenset(m.aspect for m in tag_sentences([s[2] for s in spans], lexicon))
        out.append(ModeReview(app, flags[doc], ts, found))
    logger.info("Full DB %s split: %d reviews read, %d kept.", split, len(rows), len(out))
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def compute(
    study: Study, reviews: Sequence[ModeReview], events: dict[int, ModeEvent], aspects: Sequence[str]
) -> dict[str, Any]:
    """Build both views for every dual-mode game.

    Args:
        study: The study.
        reviews: Kept reviews of the dual-mode games in one split.
        events: Mode events by app ID.
        aspects: Aspect names for the before-and-after view.

    Returns:
        ``{app_id: {name, mode_mentions, before_after}}`` as strings keys;
        ``before_after`` is None for a game without an event.
    """
    out: dict[str, Any] = {}
    for app in study.apps:
        if DUAL_MODE_TAG not in app.tags:
            continue
        rs = [r for r in reviews if r.app_id == app.app_id]
        event = events.get(app.app_id)
        out[str(app.app_id)] = {
            "name": app.name,
            "tags": list(app.tags),
            "kept_reviews": len(rs),
            "mode_mentions": mode_mention_table(rs),
            "before_after": before_after(rs, event, aspects) if event else None,
        }
    return out


def render_markdown(result: dict[str, Any]) -> str:
    """Render the views as a short Markdown report.

    Args:
        result: Output of ``run`` (with ``games``).

    Returns:
        Markdown text.
    """

    def pct(rate: dict[str, Any]) -> str:
        """Format a rate as 'share% (lo to hi), k/n'."""
        if rate["share"] is None:
            return f"n/a, 0/{rate['n']}"
        lo, hi = rate["ci"]
        return f"{100 * rate['share']:.1f}% ({100 * lo:.1f} to {100 * hi:.1f}), {rate['k']}/{rate['n']}"

    lines = [
        f"# Dual-mode views: {result['study']}",
        "",
        f"Split: {result['split']}. Source: {result['source']}. Lexicon: {result['lexicon_version']}. "
        "Every view is descriptive.",
        "",
        "## Recommend share by mode mention",
        "",
        "| Game | Kept | TB only | RTwP only | Both | Neither |",
        "|---|---|---|---|---|---|",
    ]
    for g in result["games"].values():
        mm = g["mode_mentions"]
        lines.append(
            f"| {g['name']} | {g['kept_reviews']} | "
            + " | ".join(pct(mm[k]["recommend"]) for k in GROUPS) + " |"
        )
    lines += ["", "## Before and after the turn-based mode reached the default branch", ""]
    for g in result["games"].values():
        ba = g["before_after"]
        if not ba:
            continue
        lines += [
            f"### {g['name']} (event {ba['event_date']})",
            "",
            "| Period | Reviews | Dates | Recommend | Mentions TB mode | Mentions RTwP |",
            "|---|---|---|---|---|---|",
        ]
        for period in ("before", "after"):
            p = ba[period]
            tb = p["aspects"].get(TB_ASPECT, {}).get("share")
            rt = p["aspects"].get(RT_ASPECT, {}).get("share")
            lines.append(
                f"| {period} | {p['reviews']} | {p['first_created']} to {p['last_created']} | "
                f"{pct(p['recommend'])} | {pct(tb) if tb else 'n/a'} | {pct(rt) if rt else 'n/a'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run(study_path: Path, source: str = "seed", split: str = "dev", out_dir: Path | None = None) -> Path:
    """Compute the dual-mode views and write JSON and Markdown.

    Args:
        study_path: Study TOML file.
        source: ``seed`` or ``full``.
        split: ``dev`` or ``test``.
        out_dir: Output directory (default ``outputs/<study>/exploratory``).

    Returns:
        Path of the JSON file.

    Raises:
        ValueError: For an unknown source or split, or a study without
            dual-mode games or a lexicon.
    """
    from src.aspects import load_lexicon

    if split not in STATS_SPLITS:
        raise ValueError(f"split must be one of {sorted(STATS_SPLITS)}.")
    if source not in ("seed", "full"):
        raise ValueError("source must be 'seed' or 'full'.")
    study = load_study(study_path)
    if study.aspect_lexicon is None:
        raise ValueError(f"Study {study.name} has no aspect lexicon.")
    app_ids = [a.app_id for a in study.apps if DUAL_MODE_TAG in a.tags]
    if not app_ids:
        raise ValueError(f"Study {study.name} has no app tagged {DUAL_MODE_TAG}.")
    lexicon = load_lexicon(study.aspect_lexicon)
    events = load_events(study_path)
    reviews = (
        seed_reviews(study, app_ids, split, lexicon.version) if source == "seed"
        else full_reviews(study, app_ids, split)
    )
    result = {
        "study": study.name,
        "split": split,
        "source": source,
        "lexicon_version": lexicon.version,
        "computed_at": date.today().isoformat(),
        "games": compute(study, reviews, events, [a.name for a in lexicon.aspects]),
    }
    out = out_dir or OUTPUTS_DIR / study.name / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"dual_mode_{source}_{split}"
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text(render_markdown(result), encoding="utf-8")
    logger.info("Wrote %s", json_path)
    return json_path


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: ``--study PATH [--source seed|full] [--split dev|test]``.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description="Descriptive views for dual-mode games.")
    parser.add_argument("--study", type=Path, required=True, help="Study TOML file.")
    parser.add_argument("--source", default="seed", choices=["seed", "full"])
    parser.add_argument("--split", default="dev", choices=sorted(STATS_SPLITS))
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    args = parser.parse_args(argv)
    try:
        run(args.study, args.source, args.split, args.out)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
