"""Run every stage of a natural-sampling study in order, idempotently.

Stages: ``pull`` (census pull, cached apps skipped), ``load`` (raw pages into
the full database), ``seed`` (uniform per-game sample), ``preprocess``,
``sentiment`` (VADER), ``topics`` (BERTopic), ``aspects`` (phrase lexicon)
and ``stats``. The first three run only with ``--full``; without it the
pipeline works on the existing seed.

A second run with unchanged inputs changes nothing. ``pull``, ``load``,
``preprocess``, ``sentiment`` and ``topics`` are idempotent by design: they
skip cached apps, ignore duplicate reviews and handle only rows that lack
output. Each is reported as ``no-op`` when the study's raw files and table
row counts are the same after it runs. ``seed``, ``aspects`` and ``stats`` rebuild their output, so each is
guarded by a fingerprint: a hash of the study settings, the files it reads
(lexicon, audit labels) and the full content of the tables it reads and
writes. The fingerprint is stored in a ``pipeline_state`` table after the
stage runs; when the stored value matches, the stage is skipped. Any change
to its inputs, or to its outputs made outside the pipeline, runs it again.

Transformer scoring and its validation stay separate commands
(``src/sentiment_transformer.py``), because adopting a model follows a
pre-registered comparison, not a pipeline default.

The April 2026 baseline (``method = "stratified_legacy"``) is not run by the
pipeline; its original commands reproduce it. Shipped studies set
``frozen = true``: the pipeline refuses them, and every writing stage refuses
their files, unless ``--unfreeze-i-know`` is passed.
"""

import argparse
import hashlib
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.study import (
    UNFREEZE_FLAG,
    UNFREEZE_HELP,
    FrozenStudyError,
    Study,
    guard_writable,
    load_study,
    study_files,
)
from src.utils import get_logger

logger = get_logger(__name__)

_STATE_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_state (
    stage       TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    updated_at  TEXT DEFAULT (datetime('now'))
);
"""

STATS_TABLES: tuple[str, ...] = (
    "metadata", "study_apps", "pull_runs", "documents", "review_language",
    "processed_documents", "nlp_results", "topics", "aspect_mentions",
    "sentiment_models", "sentiment_scores", "analysis_runs", "game_stats",
    "cohort_stats", "theme_stats",
)
"""Tables the stats stage reads or writes."""


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def table_signature(conn: sqlite3.Connection, table: str) -> str:
    """Hash the full content of a table, independent of physical row order.

    Args:
        conn: Open connection.
        table: Table name (a constant from this module, never user input).

    Returns:
        A hex digest, or ``"absent"`` when the table does not exist.
    """
    cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not cols:
        return "absent"
    order = ", ".join(str(i + 1) for i in range(len(cols)))
    h = hashlib.sha256()
    for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY {order}'):
        h.update(repr(row).encode("utf-8"))
    return h.hexdigest()


def file_signature(path: Path | None) -> str:
    """Hash a file's bytes.

    Args:
        path: File, or None.

    Returns:
        A hex digest, ``"none"`` for None, or ``"missing"``.
    """
    if path is None:
        return "none"
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(parts: Sequence[str]) -> str:
    """Combine signature parts into one digest.

    Args:
        parts: Strings to combine.

    Returns:
        A hex digest.
    """
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def stored_fingerprint(conn: sqlite3.Connection, stage: str) -> str | None:
    """Return the fingerprint stored for a stage, if any.

    Args:
        conn: Connection to the database that holds the stage's state.
        stage: Stage name.

    Returns:
        The stored digest or None.
    """
    has = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pipeline_state'"
    ).fetchone()
    if not has:
        return None
    row = conn.execute("SELECT fingerprint FROM pipeline_state WHERE stage = ?", (stage,)).fetchone()
    return row[0] if row else None


def store_fingerprint(conn: sqlite3.Connection, stage: str, value: str) -> None:
    """Upsert a stage fingerprint.

    Args:
        conn: Connection to the database that holds the stage's state.
        stage: Stage name.
        value: Digest to store.
    """
    conn.executescript(_STATE_SQL)
    conn.execute(
        "INSERT INTO pipeline_state (stage, fingerprint) VALUES (?, ?) "
        "ON CONFLICT(stage) DO UPDATE SET fingerprint = excluded.fingerprint, "
        "updated_at = datetime('now')",
        (stage, value),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def footprint(study: Study) -> str:
    """Cheap signature of what the idempotent stages can change.

    Covers the raw directory listing (name, size, modification time) and, for
    the full and seed databases, the row count of every table plus the
    ``sqlite_sequence`` counters.

    Args:
        study: The study.

    Returns:
        A hex digest.
    """
    parts: list[str] = []
    if study.raw_dir.is_dir():
        for f in sorted(study.raw_dir.iterdir()):
            st = f.stat()
            parts.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
    for path in (study.full_db, study.seed_db):
        if not path.exists():
            parts.append("no-db")
            continue
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
            for table in tables:
                n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                parts.append(f"{path.name}:{table}:{n}")
            if "sqlite_sequence" in tables:
                parts.append(repr(conn.execute(
                    "SELECT name, seq FROM sqlite_sequence ORDER BY name").fetchall()))
        finally:
            conn.close()
    return fingerprint(parts)


def _pull(study: Study, unfreeze: bool) -> None:
    from src.data_pull import pull_study

    pull_study(study, unfreeze=unfreeze)


def _load(study: Study, unfreeze: bool) -> None:
    from src.db_setup import load_study_full

    load_study_full(study, unfreeze=unfreeze)


def _seed_parts(study: Study) -> list[str]:
    """Fingerprint parts for the seed stage.

    Args:
        study: The study.

    Returns:
        Settings plus full and seed document signatures.
    """
    s = study.sampling
    parts = [repr((s.seed_cap_per_game, s.random_seed, s.window_start, s.window_end))]
    for path in (study.full_db, study.seed_db):
        if path.exists():
            conn = sqlite3.connect(path)
            try:
                parts += [table_signature(conn, t) for t in ("documents", "study_apps", "pull_runs")]
            finally:
                conn.close()
        else:
            parts.append("no-db")
    return parts


def _seed(study: Study, unfreeze: bool) -> None:
    from src.db_setup import build_natural_seed, vacuum

    full = sqlite3.connect(study.full_db)
    try:
        build_natural_seed(
            full, study.seed_db, study.sampling.seed_cap_per_game, study.sampling.random_seed,
            (study.sampling.window_start, study.sampling.window_end), unfreeze=unfreeze,
        )
    finally:
        full.close()
    vacuum(study.seed_db)


def _preprocess(study: Study, unfreeze: bool) -> None:
    from src.preprocessing import run

    run(mode="seed", study=study, unfreeze=unfreeze)


def _sentiment(study: Study, unfreeze: bool) -> None:
    from src.sentiment import run

    run(mode="seed", study=study, unfreeze=unfreeze)


def _topics(study: Study, unfreeze: bool) -> None:
    from src.topics import run

    run(mode="seed", study=study, min_topic_size=study.topic_min_size,
        nr_topics=study.topic_nr_topics, unfreeze=unfreeze)


def _seed_tables_parts(study: Study, tables: Sequence[str], files: Sequence[Path | None]) -> list[str]:
    """Fingerprint parts from seed tables, files and the study settings.

    Args:
        study: The study.
        tables: Seed tables the stage reads or writes.
        files: Files the stage reads.

    Returns:
        Signature strings.
    """
    parts = [repr(study), *(file_signature(f) for f in files)]
    conn = sqlite3.connect(study.seed_db)
    try:
        parts += [table_signature(conn, t) for t in tables]
    finally:
        conn.close()
    return parts


def _label_files(study: Study) -> list[Path | None]:
    """Return the aspect audit label file for the study's lexicon, if any.

    Args:
        study: The study.

    Returns:
        A one-item list (None when the study has no lexicon).
    """
    if study.aspect_lexicon is None:
        return [None]
    from src.aspects import audit_paths, load_lexicon

    return [audit_paths(study, load_lexicon(study.aspect_lexicon))[1]]


def _aspects(study: Study, unfreeze: bool) -> None:
    from src.aspects import run

    run(study, unfreeze=unfreeze)


def _stats(study: Study, unfreeze: bool) -> None:
    from src import stats

    stats.run(study, n_boot=stats.N_BOOT, unfreeze=unfreeze)


@dataclass(frozen=True)
class Stage:
    """One pipeline stage.

    Attributes:
        name: Stage name.
        run: Function that runs the stage for a study; the second argument
            is the unfreeze flag.
        needs_full: True for stages that run only with ``--full``.
        parts: For rebuild stages, a function returning fingerprint parts;
            None for stages that are idempotent by design.
        state_db: ``full`` or ``seed``: where a rebuild stage stores its state.
        applies: Predicate deciding whether the stage runs for this study.
    """

    name: str
    run: Callable[[Study, bool], None]
    needs_full: bool = False
    parts: Callable[[Study], list[str]] | None = None
    state_db: str = "seed"
    applies: Callable[[Study], bool] = lambda study: True


STAGES: list[Stage] = [
    Stage("pull", _pull, needs_full=True),
    Stage("load", _load, needs_full=True),
    Stage("seed", _seed, needs_full=True, parts=_seed_parts, state_db="full"),
    Stage("preprocess", _preprocess),
    Stage("sentiment", _sentiment),
    Stage("topics", _topics),
    Stage(
        "aspects", _aspects,
        parts=lambda s: _seed_tables_parts(
            s, ("processed_documents", "aspect_mentions"), [s.aspect_lexicon]
        ),
        applies=lambda s: s.aspect_lexicon is not None,
    ),
    Stage(
        "stats", _stats,
        parts=lambda s: _seed_tables_parts(
            s, STATS_TABLES, [s.aspect_lexicon, *_label_files(s)]
        ),
        applies=lambda s: s.aspect_lexicon is not None,
    ),
]
"""Stages in run order. Tests may replace entries (for example ``topics``)."""


def run(
    study: Study,
    full: bool = False,
    stages: Sequence[Stage] | None = None,
    unfreeze: bool = False,
) -> dict[str, str]:
    """Run the pipeline for a study.

    Args:
        study: A natural-sampling study.
        full: Also pull, load and rebuild the seed.
        stages: Stage list (default ``STAGES``).
        unfreeze: Allow running a frozen (shipped) study. Only for
            regenerating a shipped study on purpose.

    Returns:
        ``{stage: outcome}``: ``ran``; ``no-op`` for an idempotent stage that
        changed nothing; ``skipped`` for a rebuild stage whose fingerprint
        matched; ``not applicable``.

    Raises:
        ValueError: For the April baseline or a missing seed without ``--full``.
        FrozenStudyError: For a frozen study, or a study whose files belong
            to a frozen study, without ``unfreeze``.
    """
    if study.sampling.method != "natural":
        raise ValueError(
            f"{study.name} uses method={study.sampling.method!r}. The pipeline runs natural "
            "studies only; the April baseline keeps its original commands."
        )
    if study.frozen and not unfreeze:
        raise FrozenStudyError(
            f"Study {study.name} is frozen (frozen = true in its TOML): its databases are "
            f"shipped and the pipeline would rewrite them. To regenerate it on purpose, rerun "
            f"with {UNFREEZE_FLAG}."
        )
    guard_writable(study_files(study), unfreeze)
    if not full and not study.seed_db.exists():
        raise ValueError(f"No seed at {study.seed_db}. Run with --full first.")
    outcome: dict[str, str] = {}
    for stage in stages if stages is not None else STAGES:
        if (stage.needs_full and not full) or not stage.applies(study):
            outcome[stage.name] = "not applicable"
            continue
        if stage.parts is None:
            logger.info("Stage %s", stage.name)
            before = footprint(study)
            stage.run(study, unfreeze)
            outcome[stage.name] = "no-op" if footprint(study) == before else "ran"
            continue
        state_path = study.full_db if stage.state_db == "full" else study.seed_db
        before = fingerprint(stage.parts(study))
        if state_path.exists():
            conn = sqlite3.connect(state_path)
            try:
                unchanged = stored_fingerprint(conn, stage.name) == before
            finally:
                conn.close()
            if unchanged:
                logger.info("Stage %s: inputs unchanged, skipped.", stage.name)
                outcome[stage.name] = "skipped"
                continue
        logger.info("Stage %s", stage.name)
        stage.run(study, unfreeze)
        after = fingerprint(stage.parts(study))
        conn = sqlite3.connect(state_path)
        try:
            store_fingerprint(conn, stage.name, after)
        finally:
            conn.close()
        outcome[stage.name] = "ran"
    logger.info("Pipeline %s: %s", study.name, outcome)
    return outcome


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point: ``--study PATH [--full] [--unfreeze-i-know]``.

    Refusals (a frozen or April study, a missing seed) print ``error: ...``
    and exit with status 2.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description="Run every stage of a study in order.")
    parser.add_argument("--study", type=Path, required=True, help="Study TOML file.")
    parser.add_argument("--full", action="store_true",
                        help="Also pull from Steam (cached apps skipped), load and rebuild the seed.")
    parser.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    args = parser.parse_args(argv)
    try:
        run(load_study(args.study), full=args.full, unfreeze=args.unfreeze)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
