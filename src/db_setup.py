"""Database setup and loading. Creates SQLite schema, loads data in seed or full mode."""

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.study import (
    DEFAULT_STUDY_PATH,
    UNFREEZE_FLAG,
    UNFREEZE_HELP,
    Study,
    guard_writable,
    load_study,
)
from src.utils import (
    FULL_DB_PATH,
    GAME_CATALOG,
    GAME_IDS,
    RAW_DIR,
    SEED_DB_PATH,
    get_db_path,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS metadata (
    app_id       INTEGER PRIMARY KEY,
    app_name     TEXT NOT NULL,
    genre        TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    steam_review_id   TEXT UNIQUE NOT NULL,
    app_id            INTEGER NOT NULL REFERENCES metadata(app_id),
    review_text       TEXT NOT NULL,
    recommended       BOOLEAN NOT NULL,
    playtime_at_review INTEGER,
    playtime_forever  INTEGER,
    timestamp_created INTEGER,
    helpful_votes     INTEGER DEFAULT 0,
    funny_votes       INTEGER DEFAULT 0,
    early_access      BOOLEAN DEFAULT 0,
    loaded_at         TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS nlp_results (
    document_id       INTEGER PRIMARY KEY REFERENCES documents(id),
    sentiment_compound REAL,
    sentiment_pos      REAL,
    sentiment_neg      REAL,
    sentiment_neu      REAL,
    topic_id           INTEGER,
    topic_label        TEXT,
    processed_at       TEXT
);

CREATE TABLE IF NOT EXISTS processed_documents (
    document_id    INTEGER PRIMARY KEY REFERENCES documents(id),
    cleaned_text   TEXT NOT NULL,
    tokens         TEXT NOT NULL,
    token_count    INTEGER NOT NULL,
    char_count     INTEGER NOT NULL,
    processed_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id     INTEGER PRIMARY KEY,
    label        TEXT NOT NULL,
    top_words    TEXT NOT NULL,
    doc_count    INTEGER NOT NULL,
    processed_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_app_id ON documents(app_id);
CREATE INDEX IF NOT EXISTS idx_documents_recommended ON documents(recommended);
CREATE INDEX IF NOT EXISTS idx_nlp_results_topic_id ON nlp_results(topic_id);
CREATE INDEX IF NOT EXISTS idx_processed_documents_token_count
    ON processed_documents(token_count);
"""


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all project tables and indexes if they don't exist.

    Creates metadata, documents, nlp_results, and processed_documents.
    Safe to call on an existing database — every DDL statement uses
    IF NOT EXISTS so this doubles as an additive migration helper.

    Args:
        conn: Active SQLite connection.
    """
    conn.executescript(_SCHEMA_SQL)
    logger.info(
        "Schema ensured (metadata, documents, nlp_results, processed_documents, topics)."
    )


# Study tables. Kept apart from _SCHEMA_SQL so that stages run against the
# April seed never add tables to it (its sha256 is frozen after Phase 7.0).
_STUDY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS study_apps (
    app_id        INTEGER PRIMARY KEY REFERENCES metadata(app_id),
    cohort        TEXT NOT NULL,
    subgroup      TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    display_order INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pull_runs (
    run_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id                  INTEGER NOT NULL REFERENCES metadata(app_id),
    params                  TEXT NOT NULL,
    baseline_params         TEXT,
    pulled_at               TEXT NOT NULL,
    pages                   INTEGER NOT NULL,
    reviews_collected       INTEGER NOT NULL,
    cap                     INTEGER NOT NULL,
    cap_hit                 BOOLEAN NOT NULL,
    stop_reason             TEXT NOT NULL,
    effective_window_start  INTEGER,
    effective_window_end    INTEGER,
    total_positive          INTEGER,
    total_negative          INTEGER,
    total_reviews           INTEGER,
    baseline_total_positive INTEGER,
    baseline_total_negative INTEGER,
    baseline_total_reviews  INTEGER,
    stratified              BOOLEAN NOT NULL DEFAULT 0,  -- review_type filtered the pull
    UNIQUE (app_id, pulled_at)
);

CREATE TABLE IF NOT EXISTS review_language (
    document_id    INTEGER PRIMARY KEY REFERENCES documents(id),
    steam_language TEXT,
    detected_lang  TEXT,
    confidence     REAL
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    stage      TEXT NOT NULL,
    params     TEXT NOT NULL,
    git_sha    TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    metrics    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS aspect_mentions (
    document_id     INTEGER NOT NULL REFERENCES documents(id),
    aspect          TEXT NOT NULL,
    sentence_idx    INTEGER NOT NULL,
    matched_term    TEXT NOT NULL,
    char_start      INTEGER NOT NULL,
    char_end        INTEGER NOT NULL,
    lexicon_version TEXT NOT NULL,
    PRIMARY KEY (document_id, aspect, sentence_idx, lexicon_version)
);

CREATE TABLE IF NOT EXISTS game_stats (
    run_id                 INTEGER NOT NULL REFERENCES analysis_runs(run_id),
    app_id                 INTEGER NOT NULL REFERENCES metadata(app_id),
    n_pulled               INTEGER,
    n_seed                 INTEGER NOT NULL,
    n_too_short            INTEGER NOT NULL,
    n_too_few_tokens       INTEGER NOT NULL,
    n_non_english          INTEGER NOT NULL,
    n_kept                 INTEGER NOT NULL,
    n_recommended          INTEGER NOT NULL,
    n_not_recommended      INTEGER NOT NULL,
    seed_rec_share         REAL,
    seed_rec_ci_low        REAL,
    seed_rec_ci_high       REAL,
    rec_share              REAL,
    rec_ci_low             REAL,
    rec_ci_high            REAL,
    pop_share_english      REAL,
    pop_share_all          REAL,
    window_start           INTEGER,
    window_end             INTEGER,
    cap_hit                BOOLEAN,
    median_hours_rec       REAL,
    median_hours_not_rec   REAL,
    early_access_share     REAL,
    mean_compound          REAL,
    vader_pos_share        REAL,
    model_pos_share        REAL,
    model_pos_ci_low       REAL,
    model_pos_ci_high      REAL,
    eligible               BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, app_id)
);

CREATE TABLE IF NOT EXISTS cohort_stats (
    run_id               INTEGER NOT NULL REFERENCES analysis_runs(run_id),
    scope_type           TEXT NOT NULL,
    scope_id             TEXT NOT NULL,
    n_games              INTEGER NOT NULL,
    n_eligible           INTEGER NOT NULL,
    n_kept               INTEGER NOT NULL,
    macro_rec_share      REAL,
    macro_rec_ci_low     REAL,
    macro_rec_ci_high    REAL,
    pooled_rec_share     REAL,
    median_hours_rec     REAL,
    median_hours_not_rec REAL,
    macro_mean_compound  REAL,
    macro_compound_ci_low  REAL,
    macro_compound_ci_high REAL,
    macro_model_pos_share  REAL,
    macro_model_pos_ci_low REAL,
    macro_model_pos_ci_high REAL,
    PRIMARY KEY (run_id, scope_type, scope_id)
);

CREATE TABLE IF NOT EXISTS theme_stats (
    run_id          INTEGER NOT NULL REFERENCES analysis_runs(run_id),
    lens            TEXT NOT NULL,
    theme           TEXT NOT NULL,
    scope_type      TEXT NOT NULL,
    scope_id        TEXT NOT NULL,
    n_docs          INTEGER NOT NULL,
    k_docs          INTEGER NOT NULL,
    share           REAL,
    share_ci_low    REAL,
    share_ci_high   REAL,
    complaint_share REAL,
    complaint_ci_low  REAL,
    complaint_ci_high REAL,
    praise_share    REAL,
    praise_ci_low   REAL,
    praise_ci_high  REAL,
    lift            REAL,
    lift_ci_low     REAL,
    lift_ci_high    REAL,
    rec_share_in_theme REAL,
    mean_compound   REAL,
    vader_pos_share REAL,
    model_pos_share REAL,     -- share scored positive by the adopted transformer (NULL if none)
    logo_stable     BOOLEAN,  -- NULL where the theme failed the precision gate
    finding         TEXT,
    PRIMARY KEY (run_id, lens, theme, scope_type, scope_id)
);

CREATE INDEX IF NOT EXISTS idx_aspect_mentions_aspect ON aspect_mentions(aspect);
CREATE INDEX IF NOT EXISTS idx_theme_stats_scope ON theme_stats(run_id, scope_type, scope_id);
"""

# Transformer score tables. Also used on their own for a scores sidecar next
# to the frozen April seed, which has no documents table, so document_id
# carries no foreign key. sentence_idx is -1 for document-level rows. Each
# score row carries a small model_id; sentiment_models maps it to the model
# name and pinned revision, which keeps the committed seed inside its size cap.
_SCORES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    stage      TEXT NOT NULL,
    params     TEXT NOT NULL,
    git_sha    TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    metrics    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentiment_models (
    model_id       INTEGER PRIMARY KEY,
    model_name     TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    UNIQUE (model_name, model_revision)
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    document_id    INTEGER NOT NULL,
    model_id       INTEGER NOT NULL REFERENCES sentiment_models(model_id),
    unit           TEXT NOT NULL CHECK (unit IN ('doc', 'sentence')),
    sentence_idx   INTEGER NOT NULL,
    positive_prob  REAL NOT NULL,
    PRIMARY KEY (document_id, model_id, unit, sentence_idx)
) WITHOUT ROWID;
"""


def create_study_tables(conn: sqlite3.Connection) -> None:
    """Create the base tables plus the study provenance tables.

    Used only for study databases (never the April seed), so the April
    seed's file hash stays fixed.

    Args:
        conn: Active SQLite connection.
    """
    create_tables(conn)
    conn.executescript(_STUDY_SCHEMA_SQL)
    conn.executescript(_SCORES_SCHEMA_SQL)


def create_scores_tables(conn: sqlite3.Connection) -> None:
    """Create only ``analysis_runs``, ``sentiment_models`` and ``sentiment_scores``.

    Used for a transformer-scores sidecar database, such as the one beside the
    April seed, whose own file must not change.

    Args:
        conn: Active SQLite connection.
    """
    conn.executescript(_SCORES_SCHEMA_SQL)


def load_raw_json(json_path: Path) -> list[dict]:
    """Parse a single raw JSON file into a list of review dicts.

    Args:
        json_path: Path to a raw JSON file in data/raw/.

    Returns:
        List of review dicts with normalized keys.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "reviews" in data:
        return list(data["reviews"].values()) if isinstance(data["reviews"], dict) else data["reviews"]

    logger.warning("Unexpected JSON structure in %s. Returning empty list.", json_path)
    return []


def insert_documents(
    conn: sqlite3.Connection,
    reviews: list[dict],
    app_id: int,
    app_name: str,
) -> int:
    """Insert reviews into the documents table. Skip duplicates by Steam review ID.

    Args:
        conn: Active SQLite connection.
        reviews: List of review dicts from load_raw_json.
        app_id: Steam application ID.
        app_name: Human-readable game name.

    Returns:
        Number of rows inserted.
    """
    rows = []
    for r in reviews:
        review_text = r.get("review", "")
        if not review_text or not review_text.strip():
            continue

        steam_id = str(r.get("recommendationid", ""))
        if not steam_id:
            continue

        rows.append((
            steam_id,
            app_id,
            review_text,
            bool(r.get("voted_up", False)),
            r.get("author", {}).get("playtime_at_review", None),
            r.get("author", {}).get("playtime_forever", None),
            r.get("timestamp_created", None),
            r.get("votes_up", 0),
            r.get("votes_funny", 0),
            bool(r.get("written_during_early_access", False)),
        ))

    count_before = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO documents
            (steam_review_id, app_id, review_text, recommended,
             playtime_at_review, playtime_forever, timestamp_created,
             helpful_votes, funny_votes, early_access)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    count_after = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    inserted = count_after - count_before
    logger.info("  %s (%d): %d/%d reviews inserted", app_name, app_id, inserted, len(rows))
    return inserted


def insert_metadata(conn: sqlite3.Connection) -> None:
    """Populate the metadata table with game info from GAME_CATALOG.

    Args:
        conn: Active SQLite connection.
    """
    for name, (app_id, genre) in GAME_CATALOG.items():
        conn.execute(
            "INSERT OR IGNORE INTO metadata (app_id, app_name, genre) VALUES (?, ?, ?)",
            (app_id, name, genre),
        )
    conn.commit()
    logger.info("Metadata table populated with %d games.", len(GAME_CATALOG))


def build_seed_db(full_conn: sqlite3.Connection, reviews_per_game: int = 200) -> None:
    """Sample a representative subset from the full database and write to seed.db.

    Takes a stratified sample (50% recommended, 50% not recommended) per game.
    Overwrites any existing seed.db.

    Note:
        Some games may have fewer negative reviews than half of reviews_per_game
        (e.g. Stardew Valley has ~82 negative reviews). In those cases the sample
        is smaller than the target. This is expected, not a bug.

    Args:
        full_conn: Connection to the FULL database (source).
        reviews_per_game: Max reviews to sample per game for the seed.
    """
    if SEED_DB_PATH.exists():
        SEED_DB_PATH.unlink()

    seed_conn = sqlite3.connect(SEED_DB_PATH)
    try:
        create_tables(seed_conn)
        insert_metadata(seed_conn)

        half = reviews_per_game // 2
        app_ids = [row[0] for row in full_conn.execute("SELECT DISTINCT app_id FROM documents").fetchall()]

        for app_id in app_ids:
            for recommended_val in [1, 0]:
                rows = full_conn.execute(
                    """
                    SELECT steam_review_id, app_id, review_text, recommended,
                           playtime_at_review, playtime_forever, timestamp_created,
                           helpful_votes, funny_votes, early_access
                    FROM documents
                    WHERE app_id = ? AND recommended = ?
                    ORDER BY RANDOM()
                    LIMIT ?
                    """,
                    (app_id, recommended_val, half),
                ).fetchall()

                seed_conn.executemany(
                    """
                    INSERT OR IGNORE INTO documents
                        (steam_review_id, app_id, review_text, recommended,
                         playtime_at_review, playtime_forever, timestamp_created,
                         helpful_votes, funny_votes, early_access)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        seed_conn.commit()

        total_inserted = seed_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        logger.info("Seed DB built: %d documents from %d games.", total_inserted, len(app_ids))

        size_mb = SEED_DB_PATH.stat().st_size / (1024 * 1024)
        logger.info("Seed DB size: %.2f MB", size_mb)
        if size_mb > 25:
            logger.warning("seed.db exceeds 25 MB (%.2f MB). Reduce reviews_per_game.", size_mb)
    finally:
        seed_conn.close()


# ---------------------------------------------------------------------------
# Study (natural sampling) loading
# ---------------------------------------------------------------------------

_DOC_COLUMNS: str = """steam_review_id, app_id, review_text, recommended,
    playtime_at_review, playtime_forever, timestamp_created,
    helpful_votes, funny_votes, early_access"""


def insert_study_apps(conn: sqlite3.Connection, study: Study) -> None:
    """Write metadata and study_apps rows for every app in a study.

    Args:
        conn: Active SQLite connection with study tables created.
        study: The study.
    """
    for order, app in enumerate(study.apps):
        conn.execute(
            "INSERT OR IGNORE INTO metadata (app_id, app_name, genre) VALUES (?, ?, ?)",
            (app.app_id, app.name, app.genre),
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO study_apps (app_id, cohort, subgroup, tags, display_order)
            VALUES (?, ?, ?, ?, ?)
            """,
            (app.app_id, app.cohort, app.subgroup, json.dumps(list(app.tags)), order),
        )
    conn.commit()


def insert_pull_run(conn: sqlite3.Connection, manifest: dict) -> None:
    """Record one census pull manifest in pull_runs (idempotent).

    Args:
        conn: Active SQLite connection with study tables created.
        manifest: Manifest dict written by ``data_pull.census_pull``.
    """
    summary = manifest.get("query_summary") or {}
    base = manifest.get("baseline_summary") or {}
    # Databases created before Phase 8 have no stratified column; they only
    # ever held natural pulls.
    has_flag = any(r[1] == "stratified" for r in conn.execute("PRAGMA table_info(pull_runs)"))
    stratified = bool(manifest.get("stratified", False))
    conn.execute(
        f"""
        INSERT OR IGNORE INTO pull_runs
            (app_id, params, baseline_params, pulled_at, pages, reviews_collected,
             cap, cap_hit, stop_reason, effective_window_start, effective_window_end,
             total_positive, total_negative, total_reviews,
             baseline_total_positive, baseline_total_negative, baseline_total_reviews
             {", stratified" if has_flag else ""})
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?{", ?" if has_flag else ""})
        """,
        (
            manifest["app_id"],
            json.dumps(manifest["params"], sort_keys=True),
            json.dumps(manifest.get("baseline_params"), sort_keys=True),
            manifest["finished_at"],
            manifest["pages"],
            manifest["reviews_collected"],
            manifest["cap"],
            bool(manifest["cap_hit"]),
            manifest["stop_reason"],
            manifest.get("effective_window_start"),
            manifest.get("effective_window_end"),
            summary.get("total_positive"),
            summary.get("total_negative"),
            summary.get("total_reviews"),
            base.get("total_positive"),
            base.get("total_negative"),
            base.get("total_reviews"),
            *((stratified,) if has_flag else ()),
        ),
    )
    if stratified and not has_flag:
        raise ValueError("A stratified pull needs a pull_runs table with the stratified column.")


def insert_steam_languages(conn: sqlite3.Connection, reviews: list[dict]) -> None:
    """Record each review's reviewer-declared Steam language.

    Args:
        conn: Active SQLite connection with study tables created.
        reviews: Raw review dicts already inserted into documents.
    """
    rows = [
        (r.get("language"), str(r.get("recommendationid", "")))
        for r in reviews
        if r.get("recommendationid")
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO review_language (document_id, steam_language)
        SELECT id, ? FROM documents WHERE steam_review_id = ?
        """,
        rows,
    )


def load_study_full(study: Study, unfreeze: bool = False) -> int:
    """Load every cached raw file of a study into its full database.

    Raw files are located by app ID from the study (``<app_id>.json`` plus
    the ``<app_id>.pull.json`` manifest), never by parsing file names.
    Reviews whose ``timestamp_created`` falls outside the study window
    (``sampling.window_start``/``window_end``) are not loaded.
    Loading is idempotent: documents dedupe on ``steam_review_id`` and pull
    runs on ``(app_id, pulled_at)``.

    Args:
        study: The study to load.
        unfreeze: Allow writing a frozen study's full database.

    Returns:
        Total document count in the full database after loading.

    Raises:
        FrozenStudyError: If the full database belongs to a frozen study and
            ``unfreeze`` is False.
    """
    guard_writable([study.full_db], unfreeze)
    study.full_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(study.full_db)
    try:
        create_study_tables(conn)
        insert_study_apps(conn, study)
        for app in study.apps:
            raw_path = study.raw_dir / f"{app.app_id}.json"
            manifest_path = study.raw_dir / f"{app.app_id}.pull.json"
            if not (raw_path.exists() and manifest_path.exists()):
                logger.warning("No cached pull for %s (%d). Skipping.", app.name, app.app_id)
                continue
            # Skip rows already loaded: an ignored INSERT still advances the
            # AUTOINCREMENT counter, and a rerun should change nothing.
            known = {
                r[0] for r in conn.execute(
                    "SELECT steam_review_id FROM documents WHERE app_id = ?", (app.app_id,)
                )
            }
            reviews = [
                r for r in load_raw_json(raw_path)
                if study.sampling.in_window(r.get("timestamp_created"))
                and str(r.get("recommendationid", "")) not in known
            ]
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            loaded = conn.execute(
                "SELECT 1 FROM pull_runs WHERE app_id = ? AND pulled_at = ?",
                (manifest["app_id"], manifest["finished_at"]),
            ).fetchone()
            conn.execute("BEGIN")
            insert_documents(conn, reviews, app.app_id, app.name)
            insert_steam_languages(conn, reviews)
            if not loaded:
                insert_pull_run(conn, manifest)
            conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        logger.info("Study %s full DB: %d documents.", study.name, total)
        return total
    finally:
        conn.close()


def build_natural_seed(
    full_conn: sqlite3.Connection,
    seed_path: Path,
    cap_per_game: int,
    random_seed: int,
    window: tuple[int | None, int | None] = (None, None),
    unfreeze: bool = False,
) -> dict[int, int]:
    """Write a uniform random per-game sample of the full DB to a new seed DB.

    Each game contributes ``min(n, cap_per_game)`` reviews drawn uniformly
    without regard to the recommend flag, so the seed keeps each game's
    natural recommend rate. Games below the cap are included whole. The draw
    is deterministic: review IDs are sorted before sampling with a per-game
    RNG seeded from ``random_seed`` and the app ID. Metadata, study_apps,
    pull_runs and review_language rows are copied for the sampled reviews.

    Args:
        full_conn: Connection to the study's full database (source).
        seed_path: Destination seed DB path. An existing file is replaced.
        cap_per_game: Maximum reviews per game.
        random_seed: Base seed for the per-game samples.
        window: Inclusive ``timestamp_created`` bounds in Unix seconds; reviews
            outside are not sampled, so a narrowed window takes effect even
            when the full database holds older loads.
        unfreeze: Allow replacing a frozen study's seed database.

    Returns:
        ``{app_id: sampled_count}``.

    Raises:
        FrozenStudyError: If ``seed_path`` belongs to a frozen study and
            ``unfreeze`` is False.
    """
    guard_writable([seed_path], unfreeze)
    if seed_path.exists():
        seed_path.unlink()
    seed_path.parent.mkdir(parents=True, exist_ok=True)

    seed_conn = sqlite3.connect(seed_path)
    counts: dict[int, int] = {}
    try:
        create_study_tables(seed_conn)
        for table in ("metadata", "study_apps", "pull_runs"):
            cols = [r[1] for r in full_conn.execute(f"PRAGMA table_info({table})")]
            rows = full_conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()
            placeholders = ", ".join("?" * len(cols))
            seed_conn.executemany(
                f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                rows,
            )

        app_ids = [
            r[0]
            for r in full_conn.execute("SELECT DISTINCT app_id FROM documents ORDER BY app_id")
        ]
        for app_id in app_ids:
            lo, hi = window
            ids = sorted(
                r[0]
                for r in full_conn.execute(
                    "SELECT steam_review_id FROM documents WHERE app_id = ? "
                    "AND (? IS NULL OR timestamp_created >= ?) "
                    "AND (? IS NULL OR timestamp_created <= ?)",
                    (app_id, lo, lo, hi, hi),
                )
            )
            rng = random.Random(f"{random_seed}:{app_id}")
            chosen = sorted(rng.sample(ids, min(len(ids), cap_per_game)))
            rows = []
            for start in range(0, len(chosen), 500):
                chunk = chosen[start : start + 500]
                marks = ",".join("?" * len(chunk))
                rows.extend(
                    full_conn.execute(
                        f"SELECT {_DOC_COLUMNS} FROM documents "
                        f"WHERE steam_review_id IN ({marks}) ORDER BY steam_review_id",
                        chunk,
                    ).fetchall()
                )
            seed_conn.executemany(
                f"INSERT OR IGNORE INTO documents ({_DOC_COLUMNS}) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            lang_rows = []
            for start in range(0, len(chosen), 500):
                chunk = chosen[start : start + 500]
                marks = ",".join("?" * len(chunk))
                lang_rows.extend(
                    full_conn.execute(
                        f"SELECT d.steam_review_id, l.steam_language FROM review_language l "
                        f"JOIN documents d ON d.id = l.document_id "
                        f"WHERE d.steam_review_id IN ({marks})",
                        chunk,
                    ).fetchall()
                )
            seed_conn.executemany(
                """
                INSERT OR IGNORE INTO review_language (document_id, steam_language)
                SELECT id, ? FROM documents WHERE steam_review_id = ?
                """,
                [(lang, rid) for rid, lang in lang_rows],
            )
            counts[app_id] = len(chosen)
        seed_conn.commit()
        logger.info(
            "Natural seed built: %d documents from %d games (cap %d, seed %d).",
            sum(counts.values()),
            len(counts),
            cap_per_game,
            random_seed,
        )
    finally:
        seed_conn.close()
    return counts


def vacuum(db_path: Path) -> float:
    """VACUUM a database file and return its size in MB.

    Args:
        db_path: SQLite file to compact.

    Returns:
        File size in megabytes after VACUUM.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()
    return db_path.stat().st_size / (1024 * 1024)


def main_study(study: Study, full: bool, unfreeze: bool = False) -> None:
    """Study entry point: load raw pulls into the full DB and build the seed.

    Args:
        study: A natural-sampling study.
        full: If False, only report the seed's document count.
        unfreeze: Allow writing a frozen study's databases.
    """
    if not full:
        if study.seed_db.exists():
            conn = sqlite3.connect(study.seed_db)
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            conn.close()
            logger.info("Seed DB %s has %d documents. Nothing to do.", study.seed_db, count)
        else:
            logger.error("No seed DB at %s. Run with --full first.", study.seed_db)
        return
    guard_writable([study.full_db, study.seed_db], unfreeze)
    load_study_full(study, unfreeze)
    full_conn = sqlite3.connect(study.full_db)
    try:
        build_natural_seed(
            full_conn,
            study.seed_db,
            study.sampling.seed_cap_per_game,
            study.sampling.random_seed,
            (study.sampling.window_start, study.sampling.window_end),
            unfreeze,
        )
    finally:
        full_conn.close()
    logger.info("Seed DB size after VACUUM: %.2f MB", vacuum(study.seed_db))


def main(mode: str = "seed", unfreeze: bool = False) -> None:
    """Entry point. Creates tables and loads data in the given mode.

    Args:
        mode: 'seed' (default) or 'full'.
        unfreeze: Allow the full mode to rebuild the frozen April databases.
    """
    db_path = get_db_path(mode)

    if mode == "seed":
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            create_tables(conn)
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            logger.info("Seed DB already exists with %d documents. Nothing to do.", count)
            conn.close()
            return
        logger.error("No seed.db found at %s. Run with --full first to generate it.", db_path)
        return

    # Full mode: load all raw JSON into full.db, then build seed.db
    guard_writable([FULL_DB_PATH, SEED_DB_PATH], unfreeze)
    logger.info("Full mode: loading raw JSON into %s", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        create_tables(conn)
        insert_metadata(conn)

        json_files = sorted(RAW_DIR.glob("*.json"))
        if not json_files:
            logger.error("No JSON files found in %s. Run data_pull.py first.", RAW_DIR)
            conn.close()
            return

        total = 0
        for json_path in json_files:
            reviews = load_raw_json(json_path)
            if not reviews:
                logger.warning("Empty or unreadable: %s", json_path)
                continue

            # Extract app_id from filename pattern: <name>_<app_id>.json
            stem = json_path.stem
            app_id_str = stem.rsplit("_", 1)[-1]
            try:
                app_id = int(app_id_str)
            except ValueError:
                logger.warning("Cannot parse app_id from filename %s. Skipping.", json_path.name)
                continue

            app_name = next((n for n, aid in GAME_IDS.items() if aid == app_id), f"unknown_{app_id}")

            conn.execute("BEGIN")
            inserted = insert_documents(conn, reviews, app_id, app_name)
            conn.commit()
            total += inserted

        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        logger.info("Full DB loaded: %d total documents.", count)
    finally:
        conn.close()

    # Build seed.db from full.db
    logger.info("Building seed.db from full.db...")
    full_conn = sqlite3.connect(FULL_DB_PATH)
    try:
        build_seed_db(full_conn)
    finally:
        full_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up the NLP dashboard database.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Load all raw JSON into full.db and generate seed.db. Default is seed mode.",
    )
    parser.add_argument(
        "--study",
        type=Path,
        default=DEFAULT_STUDY_PATH,
        help="Study TOML file (default: the April 2026 baseline).",
    )
    parser.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    args = parser.parse_args()
    selected = load_study(args.study)
    try:
        if selected.sampling.method == "natural":
            main_study(selected, full=args.full, unfreeze=args.unfreeze)
        else:
            main(mode="full" if args.full else "seed", unfreeze=args.unfreeze)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
