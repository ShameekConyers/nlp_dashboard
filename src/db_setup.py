"""Database setup and loading. Creates SQLite schema, loads data in seed or full mode."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
        "Schema ensured (metadata, documents, nlp_results, processed_documents)."
    )


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


def main(mode: str = "seed") -> None:
    """Entry point. Creates tables and loads data in the given mode.

    Args:
        mode: 'seed' (default) or 'full'.
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
    args = parser.parse_args()
    main(mode="full" if args.full else "seed")
