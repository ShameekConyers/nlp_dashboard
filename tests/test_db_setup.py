"""Smoke tests for database schema creation and seed loading."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.db_setup import create_tables, insert_documents, insert_metadata, load_raw_json, main
from src.utils import SEED_DB_PATH


def _make_in_memory_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with the full schema.

    Returns:
        Connection to an in-memory database with all tables created.
    """
    conn = sqlite3.connect(":memory:")
    create_tables(conn)
    insert_metadata(conn)
    return conn


def test_create_tables_creates_all_three() -> None:
    """Verify documents, nlp_results, and metadata tables exist after create_tables."""
    conn = sqlite3.connect(":memory:")
    create_tables(conn)

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    assert "metadata" in tables
    assert "documents" in tables
    assert "nlp_results" in tables


def test_insert_documents_deduplicates() -> None:
    """Verify inserting the same review twice doesn't create duplicates."""
    conn = _make_in_memory_db()

    fake_reviews = [
        {
            "recommendationid": "12345",
            "review": "Great game!",
            "voted_up": True,
            "author": {"playtime_at_review": 100, "playtime_forever": 500},
            "timestamp_created": 1700000000,
            "votes_up": 3,
            "votes_funny": 0,
            "written_during_early_access": False,
        },
        {
            "recommendationid": "12345",
            "review": "Great game! (duplicate)",
            "voted_up": True,
            "author": {"playtime_at_review": 100, "playtime_forever": 500},
            "timestamp_created": 1700000000,
            "votes_up": 3,
            "votes_funny": 0,
            "written_during_early_access": False,
        },
    ]

    insert_documents(conn, fake_reviews, app_id=367520, app_name="Hollow Knight")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()

    assert count == 1, f"Expected 1 row, got {count} (dedup failed)"


def test_insert_documents_skips_empty_text() -> None:
    """Verify reviews with empty text are not inserted."""
    conn = _make_in_memory_db()

    fake_reviews = [
        {
            "recommendationid": "99999",
            "review": "",
            "voted_up": True,
            "author": {},
            "timestamp_created": 1700000000,
            "votes_up": 0,
            "votes_funny": 0,
            "written_during_early_access": False,
        },
        {
            "recommendationid": "99998",
            "review": "Actual content here.",
            "voted_up": False,
            "author": {},
            "timestamp_created": 1700000000,
            "votes_up": 0,
            "votes_funny": 0,
            "written_during_early_access": False,
        },
    ]

    insert_documents(conn, fake_reviews, app_id=367520, app_name="Hollow Knight")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()

    assert count == 1, f"Expected 1 row (empty review skipped), got {count}"


def test_seed_db_under_25mb() -> None:
    """Verify seed.db file size is under 25 MB."""
    if not SEED_DB_PATH.exists():
        pytest.skip("seed.db not yet generated")

    size_mb = SEED_DB_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 25, f"seed.db is {size_mb:.2f} MB, exceeds 25 MB limit"


def test_seed_db_has_data() -> None:
    """Verify seed.db has rows in documents and metadata tables."""
    if not SEED_DB_PATH.exists():
        pytest.skip("seed.db not yet generated")

    conn = sqlite3.connect(SEED_DB_PATH)
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    meta_count = conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    conn.close()

    assert doc_count > 0, "documents table is empty"
    assert meta_count > 0, "metadata table is empty"


def test_main_seed_mode_with_existing_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main('seed') succeeds when a valid seed.db already exists.

    Creates a temporary seed.db with schema and one row, then confirms
    main('seed') runs without error and leaves the data intact.
    """
    fake_seed = tmp_path / "seed.db"
    conn = sqlite3.connect(fake_seed)
    create_tables(conn)
    insert_metadata(conn)
    conn.execute(
        """
        INSERT INTO documents
            (steam_review_id, app_id, review_text, recommended)
        VALUES (?, ?, ?, ?)
        """,
        ("test_001", 367520, "A test review.", True),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("src.db_setup.SEED_DB_PATH", fake_seed)
    monkeypatch.setattr("src.db_setup.get_db_path", lambda mode="seed": fake_seed)

    main("seed")

    conn = sqlite3.connect(fake_seed)
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()

    assert count == 1, f"Expected 1 document after seed-mode main(), got {count}"


# ---------------------------------------------------------------------------
# load_raw_json tests
# ---------------------------------------------------------------------------


def test_load_raw_json_list_format(tmp_path: Path) -> None:
    """Verify load_raw_json handles a plain list of review dicts."""
    reviews = [{"recommendationid": "1", "review": "Good"}, {"recommendationid": "2", "review": "Bad"}]
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(reviews), encoding="utf-8")

    result = load_raw_json(json_path)

    assert len(result) == 2
    assert result[0]["review"] == "Good"


def test_load_raw_json_dict_with_reviews_key(tmp_path: Path) -> None:
    """Verify load_raw_json handles a dict with a 'reviews' key containing a list."""
    data = {"reviews": [{"recommendationid": "1", "review": "Great"}]}
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    result = load_raw_json(json_path)

    assert len(result) == 1
    assert result[0]["review"] == "Great"


def test_load_raw_json_dict_with_reviews_as_dict(tmp_path: Path) -> None:
    """Verify load_raw_json handles a dict with 'reviews' as a nested dict (keyed by ID)."""
    data = {
        "reviews": {
            "abc": {"recommendationid": "abc", "review": "Fun"},
            "def": {"recommendationid": "def", "review": "Boring"},
        }
    }
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    result = load_raw_json(json_path)

    assert len(result) == 2


def test_load_raw_json_unknown_structure(tmp_path: Path) -> None:
    """Verify load_raw_json returns an empty list for unexpected JSON structures."""
    data = {"something_else": 42}
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    result = load_raw_json(json_path)

    assert result == []
