"""Smoke tests for database schema creation and seed loading."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.db_setup import create_tables, insert_documents, insert_metadata, load_raw_json, main
from src.study import Study
from src.utils import DATA_DIR, SEED_DB_PATH

MARKET_SEED_DB_PATH: Path = DATA_DIR / "market_seed.db"
APRIL_SCORES_DB_PATH: Path = DATA_DIR / "seed_transformer_scores.db"


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


def test_committed_dbs_under_25mb() -> None:
    """Verify every seed DB plus the April sidecar stays under 25 MiB and 25 MB.

    Any ``data/*_seed.db`` counts, so a new study's seed is caught before it
    is committed past the budget.
    """
    files = {SEED_DB_PATH, APRIL_SCORES_DB_PATH, *SEED_DB_PATH.parent.glob("*_seed.db")}
    assert MARKET_SEED_DB_PATH in files or not MARKET_SEED_DB_PATH.exists()
    total = sum(p.stat().st_size for p in files if p.exists())
    assert total < 25 * 1024 * 1024, f"seed DBs total {total / 1024**2:.2f} MiB, over 25 MiB"
    assert total < 25_000_000, f"seed DBs total {total / 1e6:.2f} MB, over 25 MB"


def test_market_seed_db_under_20mib() -> None:
    """Verify the market seed alone stays under its 20 MiB cap (Decision 5, measured in MiB)."""
    if not MARKET_SEED_DB_PATH.exists():
        return
    size_mib = MARKET_SEED_DB_PATH.stat().st_size / (1024 * 1024)
    assert size_mib < 20, f"market seed is {size_mib:.2f} MiB, exceeds the 20 MiB cap"


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


# ---------------------------------------------------------------------------
# Study loading and natural seed
# ---------------------------------------------------------------------------


def _study(tmp_path: Path, app_ids: tuple[int, ...] = (1,)) -> Study:
    """Build a natural study whose files live under tmp_path.

    Args:
        tmp_path: pytest temporary directory.
        app_ids: App IDs to include.

    Returns:
        A Study instance.
    """
    from src.study import AppSpec, Sampling, Study

    return Study(
        name="t",
        title="t",
        seed_db=tmp_path / "seed.db",
        full_db=tmp_path / "full.db",
        raw_dir=tmp_path / "raw",
        language_filter=False,
        aspect_lexicon=None,
        sampling=Sampling("natural", "recent", "english", "all", "all", 1, 100, 1000, 100, 42),
        baseline=None,
        apps=tuple(AppSpec(a, f"Game {a}", "Tactics", "c", "", ()) for a in app_ids),
    )


def _write_pull(raw_dir: Path, app_id: int, ids: list[int], finished_at: str) -> None:
    """Write a synthetic raw file and manifest for one app.

    Args:
        raw_dir: Study raw directory.
        app_id: App ID.
        ids: Review IDs to include.
        finished_at: Manifest finish time (distinguishes pulls).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    reviews = [
        {"recommendationid": str(i), "review": f"review {i}", "voted_up": i % 2 == 0,
         "language": "english", "timestamp_created": 1000 + i, "author": {}}
        for i in ids
    ]
    (raw_dir / f"{app_id}.json").write_text(json.dumps(reviews), encoding="utf-8")
    manifest = {
        "app_id": app_id, "params": {}, "baseline_params": None, "finished_at": finished_at,
        "pages": 1, "reviews_collected": len(ids), "cap": 1000, "cap_hit": False,
        "stop_reason": "exhausted", "query_summary": {"total_reviews": len(ids)},
    }
    (raw_dir / f"{app_id}.pull.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_load_study_full_is_idempotent_across_overlapping_pulls(tmp_path: Path) -> None:
    """Loading twice, including an overlapping re-pull, inserts each review once."""
    from src.db_setup import load_study_full

    study = _study(tmp_path)
    _write_pull(study.raw_dir, 1, [1, 2, 3], "2026-09-26T00:00:00")
    assert load_study_full(study) == 3
    assert load_study_full(study) == 3

    _write_pull(study.raw_dir, 1, [2, 3, 4], "2026-09-27T00:00:00")
    assert load_study_full(study) == 4

    conn = sqlite3.connect(study.full_db)
    runs = conn.execute("SELECT COUNT(*) FROM pull_runs").fetchone()[0]
    langs = conn.execute("SELECT COUNT(*) FROM review_language").fetchone()[0]
    conn.close()
    assert runs == 2
    assert langs == 4


def _full_db_with_rates(path: Path, per_app: dict[int, tuple[int, int]]) -> sqlite3.Connection:
    """Create a full DB with a given number of recommended and not-recommended reviews.

    Args:
        path: Database file path.
        per_app: ``{app_id: (n_recommended, n_not_recommended)}``.

    Returns:
        Open connection to the new database.
    """
    from src.db_setup import create_study_tables

    conn = sqlite3.connect(path)
    create_study_tables(conn)
    rid = 0
    for app_id, (n_pos, n_neg) in per_app.items():
        conn.execute("INSERT INTO metadata (app_id, app_name) VALUES (?, ?)", (app_id, f"G{app_id}"))
        for flag, n in ((1, n_pos), (0, n_neg)):
            for _ in range(n):
                rid += 1
                conn.execute(
                    "INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                    "VALUES (?, ?, ?, ?)",
                    (f"r{rid:06d}", app_id, "text", flag),
                )
    conn.commit()
    return conn


def test_build_natural_seed_keeps_natural_rate_and_is_deterministic(tmp_path: Path) -> None:
    """A 90/10 game keeps ~90% recommended, reruns match, the cap binds, small games stay whole."""
    from src.db_setup import build_natural_seed

    full = _full_db_with_rates(tmp_path / "full.db", {1: (3600, 400), 2: (50, 30)})
    seed = tmp_path / "seed.db"

    counts = build_natural_seed(full, seed, cap_per_game=400, random_seed=42)
    conn = sqlite3.connect(seed)
    share = conn.execute(
        "SELECT AVG(recommended) FROM documents WHERE app_id = 1"
    ).fetchone()[0]
    first = conn.execute("SELECT steam_review_id FROM documents ORDER BY 1").fetchall()
    conn.close()

    assert counts == {1: 400, 2: 80}
    assert abs(share - 0.9) <= 0.05

    build_natural_seed(full, seed, cap_per_game=400, random_seed=42)
    conn = sqlite3.connect(seed)
    second = conn.execute("SELECT steam_review_id FROM documents ORDER BY 1").fetchall()
    conn.close()
    full.close()
    assert first == second
