"""Tests for the VADER sentiment scoring pipeline.

Covers the SentimentScorer class, validation metrics, database helpers,
and end-to-end scoring with idempotency. VADER loads fast (~50ms), so
no special fixture caching is needed.
"""

import json
import sqlite3

import pytest

from src.db_setup import create_tables, insert_metadata
from src.sentiment import (
    SentimentScorer,
    compute_validation_metrics,
    fetch_unscored_documents,
    find_optimal_threshold,
    run,
    write_sentiment_rows,
)
from src.utils import MIN_TOKEN_COUNT, VADER_POSITIVE_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with full schema and metadata.

    Returns:
        Connection to the prepared in-memory database.
    """
    conn = sqlite3.connect(":memory:")
    create_tables(conn)
    insert_metadata(conn)
    return conn


def _insert_fake_documents(
    conn: sqlite3.Connection,
    reviews: list[tuple[str, str, bool]],
) -> None:
    """Insert fake documents into the in-memory database.

    Args:
        conn: Active SQLite connection with schema already created.
        reviews: List of (steam_review_id, review_text, recommended) tuples.
    """
    for steam_id, text, recommended in reviews:
        conn.execute(
            """
            INSERT INTO documents (steam_review_id, app_id, review_text, recommended)
            VALUES (?, ?, ?, ?)
            """,
            (steam_id, 367520, text, recommended),
        )
    conn.commit()


def _insert_fake_processed(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, str, int, int]],
) -> None:
    """Insert fake processed_documents rows.

    Args:
        conn: Active SQLite connection with schema already created.
        rows: List of (document_id, cleaned_text, tokens_json, token_count,
            char_count) tuples.
    """
    conn.executemany(
        """
        INSERT OR IGNORE INTO processed_documents
            (document_id, cleaned_text, tokens, token_count, char_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_text_positive() -> None:
    """Verify a clearly positive sentence gets compound > 0."""
    scorer = SentimentScorer()
    scores = scorer.score_text("This is absolutely wonderful and amazing!")
    assert scores["compound"] > 0


def test_score_text_negative() -> None:
    """Verify a clearly negative sentence gets compound < 0."""
    scorer = SentimentScorer()
    scores = scorer.score_text("This is terrible, awful, and disgusting.")
    assert scores["compound"] < 0


def test_score_text_empty_returns_zeros() -> None:
    """Empty string returns all-zero scores (sentinel behavior)."""
    scorer = SentimentScorer()
    scores = scorer.score_text("")
    assert scores == {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}

    scores_ws = scorer.score_text("   ")
    assert scores_ws == {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}


def test_score_text_all_caps_amplifies() -> None:
    """VADER treats ALL CAPS as intensifiers. Compound should be more extreme."""
    scorer = SentimentScorer()
    normal = scorer.score_text("This is great")
    caps = scorer.score_text("This is GREAT")
    assert abs(caps["compound"]) > abs(normal["compound"])


def test_score_batch_aligns_with_input() -> None:
    """Batch scoring returns one result per input, positionally aligned."""
    scorer = SentimentScorer()
    texts = ["I love this", "I hate this", ""]
    results = scorer.score_batch(texts)
    assert len(results) == 3
    assert results[0]["compound"] > 0
    assert results[1]["compound"] < 0
    assert results[2]["compound"] == 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_compute_validation_metrics_perfect_agreement() -> None:
    """When predictions match ground truth perfectly, accuracy = 1.0."""
    compounds = [0.5, 0.8, -0.3, -0.6]
    recommended = [True, True, False, False]
    metrics = compute_validation_metrics(compounds, recommended)
    assert metrics["accuracy"] == 1.0
    assert metrics["true_pos"] == 2
    assert metrics["true_neg"] == 2
    assert metrics["false_pos"] == 0
    assert metrics["false_neg"] == 0


def test_compute_validation_metrics_partial_agreement() -> None:
    """Verify TP/FP/TN/FN counts for a known mixed input."""
    # compound >= 0.05 -> predicted positive
    compounds = [0.5, 0.3, -0.2, -0.5]
    recommended = [True, False, False, True]
    # pred: [+, +, -, -], actual: [+, -, -, +]
    # TP=1, FP=1, TN=1, FN=1
    metrics = compute_validation_metrics(compounds, recommended)
    assert metrics["true_pos"] == 1
    assert metrics["false_pos"] == 1
    assert metrics["true_neg"] == 1
    assert metrics["false_neg"] == 1
    assert metrics["accuracy"] == 0.5


def test_find_optimal_threshold_returns_best() -> None:
    """Verify the optimal threshold beats or ties the default threshold."""
    compounds = [0.5, 0.3, 0.1, -0.2, -0.5, -0.8]
    recommended = [True, True, False, False, False, False]

    default_metrics = compute_validation_metrics(
        compounds, recommended, VADER_POSITIVE_THRESHOLD,
    )
    best_thresh, best_acc = find_optimal_threshold(compounds, recommended)
    assert best_acc >= default_metrics["accuracy"]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def test_write_sentiment_rows_inserts() -> None:
    """Verify rows land in nlp_results with correct sentiment columns."""
    conn = _make_test_db()
    _insert_fake_documents(conn, [("r1", "Great review", True)])
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE steam_review_id='r1'"
    ).fetchone()[0]

    rows = [(doc_id, 0.75, 0.6, 0.0, 0.4)]
    inserted = write_sentiment_rows(conn, rows)
    conn.commit()

    assert inserted == 1
    stored = conn.execute(
        "SELECT sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu "
        "FROM nlp_results WHERE document_id=?",
        (doc_id,),
    ).fetchone()
    assert stored[0] == 0.75
    assert stored[1] == 0.6
    assert stored[2] == 0.0
    assert stored[3] == 0.4
    conn.close()


def test_write_sentiment_rows_dedupes_on_document_id() -> None:
    """Re-inserting the same document_id must not duplicate rows."""
    conn = _make_test_db()
    _insert_fake_documents(conn, [("r1", "Great review", True)])
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE steam_review_id='r1'"
    ).fetchone()[0]

    rows = [(doc_id, 0.75, 0.6, 0.0, 0.4)]
    write_sentiment_rows(conn, rows)
    conn.commit()
    second = write_sentiment_rows(conn, rows)
    conn.commit()

    assert second == 0
    count = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
    assert count == 1
    conn.close()


def test_fetch_unscored_documents_excludes_already_scored() -> None:
    """Documents with existing nlp_results rows are excluded."""
    conn = _make_test_db()
    _insert_fake_documents(conn, [
        ("r1", "Review one", True),
        ("r2", "Review two", False),
    ])
    doc_id_1 = conn.execute(
        "SELECT id FROM documents WHERE steam_review_id='r1'"
    ).fetchone()[0]
    doc_id_2 = conn.execute(
        "SELECT id FROM documents WHERE steam_review_id='r2'"
    ).fetchone()[0]

    # Insert processed rows for both
    _insert_fake_processed(conn, [
        (doc_id_1, "Review one", json.dumps(["review", "one", "word"]), 3, 10),
        (doc_id_2, "Review two", json.dumps(["review", "two", "word"]), 3, 10),
    ])

    # Score only doc 1
    write_sentiment_rows(conn, [(doc_id_1, 0.5, 0.4, 0.0, 0.6)])
    conn.commit()

    unscored = fetch_unscored_documents(conn)
    unscored_ids = [row[0] for row in unscored]
    assert doc_id_1 not in unscored_ids
    assert doc_id_2 in unscored_ids
    conn.close()


def test_fetch_unscored_documents_includes_sentinels() -> None:
    """Sentinel rows (token_count=0) are included in fetch results."""
    conn = _make_test_db()
    _insert_fake_documents(conn, [("r1", "ok", True)])
    doc_id = conn.execute(
        "SELECT id FROM documents WHERE steam_review_id='r1'"
    ).fetchone()[0]

    # Insert a sentinel processed row (token_count=0)
    _insert_fake_processed(conn, [
        (doc_id, "", json.dumps([]), 0, 0),
    ])

    unscored = fetch_unscored_documents(conn)
    unscored_ids = [row[0] for row in unscored]
    assert doc_id in unscored_ids
    # Verify token_count is 0
    sentinel_row = [r for r in unscored if r[0] == doc_id][0]
    assert sentinel_row[3] == 0
    conn.close()


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def _setup_e2e_db(
    tmp_path: "Path",
    reviews: list[tuple[str, str, bool]],
) -> "Path":
    """Create a temp DB with fake documents and processed rows for e2e tests.

    Args:
        tmp_path: Pytest temporary directory.
        reviews: List of (steam_review_id, review_text, recommended) tuples.

    Returns:
        Path to the temporary database file.
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    insert_metadata(conn)
    _insert_fake_documents(conn, reviews)

    # Create matching processed_documents rows
    for steam_id, text, _rec in reviews:
        doc_id = conn.execute(
            "SELECT id FROM documents WHERE steam_review_id=?",
            (steam_id,),
        ).fetchone()[0]
        is_short = len(text.strip()) < 10
        token_count = 0 if is_short else 5
        char_count = 0 if is_short else len(text)
        cleaned = "" if is_short else text
        tokens = json.dumps([]) if is_short else json.dumps(["tok"] * 5)
        _insert_fake_processed(conn, [
            (doc_id, cleaned, tokens, token_count, char_count),
        ])

    conn.close()
    return db_path


def test_run_scores_seed_documents(tmp_path: "Path", monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed a temp DB with fake documents + processed rows, run scoring, assert nlp_results populated."""
    reviews = [
        ("e2e_1", "This extraordinary masterpiece features incredible storytelling", True),
        ("e2e_2", "The combat system handles remarkably well with responsive controls", True),
        ("e2e_3", "Absolutely horrible experience with constant crashes and bugs", False),
        ("e2e_4", "ok", True),
        ("e2e_5", "Graphics and soundtrack create an immersive atmosphere", True),
    ]
    db_path = _setup_e2e_db(tmp_path, reviews)
    monkeypatch.setattr("src.sentiment.get_db_path", lambda mode="seed": db_path)

    run(mode="seed", rebuild=False)

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
    assert count == 5
    conn.close()


def test_run_rebuild_flag_wipes_existing_rows(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --rebuild deletes prior nlp_results rows before re-scoring."""
    reviews = [
        ("rb_1", "This extraordinary masterpiece features incredible storytelling", True),
        ("rb_2", "The combat system handles remarkably well with responsive controls", False),
    ]
    db_path = _setup_e2e_db(tmp_path, reviews)
    monkeypatch.setattr("src.sentiment.get_db_path", lambda mode="seed": db_path)

    run(mode="seed", rebuild=False)

    conn = sqlite3.connect(db_path)
    count_first = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
    conn.close()
    assert count_first == 2

    # Rebuild should wipe and re-score
    run(mode="seed", rebuild=True)

    conn = sqlite3.connect(db_path)
    count_rebuild = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
    conn.close()
    assert count_rebuild == 2


def test_run_sentinel_rows_get_zero_scores(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sentinel rows (token_count=0) should get compound=0.0 in nlp_results."""
    reviews = [
        ("sent_1", "ok", True),
        ("sent_2", "This extraordinary masterpiece features incredible storytelling", True),
    ]
    db_path = _setup_e2e_db(tmp_path, reviews)
    monkeypatch.setattr("src.sentiment.get_db_path", lambda mode="seed": db_path)

    run(mode="seed", rebuild=False)

    conn = sqlite3.connect(db_path)
    # Get the sentinel doc (the one with "ok")
    sentinel_id = conn.execute(
        "SELECT id FROM documents WHERE steam_review_id='sent_1'"
    ).fetchone()[0]
    scores = conn.execute(
        "SELECT sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu "
        "FROM nlp_results WHERE document_id=?",
        (sentinel_id,),
    ).fetchone()
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] == 0.0
    assert scores[3] == 0.0
    conn.close()
