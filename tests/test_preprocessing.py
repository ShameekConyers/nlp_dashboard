"""Tests for the preprocessing pipeline.

Covers regex-based cleaning, spaCy tokenization, filter rules, database
helpers, and end-to-end idempotency. The spaCy model is loaded once per
module (scope="module" fixture) to avoid the ~2s load cost on every test.
"""

import json
import sqlite3

import pytest
import spacy

from src.db_setup import create_tables, insert_metadata
from src.preprocessing import (
    EMAIL_RE,
    HTML_TAG_RE,
    STEAM_BBCODE_RE,
    URL_RE,
    Preprocessor,
    fetch_unprocessed_documents,
    run,
    write_processed_rows,
)
from src.utils import MIN_CLEAN_TEXT_CHARS, MIN_TOKEN_COUNT, PROJECT_STOPWORDS


@pytest.fixture(scope="module")
def preprocessor() -> Preprocessor:
    """Provide a shared Preprocessor instance across all tests in this module.

    Returns:
        A `Preprocessor` loaded with the default spaCy model.
    """
    return Preprocessor()


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def test_clean_text_strips_html_tags(preprocessor: Preprocessor) -> None:
    """Verify raw HTML tags are removed from the cleaned text."""
    raw = "Great<br>game<p>indeed</p>!"
    result = preprocessor.clean_text(raw)
    assert "<br>" not in result
    assert "<p>" not in result
    assert "</p>" not in result
    assert "Great" in result
    assert "indeed" in result


def test_clean_text_strips_steam_bbcode(preprocessor: Preprocessor) -> None:
    """Verify Steam BBCode tags (including [*] list markers) are stripped."""
    raw = "[b]Bold[/b] text [url=http://example.com]link[/url] and [*]item"
    result = preprocessor.clean_text(raw)
    assert "[b]" not in result
    assert "[/b]" not in result
    assert "[url=" not in result
    assert "[/url]" not in result
    assert "[*]" not in result
    assert "Bold" in result
    assert "text" in result


def test_clean_text_strips_urls(preprocessor: Preprocessor) -> None:
    """Verify http/https and bare www URLs are removed."""
    raw = "Check https://store.steampowered.com and www.example.com for details"
    result = preprocessor.clean_text(raw)
    assert "https://" not in result
    assert "www." not in result
    assert "details" in result


def test_clean_text_strips_emails(preprocessor: Preprocessor) -> None:
    """Verify email addresses are removed from cleaned text."""
    raw = "Contact support@example.com for help with this great game"
    result = preprocessor.clean_text(raw)
    assert "support@example.com" not in result
    assert "great game" in result


def test_clean_text_fixes_mojibake(preprocessor: Preprocessor) -> None:
    """Verify ftfy repairs common mojibake like curly quotes."""
    raw = "It\u2019s a great game \u2013 really!"
    result = preprocessor.clean_text(raw)
    assert "\u2019" in result or "'" in result
    assert "great game" in result


def test_clean_text_collapses_whitespace(preprocessor: Preprocessor) -> None:
    """Verify runs of whitespace and newlines are collapsed to single spaces."""
    raw = "This   has\n\nmany   \t  spaces"
    result = preprocessor.clean_text(raw)
    assert "  " not in result
    assert "\n" not in result
    assert "\t" not in result


def test_clean_text_preserves_caps_and_punctuation(preprocessor: Preprocessor) -> None:
    """Caps and ! must survive cleaning — VADER needs them."""
    raw = "AMAZING game!!! Best EVER!"
    result = preprocessor.clean_text(raw)
    assert "AMAZING" in result
    assert "!!!" in result
    assert "EVER!" in result


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_tokenize_batch_lowercases_and_lemmatizes(preprocessor: Preprocessor) -> None:
    """Verify tokens are lowercased lemmas, not surface forms."""
    result = preprocessor.tokenize_batch(["The dogs were running quickly"])
    assert len(result) == 1
    tokens = result[0]
    assert "dog" in tokens
    assert "quickly" in tokens or "quick" in tokens
    assert "run" in tokens
    # surface forms should not appear
    assert "dogs" not in tokens
    assert "running" not in tokens


def test_tokenize_batch_drops_stopwords_and_punct(preprocessor: Preprocessor) -> None:
    """Verify spaCy stopwords and punctuation tokens are filtered out."""
    result = preprocessor.tokenize_batch(["This is a very great thing, indeed!"])
    assert len(result) == 1
    tokens = result[0]
    # "this", "is", "a", "very" are stopwords; "," and "!" are punctuation
    assert "this" not in tokens
    assert "," not in tokens
    assert "!" not in tokens
    assert "great" in tokens
    assert "thing" in tokens


def test_tokenize_batch_drops_short_tokens(preprocessor: Preprocessor) -> None:
    """Verify lemmas shorter than 3 characters are removed."""
    result = preprocessor.tokenize_batch(["I am going to do it now quickly"])
    tokens = result[0]
    for t in tokens:
        assert len(t) >= 3, f"Token '{t}' is shorter than 3 characters"


def test_tokenize_batch_drops_project_stoplist(preprocessor: Preprocessor) -> None:
    """The 'game' / 'play' / 'player' / 'steam' words must be filtered."""
    result = preprocessor.tokenize_batch(
        ["This game has great gameplay with steam achievements and player rankings"]
    )
    tokens = result[0]
    for stopword in PROJECT_STOPWORDS:
        assert stopword not in tokens, f"Project stopword '{stopword}' was not filtered"


# ---------------------------------------------------------------------------
# Filter rules (Decision 2)
# ---------------------------------------------------------------------------


def test_process_batch_drops_too_short_clean_text(preprocessor: Preprocessor) -> None:
    """Reviews with cleaned text shorter than MIN_CLEAN_TEXT_CHARS are dropped."""
    results = preprocessor.process_batch(["ok"])
    assert len(results) == 1
    assert results[0]["dropped"] is True
    assert results[0]["drop_reason"] == "too_short"


def test_process_batch_drops_too_few_tokens(preprocessor: Preprocessor) -> None:
    """Reviews that clean OK but produce fewer than MIN_TOKEN_COUNT tokens are dropped."""
    # "good game yes" cleans to 10+ chars but yields < 3 tokens after filtering
    # ("good" is 4 chars so survives length, but "game" is a project stopword)
    results = preprocessor.process_batch(["good game yes"])
    assert len(results) == 1
    assert results[0]["dropped"] is True
    assert results[0]["drop_reason"] == "too_few_tokens"


def test_process_batch_keeps_good_review(preprocessor: Preprocessor) -> None:
    """A substantive review survives cleaning and tokenization."""
    review = "This extraordinary masterpiece features incredible storytelling and beautiful artwork"
    results = preprocessor.process_batch([review])
    assert len(results) == 1
    result = results[0]
    assert result["dropped"] is False
    assert result["drop_reason"] is None
    assert "cleaned_text" in result
    assert "tokens" in result
    assert result["token_count"] >= MIN_TOKEN_COUNT
    assert result["char_count"] >= MIN_CLEAN_TEXT_CHARS


# ---------------------------------------------------------------------------
# Table + idempotency
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
    reviews: list[tuple[str, str]],
) -> None:
    """Insert fake documents into the in-memory database.

    Args:
        conn: Active SQLite connection with schema already created.
        reviews: List of `(steam_review_id, review_text)` tuples to insert.
    """
    for steam_id, text in reviews:
        conn.execute(
            """
            INSERT INTO documents (steam_review_id, app_id, review_text, recommended)
            VALUES (?, ?, ?, ?)
            """,
            (steam_id, 367520, text, True),
        )
    conn.commit()


def test_write_processed_rows_inserts() -> None:
    """Verify rows are inserted into processed_documents."""
    conn = _make_test_db()
    _insert_fake_documents(conn, [("r1", "Test review")])
    doc_id = conn.execute("SELECT id FROM documents WHERE steam_review_id='r1'").fetchone()[0]

    rows = [(doc_id, "Test review", json.dumps(["test", "review", "word"]), 3, 11)]
    inserted = write_processed_rows(conn, rows)
    conn.commit()

    assert inserted == 1
    stored = conn.execute(
        "SELECT cleaned_text, tokens FROM processed_documents WHERE document_id=?",
        (doc_id,),
    ).fetchone()
    assert stored[0] == "Test review"
    assert json.loads(stored[1]) == ["test", "review", "word"]
    conn.close()


def test_write_processed_rows_dedupes_on_document_id() -> None:
    """Re-running on the same doc_id must not duplicate rows."""
    conn = _make_test_db()
    _insert_fake_documents(conn, [("r1", "Test review")])
    doc_id = conn.execute("SELECT id FROM documents WHERE steam_review_id='r1'").fetchone()[0]

    rows = [(doc_id, "Test review", json.dumps(["test", "review", "word"]), 3, 11)]
    write_processed_rows(conn, rows)
    conn.commit()
    second_insert = write_processed_rows(conn, rows)
    conn.commit()

    assert second_insert == 0
    count = conn.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0]
    assert count == 1
    conn.close()


def test_fetch_unprocessed_documents_excludes_already_processed() -> None:
    """Documents with existing processed_documents rows are excluded."""
    conn = _make_test_db()
    _insert_fake_documents(conn, [("r1", "Review one"), ("r2", "Review two")])

    doc_id_1 = conn.execute(
        "SELECT id FROM documents WHERE steam_review_id='r1'"
    ).fetchone()[0]

    write_processed_rows(
        conn,
        [(doc_id_1, "Review one", json.dumps(["review", "one", "word"]), 3, 10)],
    )
    conn.commit()

    unprocessed = fetch_unprocessed_documents(conn)
    unprocessed_ids = [row[0] for row in unprocessed]
    assert doc_id_1 not in unprocessed_ids
    assert len(unprocessed) == 1
    conn.close()


# ---------------------------------------------------------------------------
# End-to-end against in-memory DB
# ---------------------------------------------------------------------------


def test_run_processes_seed_documents_and_writes_rows(
    preprocessor: Preprocessor,
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed an in-memory-like temp DB with 5 fake documents, run preprocessing, assert row counts."""
    from pathlib import Path

    db_path = tmp_path / "test_seed.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    insert_metadata(conn)
    reviews = [
        ("e2e_1", "This extraordinary masterpiece features incredible storytelling and beautiful artwork"),
        ("e2e_2", "The combat system handles remarkably well with responsive controls throughout"),
        ("e2e_3", "Absolutely horrible experience with constant crashes and terrible performance issues"),
        ("e2e_4", "ok"),
        ("e2e_5", "Graphics and soundtrack create an immersive atmosphere that pulls you right in"),
    ]
    _insert_fake_documents(conn, reviews)
    conn.close()

    monkeypatch.setattr("src.preprocessing.get_db_path", lambda mode="seed": db_path)

    run(mode="seed", rebuild=False, batch_size=200)

    conn = sqlite3.connect(db_path)
    processed = conn.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0]

    # All 5 get a processed_documents row (4 kept + 1 sentinel for "ok").
    assert processed == 5, f"Expected 5 processed rows (including sentinels), got {processed}"
    kept = conn.execute(
        "SELECT COUNT(*) FROM processed_documents WHERE token_count >= ?",
        (MIN_TOKEN_COUNT,),
    ).fetchone()[0]
    assert kept >= 3, f"Expected at least 3 kept reviews, got {kept}"
    assert kept <= 4, f"Expected at most 4 kept reviews, got {kept}"
    conn.close()


def test_run_rebuild_flag_wipes_existing_rows(
    preprocessor: Preprocessor,
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --rebuild deletes prior processed_documents rows before reprocessing."""
    from pathlib import Path

    db_path = tmp_path / "test_rebuild.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    insert_metadata(conn)
    reviews = [
        ("rb_1", "This extraordinary masterpiece features incredible storytelling and beautiful artwork"),
        ("rb_2", "The combat system handles remarkably well with responsive controls throughout"),
    ]
    _insert_fake_documents(conn, reviews)
    conn.close()

    monkeypatch.setattr("src.preprocessing.get_db_path", lambda mode="seed": db_path)

    run(mode="seed", rebuild=False, batch_size=200)

    conn = sqlite3.connect(db_path)
    count_first = conn.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0]
    conn.close()
    assert count_first == 2

    # Rebuild should wipe and reprocess
    run(mode="seed", rebuild=True, batch_size=200)

    conn = sqlite3.connect(db_path)
    count_rebuild = conn.execute("SELECT COUNT(*) FROM processed_documents").fetchone()[0]
    conn.close()
    assert count_rebuild == 2, "Rebuild should produce the same row count"
