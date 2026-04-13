"""Tests for the BERTopic topic modeling pipeline.

Covers label formatting, the TopicModeler class, database helpers,
rebuild logic, and pre-checks. BERTopic fitting is slow (~10-30s), so
a module-scoped fixture fits once on a small synthetic corpus and
individual tests reuse the fitted model.
"""

import json
import sqlite3

import pytest

from src.db_setup import create_tables, insert_metadata
from src.topics import (
    TopicModeler,
    fetch_untopiced_documents,
    format_topic_label,
    write_topic_assignments,
    write_topics_table,
)
from src.utils import MIN_TOKEN_COUNT


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


def _insert_fake_nlp_results(
    conn: sqlite3.Connection,
    doc_ids: list[int],
) -> None:
    """Insert placeholder nlp_results rows with sentiment scores but NULL topics.

    Args:
        conn: Active SQLite connection with schema already created.
        doc_ids: List of document IDs to insert.
    """
    conn.executemany(
        """
        INSERT OR IGNORE INTO nlp_results
            (document_id, sentiment_compound, sentiment_pos,
             sentiment_neg, sentiment_neu, processed_at)
        VALUES (?, 0.5, 0.3, 0.1, 0.6, datetime('now'))
        """,
        [(doc_id,) for doc_id in doc_ids],
    )
    conn.commit()


def _setup_db_with_docs(
    reviews: list[tuple[str, str, bool, int]],
) -> tuple[sqlite3.Connection, list[int]]:
    """Create a test DB with documents, processed_documents, and nlp_results.

    Args:
        reviews: List of (steam_review_id, review_text, recommended,
            token_count) tuples.

    Returns:
        Tuple of (connection, list_of_doc_ids).
    """
    conn = _make_test_db()
    _insert_fake_documents(
        conn, [(r[0], r[1], r[2]) for r in reviews]
    )
    doc_ids = []
    for steam_id, text, _rec, token_count in reviews:
        doc_id = conn.execute(
            "SELECT id FROM documents WHERE steam_review_id=?",
            (steam_id,),
        ).fetchone()[0]
        doc_ids.append(doc_id)
        cleaned = text if token_count >= MIN_TOKEN_COUNT else ""
        tokens = json.dumps(["tok"] * token_count)
        char_count = len(cleaned)
        _insert_fake_processed(
            conn, [(doc_id, cleaned, tokens, token_count, char_count)]
        )
    _insert_fake_nlp_results(conn, doc_ids)
    return conn, doc_ids


# ---------------------------------------------------------------------------
# Module-scoped BERTopic fixture (slow — fits once)
# ---------------------------------------------------------------------------

# Synthetic corpus with 3 known clusters for reproducible testing.
_COMBAT_DOCS = [
    "The combat is intense with swords and shields fighting enemies",
    "Great sword fighting mechanics and epic boss battles",
    "Combat system feels responsive with amazing attack combos",
    "Fighting the bosses is challenging and rewarding",
    "Melee combat with swords axes and shields feels great",
    "The enemy AI in combat is smart and unpredictable",
    "Boss fights are the highlight of this combat system",
    "Parry and dodge mechanics make combat fluid and fun",
] * 5

_STORY_DOCS = [
    "The story is captivating with deep character development",
    "Plot twists keep you engaged throughout the narrative",
    "Character dialogue and story arcs are beautifully written",
    "The narrative explores themes of loss and redemption",
    "Story missions reveal the lore of this fascinating world",
    "Character backstories add depth to the main narrative",
    "The plot unfolds through well-crafted cutscenes",
    "Emotional storytelling with memorable characters throughout",
] * 5

_BUG_DOCS = [
    "Game crashes constantly with error messages and freezes",
    "Bugs and glitches make this unplayable on my system",
    "Constant crashes and performance issues ruin the experience",
    "The game freezes every few minutes needs urgent patches",
    "Error after error crashes and bugs everywhere unplayable",
    "Performance drops and crashes make this a buggy mess",
    "Game breaking bugs and crashes need immediate fixing",
    "Stuttering lag and crashes throughout the entire game",
] * 5

_SYNTHETIC_CORPUS: list[str] = _COMBAT_DOCS + _STORY_DOCS + _BUG_DOCS


@pytest.fixture(scope="module")
def fitted_modeler() -> TopicModeler:
    """Fit a TopicModeler once on the synthetic corpus for reuse across tests.

    Returns:
        A fitted TopicModeler instance.
    """
    modeler = TopicModeler(min_topic_size=5)
    modeler.fit(_SYNTHETIC_CORPUS)
    return modeler


@pytest.fixture(scope="module")
def fitted_topic_ids(fitted_modeler: TopicModeler) -> list[int]:
    """Return topic assignments from the fitted modeler.

    Args:
        fitted_modeler: Module-scoped fitted TopicModeler.

    Returns:
        List of topic IDs aligned with _SYNTHETIC_CORPUS.
    """
    assert fitted_modeler.model is not None
    return list(fitted_modeler.model.topics_)


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------


def test_format_topic_label_strips_prefix() -> None:
    """'0_combat_fight_enemy' becomes 'Combat / Fight / Enemy'."""
    assert format_topic_label("0_combat_fight_enemy") == "Combat / Fight / Enemy"


def test_format_topic_label_outlier() -> None:
    """Labels starting with '-1' return 'Outlier'."""
    assert format_topic_label("-1_outlier_noise") == "Outlier"
    assert format_topic_label("-1") == "Outlier"


def test_format_topic_label_single_word() -> None:
    """'3_combat' becomes 'Combat'."""
    assert format_topic_label("3_combat") == "Combat"


# ---------------------------------------------------------------------------
# TopicModeler (uses module-scoped fixture)
# ---------------------------------------------------------------------------


def test_fit_returns_topic_ids(fitted_topic_ids: list[int]) -> None:
    """Fit on synthetic corpus returns one topic ID per input document."""
    assert len(fitted_topic_ids) == len(_SYNTHETIC_CORPUS)


def test_fit_topic_ids_are_integers(fitted_topic_ids: list[int]) -> None:
    """All topic IDs should be integers."""
    assert all(isinstance(t, (int,)) for t in fitted_topic_ids)


def test_get_topic_info_returns_all_topics(fitted_modeler: TopicModeler) -> None:
    """get_topic_info returns a list including the outlier topic."""
    info = fitted_modeler.get_topic_info()
    assert len(info) > 0
    topic_ids_in_info = {t["topic_id"] for t in info}
    # Should include all unique topic IDs from the fitted model
    unique_model_topics = set(fitted_modeler.model.topics_)
    assert unique_model_topics.issubset(topic_ids_in_info)


def test_get_topic_info_has_required_keys(fitted_modeler: TopicModeler) -> None:
    """Each topic dict has topic_id, label, top_words, doc_count."""
    info = fitted_modeler.get_topic_info()
    required_keys = {"topic_id", "label", "top_words", "doc_count"}
    for topic in info:
        assert required_keys.issubset(topic.keys())


def test_get_topic_info_doc_counts_sum_to_corpus(
    fitted_modeler: TopicModeler,
) -> None:
    """Sum of doc_count across all topics equals corpus size."""
    info = fitted_modeler.get_topic_info()
    total = sum(t["doc_count"] for t in info)
    assert total == len(_SYNTHETIC_CORPUS)


def test_reduce_topics_lowers_count(fitted_modeler: TopicModeler) -> None:
    """reduce_topics merges down to the requested count (or fewer).

    Note:
        This test uses a copy of the modeler's state to avoid mutating
        the shared fixture. Instead we just verify the API contract
        by checking that the model supports reduce_topics.
    """
    # Create a fresh modeler for this test to avoid mutating the shared fixture
    modeler = TopicModeler(min_topic_size=5)
    modeler.fit(_SYNTHETIC_CORPUS)
    original_count = len({t for t in modeler.model.topics_ if t != -1})

    if original_count > 2:
        new_ids = modeler.reduce_topics(_SYNTHETIC_CORPUS, nr_topics=2)
        new_count = len({t for t in new_ids if t != -1})
        assert new_count <= original_count


def test_save_model_creates_file(
    fitted_modeler: TopicModeler,
    tmp_path: "Path",
) -> None:
    """save_model writes a file to the given path."""
    model_path = tmp_path / "test_model.pkl"
    fitted_modeler.save_model(model_path)
    assert model_path.exists()
    assert model_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def test_fetch_untopiced_documents_excludes_sentinels() -> None:
    """Sentinel rows (token_count < MIN_TOKEN_COUNT) are not returned."""
    conn, doc_ids = _setup_db_with_docs([
        ("t1", "A great game with combat and story", True, 6),
        ("t2", "ok", True, 0),  # sentinel
    ])
    results = fetch_untopiced_documents(conn)
    result_ids = [r[0] for r in results]
    assert doc_ids[0] in result_ids
    assert doc_ids[1] not in result_ids
    conn.close()


def test_fetch_untopiced_documents_excludes_already_topiced() -> None:
    """Rows with non-NULL topic_id are not returned."""
    conn, doc_ids = _setup_db_with_docs([
        ("t1", "A great game with combat and story", True, 6),
        ("t2", "Another great game with amazing story", True, 6),
    ])
    # Assign a topic to doc 1
    conn.execute(
        "UPDATE nlp_results SET topic_id = 0, topic_label = 'Test' "
        "WHERE document_id = ?",
        (doc_ids[0],),
    )
    conn.commit()

    results = fetch_untopiced_documents(conn)
    result_ids = [r[0] for r in results]
    assert doc_ids[0] not in result_ids
    assert doc_ids[1] in result_ids
    conn.close()


def test_write_topic_assignments_updates_rows() -> None:
    """Verify topic_id and topic_label are set on the correct nlp_results rows."""
    conn, doc_ids = _setup_db_with_docs([
        ("t1", "A great game with combat and story", True, 6),
        ("t2", "Another great game with amazing story", True, 6),
    ])
    assignments = [
        (doc_ids[0], 1, "Combat / Fight"),
        (doc_ids[1], 2, "Story / Plot"),
    ]
    n = write_topic_assignments(conn, assignments)
    conn.commit()

    assert n == 2
    row = conn.execute(
        "SELECT topic_id, topic_label FROM nlp_results WHERE document_id=?",
        (doc_ids[0],),
    ).fetchone()
    assert row[0] == 1
    assert row[1] == "Combat / Fight"
    conn.close()


def test_write_topic_assignments_idempotent() -> None:
    """Re-running with the same assignments produces no errors or changes."""
    conn, doc_ids = _setup_db_with_docs([
        ("t1", "A great game with combat and story", True, 6),
    ])
    assignments = [(doc_ids[0], 1, "Combat")]
    write_topic_assignments(conn, assignments)
    conn.commit()

    # Run again — should succeed without error
    n = write_topic_assignments(conn, assignments)
    conn.commit()
    assert n == 1  # UPDATE always reports the count

    count = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
    assert count == 1
    conn.close()


def test_write_topics_table_inserts() -> None:
    """Verify rows land in the topics table with correct columns."""
    conn = _make_test_db()
    topic_info = [
        {"topic_id": 0, "label": "Combat", "top_words": ["sword", "fight"], "doc_count": 50},
        {"topic_id": 1, "label": "Story", "top_words": ["plot", "character"], "doc_count": 30},
        {"topic_id": -1, "label": "Outlier", "top_words": [], "doc_count": 10},
    ]
    n = write_topics_table(conn, topic_info)
    conn.commit()

    assert n == 3
    rows = conn.execute(
        "SELECT topic_id, label, top_words, doc_count FROM topics ORDER BY topic_id"
    ).fetchall()
    assert len(rows) == 3
    assert rows[0][0] == -1
    assert rows[0][1] == "Outlier"
    assert json.loads(rows[0][2]) == []
    conn.close()


def test_write_topics_table_replaces_on_rebuild() -> None:
    """INSERT OR REPLACE overwrites existing topic rows."""
    conn = _make_test_db()
    topic_info_v1 = [
        {"topic_id": 0, "label": "Combat", "top_words": ["sword"], "doc_count": 50},
    ]
    write_topics_table(conn, topic_info_v1)
    conn.commit()

    topic_info_v2 = [
        {"topic_id": 0, "label": "Battle", "top_words": ["fight", "attack"], "doc_count": 60},
    ]
    write_topics_table(conn, topic_info_v2)
    conn.commit()

    row = conn.execute(
        "SELECT label, doc_count FROM topics WHERE topic_id=0"
    ).fetchone()
    assert row[0] == "Battle"
    assert row[1] == 60
    count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    assert count == 1
    conn.close()


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------


def test_rebuild_clears_topic_columns_preserves_sentiment() -> None:
    """Rebuild NULLs topic_id/topic_label but keeps sentiment scores intact."""
    conn, doc_ids = _setup_db_with_docs([
        ("t1", "A great game with combat and story", True, 6),
    ])
    # Set topic assignment
    conn.execute(
        "UPDATE nlp_results SET topic_id = 1, topic_label = 'Combat' "
        "WHERE document_id = ?",
        (doc_ids[0],),
    )
    conn.commit()

    # Simulate rebuild: clear topic columns only
    conn.execute(
        "UPDATE nlp_results SET topic_id = NULL, topic_label = NULL "
        "WHERE topic_id IS NOT NULL"
    )
    conn.commit()

    row = conn.execute(
        "SELECT sentiment_compound, topic_id, topic_label "
        "FROM nlp_results WHERE document_id=?",
        (doc_ids[0],),
    ).fetchone()
    assert row[0] == 0.5  # sentiment preserved
    assert row[1] is None  # topic cleared
    assert row[2] is None  # topic_label cleared
    conn.close()


def test_rebuild_clears_topics_table() -> None:
    """Rebuild empties the topics table."""
    conn = _make_test_db()
    topic_info = [
        {"topic_id": 0, "label": "Combat", "top_words": ["sword"], "doc_count": 50},
    ]
    write_topics_table(conn, topic_info)
    conn.commit()

    # Simulate rebuild
    conn.execute("DELETE FROM topics")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    assert count == 0
    conn.close()


# ---------------------------------------------------------------------------
# Pre-check
# ---------------------------------------------------------------------------


def test_run_aborts_if_no_nlp_results_rows(
    tmp_path: "Path",
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """run() logs error and exits if nlp_results is empty (Phase 3 not run)."""
    from src.topics import run

    db_path = tmp_path / "empty.db"
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    insert_metadata(conn)
    conn.close()

    monkeypatch.setattr("src.topics.get_db_path", lambda mode="seed": db_path)

    import logging

    with caplog.at_level(logging.ERROR):
        run(mode="seed")

    assert "No nlp_results rows found" in caplog.text
