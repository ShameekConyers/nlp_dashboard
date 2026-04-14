"""Unit tests for dashboard.queries against in-memory SQLite."""

import sqlite3
from datetime import datetime, timezone

import pandas as pd
import pytest

from dashboard.queries import (
    _build_where,
    get_filter_options,
    get_overview_kpis,
    get_reviews_count,
    get_reviews_table,
    get_sentiment_by_genre,
    get_sentiment_by_topic,
    get_sentiment_distribution,
    get_sentiment_over_time,
    get_sentiment_vs_recommendation,
    get_topic_distribution,
)


# ---------------------------------------------------------------------------
# Fixture — in-memory SQLite with representative test data
# ---------------------------------------------------------------------------


def _ts(year: int, month: int) -> int:
    """Return a Unix timestamp for the first day of the given month."""
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """Create an in-memory SQLite database with test data.

    Returns:
        A populated sqlite3.Connection.
    """
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE metadata (
            app_id   INTEGER PRIMARY KEY,
            app_name TEXT NOT NULL,
            genre    TEXT
        );
        CREATE TABLE documents (
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
        CREATE TABLE processed_documents (
            document_id  INTEGER PRIMARY KEY REFERENCES documents(id),
            cleaned_text TEXT NOT NULL,
            tokens       TEXT NOT NULL,
            token_count  INTEGER NOT NULL,
            char_count   INTEGER NOT NULL,
            processed_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE nlp_results (
            document_id       INTEGER PRIMARY KEY REFERENCES documents(id),
            sentiment_compound REAL,
            sentiment_pos      REAL,
            sentiment_neg      REAL,
            sentiment_neu      REAL,
            topic_id           INTEGER,
            topic_label        TEXT,
            processed_at       TEXT
        );
        CREATE TABLE topics (
            topic_id   INTEGER PRIMARY KEY,
            label      TEXT NOT NULL,
            top_words  TEXT NOT NULL,
            doc_count  INTEGER NOT NULL,
            processed_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX idx_documents_app_id ON documents(app_id);
        CREATE INDEX idx_documents_recommended ON documents(recommended);
        CREATE INDEX idx_nlp_results_topic_id ON nlp_results(topic_id);
        CREATE INDEX idx_processed_documents_token_count
            ON processed_documents(token_count);
    """)

    # Two games, two genres
    db.executemany(
        "INSERT INTO metadata (app_id, app_name, genre) VALUES (?, ?, ?)",
        [(100, "TestRPG", "RPG"), (200, "TestFPS", "FPS")],
    )

    # Topics (including outlier)
    db.executemany(
        "INSERT INTO topics (topic_id, label, top_words, doc_count) VALUES (?, ?, ?, ?)",
        [
            (-1, "Outlier", "misc random noise", 2),
            (0, "Combat", "fight sword enemy battle", 3),
            (1, "Story", "plot character narrative arc", 2),
        ],
    )

    # Reviews — mix of kept (token_count >= 3) and sentinels (token_count < 3)
    ts_jan = _ts(2025, 1)
    ts_mar = _ts(2025, 3)
    ts_jun = _ts(2025, 6)

    reviews = [
        # id, steam_id, app_id, text, recommended, ts
        (1, "r1", 100, "Great combat system and story", True, ts_jan),
        (2, "r2", 100, "Terrible bugs everywhere in this RPG", False, ts_jan),
        (3, "r3", 100, "Amazing game with deep lore and quests", True, ts_mar),
        (4, "r4", 200, "Fast paced FPS action with great gunplay", True, ts_mar),
        (5, "r5", 200, "Hackers ruin this game completely", False, ts_jun),
        # Sentinel — will be excluded by token_count filter
        (6, "r6", 200, "ok", True, ts_jun),
        # Outlier topic review
        (7, "r7", 100, "Random miscellaneous review text", True, ts_jun),
    ]
    db.executemany(
        """INSERT INTO documents
           (id, steam_review_id, app_id, review_text, recommended,
            playtime_at_review, playtime_forever, timestamp_created)
           VALUES (?, ?, ?, ?, ?, 100, 200, ?)""",
        reviews,
    )

    # Processed documents — sentinel has token_count=1
    processed = [
        (1, "great combat system story", "great combat system story", 4, 25),
        (2, "terrible bugs everywhere rpg", "terrible bugs everywhere rpg", 4, 28),
        (3, "amazing game deep lore quests", "amazing game deep lore quests", 5, 29),
        (4, "fast paced fps action gunplay", "fast paced fps action gunplay", 5, 29),
        (5, "hackers ruin game completely", "hackers ruin game completely", 4, 27),
        (6, "ok", "ok", 1, 2),  # sentinel
        (7, "random miscellaneous review text", "random miscellaneous review text", 4, 30),
    ]
    db.executemany(
        """INSERT INTO processed_documents
           (document_id, cleaned_text, tokens, token_count, char_count)
           VALUES (?, ?, ?, ?, ?)""",
        processed,
    )

    # NLP results — sentiment + topics
    nlp = [
        (1, 0.65, 0.3, 0.0, 0.7, 0, "Combat"),
        (2, -0.55, 0.0, 0.4, 0.6, 0, "Combat"),
        (3, 0.80, 0.4, 0.0, 0.6, 1, "Story"),
        (4, 0.70, 0.35, 0.0, 0.65, 0, "Combat"),
        (5, -0.60, 0.0, 0.45, 0.55, 1, "Story"),
        (6, 0.00, 0.0, 0.0, 1.0, None, None),  # sentinel
        (7, 0.30, 0.2, 0.05, 0.75, -1, "Outlier"),  # outlier topic
    ]
    db.executemany(
        """INSERT INTO nlp_results
           (document_id, sentiment_compound, sentiment_pos, sentiment_neg,
            sentiment_neu, topic_id, topic_label)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        nlp,
    )

    db.commit()
    return db


# ---------------------------------------------------------------------------
# _build_where tests
# ---------------------------------------------------------------------------


class TestBuildWhere:
    """Tests for the _build_where helper."""

    def test_no_filters(self) -> None:
        """With no filters, only the token_count clause is present."""
        where, params = _build_where()
        assert "token_count" in where
        assert len(params) == 1

    def test_genre_filter(self) -> None:
        """Genre filter produces an IN clause."""
        where, params = _build_where(genres=["RPG", "FPS"])
        assert "m.genre IN (?,?)" in where
        assert "RPG" in params
        assert "FPS" in params

    def test_multiple_filters(self) -> None:
        """Multiple filters produce multiple AND clauses."""
        where, params = _build_where(
            genres=["RPG"],
            sentiment_min=-0.5,
            date_start=1000000,
        )
        assert where.count(" AND ") == 3
        assert len(params) == 4

    def test_sentiment_range(self) -> None:
        """Sentiment min/max produce >= and <= clauses."""
        where, params = _build_where(sentiment_min=-0.3, sentiment_max=0.7)
        assert "sentiment_compound >= ?" in where
        assert "sentiment_compound <= ?" in where
        assert -0.3 in params
        assert 0.7 in params


# ---------------------------------------------------------------------------
# get_filter_options tests
# ---------------------------------------------------------------------------


class TestGetFilterOptions:
    """Tests for get_filter_options."""

    def test_returns_all_genres(self, conn: sqlite3.Connection) -> None:
        """All genres present in metadata are returned."""
        opts = get_filter_options(conn)
        assert set(opts["genres"]) == {"RPG", "FPS"}

    def test_excludes_outlier_from_topics(self, conn: sqlite3.Connection) -> None:
        """Outlier topic (-1) is excluded from topic_labels."""
        opts = get_filter_options(conn)
        assert -1 not in opts["topic_labels"]


# ---------------------------------------------------------------------------
# get_overview_kpis tests
# ---------------------------------------------------------------------------


class TestGetOverviewKpis:
    """Tests for get_overview_kpis."""

    def test_keys(self, conn: sqlite3.Connection) -> None:
        """All expected keys are present in the result dict."""
        kpis = get_overview_kpis(conn)
        expected_keys = {
            "total_reviews", "avg_sentiment", "pct_positive",
            "pct_negative", "num_topics", "num_games",
        }
        assert set(kpis.keys()) == expected_keys

    def test_with_genre_filter(self, conn: sqlite3.Connection) -> None:
        """Filtering by genre produces fewer reviews than the total."""
        kpis_all = get_overview_kpis(conn)
        kpis_rpg = get_overview_kpis(conn, genres=["RPG"])
        assert kpis_rpg["total_reviews"] < kpis_all["total_reviews"]

    def test_excludes_sentinels(self, conn: sqlite3.Connection) -> None:
        """Sentinel review (id=6) is excluded from the count."""
        kpis = get_overview_kpis(conn)
        # 7 total rows, 1 sentinel => 6 kept
        assert kpis["total_reviews"] == 6


# ---------------------------------------------------------------------------
# Sentiment query tests
# ---------------------------------------------------------------------------


class TestSentimentQueries:
    """Tests for sentiment-related query functions."""

    def test_distribution_returns_dataframe(self, conn: sqlite3.Connection) -> None:
        """get_sentiment_distribution returns a DataFrame with correct column."""
        df = get_sentiment_distribution(conn)
        assert isinstance(df, pd.DataFrame)
        assert "sentiment_compound" in df.columns
        assert len(df) == 6  # excludes sentinel

    def test_by_genre_columns(self, conn: sqlite3.Connection) -> None:
        """get_sentiment_by_genre returns expected columns."""
        df = get_sentiment_by_genre(conn)
        assert list(df.columns) == ["genre", "avg_sentiment", "review_count"]
        assert len(df) == 2  # RPG and FPS

    def test_vs_recommendation_two_groups(self, conn: sqlite3.Connection) -> None:
        """get_sentiment_vs_recommendation returns two groups (True/False)."""
        df = get_sentiment_vs_recommendation(conn)
        assert len(df) == 2
        assert set(df["recommended"]) == {0, 1}

    def test_over_time_monthly_grouping(self, conn: sqlite3.Connection) -> None:
        """get_sentiment_over_time groups by YYYY-MM format."""
        df = get_sentiment_over_time(conn)
        assert "month" in df.columns
        # All months should match YYYY-MM pattern
        for month_val in df["month"]:
            assert len(month_val) == 7
            assert month_val[4] == "-"


# ---------------------------------------------------------------------------
# Topic query tests
# ---------------------------------------------------------------------------


class TestTopicQueries:
    """Tests for topic-related query functions."""

    def test_distribution_excludes_outliers(self, conn: sqlite3.Connection) -> None:
        """get_topic_distribution excludes topic_id -1."""
        df = get_topic_distribution(conn)
        assert -1 not in df["topic_id"].values

    def test_sentiment_by_topic_columns(self, conn: sqlite3.Connection) -> None:
        """get_sentiment_by_topic returns expected columns."""
        df = get_sentiment_by_topic(conn)
        expected = {"topic_id", "label", "avg_sentiment", "review_count", "pct_positive"}
        assert set(df.columns) == expected


# ---------------------------------------------------------------------------
# Reviews table tests
# ---------------------------------------------------------------------------


class TestReviewsTable:
    """Tests for get_reviews_table and get_reviews_count."""

    def test_pagination(self, conn: sqlite3.Connection) -> None:
        """Limit and offset control result size."""
        df_full = get_reviews_table(conn, limit=100)
        df_page = get_reviews_table(conn, limit=2, offset=0)
        assert len(df_page) == 2
        assert len(df_page) < len(df_full)

    def test_truncates_text(self, conn: sqlite3.Connection) -> None:
        """Review text is truncated to at most 300 characters."""
        df = get_reviews_table(conn, limit=100)
        for text in df["review_text"]:
            assert len(text) <= 300

    def test_count_matches_table(self, conn: sqlite3.Connection) -> None:
        """get_reviews_count matches the total number of rows from get_reviews_table."""
        count = get_reviews_count(conn)
        df = get_reviews_table(conn, limit=1000)
        assert count == len(df)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and empty results."""

    def test_empty_result_no_error(self, conn: sqlite3.Connection) -> None:
        """Filters that match nothing return an empty DataFrame, not an error."""
        df = get_sentiment_distribution(
            conn, game_names=["NonexistentGame"]
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_kpis_no_error(self, conn: sqlite3.Connection) -> None:
        """get_overview_kpis with zero-result filters returns zeroed values."""
        kpis = get_overview_kpis(conn, game_names=["NonexistentGame"])
        assert kpis["total_reviews"] == 0
        assert kpis["avg_sentiment"] == 0.0
