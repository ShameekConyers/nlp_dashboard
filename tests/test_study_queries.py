"""Unit tests for the dashboard's study queries against in-memory SQLite."""

import json
import sqlite3

import pytest

from dashboard.queries import (
    get_aspect_examples,
    get_cohort_options,
    get_cohort_stats,
    get_filtered_theme_points,
    get_game_stats,
    get_latest_run,
    get_pull_windows,
    get_recommend_share_by_game,
    get_theme_stats,
    has_study_tables,
)
from src.db_setup import create_study_tables


@pytest.fixture()
def conn() -> sqlite3.Connection:
    """Build a two-game study database with one stats run.

    Game 1 (cohort a) has three kept reviews, game 2 (cohort b) has one.
    Review 1 mentions ``map_design`` and sits in topic 0.

    Returns:
        A populated connection.
    """
    db = sqlite3.connect(":memory:")
    create_study_tables(db)
    db.executemany("INSERT INTO metadata VALUES (?, ?, ?)", [(1, "G1", "Tactics"), (2, "G2", "4X")])
    db.executemany(
        "INSERT INTO study_apps VALUES (?, ?, '', '[]', ?)", [(1, "a", 0), (2, "b", 1)]
    )
    docs = [(1, 1, "Great maps. Fun.", 1), (2, 1, "Bad AI overall here.", 0),
            (3, 1, "Solid game really.", 1), (4, 2, "Too slow for me.", 0)]
    for doc_id, app_id, text, rec in docs:
        db.execute(
            "INSERT INTO documents (id, steam_review_id, app_id, review_text, recommended, "
            "timestamp_created) VALUES (?, ?, ?, ?, ?, ?)",
            (doc_id, str(doc_id), app_id, text, rec, 1700000000 + doc_id),
        )
        db.execute("INSERT INTO processed_documents VALUES (?, ?, '[]', 4, ?, NULL)",
                   (doc_id, text, len(text)))
        db.execute("INSERT INTO nlp_results (document_id, sentiment_compound, topic_id) "
                   "VALUES (?, ?, ?)", (doc_id, 0.5 if rec else -0.5, 0 if doc_id == 1 else -1))
    db.execute("INSERT INTO topics VALUES (0, 'Maps', '[]', 1, NULL)")
    db.execute("INSERT INTO aspect_mentions VALUES (1, 'map_design', 0, 'maps', 0, 11, 'v1')")
    db.execute("INSERT INTO analysis_runs (run_id, stage, params, metrics) "
               "VALUES (1, 'stats', '{}', ?)", (json.dumps({"n_comparisons": 3}),))
    db.execute("INSERT INTO game_stats (run_id, app_id, n_seed, n_too_short, n_too_few_tokens, "
               "n_non_english, n_kept, n_recommended, n_not_recommended, eligible) "
               "VALUES (1, 1, 3, 0, 0, 0, 3, 2, 1, 1)")
    db.execute("INSERT INTO cohort_stats (run_id, scope_type, scope_id, n_games, n_eligible, "
               "n_kept) VALUES (1, 'cohort', 'a', 1, 1, 3)")
    db.execute("INSERT INTO theme_stats (run_id, lens, theme, scope_type, scope_id, n_docs, "
               "k_docs) VALUES (1, 'topic', '0', 'game', '1', 3, 1)")
    db.execute("INSERT INTO pull_runs (app_id, params, pulled_at, pages, reviews_collected, cap, "
               "cap_hit, stop_reason) VALUES (1, '{}', 't', 1, 3, 100, 0, 'exhausted')")
    db.commit()
    return db


def test_has_study_tables(conn: sqlite3.Connection) -> None:
    """A study DB with stats is detected; a bare DB is not."""
    assert has_study_tables(conn)
    assert not has_study_tables(sqlite3.connect(":memory:"))


def test_get_latest_run(conn: sqlite3.Connection) -> None:
    """The newest stats run and its metrics come back."""
    assert get_latest_run(conn) == (1, {"n_comparisons": 3})
    assert get_latest_run(conn, "reference_checks") is None


def test_get_cohort_options(conn: sqlite3.Connection) -> None:
    """Cohorts come back in display order."""
    assert get_cohort_options(conn) == ["a", "b"]


def test_get_game_stats(conn: sqlite3.Connection) -> None:
    """Game rows carry the name and cohort."""
    df = get_game_stats(conn, 1)
    assert df.loc[0, "app_name"] == "G1"
    assert df.loc[0, "cohort"] == "a"


def test_get_cohort_stats(conn: sqlite3.Connection) -> None:
    """Cohort rows for the run come back."""
    assert get_cohort_stats(conn, 1)["scope_id"].tolist() == ["a"]


def test_get_theme_stats_joins_topic_label_and_game(conn: sqlite3.Connection) -> None:
    """Game-scope topic rows carry the game name and topic label."""
    df = get_theme_stats(conn, 1, "topic", "game")
    assert df.loc[0, "app_name"] == "G1"
    assert df.loc[0, "topic_label"] == "Maps"


def test_get_filtered_theme_points_respects_cohort(conn: sqlite3.Connection) -> None:
    """Cohort filter narrows the denominator; shares split by the recommend flag."""
    df = get_filtered_theme_points(conn, "aspect", cohorts=["a"])
    row = df.iloc[0]
    assert row["theme"] == "map_design"
    assert row["n_docs"] == 3
    assert row["share"] == pytest.approx(1 / 3)
    assert row["praise_share"] == pytest.approx(1 / 2)
    assert row["complaint_share"] == 0


def test_get_aspect_examples_slices_sentence(conn: sqlite3.Connection) -> None:
    """The example is the matched sentence, not the whole review."""
    df = get_aspect_examples(conn, "map_design")
    assert df["sentence"].tolist() == ["Great maps."]


def test_get_pull_windows(conn: sqlite3.Connection) -> None:
    """The latest pull per app comes back with its stop reason."""
    df = get_pull_windows(conn)
    assert df.loc[0, "stop_reason"] == "exhausted"


def test_get_recommend_share_by_game_with_aspect_filter(conn: sqlite3.Connection) -> None:
    """The aspect filter keeps only reviews that mention the aspect."""
    df = get_recommend_share_by_game(conn, aspects=["map_design"])
    assert df["n"].tolist() == [1]
    assert df["rec_share"].tolist() == [1.0]
