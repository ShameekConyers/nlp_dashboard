"""SQL query builders for the Streamlit dashboard.

Every function takes a sqlite3.Connection and optional filter parameters,
builds a parameterized query, executes it, and returns a pd.DataFrame (or
dict/int for scalar results).  All filter values go through ``?`` placeholders.
Sentinels (reviews with token_count < MIN_TOKEN_COUNT) are excluded from every
query via the shared ``_build_where`` helper.
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — dashboard/ is outside src/, so add project root to sys.path.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import MIN_TOKEN_COUNT  # noqa: E402

# ---------------------------------------------------------------------------
# Standard four-table join used by most queries.
# ---------------------------------------------------------------------------

_BASE_FROM: str = """
FROM nlp_results n
JOIN documents d ON n.document_id = d.id
JOIN processed_documents p ON n.document_id = p.document_id
JOIN metadata m ON d.app_id = m.app_id
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_where(
    genres: list[str] | None = None,
    game_names: list[str] | None = None,
    sentiment_min: float | None = None,
    sentiment_max: float | None = None,
    topic_ids: list[int] | None = None,
    date_start: int | None = None,
    date_end: int | None = None,
    cohorts: list[str] | None = None,
    aspects: list[str] | None = None,
) -> tuple[str, list[Any]]:
    """Build a WHERE clause and parameter list from active filters.

    The clause always includes the sentinel-exclusion filter
    ``p.token_count >= MIN_TOKEN_COUNT``.  Additional clauses are appended
    only when the corresponding argument is not ``None`` (or not empty for
    list arguments).

    Args:
        genres: Genre strings to include (e.g. ``["RPG", "FPS"]``).
        game_names: Game display names to include.
        sentiment_min: Lower bound for ``sentiment_compound`` (inclusive).
        sentiment_max: Upper bound for ``sentiment_compound`` (inclusive).
        topic_ids: Topic ID integers to include.
        date_start: Unix-timestamp lower bound (inclusive).
        date_end: Unix-timestamp upper bound (inclusive).
        cohorts: Study cohorts to include (study databases only).
        aspects: Aspect names; keeps reviews with at least one mention
            (study databases only).

    Returns:
        A ``(where_clause, params)`` tuple ready to splice into a query.
    """
    clauses: list[str] = ["p.token_count >= ?"]
    params: list[Any] = [MIN_TOKEN_COUNT]

    if cohorts:
        placeholders = ",".join("?" * len(cohorts))
        clauses.append(
            f"d.app_id IN (SELECT app_id FROM study_apps WHERE cohort IN ({placeholders}))"
        )
        params.extend(cohorts)

    if aspects:
        placeholders = ",".join("?" * len(aspects))
        clauses.append(
            "n.document_id IN (SELECT document_id FROM aspect_mentions "
            f"WHERE aspect IN ({placeholders}))"
        )
        params.extend(aspects)

    if genres:
        placeholders = ",".join("?" * len(genres))
        clauses.append(f"m.genre IN ({placeholders})")
        params.extend(genres)

    if game_names:
        placeholders = ",".join("?" * len(game_names))
        clauses.append(f"m.app_name IN ({placeholders})")
        params.extend(game_names)

    if sentiment_min is not None:
        clauses.append("n.sentiment_compound >= ?")
        params.append(sentiment_min)

    if sentiment_max is not None:
        clauses.append("n.sentiment_compound <= ?")
        params.append(sentiment_max)

    if topic_ids:
        placeholders = ",".join("?" * len(topic_ids))
        clauses.append(f"n.topic_id IN ({placeholders})")
        params.extend(topic_ids)

    if date_start is not None:
        clauses.append("d.timestamp_created >= ?")
        params.append(date_start)

    if date_end is not None:
        clauses.append("d.timestamp_created <= ?")
        params.append(date_end)

    where = " AND ".join(clauses)
    return where, params


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def get_filter_options(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return available filter values for sidebar widgets.

    Args:
        conn: Open SQLite connection.

    Returns:
        Dict with keys ``genres``, ``game_names``, ``topic_labels``,
        ``date_min``, ``date_max``.
    """
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT genre FROM metadata ORDER BY genre")
    genres = [row[0] for row in cur.fetchall()]

    cur.execute("SELECT DISTINCT app_name FROM metadata ORDER BY app_name")
    game_names = [row[0] for row in cur.fetchall()]

    cur.execute(
        "SELECT topic_id, label FROM topics WHERE topic_id != -1 ORDER BY topic_id"
    )
    topic_labels = {row[0]: row[1] for row in cur.fetchall()}

    cur.execute(
        "SELECT MIN(timestamp_created), MAX(timestamp_created) FROM documents"
    )
    date_min, date_max = cur.fetchone()

    return {
        "genres": genres,
        "game_names": game_names,
        "topic_labels": topic_labels,
        "date_min": date_min,
        "date_max": date_max,
    }


def get_overview_kpis(
    conn: sqlite3.Connection,
    **filters: Any,
) -> dict[str, Any]:
    """Return high-level KPI values for the overview cards.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        Dict with keys ``total_reviews``, ``avg_sentiment``,
        ``pct_positive``, ``pct_negative``, ``num_topics``, ``num_games``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            COUNT(*)                                          AS total_reviews,
            AVG(n.sentiment_compound)                         AS avg_sentiment,
            SUM(CASE WHEN n.sentiment_compound >= 0.05 THEN 1 ELSE 0 END)
                * 100.0 / MAX(COUNT(*), 1)                   AS pct_positive,
            SUM(CASE WHEN n.sentiment_compound <= -0.05 THEN 1 ELSE 0 END)
                * 100.0 / MAX(COUNT(*), 1)                   AS pct_negative,
            COUNT(DISTINCT n.topic_id)                        AS num_topics,
            COUNT(DISTINCT d.app_id)                          AS num_games
        {_BASE_FROM}
        WHERE {where}
    """
    cur = conn.execute(sql, params)
    row = cur.fetchone()
    return {
        "total_reviews": row[0],
        "avg_sentiment": row[1] if row[1] is not None else 0.0,
        "pct_positive": row[2] if row[2] is not None else 0.0,
        "pct_negative": row[3] if row[3] is not None else 0.0,
        "num_topics": row[4],
        "num_games": row[5],
    }


def get_sentiment_distribution(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return sentiment compound scores for histogram plotting.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with column ``sentiment_compound``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT n.sentiment_compound
        {_BASE_FROM}
        WHERE {where}
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_sentiment_by_genre(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return average sentiment and review count per genre.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``genre``, ``avg_sentiment``, ``review_count``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            m.genre,
            AVG(n.sentiment_compound) AS avg_sentiment,
            COUNT(*)                  AS review_count
        {_BASE_FROM}
        WHERE {where}
        GROUP BY m.genre
        ORDER BY avg_sentiment DESC
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_sentiment_by_game(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return average sentiment and review count per game.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``app_name``, ``genre``, ``avg_sentiment``,
        ``review_count``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            m.app_name,
            m.genre,
            AVG(n.sentiment_compound) AS avg_sentiment,
            COUNT(*)                  AS review_count
        {_BASE_FROM}
        WHERE {where}
        GROUP BY m.app_name, m.genre
        ORDER BY avg_sentiment DESC
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_sentiment_vs_recommendation(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return average sentiment grouped by recommendation flag.

    This is the ground-truth validation view: does VADER sentiment align
    with the thumbs-up/thumbs-down recommendation?

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``recommended``, ``avg_sentiment``,
        ``review_count``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            d.recommended,
            AVG(n.sentiment_compound) AS avg_sentiment,
            COUNT(*)                  AS review_count
        {_BASE_FROM}
        WHERE {where}
        GROUP BY d.recommended
        ORDER BY d.recommended
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_sentiment_over_time(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return monthly average sentiment and review count.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``month``, ``avg_sentiment``, ``review_count``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            strftime('%Y-%m', d.timestamp_created, 'unixepoch') AS month,
            AVG(n.sentiment_compound)                           AS avg_sentiment,
            COUNT(*)                                            AS review_count
        {_BASE_FROM}
        WHERE {where}
        GROUP BY month
        ORDER BY month
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_topic_distribution(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return document counts per topic, excluding the outlier topic (-1).

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``topic_id``, ``label``, ``top_words``,
        ``doc_count``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            n.topic_id,
            t.label,
            t.top_words,
            COUNT(*) AS doc_count
        {_BASE_FROM}
        JOIN topics t ON n.topic_id = t.topic_id
        WHERE {where}
          AND n.topic_id != -1
        GROUP BY n.topic_id, t.label, t.top_words
        ORDER BY doc_count DESC
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_sentiment_by_topic(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return average sentiment per topic with positive-review percentage.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``topic_id``, ``label``, ``avg_sentiment``,
        ``review_count``, ``pct_positive``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            n.topic_id,
            t.label,
            AVG(n.sentiment_compound) AS avg_sentiment,
            COUNT(*)                  AS review_count,
            SUM(CASE WHEN n.sentiment_compound >= 0.05 THEN 1 ELSE 0 END)
                * 100.0 / MAX(COUNT(*), 1) AS pct_positive
        {_BASE_FROM}
        JOIN topics t ON n.topic_id = t.topic_id
        WHERE {where}
          AND n.topic_id != -1
        GROUP BY n.topic_id, t.label
        ORDER BY avg_sentiment DESC
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_topic_words(conn: sqlite3.Connection, topic_id: int) -> str:
    """Return the top_words string for a single topic.

    Args:
        conn: Open SQLite connection.
        topic_id: The topic to look up.

    Returns:
        JSON-encoded list of representative words (e.g.
        ``'["fight", "sword", "enemy"]'``), or empty string if the topic
        is not found.
    """
    cur = conn.execute(
        "SELECT top_words FROM topics WHERE topic_id = ?", (topic_id,)
    )
    row = cur.fetchone()
    return row[0] if row else ""


def get_sentiment_with_genre(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return per-review sentiment with genre for histogram overlay.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``sentiment_compound`` and ``genre``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT n.sentiment_compound, m.genre
        {_BASE_FROM}
        WHERE {where}
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_reviews_table(
    conn: sqlite3.Connection,
    limit: int = 100,
    offset: int = 0,
    **filters: Any,
) -> pd.DataFrame:
    """Return a paginated table of individual reviews with NLP scores.

    Review text is truncated to 300 characters for display.

    Args:
        conn: Open SQLite connection.
        limit: Maximum rows to return.
        offset: Number of rows to skip.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with columns ``app_name``, ``genre``, ``review_text``,
        ``sentiment_compound``, ``recommended``, ``topic_label``,
        ``timestamp_created``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT
            m.app_name,
            m.genre,
            SUBSTR(d.review_text, 1, 300) AS review_text,
            n.sentiment_compound,
            d.recommended,
            n.topic_label,
            d.timestamp_created
        {_BASE_FROM}
        WHERE {where}
        ORDER BY d.timestamp_created DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    return pd.read_sql_query(sql, conn, params=params)


def get_reviews_count(
    conn: sqlite3.Connection,
    **filters: Any,
) -> int:
    """Return total count of reviews matching the current filters.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        Integer count.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT COUNT(*)
        {_BASE_FROM}
        WHERE {where}
    """
    cur = conn.execute(sql, params)
    return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Study (market analysis) queries over precomputed tables
# ---------------------------------------------------------------------------


def has_study_tables(conn: sqlite3.Connection) -> bool:
    """Return whether the database carries study statistics tables.

    Args:
        conn: Open SQLite connection.

    Returns:
        True if ``game_stats`` exists and has at least one row.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'game_stats'"
    ).fetchone()
    if row is None:
        return False
    return conn.execute("SELECT 1 FROM game_stats LIMIT 1").fetchone() is not None


def get_latest_run(conn: sqlite3.Connection, stage: str = "stats") -> tuple[int, dict] | None:
    """Return the newest analysis run of a stage and its metrics.

    Args:
        conn: Open SQLite connection.
        stage: ``stats`` or ``reference_checks``.

    Returns:
        ``(run_id, metrics_dict)`` or None if no run exists.
    """
    row = conn.execute(
        "SELECT run_id, metrics FROM analysis_runs WHERE stage = ? ORDER BY run_id DESC LIMIT 1",
        (stage,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), json.loads(row[1])


def get_cohort_options(conn: sqlite3.Connection) -> list[str]:
    """Return the study's cohorts in display order.

    Args:
        conn: Open SQLite connection.

    Returns:
        Cohort names.
    """
    return [
        r[0]
        for r in conn.execute(
            "SELECT cohort FROM study_apps GROUP BY cohort ORDER BY MIN(display_order)"
        )
    ]


def get_game_stats(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Return per-game statistics with names, cohorts and tags.

    Args:
        conn: Open SQLite connection.
        run_id: Stats run.

    Returns:
        DataFrame of ``game_stats`` columns plus ``app_name``, ``cohort``,
        ``subgroup``, ``tags`` and ``display_order``.
    """
    sql = """
        SELECT g.*, m.app_name, s.cohort, s.subgroup, s.tags, s.display_order
        FROM game_stats g
        JOIN metadata m ON m.app_id = g.app_id
        JOIN study_apps s ON s.app_id = g.app_id
        WHERE g.run_id = ?
        ORDER BY s.display_order
    """
    return pd.read_sql_query(sql, conn, params=[run_id])


def get_cohort_stats(conn: sqlite3.Connection, run_id: int) -> pd.DataFrame:
    """Return cohort, subgroup and matched-basket statistics.

    Args:
        conn: Open SQLite connection.
        run_id: Stats run.

    Returns:
        DataFrame of ``cohort_stats`` rows.
    """
    return pd.read_sql_query(
        "SELECT * FROM cohort_stats WHERE run_id = ? ORDER BY scope_type, scope_id",
        conn,
        params=[run_id],
    )


def get_theme_stats(
    conn: sqlite3.Connection,
    run_id: int,
    lens: str,
    scope_type: str,
) -> pd.DataFrame:
    """Return theme statistics for one lens and scope type.

    Game-scope rows carry ``app_name``; topic rows carry the topic ``label``.

    Args:
        conn: Open SQLite connection.
        run_id: Stats run.
        lens: ``topic`` or ``aspect``.
        scope_type: ``game``, ``cohort``, ``subgroup``, ``subgroup_matched`` or ``all``.

    Returns:
        DataFrame of ``theme_stats`` rows plus ``app_name`` and ``topic_label``.
    """
    sql = """
        SELECT t.*, m.app_name, tp.label AS topic_label
        FROM theme_stats t
        LEFT JOIN metadata m
               ON t.scope_type = 'game' AND m.app_id = CAST(t.scope_id AS INTEGER)
        LEFT JOIN topics tp
               ON t.lens = 'topic' AND tp.topic_id = CAST(t.theme AS INTEGER)
        WHERE t.run_id = ? AND t.lens = ? AND t.scope_type = ?
        ORDER BY t.scope_id, t.theme
    """
    return pd.read_sql_query(sql, conn, params=[run_id, lens, scope_type])


def get_filtered_theme_points(
    conn: sqlite3.Connection,
    lens: str,
    **filters: Any,
) -> pd.DataFrame:
    """Return pooled point estimates per theme for the current filters (no intervals).

    Used when sidebar filters are active and precomputed intervals no longer
    apply.

    Args:
        conn: Open SQLite connection.
        lens: ``topic`` or ``aspect``.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with ``theme``, ``k_docs``, ``n_docs``, ``share``,
        ``complaint_share``, ``praise_share``, ``mean_compound``.
    """
    where, params = _build_where(**filters)
    if lens == "aspect":
        theme_sql = """
            SELECT DISTINCT a.document_id, a.aspect AS theme
            FROM aspect_mentions a
        """
    else:
        theme_sql = """
            SELECT n2.document_id, CAST(n2.topic_id AS TEXT) AS theme
            FROM nlp_results n2 WHERE n2.topic_id != -1
        """
    sql = f"""
        WITH scoped AS (
            SELECT n.document_id, d.recommended, n.sentiment_compound
            {_BASE_FROM}
            WHERE {where}
        ),
        totals AS (
            SELECT COUNT(*) AS n_docs,
                   SUM(recommended) AS n_rec,
                   SUM(1 - recommended) AS n_not
            FROM scoped
        )
        SELECT th.theme,
               COUNT(*) AS k_docs,
               (SELECT n_docs FROM totals) AS n_docs,
               COUNT(*) * 1.0 / MAX((SELECT n_docs FROM totals), 1) AS share,
               SUM(1 - s.recommended) * 1.0 / MAX((SELECT n_not FROM totals), 1)
                   AS complaint_share,
               SUM(s.recommended) * 1.0 / MAX((SELECT n_rec FROM totals), 1) AS praise_share,
               AVG(s.sentiment_compound) AS mean_compound
        FROM scoped s
        JOIN ({theme_sql}) th ON th.document_id = s.document_id
        GROUP BY th.theme
        ORDER BY k_docs DESC
    """
    return pd.read_sql_query(sql, conn, params=params)


def get_aspect_examples(
    conn: sqlite3.Connection,
    aspect: str,
    limit: int = 5,
    **filters: Any,
) -> pd.DataFrame:
    """Return example sentences for an aspect, newest reviews first.

    Args:
        conn: Open SQLite connection.
        aspect: Aspect name.
        limit: Maximum sentences.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with ``app_name``, ``recommended``, ``sentence``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT m.app_name, d.recommended,
               SUBSTR(p.cleaned_text, a.char_start + 1, a.char_end - a.char_start) AS sentence
        {_BASE_FROM}
        JOIN aspect_mentions a ON a.document_id = n.document_id
        WHERE {where} AND a.aspect = ?
        ORDER BY d.timestamp_created DESC
        LIMIT ?
    """
    return pd.read_sql_query(sql, conn, params=[*params, aspect, limit])


def get_pull_windows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the latest pull per app with its window and population totals.

    Args:
        conn: Open SQLite connection.

    Returns:
        DataFrame with app name, pull counts, cap flag, stop reason, window
        bounds and ``query_summary`` totals.
    """
    sql = """
        SELECT m.app_name, r.reviews_collected, r.cap_hit, r.stop_reason,
               r.effective_window_start, r.effective_window_end,
               r.total_reviews, r.total_positive,
               r.baseline_total_reviews, r.baseline_total_positive, r.pulled_at
        FROM pull_runs r
        JOIN metadata m ON m.app_id = r.app_id
        JOIN study_apps s ON s.app_id = r.app_id
        WHERE r.run_id IN (SELECT MAX(run_id) FROM pull_runs GROUP BY app_id)
        ORDER BY s.display_order
    """
    return pd.read_sql_query(sql, conn)


def get_recommend_share_by_game(
    conn: sqlite3.Connection,
    **filters: Any,
) -> pd.DataFrame:
    """Return per-game recommend share, review count and mean VADER for the filters.

    Args:
        conn: Open SQLite connection.
        **filters: Keyword arguments forwarded to ``_build_where``.

    Returns:
        DataFrame with ``app_name``, ``n``, ``rec_share``, ``mean_compound``.
    """
    where, params = _build_where(**filters)
    sql = f"""
        SELECT m.app_name,
               COUNT(*)                  AS n,
               AVG(d.recommended)        AS rec_share,
               AVG(n.sentiment_compound) AS mean_compound
        {_BASE_FROM}
        WHERE {where}
        GROUP BY m.app_name
        ORDER BY rec_share DESC
    """
    return pd.read_sql_query(sql, conn, params=params)
