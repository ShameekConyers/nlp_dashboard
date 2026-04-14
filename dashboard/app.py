"""Streamlit dashboard for NLP Review Analysis.

Launch with::

    .venv/bin/streamlit run dashboard/app.py

Set ``NLP_DASHBOARD_MODE=full`` to use full.db instead of the default seed.db.
"""

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — add project root so src.* and dashboard.* imports resolve.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dashboard.queries import (  # noqa: E402
    get_filter_options,
    get_overview_kpis,
    get_reviews_count,
    get_reviews_table,
    get_sentiment_by_game,
    get_sentiment_by_genre,
    get_sentiment_by_topic,
    get_sentiment_distribution,
    get_sentiment_over_time,
    get_sentiment_vs_recommendation,
    get_sentiment_with_genre,
    get_topic_distribution,
    get_topic_words,
)
from src.utils import get_db_path  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NLP Review Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------


@st.cache_resource
def _get_connection() -> sqlite3.Connection:
    """Open a read-only SQLite connection to the project database.

    Returns:
        An open sqlite3.Connection.
    """
    mode = os.environ.get("NLP_DASHBOARD_MODE", "seed")
    db_path = get_db_path(mode)
    if not db_path.exists():
        st.error(
            f"Database not found at `{db_path}`.\n\n"
            "Run the pipeline first:\n"
            "```\n"
            ".venv/bin/python src/data_pull.py\n"
            ".venv/bin/python src/db_setup.py --full\n"
            ".venv/bin/python src/preprocessing.py --full\n"
            ".venv/bin/python src/sentiment.py --full\n"
            ".venv/bin/python src/topics.py --full\n"
            "```"
        )
        st.stop()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return conn


conn = _get_connection()


# ---------------------------------------------------------------------------
# Sidebar — view selector and filters
# ---------------------------------------------------------------------------

st.sidebar.title("NLP Review Dashboard")

view = st.sidebar.radio(
    "View",
    ["Overview", "Sentiment Analysis", "Topic Analysis", "Review Explorer"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

options = get_filter_options(conn)

# Genre multiselect
selected_genres: list[str] = st.sidebar.multiselect(
    "Genre", options["genres"]
)

# Game multiselect — SQL handles genre filtering, so show all games here
selected_games: list[str] = st.sidebar.multiselect("Game", options["game_names"])

# Sentiment range slider
sentiment_range: tuple[float, float] = st.sidebar.slider(
    "Sentiment range",
    min_value=-1.0,
    max_value=1.0,
    value=(-1.0, 1.0),
    step=0.05,
)

# Topic multiselect
topic_choices = {
    f"{tid}: {label}": tid
    for tid, label in options["topic_labels"].items()
}
selected_topic_labels: list[str] = st.sidebar.multiselect("Topic", list(topic_choices.keys()))
selected_topic_ids: list[int] = [topic_choices[t] for t in selected_topic_labels]

# Date range
_date_min = datetime.fromtimestamp(options["date_min"], tz=timezone.utc).date()
_date_max = datetime.fromtimestamp(options["date_max"], tz=timezone.utc).date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(_date_min, _date_max),
    min_value=_date_min,
    max_value=_date_max,
)

# Build the filters dict
filters: dict = {}
if selected_genres:
    filters["genres"] = selected_genres
if selected_games:
    filters["game_names"] = selected_games
if sentiment_range != (-1.0, 1.0):
    filters["sentiment_min"] = sentiment_range[0]
    filters["sentiment_max"] = sentiment_range[1]
if selected_topic_ids:
    filters["topic_ids"] = selected_topic_ids
if isinstance(date_range, tuple) and len(date_range) == 2:
    ds = datetime.combine(date_range[0], datetime.min.time(), tzinfo=timezone.utc)
    de = datetime.combine(date_range[1], datetime.max.time(), tzinfo=timezone.utc)
    date_start_ts = int(ds.timestamp())
    date_end_ts = int(de.timestamp())
    if date_range[0] != _date_min or date_range[1] != _date_max:
        filters["date_start"] = date_start_ts
        filters["date_end"] = date_end_ts


# ---------------------------------------------------------------------------
# View: Overview
# ---------------------------------------------------------------------------


def _render_overview() -> None:
    """Render the Overview dashboard view."""
    st.header("Overview")

    kpis = get_overview_kpis(conn, **filters)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", f"{kpis['total_reviews']:,}")
    col2.metric("Avg Sentiment", f"{kpis['avg_sentiment']:.3f}")
    col3.metric("Positive", f"{kpis['pct_positive']:.1f}%")
    col4.metric("Negative", f"{kpis['pct_negative']:.1f}%")

    if kpis["total_reviews"] == 0:
        st.info("No reviews match the current filters.")
        return

    # Two-column layout: sentiment histogram + genre breakdown
    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Distribution")
        dist_df = get_sentiment_distribution(conn, **filters)
        fig = px.histogram(
            dist_df,
            x="sentiment_compound",
            nbins=30,
            color_discrete_sequence=["#636EFA"],
            labels={"sentiment_compound": "Compound Score"},
        )
        fig.update_layout(
            xaxis_title="Compound Sentiment Score",
            yaxis_title="Number of Reviews",
            showlegend=False,
            margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Reviews by Genre")
        genre_df = get_sentiment_by_genre(conn, **filters)
        fig = px.bar(
            genre_df,
            x="genre",
            y="review_count",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            labels={
                "genre": "Genre",
                "review_count": "Reviews",
                "avg_sentiment": "Avg Sentiment",
            },
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Sentiment over time — full width
    st.subheader("Sentiment Over Time")
    time_df = get_sentiment_over_time(conn, **filters)
    if len(time_df) > 0:
        fig = px.line(
            time_df,
            x="month",
            y="avg_sentiment",
            markers=True,
            labels={
                "month": "Month",
                "avg_sentiment": "Avg Sentiment",
            },
        )
        fig.update_layout(margin=dict(t=10))
        # Add a reference line at zero
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to show trends.")


# ---------------------------------------------------------------------------
# View: Sentiment Analysis
# ---------------------------------------------------------------------------


def _render_sentiment() -> None:
    """Render the Sentiment Analysis dashboard view."""
    st.header("Sentiment Analysis")

    kpis = get_overview_kpis(conn, **filters)
    if kpis["total_reviews"] == 0:
        st.info("No reviews match the current filters.")
        return

    # Ground truth validation: sentiment vs recommendation
    st.subheader("Sentiment vs. Recommendation (Ground Truth Validation)")
    rec_df = get_sentiment_vs_recommendation(conn, **filters)
    rec_df["recommended_label"] = rec_df["recommended"].map(
        {0: "Not Recommended", 1: "Recommended"}
    )

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            rec_df,
            x="recommended_label",
            y="avg_sentiment",
            color="recommended_label",
            color_discrete_map={
                "Recommended": "#2ecc71",
                "Not Recommended": "#e74c3c",
            },
            labels={
                "recommended_label": "",
                "avg_sentiment": "Avg Sentiment",
            },
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.bar(
            rec_df,
            x="recommended_label",
            y="review_count",
            color="recommended_label",
            color_discrete_map={
                "Recommended": "#2ecc71",
                "Not Recommended": "#e74c3c",
            },
            labels={
                "recommended_label": "",
                "review_count": "Review Count",
            },
        )
        fig.update_layout(showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    # Sentiment by game
    st.subheader("Sentiment by Game")
    game_df = get_sentiment_by_game(conn, **filters)
    fig = px.bar(
        game_df,
        y="app_name",
        x="avg_sentiment",
        color="genre",
        orientation="h",
        labels={
            "app_name": "Game",
            "avg_sentiment": "Avg Sentiment",
            "genre": "Genre",
        },
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment distribution by genre
    st.subheader("Sentiment Distribution by Genre")
    review_genre_df = get_sentiment_with_genre(conn, **filters)

    if len(review_genre_df) > 0:
        fig = px.histogram(
            review_genre_df,
            x="sentiment_compound",
            color="genre",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            labels={
                "sentiment_compound": "Compound Score",
                "genre": "Genre",
            },
        )
        fig.update_layout(margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# View: Topic Analysis
# ---------------------------------------------------------------------------


def _render_topics() -> None:
    """Render the Topic Analysis dashboard view."""
    st.header("Topic Analysis")

    kpis = get_overview_kpis(conn, **filters)
    if kpis["total_reviews"] == 0:
        st.info("No reviews match the current filters.")
        return

    # Topic distribution bar chart
    st.subheader("Topic Distribution")
    topic_df = get_topic_distribution(conn, **filters)
    if len(topic_df) == 0:
        st.info("No topic data available for the current filters.")
        return

    fig = px.bar(
        topic_df,
        y="label",
        x="doc_count",
        orientation="h",
        labels={"label": "Topic", "doc_count": "Reviews"},
        color="doc_count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(t=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment by topic
    st.subheader("Sentiment by Topic")
    sent_topic_df = get_sentiment_by_topic(conn, **filters)
    if len(sent_topic_df) > 0:
        fig = px.bar(
            sent_topic_df,
            y="label",
            x="avg_sentiment",
            orientation="h",
            color="avg_sentiment",
            color_continuous_scale="RdYlGn",
            labels={"label": "Topic", "avg_sentiment": "Avg Sentiment"},
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

    # Word cloud for selected topic
    st.subheader("Topic Word Cloud")
    topic_options_wc = {
        f"{row['topic_id']}: {row['label']}": row["topic_id"]
        for _, row in topic_df.iterrows()
    }

    if topic_options_wc:
        selected_topic_label_wc = st.selectbox(
            "Select a topic to visualize",
            list(topic_options_wc.keys()),
        )
        selected_tid = topic_options_wc[selected_topic_label_wc]
        words_str = get_topic_words(conn, selected_tid)

        if words_str:
            _render_word_cloud(words_str)
        else:
            st.info("No words available for this topic.")


def _render_word_cloud(words_str: str) -> None:
    """Generate and display a word cloud from topic representative words.

    Args:
        words_str: JSON array string of representative words (e.g.
            ``'["fight", "sword", "enemy"]'``).
    """
    import json

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # top_words are stored as a JSON list in the topics table
    try:
        word_list = json.loads(words_str)
        clean_words = " ".join(word_list)
    except (json.JSONDecodeError, TypeError):
        clean_words = words_str.replace(" / ", " ").replace("/", " ")

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=50,
    ).generate(clean_words)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# View: Review Explorer
# ---------------------------------------------------------------------------

_PAGE_SIZE: int = 50


def _render_explorer() -> None:
    """Render the Review Explorer dashboard view."""
    st.header("Review Explorer")

    total_count = get_reviews_count(conn, **filters)

    if total_count == 0:
        st.info("No reviews match the current filters.")
        return

    st.markdown(f"**{total_count:,}** reviews match the current filters.")

    # Pagination
    total_pages = max(1, (total_count + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
    )
    offset = (page - 1) * _PAGE_SIZE

    df = get_reviews_table(conn, limit=_PAGE_SIZE, offset=offset, **filters)

    # Format the timestamp for display
    if "timestamp_created" in df.columns and len(df) > 0:
        df["date"] = df["timestamp_created"].apply(
            lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            if ts else ""
        )
        df = df.drop(columns=["timestamp_created"])

    # Format recommended as text
    if "recommended" in df.columns:
        df["recommended"] = df["recommended"].map({0: "No", 1: "Yes"})

    # Rename columns for display
    df = df.rename(columns={
        "app_name": "Game",
        "genre": "Genre",
        "review_text": "Review (truncated)",
        "sentiment_compound": "Sentiment",
        "recommended": "Recommended",
        "topic_label": "Topic",
        "date": "Date",
    })

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sentiment": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.caption(
        f"Page {page} of {total_pages} "
        f"(showing rows {offset + 1}--{min(offset + _PAGE_SIZE, total_count)})"
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

if view == "Overview":
    _render_overview()
elif view == "Sentiment Analysis":
    _render_sentiment()
elif view == "Topic Analysis":
    _render_topics()
elif view == "Review Explorer":
    _render_explorer()
