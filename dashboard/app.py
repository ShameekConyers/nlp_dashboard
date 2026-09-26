"""Streamlit dashboard for NLP Review Analysis.

Launch with::

    .venv/bin/streamlit run dashboard/app.py

The sidebar selects a study. The market analysis of historical and tactical
strategy games is the default; the April 2026 cross-genre baseline stays
selectable. Set ``NLP_DASHBOARD_MODE=full`` to use a study's full database
instead of its committed seed, and ``NLP_DASHBOARD_STUDY`` to a study name to
change the default study.
"""

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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
    get_aspect_examples,
    get_cohort_options,
    get_cohort_stats,
    get_filter_options,
    get_filtered_theme_points,
    get_game_stats,
    get_latest_run,
    get_overview_kpis,
    get_pull_windows,
    get_recommend_share_by_game,
    get_reviews_count,
    get_reviews_table,
    get_sentiment_by_game,
    get_sentiment_by_genre,
    get_sentiment_by_topic,
    get_sentiment_distribution,
    get_sentiment_over_time,
    get_sentiment_vs_recommendation,
    get_sentiment_with_genre,
    get_theme_stats,
    get_topic_distribution,
    get_topic_words,
    has_study_tables,
)
from src.study import STUDIES_DIR, Study, load_study  # noqa: E402
from src.utils import get_db_path  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NLP Review Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

DEFAULT_STUDY_NAME: str = os.environ.get("NLP_DASHBOARD_STUDY", "strategy_market_2026")
"""Study selected when the dashboard opens."""

VADER_CAVEAT: str = (
    "VADER is a lexicon scorer. On strategy reviews it often scores critical, "
    "not-recommended reviews as positive (see Methods for the measured false-positive "
    "rate), so treat VADER columns as tone, not verdict. The recommend flag is the verdict."
)

SUBGROUP_LABELS: dict[str, str] = {
    "igo_ugo": "IGO-UGO",
    "alternating": "Alternating",
    "wego": "WEGO",
}


# ---------------------------------------------------------------------------
# Study and database connection
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_studies() -> list[Study]:
    """Load every study TOML whose seed database exists, default study first.

    Returns:
        Studies in display order.
    """
    studies: list[Study] = []
    for path in sorted(STUDIES_DIR.glob("*.toml")):
        if path.name.startswith("aspects_"):
            continue
        study = load_study(path)
        if study.seed_db.exists():
            studies.append(study)
    studies.sort(key=lambda s: s.name != DEFAULT_STUDY_NAME)
    return studies


@st.cache_resource
def _get_connection(db_path: str) -> sqlite3.Connection:
    """Open a shared read-only SQLite connection to a study database.

    The connection is cached per path and reused across sessions and
    threads. The dashboard only reads from it.

    Args:
        db_path: Database file path.

    Returns:
        An open sqlite3.Connection.
    """
    uri = Path(db_path).as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


st.sidebar.title("NLP Review Dashboard")

_studies = _load_studies()
if not _studies:
    st.error("No study database found. Run the pipeline first (see README Quick Start).")
    st.stop()

_study_titles = [s.title for s in _studies]
_selected_title: str = st.sidebar.selectbox("Study", _study_titles, index=0)
study: Study = _studies[_study_titles.index(_selected_title)]

_mode = os.environ.get("NLP_DASHBOARD_MODE", "seed")
_db_path = get_db_path(_mode, study)
if not _db_path.exists():
    st.error(f"Database not found at `{_db_path}`. Run the pipeline for this study first.")
    st.stop()
conn = _get_connection(str(_db_path))
is_market: bool = has_study_tables(conn)
_run = get_latest_run(conn, "stats") if is_market else None
run_id: int | None = _run[0] if _run else None
run_metrics: dict = _run[1] if _run else {}
_ref = get_latest_run(conn, "reference_checks") if is_market else None
ref_metrics: dict = _ref[1] if _ref else {}


# ---------------------------------------------------------------------------
# Sidebar — view selector and filters
# ---------------------------------------------------------------------------

_BASE_VIEWS = ["Overview", "Sentiment Analysis", "Topic Analysis", "Review Explorer"]
_MARKET_VIEWS = ["Market Overview", "Topics x Sentiment", "Design Themes", "Methods"]
_views = (_MARKET_VIEWS + _BASE_VIEWS) if is_market else _BASE_VIEWS
if st.session_state.get("view") not in _views:
    st.session_state["view"] = _views[0]

view = st.sidebar.radio("View", _views, key="view")

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

options = get_filter_options(conn)

if options["date_min"] is None or options["date_max"] is None:
    st.error("The database contains no reviews. Run the pipeline to load data first.")
    st.stop()

selected_cohorts: list[str] = []
selected_aspects: list[str] = []
if is_market:
    selected_cohorts = st.sidebar.multiselect("Cohort", get_cohort_options(conn))
    _aspect_labels: dict[str, str] = {
        k: v for k, v in run_metrics.get("aspect_labels", {}).items() if k != "turn_structure_any"
    }
    selected_aspects = st.sidebar.multiselect(
        "Aspect mentioned",
        list(_aspect_labels),
        format_func=lambda a: _aspect_labels.get(a, a),
        key="aspect_filter",
    )

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
if selected_cohorts:
    filters["cohorts"] = selected_cohorts
if selected_aspects:
    filters["aspects"] = selected_aspects
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
# Market study helpers
# ---------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    """Format a share as a percentage string.

    Args:
        x: Share in [0, 1], or None.
        digits: Decimal places.

    Returns:
        Formatted percentage, or "n/a" when missing.
    """
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def _scope_label(scope_type: str, scope_id: str) -> str:
    """Human label for a cohort-level scope.

    Args:
        scope_type: Scope type from cohort_stats or theme_stats.
        scope_id: Scope ID.

    Returns:
        Display label.
    """
    name = SUBGROUP_LABELS.get(scope_id, scope_id.replace("_", " "))
    if scope_type == "subgroup_matched":
        return f"{name} (matched basket)"
    if scope_type == "all":
        return "All games"
    return name


def _theme_label(lens: str, theme: str, topic_label: str | None = None) -> str:
    """Human label for a theme.

    Args:
        lens: ``topic`` or ``aspect``.
        theme: Theme key.
        topic_label: Topic label from the topics table, if any.

    Returns:
        Display label.
    """
    if lens == "topic":
        return f"T{theme}: {topic_label or ''}".strip()
    return run_metrics.get("aspect_labels", {}).get(theme, theme)


def _filters_notice() -> bool:
    """Show the notice for active filters and report whether any is active.

    Returns:
        True if sidebar filters are active.
    """
    if filters:
        st.info(
            "Filters are active. Precomputed intervals and findings apply to the unfiltered "
            "study only, so this view shows SQL point estimates and n without intervals."
        )
        return True
    return False


def _finding_badge(value: str | None) -> str:
    """Render a finding flag as a short badge.

    Args:
        value: ``complaint``, ``praise`` or None.

    Returns:
        Badge text.
    """
    return {"complaint": "Finding: complaint", "praise": "Finding: praise"}.get(
        value or "", "Descriptive"
    )


# ---------------------------------------------------------------------------
# View: Market Overview
# ---------------------------------------------------------------------------


def _render_market_overview() -> None:
    """Render natural recommend shares per game and per cohort."""
    st.header("Market Overview")
    st.caption(study.title)
    if run_id is None:
        st.info("No statistics run found for this study. Run `src/stats.py --study` first.")
        return

    if _filters_notice():
        df = get_recommend_share_by_game(conn, **filters)
        if df.empty:
            st.info("No reviews match the current filters.")
            return
        st.dataframe(
            df.rename(columns={"app_name": "Game", "n": "Kept reviews",
                               "rec_share": "Recommended share",
                               "mean_compound": "Mean VADER"}),
            width="stretch", hide_index=True,
            column_config={"Recommended share": st.column_config.NumberColumn(format="%.3f"),
                           "Mean VADER": st.column_config.NumberColumn(format="%.3f")},
        )
        return

    games = get_game_stats(conn, run_id)
    val = run_metrics.get("vader_validation", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", f"{len(games)}")
    c2.metric("Eligible for cohort estimates", f"{int(games['eligible'].sum())}")
    c3.metric("Kept English reviews", f"{int(games['n_kept'].sum()):,}")
    c4.metric("VADER balanced accuracy (test)", _pct(val.get("balanced_accuracy")))

    st.subheader("Natural recommend share per game")
    st.caption(
        "Dots: share of all seed reviews (a uniform sample of the pull) that recommend the "
        "game, with a Wilson 95% interval. Crosses: the same share among kept English "
        "reviews, the base for every theme metric; short reviews are mostly positive, so it "
        "runs lower. Diamonds: Steam's English population share for the pull parameters. "
        "Grey rows are low n (ineligible for cohort estimates)."
    )
    g = games.copy()
    g["label"] = g["app_name"] + g["eligible"].map({1: "", 0: " (low n)"})
    fig = go.Figure()
    for elig, color in ((1, "#1f77b4"), (0, "#b0b0b0")):
        sub = g[g["eligible"] == elig]
        fig.add_trace(go.Scatter(
            x=sub["seed_rec_share"], y=sub["label"], mode="markers",
            marker=dict(color=color, size=9),
            error_x=dict(type="data", symmetric=False,
                         array=sub["seed_rec_ci_high"] - sub["seed_rec_share"],
                         arrayminus=sub["seed_rec_share"] - sub["seed_rec_ci_low"]),
            name="Seed share (eligible)" if elig else "Seed share (low n)",
            customdata=sub[["n_seed"]],
            hovertemplate="%{y}: %{x:.1%} (n=%{customdata[0]})<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=g["rec_share"], y=g["label"], mode="markers",
        marker=dict(symbol="x-thin-open", size=9, color="#555555"),
        name="Kept English reviews", customdata=g[["n_kept"]],
        hovertemplate="%{y}: kept %{x:.1%} (n=%{customdata[0]})<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=g["pop_share_english"], y=g["label"], mode="markers",
        marker=dict(symbol="diamond-open", size=9, color="#d62728"),
        name="Steam population (English)",
        hovertemplate="%{y}: population %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(height=max(400, 22 * len(g)), xaxis_tickformat=".0%",
                      yaxis=dict(autorange="reversed"), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cohort recommend share (macro average)")
    cohorts = get_cohort_stats(conn, run_id)
    cohorts = cohorts[cohorts["scope_type"].isin(["cohort", "subgroup", "subgroup_matched"])]
    cohorts["scope"] = [_scope_label(t, i) for t, i in zip(cohorts["scope_type"], cohorts["scope_id"])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cohorts["macro_rec_share"], y=cohorts["scope"], mode="markers",
        marker=dict(size=11, color="#1f77b4"), name="Macro share (cluster-bootstrap 95% CI)",
        error_x=dict(type="data", symmetric=False,
                     array=cohorts["macro_rec_ci_high"] - cohorts["macro_rec_share"],
                     arrayminus=cohorts["macro_rec_share"] - cohorts["macro_rec_ci_low"]),
    ))
    dots_x, dots_y = [], []
    for _, row in cohorts.iterrows():
        if row["scope_type"] == "cohort":
            members = games[games["cohort"] == row["scope_id"]]
        elif row["scope_type"] == "subgroup":
            members = games[games["subgroup"] == row["scope_id"]]
        else:
            members = games[(games["subgroup"] == row["scope_id"])
                            & games["tags"].str.contains("matched_basket")]
        members = members[members["n_kept"] >= 100]
        dots_x.extend(members["rec_share"])
        dots_y.extend([row["scope"]] * len(members))
    fig.add_trace(go.Scatter(x=dots_x, y=dots_y, mode="markers", name="Games",
                             marker=dict(size=6, color="rgba(100,100,100,0.5)")))
    fig.update_layout(height=380, xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"),
                      margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cohort sentiment and playtime")
    st.caption(VADER_CAVEAT)
    st.dataframe(
        cohorts[["scope", "n_kept", "macro_mean_compound", "macro_compound_ci_low",
                 "macro_compound_ci_high", "median_hours_rec", "median_hours_not_rec"]].rename(
            columns={"scope": "Scope", "n_kept": "Kept reviews",
                     "macro_mean_compound": "Mean VADER (macro)",
                     "macro_compound_ci_low": "VADER low", "macro_compound_ci_high": "VADER high",
                     "median_hours_rec": "Median hours, recommended",
                     "median_hours_not_rec": "Median hours, not recommended"}),
        width="stretch", hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                       ["Mean VADER (macro)", "VADER low", "VADER high"]}
        | {c: st.column_config.NumberColumn(format="%.1f") for c in
           ["Median hours, recommended", "Median hours, not recommended"]},
    )

    st.subheader("Per-game sentiment")
    st.dataframe(
        games[["app_name", "cohort", "subgroup", "n_kept", "rec_share", "mean_compound",
               "vader_pos_share", "eligible"]].rename(
            columns={"app_name": "Game", "cohort": "Cohort", "subgroup": "Subgroup",
                     "n_kept": "Kept reviews", "rec_share": "Recommended (kept)",
                     "mean_compound": "Mean VADER", "vader_pos_share": "VADER positive share",
                     "eligible": "Eligible"}),
        width="stretch", hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                       ["Recommended (kept)", "Mean VADER", "VADER positive share"]},
    )


# ---------------------------------------------------------------------------
# View: Topics x Sentiment
# ---------------------------------------------------------------------------


def _scope_selector(key: str) -> str:
    """Radio for the cohort-level scope type.

    Args:
        key: Streamlit widget key prefix.

    Returns:
        ``cohort``, ``subgroup`` or ``subgroup_matched``.
    """
    choice = st.radio(
        "Compare", ["Cohorts", "Turn-structure subgroups", "Subgroups, matched basket"],
        horizontal=True, key=f"{key}_scope",
    )
    scope_type = {"Cohorts": "cohort", "Turn-structure subgroups": "subgroup",
                  "Subgroups, matched basket": "subgroup_matched"}[choice]
    return scope_type


def _render_top_themes(lens: str) -> None:
    """Show the stored top-three complaint and praise themes per cohort-level scope.

    Reads ``analysis_runs.metrics.top_themes`` (precision-passing themes only).
    Only a top complaint theme can be a ranking finding.

    Args:
        lens: ``topic`` or ``aspect``.
    """
    top = run_metrics.get("top_themes", {})
    rows = []
    for key, lists in top.items():
        scope_type, scope_id = key.split(":", 1)
        if scope_type == "all":
            continue
        for side in ("complaint", "praise"):
            for rank, item in enumerate(lists.get(f"{lens}_{side}", []), start=1):
                label = (run_metrics.get("topic_labels", {}).get(item["theme"], item["theme"])
                         if lens == "topic" else _theme_label("aspect", item["theme"]))
                rows.append({
                    "Scope": _scope_label(scope_type, scope_id), "Side": side, "Rank": rank,
                    "Theme": label, "Share": item["share"],
                    "Low": item["ci"][0], "High": item["ci"][1],
                    "Status": "Finding" if item["status"] == "finding" else "Descriptive",
                })
    if not rows:
        st.info("No ranked themes for this lens.")
        return
    st.subheader("Top three themes per scope")
    st.caption(
        f"Ranked by complaint share and by praise share among themes that support claims. "
        f"Ranking tests: {run_metrics.get('n_ranking_comparisons', 0)}, ranking findings: "
        f"{run_metrics.get('n_ranking_findings', 0)}."
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ["Share", "Low", "High"]})


def _render_topics_sentiment() -> None:
    """Render BERTopic topics per cohort and game, crossed with sentiment."""
    st.header("Topics x Sentiment")
    st.caption(
        "Primary analysis. Each kept English review has one BERTopic topic. Share: reviews "
        "in the topic. Complaint share: not-recommended reviews in the topic. Praise share: "
        "recommended reviews in the topic. Lift above 1 means the topic is more common in "
        "complaints. " + VADER_CAVEAT
    )
    if run_id is None:
        st.info("No statistics run found for this study.")
        return
    if _filters_notice():
        df = get_filtered_theme_points(conn, "topic", **filters)
        if df.empty:
            st.info("No reviews match the current filters.")
            return
        labels = run_metrics.get("topic_labels", {})
        df["Topic"] = [f"T{t}: {labels.get(t, '')}" for t in df["theme"]]
        df["lift"] = df["complaint_share"] / df["praise_share"]
        st.dataframe(df.drop(columns=["theme"]), width="stretch", hide_index=True)
        return

    scope_type = _scope_selector("topics")
    df = get_theme_stats(conn, run_id, "topic", scope_type)
    if df.empty:
        st.info("No eligible games in this scope.")
        return
    df["Topic"] = [_theme_label("topic", t, lbl) for t, lbl in zip(df["theme"], df["topic_label"])]
    df["Scope"] = [_scope_label(scope_type, i) for i in df["scope_id"]]

    st.subheader("Complaint lift by topic")
    scope_pick = st.selectbox("Scope", sorted(df["Scope"].unique()), key="topic_scope_pick")
    sub = df[df["Scope"] == scope_pick].sort_values("share", ascending=False)
    fig = go.Figure(go.Scatter(
        x=sub["lift"], y=sub["Topic"], mode="markers",
        marker=dict(size=9, color=["#d62728" if f == "complaint" else "#2ca02c" if f == "praise"
                                   else "#7f7f7f" for f in sub["finding"]]),
        error_x=dict(type="data", symmetric=False, array=sub["lift_ci_high"] - sub["lift"],
                     arrayminus=sub["lift"] - sub["lift_ci_low"]),
        hovertemplate="%{y}: lift %{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    fig.update_layout(height=max(350, 24 * len(sub)), xaxis_type="log",
                      yaxis=dict(autorange="reversed"), margin=dict(t=10),
                      xaxis_title="Complaint lift (log scale, 95% cluster-bootstrap CI)")
    st.plotly_chart(fig, width="stretch")

    table = sub[["Topic", "k_docs", "share", "complaint_share", "praise_share", "lift",
                 "lift_ci_low", "lift_ci_high", "rec_share_in_theme", "mean_compound",
                 "logo_stable", "finding"]].copy()
    table["finding"] = table["finding"].map(_finding_badge)
    st.dataframe(
        table.rename(columns={"k_docs": "Reviews", "share": "Share",
                              "complaint_share": "Complaint share", "praise_share": "Praise share",
                              "lift": "Lift", "lift_ci_low": "Lift low", "lift_ci_high": "Lift high",
                              "rec_share_in_theme": "Recommended in topic",
                              "mean_compound": "Mean VADER", "logo_stable": "LOGO stable",
                              "finding": "Status"}),
        width="stretch", hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                       ["Share", "Complaint share", "Praise share", "Lift", "Lift low",
                        "Lift high", "Recommended in topic", "Mean VADER"]},
    )

    _render_top_themes("topic")

    st.subheader("Topic share per game")
    per_game = get_theme_stats(conn, run_id, "topic", "game")
    if not per_game.empty:
        per_game["Topic"] = [_theme_label("topic", t, lbl)
                             for t, lbl in zip(per_game["theme"], per_game["topic_label"])]
        pivot = per_game.pivot_table(index="Topic", columns="app_name", values="share")
        fig = px.imshow(pivot, color_continuous_scale="Blues", aspect="auto",
                        labels=dict(color="Share"))
        fig.update_layout(height=max(400, 22 * len(pivot)), margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# View: Design Themes
# ---------------------------------------------------------------------------


def _open_aspect_in_explorer(aspect: str) -> None:
    """Callback: filter Review Explorer to one aspect and switch to it.

    Args:
        aspect: Aspect name.
    """
    st.session_state["aspect_filter"] = [aspect]
    st.session_state["view"] = "Review Explorer"


def _render_design_themes() -> None:
    """Render the audited phrase-lexicon lens for design questions."""
    st.header("Design Themes")
    st.caption(
        "Supplementary lens. Aspects are tagged by a versioned phrase lexicon, so shares "
        "count mentions the lexicon detects, not true prevalence. An aspect supports claims "
        "only after its precision audit passes. " + VADER_CAVEAT
    )
    if run_id is None:
        st.info("No statistics run found for this study.")
        return
    precision = run_metrics.get("aspect_precision", {})
    status = run_metrics.get("aspect_audit_status", "labels pending")
    if status != "complete":
        st.warning("Precision audit labels are pending. No aspect supports a claim yet.")
    show_unreliable = st.toggle("Show aspects without a passing precision audit", value=False)

    def reliable(aspect: str) -> bool:
        """Return whether an aspect passed its audit.

        Args:
            aspect: Aspect name.

        Returns:
            True if precision passed (derived theme needs all four parts).
        """
        if aspect == "turn_structure_any":
            parts = ["turn_waiting", "first_strike", "initiative_clarity", "preview_reliability"]
            return all(precision.get(a, {}).get("passes") for a in parts)
        return bool(precision.get(aspect, {}).get("passes"))

    if _filters_notice():
        df = get_filtered_theme_points(conn, "aspect", **filters)
        if df.empty:
            st.info("No reviews match the current filters.")
            return
        if not show_unreliable:
            df = df[[reliable(a) for a in df["theme"]]]
        df["Aspect"] = [_theme_label("aspect", a) for a in df["theme"]]
        st.dataframe(df.drop(columns=["theme"]), width="stretch", hide_index=True)
        return

    scope_type = _scope_selector("aspects")
    df = get_theme_stats(conn, run_id, "aspect", scope_type)
    if not show_unreliable:
        df = df[[reliable(a) for a in df["theme"]]]
    if df.empty:
        st.info("No aspect to show. Turn on the toggle to see aspects awaiting the audit.")
        return
    df["Aspect"] = [_theme_label("aspect", a) for a in df["theme"]]
    df["Scope"] = [_scope_label(scope_type, i) for i in df["scope_id"]]

    st.subheader("Complaint share by aspect (macro over eligible games, 95% cluster CI)")
    fig = px.scatter(
        df, x="complaint_share", y="Aspect", color="Scope",
        error_x=df["complaint_ci_high"] - df["complaint_share"],
        error_x_minus=df["complaint_share"] - df["complaint_ci_low"],
        hover_data={"lift": ":.2f", "finding": True},
    )
    fig.update_layout(height=max(350, 30 * df["Aspect"].nunique()), xaxis_tickformat=".0%",
                      margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

    table = df[["Scope", "Aspect", "k_docs", "complaint_share", "praise_share", "lift",
                "lift_ci_low", "lift_ci_high", "mean_compound", "logo_stable", "finding"]].copy()
    table["finding"] = table["finding"].map(_finding_badge)
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["complaint_share", "praise_share", "lift", "lift_ci_low",
                                 "lift_ci_high", "mean_compound"]})

    _render_top_themes("aspect")

    st.subheader("Complaint share per game")
    per_game = get_theme_stats(conn, run_id, "aspect", "game")
    if not show_unreliable:
        per_game = per_game[[reliable(a) for a in per_game["theme"]]]
    if not per_game.empty:
        per_game["Aspect"] = [_theme_label("aspect", a) for a in per_game["theme"]]
        pivot = per_game.pivot_table(index="Aspect", columns="app_name", values="complaint_share")
        fig = px.imshow(pivot, color_continuous_scale="Reds", aspect="auto",
                        labels=dict(color="Complaint share"))
        fig.update_layout(height=max(350, 30 * len(pivot)), margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Example sentences")
    aspects = sorted(a for a in df["theme"].unique() if a != "turn_structure_any")
    if aspects:
        pick = st.selectbox("Aspect", aspects, format_func=lambda a: _theme_label("aspect", a))
        ex = get_aspect_examples(conn, pick, limit=8)
        if ex.empty:
            st.info("No example sentences.")
        else:
            ex["recommended"] = ex["recommended"].map({0: "No", 1: "Yes"})
            st.dataframe(ex.rename(columns={"app_name": "Game", "recommended": "Recommended",
                                            "sentence": "Sentence"}),
                         width="stretch", hide_index=True)
        st.button("Open these reviews in Review Explorer", on_click=_open_aspect_in_explorer,
                  args=(pick,))


# ---------------------------------------------------------------------------
# View: Methods
# ---------------------------------------------------------------------------


def _render_methods() -> None:
    """Render sampling parameters, windows, filters and the finding rule."""
    st.header("Methods")
    samp = study.sampling
    st.subheader("Sampling")
    st.markdown(
        f"- Steam appreviews, `filter={samp.filter}`, `language={samp.language}`, "
        f"`review_type={samp.review_type}`, `purchase_type={samp.purchase_type}`, "
        f"`filter_offtopic_activity={samp.filter_offtopic_activity}`.\n"
        f"- Census pull, newest first, capped at {samp.raw_cap_per_game:,} reviews per game. "
        f"Capped games cover a recent window; the others cover their full history.\n"
        f"- Seed: a uniform random sample of up to {samp.seed_cap_per_game} reviews per game "
        f"(seed {samp.random_seed}), no stratification, so recommend shares are natural.\n"
        "- Kept reviews: at least 10 characters and 3 content tokens after cleaning. A review "
        "is dropped as non-English only when lingua's top language is not English at "
        "confidence 0.50 or higher."
    )
    windows = get_pull_windows(conn)
    if not windows.empty:
        for col in ("effective_window_start", "effective_window_end"):
            windows[col] = pd.to_datetime(windows[col], unit="s").dt.strftime("%Y-%m-%d")
        st.dataframe(windows, width="stretch", hide_index=True)
    if run_id is not None:
        games = get_game_stats(conn, run_id)
        st.subheader("Seed counts and drops per game")
        st.dataframe(
            games[["app_name", "cohort", "subgroup", "n_pulled", "n_seed", "n_too_short",
                   "n_too_few_tokens", "n_non_english", "n_kept", "n_not_recommended",
                   "early_access_share", "eligible"]],
            width="stretch", hide_index=True,
        )
        st.subheader("Finding rule")
        st.markdown(run_metrics.get("finding_rule", ""))
        st.markdown(
            "Two rule changes were made after results were seen: the minimum-support "
            "condition, added after the first run read collapsed [0, 0] intervals as praise "
            "findings, and a correction of the stability check to rank within each lens as "
            "the written plan specifies. See the README for their effect."
        )
        st.markdown(
            f"Comparisons evaluated: **{run_metrics.get('n_comparisons', 0)}** "
            f"({run_metrics.get('n_lift_comparisons', 0)} lift, "
            f"{run_metrics.get('n_ranking_comparisons', 0)} ranking). Lift findings: "
            f"**{run_metrics.get('n_findings', 0)}**; ranking findings: "
            f"**{run_metrics.get('n_ranking_findings', 0)}**. Intervals are "
            "per-comparison 95% and uncorrected, so findings are exploratory. Subgroup "
            "differences describe the games in each group; genre and franchise travel with "
            "turn structure, so no difference is attributed to the turn rule."
        )
        st.subheader("VADER on the natural sample (test split)")
        st.json(run_metrics.get("vader_validation", {}))
        st.subheader("Aspect precision audit")
        st.json({"status": run_metrics.get("aspect_audit_status"),
                 "precision": run_metrics.get("aspect_precision", {})})
    if ref_metrics:
        st.subheader("Reproduction of prior review shares")
        st.dataframe(pd.DataFrame(ref_metrics.get("reproduction", [])), width="stretch",
                     hide_index=True)
        st.json({k: ref_metrics.get(k) for k in
                 ("subgroup_medians", "matched_basket_medians", "direction_check",
                  "recall_proxy")})


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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")


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
    st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        width="stretch",
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

if view == "Market Overview":
    _render_market_overview()
elif view == "Topics x Sentiment":
    _render_topics_sentiment()
elif view == "Design Themes":
    _render_design_themes()
elif view == "Methods":
    _render_methods()
elif view == "Overview":
    _render_overview()
elif view == "Sentiment Analysis":
    _render_sentiment()
elif view == "Topic Analysis":
    _render_topics()
elif view == "Review Explorer":
    _render_explorer()
