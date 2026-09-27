"""Tests for study statistics: intervals, bootstrap, LOGO, splits and the runner."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.db_setup import create_study_tables
from src.stats import (
    GameData,
    balanced_accuracy,
    cluster_bootstrap_macro,
    dev_test_split,
    leave_one_game_out,
    macro_theme_metrics,
    run,
    wilson_interval,
)
from src.study import AppSpec, Sampling, Study
from src.utils import PROJECT_ROOT


def _game(app_id: int, n_rec: int, n_not: int, theme_rec: int, theme_not: int) -> GameData:
    """Build a one-theme synthetic game.

    Args:
        app_id: App ID.
        n_rec: Recommended reviews.
        n_not: Not-recommended reviews.
        theme_rec: Recommended reviews that mention the theme.
        theme_not: Not-recommended reviews that mention the theme.

    Returns:
        GameData with a single theme column.
    """
    rec = np.array([True] * n_rec + [False] * n_not)
    theme = np.zeros((n_rec + n_not, 1), dtype=bool)
    theme[:theme_rec, 0] = True
    theme[n_rec : n_rec + theme_not, 0] = True
    return GameData(app_id, rec, np.zeros(n_rec + n_not), theme)


def test_wilson_interval_known_value() -> None:
    """8 of 10 gives the textbook Wilson interval (0.490, 0.943)."""
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)


def test_balanced_accuracy_all_positive_on_90_10_is_half() -> None:
    """Predicting 'recommended' for everything scores 0.5, not 0.9."""
    y = [True] * 90 + [False] * 10
    assert balanced_accuracy(y, [True] * 100) == pytest.approx(0.5)


def test_dev_test_split_deterministic_and_disjoint() -> None:
    """The split is a pure function of the ID and yields both halves."""
    ids = [str(i) for i in range(1000)]
    first = [dev_test_split(i) for i in ids]
    assert first == [dev_test_split(i) for i in ids]
    assert set(first) == {"dev", "test"}
    assert 400 < first.count("dev") < 600


def test_cluster_bootstrap_deterministic_and_covers_point() -> None:
    """Same seed, same interval; the interval contains the macro point estimate."""
    games = [_game(i, 100, 40, 10 + i, 12 + 2 * i) for i in range(4)]
    a = cluster_bootstrap_macro(games, n_boot=300, seed=7)
    b = cluster_bootstrap_macro(games, n_boot=300, seed=7)
    point = macro_theme_metrics(games)
    for key in ("share", "complaint", "praise", "lift"):
        assert np.array_equal(a[key][0], b[key][0])
        assert a[key][0][0] <= getattr(point, key)[0] <= a[key][1][0]


def test_logo_flags_single_game_driven_lift() -> None:
    """A lift above 1 that exists only because of one game is not LOGO-stable."""
    flat = [_game(i, 100, 50, 40, 10) for i in range(3)]  # lift 0.5 each
    driver = _game(9, 100, 50, 1, 49)  # lift ~98
    unstable = leave_one_game_out(flat + [driver])
    stable = leave_one_game_out(flat)
    assert not unstable[0]
    assert stable[0]


# ---------------------------------------------------------------------------
# Runner on a synthetic seed
# ---------------------------------------------------------------------------


def _make_seed(tmp_path: Path) -> Study:
    """Write a small study seed where one game is ineligible.

    Games 1 to 3 have 120 kept reviews with 40 not recommended; game 4 has
    only 20 kept reviews and is ineligible.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        A Study pointing at the synthetic seed.
    """
    apps = tuple(AppSpec(a, f"G{a}", "Tactics", "c1", "igo_ugo", ("matched_basket",))
                 for a in (1, 2, 3, 4))
    study = Study(
        name="synthetic", title="t", seed_db=tmp_path / "seed.db", full_db=tmp_path / "full.db",
        raw_dir=tmp_path / "raw", language_filter=True,
        aspect_lexicon=PROJECT_ROOT / "studies" / "aspects_strategy_v1.toml",
        sampling=Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 42),
        baseline=None, apps=apps,
    )
    conn = sqlite3.connect(study.seed_db)
    create_study_tables(conn)
    conn.execute("INSERT INTO topics (topic_id, label, top_words, doc_count) VALUES (0, 'Maps', '[]', 1)")
    rid = 0
    for app_id, n_kept, n_not in ((1, 120, 40), (2, 120, 40), (3, 120, 40), (4, 20, 5)):
        conn.execute("INSERT INTO metadata (app_id, app_name) VALUES (?, ?)", (app_id, f"G{app_id}"))
        conn.execute(
            "INSERT INTO study_apps (app_id, cohort, subgroup, tags, display_order) "
            "VALUES (?, 'c1', 'igo_ugo', '[]', ?)", (app_id, app_id),
        )
        for i in range(n_kept):
            rid += 1
            rec = i >= n_not
            text = "The maps are great." if i % 3 == 0 else "Solid tactics overall."
            doc_id = conn.execute(
                "INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                "VALUES (?, ?, ?, ?)", (str(rid), app_id, text, rec),
            ).lastrowid
            conn.execute(
                "INSERT INTO processed_documents VALUES (?, ?, '[]', 5, ?, NULL)",
                (doc_id, text, len(text)),
            )
            conn.execute(
                "INSERT INTO nlp_results (document_id, sentiment_compound, topic_id) "
                "VALUES (?, ?, ?)", (doc_id, 0.5 if rec else -0.5, 0 if i % 3 == 0 else -1),
            )
    conn.commit()
    conn.close()
    return study


def test_run_excludes_ineligible_games_from_cohort_macro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cohort theme row covers eligible games only; the game row keeps all four."""
    from src.aspects import run as run_aspects

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path)  # no audit labels for this seed
    study = _make_seed(tmp_path)
    run_aspects(study)
    run_id = run(study, n_boot=50)

    conn = sqlite3.connect(study.seed_db)
    eligible = {r[0] for r in conn.execute(
        "SELECT app_id FROM game_stats WHERE run_id = ? AND eligible", (run_id,))}
    n_docs = conn.execute(
        "SELECT n_docs FROM theme_stats WHERE run_id = ? AND lens = 'topic' AND theme = '0' "
        "AND scope_type = 'cohort' AND scope_id = 'c1'", (run_id,)).fetchone()[0]
    game_rows = conn.execute(
        "SELECT COUNT(*) FROM game_stats WHERE run_id = ?", (run_id,)).fetchone()[0]
    metrics = json.loads(conn.execute(
        "SELECT metrics FROM analysis_runs WHERE run_id = ?", (run_id,)).fetchone()[0])
    conn.close()

    assert eligible == {1, 2, 3}
    assert game_rows == 4
    assert n_docs == 360
    assert metrics["aspect_audit_status"] == "labels pending"
    assert metrics["vader_validation"]["n"] > 0


def test_zero_support_side_is_never_a_finding() -> None:
    """A collapsed [0, 0] lift interval from zero complaint mentions stays descriptive."""
    from src.stats import classify_finding

    assert classify_finding(0.0, 0.0, 5, True, True, k_complaint=0, k_praise=40) is None
    assert classify_finding(0.2, 0.6, 5, True, True, k_complaint=6, k_praise=40) == "praise"


def test_logo_ranks_within_lens_not_mixed_pool() -> None:
    """Topics interleaving with an aspect move its mixed rank, not its within-lens rank.

    Columns: 0 and 1 are aspects, 2 and 3 are topics. Dropping game 9 lowers
    aspect 0's complaint share from 0.30 to 0.27, below both topics: its
    mixed-pool rank moves two places, its rank among aspects does not move.
    """
    from src.stats import ThemeMetrics

    games = [GameData(i, np.array([True]), np.zeros(1), np.zeros((1, 4), dtype=bool))
             for i in (1, 2, 9)]

    def estimator(subset: list[GameData]) -> ThemeMetrics:
        """Fixed metrics that depend only on whether game 9 is present."""
        a0 = 0.30 if any(g.app_id == 9 for g in subset) else 0.27
        complaint = np.array([a0, 0.10, 0.29, 0.28])
        return ThemeMetrics(complaint, complaint, complaint / 2, np.full(4, 2.0))

    mixed = leave_one_game_out(games, estimator)
    within = leave_one_game_out(games, estimator, groups=[[0, 1], [2, 3]])

    assert not mixed[0]
    assert within[0]


def test_ranking_findings_only_on_complaint_side_of_cohort_scopes() -> None:
    """Praise rankings and the all-games scope never yield a ranking finding."""
    from src.stats import ThemeMetrics, _top_themes

    keys = [("aspect", "a"), ("aspect", "b")]
    point = ThemeMetrics(np.array([0.5, 0.1]), np.array([0.5, 0.1]), np.array([0.5, 0.1]),
                         np.array([1.0, 1.0]))
    ci = {k: (np.array([0.45, 0.05]), np.array([0.55, 0.15]))
          for k in ("share", "complaint", "praise", "lift")}
    ok = {k: True for k in keys}
    logo = np.array([True, True])

    top, n = _top_themes(keys, point, ci, ok, logo, n_eligible=5, is_cohort_scope=True)
    assert top["aspect_complaint"][0]["status"] == "finding"
    assert top["aspect_praise"][0]["status"] == "descriptive"
    assert n == 1

    top_all, n_all = _top_themes(keys, point, ci, ok, logo, n_eligible=5, is_cohort_scope=False)
    assert top_all["aspect_complaint"][0]["status"] == "descriptive"
    assert n_all == 0


def test_coverage_proxy_counts_unmatched_not_recommended_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unmatched share counts not-recommended reviews with no aspect, by topic."""
    from src.aspects import run as run_aspects
    from src.stats import coverage_proxy

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path)  # no audit labels: nothing passes
    study = _make_seed(tmp_path)
    run_aspects(study)
    run_id = coverage_proxy(study)
    conn = sqlite3.connect(study.seed_db)
    metrics = json.loads(conn.execute(
        "SELECT metrics FROM analysis_runs WHERE run_id = ?", (run_id,)).fetchone()[0])
    conn.close()
    overall = metrics["scopes"][0]
    # 3 games x 40 + 5 not recommended; every third mentions maps (14 + 14 + 14 + 2 tagged).
    assert (overall["n_not_recommended"], overall["unmatched_any"]) == (125, 81)
    assert overall["unmatched_passing"] == 125
    assert overall["top_unmatched_topics"][0]["topic_id"] == -1
    assert not overall["top_unmatched_topics"][0]["large"]


def test_run_uses_adopted_transformer_for_sentiment_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With an adopted model, game, cohort and theme rows carry its positive share."""
    from src.aspects import run as run_aspects

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path)
    study = _make_seed(tmp_path)
    run_aspects(study)
    conn = sqlite3.connect(study.seed_db)
    conn.execute("INSERT INTO sentiment_models (model_id, model_name, model_revision) "
                 "VALUES (7, 'm/x', 'r1')")
    conn.execute(
        "INSERT INTO sentiment_scores SELECT id, 7, 'doc', -1, "
        "CASE WHEN recommended THEN 0.9 ELSE 0.1 END FROM documents"
    )
    conn.execute(
        "INSERT INTO analysis_runs (stage, params, metrics) VALUES ('sentiment_validation', '{}', ?)",
        (json.dumps({"adopted": True, "selected": "x",
                     "models": {"x": {"model_name": "m/x", "revision": "r1"}}}),),
    )
    conn.commit()
    conn.close()
    run_id = run(study, n_boot=50)

    conn = sqlite3.connect(study.seed_db)
    shares = conn.execute(
        "SELECT model_pos_share, seed_rec_share FROM game_stats WHERE run_id = ?", (run_id,)
    ).fetchall()
    cohort = conn.execute(
        "SELECT macro_model_pos_share, macro_model_pos_ci_low FROM cohort_stats "
        "WHERE run_id = ? AND scope_type = 'cohort'", (run_id,)
    ).fetchone()
    theme = conn.execute(
        "SELECT model_pos_share, rec_share_in_theme FROM theme_stats WHERE run_id = ? "
        "AND lens = 'topic' AND scope_type = 'cohort'", (run_id,)
    ).fetchone()
    params = json.loads(conn.execute(
        "SELECT params FROM analysis_runs WHERE run_id = ?", (run_id,)).fetchone()[0])
    conn.close()
    assert all(m == pytest.approx(r) for m, r in shares)
    assert cohort[0] is not None and cohort[1] is not None
    assert theme[0] == pytest.approx(theme[1])
    assert params["sentiment_model"]["model_name"] == "m/x"
