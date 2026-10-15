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
        "INSERT INTO sentiment_scores (document_id, model_id, unit, sentence_idx, positive_prob) "
        "SELECT id, 7, 'doc', -1, CASE WHEN recommended THEN 0.9 ELSE 0.1 END FROM documents"
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


# ---------------------------------------------------------------------------
# Split restriction, cohort-subgroup scopes and export
# ---------------------------------------------------------------------------


def _split_seed(tmp_path: Path, name: str, drop_test: bool) -> Study:
    """Write a three-game seed, optionally without its test-split reviews.

    Each game has 300 kept reviews, 90 not recommended; one in three
    mentions the map topic. Steam IDs are shared between the two variants, so
    the dev rows are identical.

    Args:
        tmp_path: pytest temporary directory.
        name: Subdirectory and study name.
        drop_test: Leave out reviews in the test split.

    Returns:
        A dev-split Study on the seed.
    """
    from dataclasses import replace

    (tmp_path / name).mkdir()
    base = _make_seed(tmp_path / name)
    study = replace(base, name=name, stats_split="dev",
                    apps=tuple(a for a in base.apps if a.app_id != 4))
    conn = sqlite3.connect(study.seed_db)
    conn.execute("DELETE FROM nlp_results")
    conn.execute("DELETE FROM processed_documents")
    conn.execute("DELETE FROM documents")
    rid = 0
    for app_id in (1, 2, 3):
        for i in range(300):
            rid += 1
            if drop_test and dev_test_split(str(rid)) == "test":
                continue
            rec = i >= 90
            text = "The maps are great." if i % 3 == 0 else "Solid tactics overall."
            doc_id = conn.execute(
                "INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                "VALUES (?, ?, ?, ?)", (str(rid), app_id, text, rec),
            ).lastrowid
            conn.execute("INSERT INTO processed_documents VALUES (?, ?, '[]', 5, ?, NULL)",
                         (doc_id, text, len(text)))
            conn.execute(
                "INSERT INTO nlp_results (document_id, sentiment_compound, topic_id) "
                "VALUES (?, ?, ?)", (doc_id, 0.5 if rec else -0.5, 0 if i % 3 == 0 else -1),
            )
    conn.commit()
    conn.close()
    return study


def _rows(study: Study, run_id: int, sql: str) -> list[tuple]:
    conn = sqlite3.connect(study.seed_db)
    try:
        return conn.execute(sql, (run_id,)).fetchall()
    finally:
        conn.close()


def test_dev_split_stats_ignore_test_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dev-split run on the whole seed equals a run on a seed holding only dev rows."""
    from src.aspects import run as run_aspects

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path)
    whole = _split_seed(tmp_path, "whole", drop_test=False)
    dev_only = _split_seed(tmp_path, "devonly", drop_test=True)
    results = []
    for study in (whole, dev_only):
        run_aspects(study)
        run_id = run(study, n_boot=50)
        game = _rows(study, run_id, "SELECT * FROM game_stats WHERE run_id = ? ORDER BY app_id")
        theme = _rows(study, run_id, "SELECT * FROM theme_stats WHERE run_id = ? "
                      "ORDER BY lens, theme, scope_type, scope_id")
        metrics = json.loads(_rows(study, run_id, "SELECT metrics FROM analysis_runs "
                                   "WHERE run_id = ?")[0][0])
        results.append(([g[1:] for g in game], [t[1:] for t in theme], metrics))

    (g_whole, t_whole, m_whole), (g_dev, t_dev, m_dev) = results
    assert g_whole == g_dev
    assert t_whole == t_dev
    assert m_whole["vader_validation"] == m_dev["vader_validation"]
    assert m_whole["vader_validation"]["split"] == "dev"
    assert m_whole["stats_split"] == "dev"
    assert m_whole["n_kept"] == m_dev["n_kept"] < 900
    assert m_whole["eligible_app_ids"] == [1, 2, 3]


def test_cohort_subgroup_scopes_only_for_shared_subgroups() -> None:
    from dataclasses import replace

    from src.stats import SCOPE_COHORT_SUBGROUP, _scopes

    apps = (
        AppSpec(1, "A", "G", "crpg", "rtwp", ()), AppSpec(2, "B", "G", "crpg", "turn_based", ()),
        AppSpec(3, "C", "G", "strategy", "rtwp", ()), AppSpec(4, "D", "G", "strategy", "turn_based", ()),
        AppSpec(5, "E", "G", "dual", "", ()), AppSpec(6, "F", "G", "other", "solo", ()),
    )
    study = Study(
        name="x", title="t", seed_db=Path("s"), full_db=Path("f"), raw_dir=Path("r"),
        language_filter=True, aspect_lexicon=None,
        sampling=Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 42),
        baseline=None, apps=apps,
    )
    scopes = {(t, i): (ids, el) for t, i, ids, el in _scopes(study, {1, 3}, {1, 3})}
    shared = {i for t, i in scopes if t == SCOPE_COHORT_SUBGROUP}
    assert shared == {"crpg/rtwp", "crpg/turn_based", "strategy/rtwp", "strategy/turn_based"}
    assert scopes[(SCOPE_COHORT_SUBGROUP, "crpg/rtwp")] == ([1], [1])
    assert scopes[("subgroup", "rtwp")] == ([1, 3], [1, 3])

    unique = replace(study, apps=tuple(a for a in apps if a.cohort in ("crpg", "other")))
    assert not [s for s in _scopes(unique, set(), set()) if s[0] == SCOPE_COHORT_SUBGROUP]


def test_export_run_writes_metrics_and_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import csv

    from src.aspects import run as run_aspects
    from src.stats import export_run

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path)
    study = _make_seed(tmp_path)
    run_aspects(study)
    run_id = run(study, n_boot=20)
    before = (tmp_path / "seed.db").read_bytes()
    path = export_run(study, tmp_path / "out")
    assert json.loads(path.read_text())["run_id"] == run_id
    with open(tmp_path / "out" / f"theme_stats_run_{run_id}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and {r["run_id"] for r in rows} == {str(run_id)}
    assert (tmp_path / "out" / f"game_stats_run_{run_id}.csv").exists()
    assert (tmp_path / "seed.db").read_bytes() == before


def test_candidate_run_assumes_precision_on_a_copy_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidates may contain aspect findings; the study seed gains no run."""
    from src.aspects import run as run_aspects
    from src.stats import candidate_run

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path)
    study = _make_seed(tmp_path)
    run_aspects(study)
    before = (tmp_path / "seed.db").read_bytes()
    path = candidate_run(study, tmp_path / "cand", n_boot=30)
    data = json.loads(path.read_text())
    assert data["metrics"]["precision_assumed"] is True
    assert all(v["passes"] and v["assumed"] for v in data["metrics"]["aspect_precision"].values())
    with open(tmp_path / "cand" / f"candidates_theme_stats_run_{data['run_id']}.csv") as f:
        import csv
        rows = [r for r in csv.DictReader(f) if r["lens"] == "aspect" and r["scope_type"] == "cohort"]
    assert rows and all(r["logo_stable"] != "" for r in rows)
    assert (tmp_path / "seed.db").read_bytes() == before
    # A second call overwrites the copy instead of stacking runs on it.
    assert json.loads(candidate_run(study, tmp_path / "cand", n_boot=30).read_text())["run_id"] == 1


def _merge_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, governs: tuple = (
    ("a", "lx_v3"), ("b", "lx_v3"), ("c", "lx_v2"),
)):
    """A synthetic retry: a and b revised in lx_v3, c unchanged from lx_v2."""
    from dataclasses import replace

    from src.aspects import Aspect, Lexicon

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path / "data")
    (tmp_path / "data" / "labels").mkdir(parents=True)
    lexicon = Lexicon("lx_v3", tuple(Aspect(n, "design", n, (), ()) for n in "abc"),
                      revised=("a", "b"), supersedes="lx_v2")
    sampling = Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 1)
    study = Study(name="m", title="t", seed_db=tmp_path / "s.db", full_db=tmp_path / "f.db",
                  raw_dir=tmp_path, language_filter=False, aspect_lexicon=None,
                  sampling=sampling, baseline=None, apps=())
    return replace(study, aspect_labels=governs), lexicon, tmp_path / "data" / "labels"


def _label_file(path: Path, relevant: dict[str, int], n: int = 40) -> None:
    rows = ["steam_review_id,sentence_idx,aspect,label"]
    for aspect, rel in relevant.items():
        rows += [f"{aspect}{i},0,{aspect},{'relevant' if i < rel else 'not_relevant'}"
                 for i in range(n)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_study_precision_takes_each_aspect_from_its_governing_label_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.stats import study_precision

    study, lexicon, labels = _merge_setup(tmp_path, monkeypatch)
    # The earlier audit also labeled a and b (both failed). Those rows must not count.
    _label_file(labels / "aspect_audit_lx_v2.csv", {"a": 10, "b": 10, "c": 36})
    _label_file(labels / "aspect_audit_lx_v3.csv", {"a": 38, "b": 20, "c": 0})
    got = study_precision(study, lexicon, required=True)
    assert list(got) == ["a", "b", "c"]
    assert {a: (p["label_version"], p["relevant"], p["passes"]) for a, p in got.items()} == {
        "a": ("lx_v3", 38, True), "b": ("lx_v3", 20, False), "c": ("lx_v2", 36, True),
    }


def test_study_precision_fails_on_a_missing_file_once_labels_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.stats import study_precision

    study, lexicon, labels = _merge_setup(tmp_path, monkeypatch)
    _label_file(labels / "aspect_audit_lx_v2.csv", {"c": 36})
    pending = study_precision(study, lexicon)
    assert list(pending) == ["c"] and pending["c"]["label_version"] == "lx_v2"
    with pytest.raises(FileNotFoundError, match="aspect_audit_lx_v3.csv"):
        study_precision(study, lexicon, required=True)
    _label_file(labels / "aspect_audit_lx_v3.csv", {"a": 38})
    with pytest.raises(ValueError, match=r"\['b'\] have no rows"):
        study_precision(study, lexicon, required=True)


@pytest.mark.parametrize(("governs", "match"), [
    ((("a", "lx_v3"), ("b", "lx_v3")), "have no governing label file"),
    ((("a", "lx_v3"), ("b", "lx_v3"), ("c", "lx_v2"), ("d", "lx_v2")), "does not define"),
    ((("a", "lx_v3"), ("b", "lx_v2"), ("c", "lx_v2")), "revised in lx_v3"),
])
def test_study_precision_rejects_a_bad_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, governs: tuple, match: str
) -> None:
    from src.stats import study_precision

    study, lexicon, _ = _merge_setup(tmp_path, monkeypatch, governs)
    with pytest.raises(ValueError, match=match):
        study_precision(study, lexicon)


def test_rtwp_study_declares_v3_labels_for_revised_aspects_and_v2_for_the_rest() -> None:
    from src.aspects import load_lexicon
    from src.study import load_study

    study = load_study(PROJECT_ROOT / "studies" / "rtwp_turn_2026.toml")
    lexicon = load_lexicon(study.aspect_lexicon)
    governs = study.label_versions()
    assert set(governs) == {a.name for a in lexicon.aspects}
    assert {a for a, v in governs.items() if v == lexicon.version} == set(lexicon.revised)
    assert {v for a, v in governs.items() if a not in lexicon.revised} == {lexicon.supersedes}
