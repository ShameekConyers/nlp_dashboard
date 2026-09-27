"""Tests for transformer scoring helpers and validation statistics (no model download)."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src import sentiment_transformer as st
from src.db_setup import create_study_tables
from src.study import load_study
from src.validation import (
    Predictor,
    bootstrap_validation,
    macro_f1,
    macro_f1_multiclass,
    roc_auc,
    tune_threshold,
)


def test_head_tail_keeps_first_256_and_last_254() -> None:
    ids = list(range(1000))
    out = st.head_tail(ids)
    assert len(out) == 510
    assert out[:256] == list(range(256))
    assert out[256:] == list(range(746, 1000))


def test_head_tail_passes_short_input_unchanged() -> None:
    assert st.head_tail(list(range(510))) == list(range(510))
    assert st.head_tail([7, 8, 9]) == [7, 8, 9]


def test_paired_bootstrap_is_deterministic_and_straddles_zero_for_identical_predictors() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.7
    score = rng.random(300)
    same = Predictor(score >= 0.5, score)
    preds = {"a": same, "b": Predictor(score >= 0.5, score.copy())}
    r1 = bootstrap_validation(y, preds, [("a", "b")], n_boot=200, seed=42)
    r2 = bootstrap_validation(y, preds, [("a", "b")], n_boot=200, seed=42)
    assert r1 == r2
    lo, hi = r1["differences"]["a_minus_b"]["ci"]
    assert lo <= 0 <= hi


def test_paired_bootstrap_detects_a_better_predictor() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(400) < 0.6
    good = Predictor(y.copy(), y.astype(float))
    coin = rng.random(400)
    preds = {"good": good, "coin": Predictor(coin >= 0.5, coin)}
    diff = bootstrap_validation(y, preds, [("good", "coin")], n_boot=200, seed=42)["differences"]
    assert diff["good_minus_coin"]["ci"][0] > 0


def test_cluster_bootstrap_runs_and_is_deterministic() -> None:
    y = np.array([True, False] * 50)
    s = np.linspace(0, 1, 100)
    preds = {"a": Predictor(s >= 0.5, s)}
    clusters = np.repeat(np.arange(5), 20)
    r1 = bootstrap_validation(y, preds, [], 50, 3, clusters=clusters)
    r2 = bootstrap_validation(y, preds, [], 50, 3, clusters=clusters)
    assert r1 == r2


def test_metrics_on_known_values() -> None:
    y = np.array([True, True, False, False])
    assert roc_auc(y, np.array([0.9, 0.8, 0.1, 0.2])) == 1.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert macro_f1(y, y) == 1.0
    assert macro_f1_multiclass(["a", "b"], ["a", "a"], ["a", "b", "c"]) == pytest.approx(
        (2 / 3 + 0 + 0) / 3
    )


def test_tune_threshold_prefers_default_on_ties() -> None:
    y = np.array([True, False])
    t, ba = tune_threshold(y, np.array([0.9, -0.9]), grid=(-0.5, 0.0, 0.05, 0.5))
    assert (t, ba) == (0.05, 1.0)


def test_scorer_output_aligns_with_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = st.TransformerScorer("fake/model", "rev", batch_size=2)
    monkeypatch.setattr(scorer, "_load", lambda: None)
    monkeypatch.setattr(scorer, "encode", lambda text: [0] * len(text))
    monkeypatch.setattr(scorer, "_forward", lambda batch: [len(ids) / 100 for ids in batch])
    texts = ["aaaa", "a", "aaaaaaa", "aa", "aaaaa"]
    assert scorer.score(texts) == [len(t) / 100 for t in texts]


def test_constructing_a_scorer_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: st.TransformerScorer) -> None:
        raise AssertionError("model loaded")

    monkeypatch.setattr(st.TransformerScorer, "_load", boom)
    scorer = st.TransformerScorer("fake/model", "rev")
    assert scorer.model_name == "fake/model"


def test_score_study_stores_model_and_revision_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = load_study()
    study = base.__class__(**{**base.__dict__, "seed_db": tmp_path / "s.db", "scores_db": None})
    conn = sqlite3.connect(study.seed_db)
    create_study_tables(conn)
    conn.executemany(
        "INSERT INTO processed_documents (document_id, cleaned_text, tokens, token_count, "
        "char_count) VALUES (?, ?, '[]', ?, 0)",
        [(1, "good game", 3), (2, "bad", 1), (3, "fine enough", 4)],
    )
    conn.commit()
    conn.close()

    scorer = st.TransformerScorer("fake/model", "rev1")
    monkeypatch.setattr(scorer, "score", lambda texts: [0.25] * len(texts))
    first = st.score_study(study, scorer, "doc")
    second = st.score_study(study, scorer, "doc")
    assert (first["n_scored"], second["n_scored"]) == (2, 0)
    conn = sqlite3.connect(study.seed_db)
    rows = conn.execute(
        "SELECT s.document_id, m.model_name, m.model_revision, s.unit, s.sentence_idx "
        "FROM sentiment_scores s JOIN sentiment_models m ON m.model_id = s.model_id"
    ).fetchall()
    n_models = conn.execute("SELECT COUNT(*) FROM sentiment_models").fetchone()[0]
    conn.close()
    assert sorted(rows) == [(1, "fake/model", "rev1", "doc", -1), (3, "fake/model", "rev1", "doc", -1)]
    assert n_models == 1  # every score row joins to one stored name and revision

    rescored = st.TransformerScorer("fake/model", "rev2")
    monkeypatch.setattr(rescored, "score", lambda texts: [0.75] * len(texts))
    assert st.score_study(study, rescored, "doc")["n_scored"] == 2
    conn = sqlite3.connect(study.seed_db)
    assert st.model_scores(conn, "fake/model", "rev2", "doc") == {(1, -1): 0.75, (3, -1): 0.75}
    conn.close()


def test_sentence_class_band_and_allocation() -> None:
    assert [st.sentence_class(p) for p in (0.9, 0.8, 0.5, 0.2, 0.1)] == [
        "positive", "positive", "neither", "negative", "negative"
    ]
    assert st.allocate(100, ["a", "b", "c", "d", "e", "f", "g"]) == {
        "a": 15, "b": 15, "c": 14, "d": 14, "e": 14, "f": 14, "g": 14
    }
