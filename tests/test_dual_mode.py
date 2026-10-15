"""Tests for the dual-mode views on synthetic data."""

import json
import sqlite3
from pathlib import Path

import pytest

from src import dual_mode
from src.db_setup import create_study_tables
from src.dual_mode import (
    ModeEvent,
    ModeReview,
    before_after,
    load_events,
    mode_group,
    mode_mention_table,
)
from src.stats import dev_test_split, wilson_interval
from src.study import load_study

TB, RT = dual_mode.TB_ASPECT, dual_mode.RT_ASPECT


def _r(rec: bool, created: int | None = 100, *aspects: str) -> ModeReview:
    return ModeReview(1, rec, created, frozenset(aspects))


def test_mode_group_covers_every_combination() -> None:
    assert mode_group([TB]) == "tb_only"
    assert mode_group([RT, "ui_clarity"]) == "rt_only"
    assert mode_group([TB, RT]) == "both"
    assert mode_group(["ui_clarity"]) == "neither"


def test_mode_mention_table_counts_and_wilson() -> None:
    reviews = [_r(True, 1, TB)] * 8 + [_r(False, 1, TB)] * 2 + [_r(False, 1, RT)] * 3
    table = mode_mention_table(reviews)
    assert table["tb_only"]["reviews"] == 10
    assert table["tb_only"]["recommend"]["share"] == pytest.approx(0.8)
    assert table["tb_only"]["recommend"]["ci"] == pytest.approx(list(wilson_interval(8, 10)))
    assert table["rt_only"]["recommend"]["k"] == 0
    assert table["both"]["reviews"] == 0 and table["both"]["recommend"]["share"] is None


def test_before_after_splits_at_the_event_and_skips_undated() -> None:
    event = ModeEvent(1, 1000, "1970-01-01", "https://example.invalid")
    reviews = [
        _r(False, 999, RT, "pause_micro"),
        _r(True, 1000, TB),  # on the event time counts as after
        _r(True, 2000, TB),
        _r(False, 3000),
        _r(True, None, TB),
    ]
    out = before_after(reviews, event, [TB, RT, "pause_micro"])
    assert out["status"] == "descriptive" and out["undated"] == 1
    assert out["before"]["reviews"] == 1 and out["after"]["reviews"] == 3
    assert out["after"]["recommend"]["k"] == 2
    assert out["after"]["mode_groups"]["tb_only"]["k"] == 2
    assert out["after"]["aspects"][TB]["praise"]["share"] == pytest.approx(1.0)
    assert out["after"]["aspects"][TB]["complaint"]["share"] == pytest.approx(0.0)
    assert out["before"]["aspects"]["pause_micro"]["complaint"]["k"] == 1


def test_load_events_reads_dates_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "s.toml"
    path.write_text(
        '[[mode_events]]\napp_id = 7\ndate = 2020-08-18\nsource = "u"\n', encoding="utf-8"
    )
    ev = load_events(path)[7]
    assert ev.day == "2020-08-18" and ev.timestamp == 1597708800 and ev.source == "u"
    path.write_text(path.read_text() + '[[mode_events]]\napp_id = 7\ndate = 2021-01-01\n')
    with pytest.raises(ValueError, match="twice"):
        load_events(path)


STUDY_TOML = """
[study]
name = "dm"
title = "t"
seed_db = "{root}/seed.db"
full_db = "{root}/full.db"
raw_dir = "{root}/raw"
language_filter = false
aspect_lexicon = "studies/aspects_rtwp_turn_v1.toml"

[sampling]
filter = "recent"
language = "english"
review_type = "all"
purchase_type = "all"
num_per_page = 100
raw_cap_per_game = 100
seed_cap_per_game = 100
random_seed = 1

[[mode_events]]
app_id = 1
date = 1970-01-02

[[apps]]
app_id = 1
name = "Dual"
genre = "CRPG"
cohort = "dual_mode"
tags = ["dual_mode"]

[[apps]]
app_id = 2
name = "Single"
genre = "CRPG"
cohort = "crpg"
subgroup = "rtwp"
"""

TEXTS = [
    "The turn-based mode is a huge improvement over the old combat.",
    "I prefer real-time with pause, it keeps the fights moving.",
    "Great story and memorable companions throughout.",
]


def _write_db(path: Path, n: int) -> None:
    """Write documents (and, for the seed, processed rows and aspect tags)."""
    from src.aspects import SentenceSplitter, load_lexicon, tag_sentences
    from src.utils import PROJECT_ROOT

    lexicon = load_lexicon(PROJECT_ROOT / "studies" / "aspects_rtwp_turn_v1.toml")
    conn = sqlite3.connect(path)
    create_study_tables(conn)
    for app_id in (1, 2):
        conn.execute("INSERT INTO metadata (app_id, app_name) VALUES (?, ?)", (app_id, f"G{app_id}"))
    splitter = SentenceSplitter()
    for i in range(n):
        text = TEXTS[i % 3]
        app = 1 if i % 4 else 2
        doc = conn.execute(
            "INSERT INTO documents (steam_review_id, app_id, review_text, recommended, "
            "timestamp_created) VALUES (?, ?, ?, ?, ?)",
            (str(5000 + i), app, text, i % 5 != 0, 50_000 + 1_000 * i),
        ).lastrowid
        conn.execute("INSERT INTO processed_documents VALUES (?, ?, '[]', 5, ?, NULL)",
                     (doc, text, len(text)))
        spans = splitter.split([text])[0]
        for m in tag_sentences([s[2] for s in spans], lexicon):
            s, e, _ = spans[m.sentence_idx]
            conn.execute(
                "INSERT INTO aspect_mentions (document_id, aspect, sentence_idx, matched_term, "
                "char_start, char_end, lexicon_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc, m.aspect, m.sentence_idx, m.matched_term, s, e, lexicon.version),
            )
    conn.commit()
    conn.close()


@pytest.fixture()
def study_path(tmp_path: Path) -> Path:
    path = tmp_path / "dm.toml"
    path.write_text(STUDY_TOML.format(root=tmp_path), encoding="utf-8")
    _write_db(tmp_path / "seed.db", 120)
    _write_db(tmp_path / "full.db", 120)
    return path


def test_seed_reviews_hold_only_the_split_and_dual_mode_apps(study_path: Path) -> None:
    study = load_study(study_path)
    reviews = dual_mode.seed_reviews(study, [1], "dev", "rtwp_turn_v1")
    expected = sum(1 for i in range(120) if i % 4 and dev_test_split(str(5000 + i)) == "dev")
    assert len(reviews) == expected
    assert {r.app_id for r in reviews} == {1}
    assert any(mode_group(r.aspects) == "tb_only" for r in reviews)
    assert any(mode_group(r.aspects) == "rt_only" for r in reviews)


def test_run_writes_json_and_markdown_without_touching_the_seed(study_path: Path, tmp_path: Path) -> None:
    before = (tmp_path / "seed.db").read_bytes()
    out = dual_mode.run(study_path, source="seed", split="dev", out_dir=tmp_path / "out")
    result = json.loads(out.read_text())
    assert result["split"] == "dev" and list(result["games"]) == ["1"]
    game = result["games"]["1"]
    assert game["before_after"]["event_date"] == "1970-01-02"
    assert game["before_after"]["before"]["reviews"] + game["before_after"]["after"]["reviews"] \
        == game["kept_reviews"]
    assert (tmp_path / "out" / "dual_mode_seed_dev.md").read_text().startswith("# Dual-mode views")
    assert (tmp_path / "seed.db").read_bytes() == before


def test_full_source_matches_the_seed_tags(study_path: Path, tmp_path: Path) -> None:
    """The full-database path re-derives the same kept reviews and aspect tags."""
    study = load_study(study_path)
    seed = dual_mode.seed_reviews(study, [1], "dev", "rtwp_turn_v1")
    full = dual_mode.full_reviews(study, [1], "dev")
    assert sorted((r.created, r.recommended, r.aspects) for r in full) == \
        sorted((r.created, r.recommended, r.aspects) for r in seed)


def test_run_rejects_an_unknown_split(study_path: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        dual_mode.run(study_path, split="holdout")
