"""End-to-end pipeline test on cached fixture pulls (no network, no model download)."""

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from src import pipeline, study_cli
from src.study import load_study

APPS = (101, 202, 303)

TEXTS_POS = [
    "The story and characters are wonderful and I loved every quest.",
    "Great content with hours of replayability, easily worth the price.",
    "Co-op with friends is a blast and the servers are stable.",
    "Smooth performance, no stutter at all on my old laptop.",
]
TEXTS_NEG = [
    "Crashes every hour and the bugs ruin the whole experience.",
    "Terrible optimization, the frame rate drops constantly.",
    "Overpriced and the DLC is a cash grab, not worth it.",
    "The menus and interface are confusing and the controls feel awful.",
]


def _write_fixture_pulls(raw_dir: Path) -> None:
    """Write cached raw pages and manifests so the pull stage skips every app."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    rid = 0
    for app_id in APPS:
        reviews = []
        for i in range(120):
            rid += 1
            up = i >= 40
            text = (TEXTS_POS if up else TEXTS_NEG)[i % 4] + f" Review number {i} for app {app_id}."
            reviews.append({
                "recommendationid": str(900000 + rid), "review": text, "voted_up": up,
                "language": "english", "timestamp_created": 1_700_000_000 + rid,
                "votes_up": 0, "votes_funny": 0, "written_during_early_access": False,
                "author": {"playtime_at_review": 600, "playtime_forever": 900},
            })
        (raw_dir / f"{app_id}.json").write_text(json.dumps(reviews), encoding="utf-8")
        (raw_dir / f"{app_id}.pull.json").write_text(json.dumps({
            "app_id": app_id, "app_name": f"G{app_id}", "study": "mini",
            "params": {"review_type": "all"}, "baseline_params": None,
            "started_at": "2026-09-26T00:00:00+00:00", "finished_at": "2026-09-26T00:01:00+00:00",
            "pages": 2, "reviews_collected": 120, "cap": 20000, "cap_hit": False,
            "stop_reason": "exhausted", "stratified": False,
            "effective_window_start": 1_700_000_000, "effective_window_end": 1_700_001_000,
            "query_summary": {"total_positive": 80, "total_negative": 40, "total_reviews": 120},
            "baseline_summary": None,
        }), encoding="utf-8")


def _fake_topics(study, unfreeze: bool = False) -> None:  # noqa: ANN001 - Study
    """Assign topic 0 to story reviews and -1 to the rest, for untopiced rows only."""
    conn = sqlite3.connect(study.seed_db)
    rows = conn.execute(
        "SELECT p.document_id, p.cleaned_text FROM processed_documents p "
        "JOIN nlp_results n ON n.document_id = p.document_id WHERE n.topic_id IS NULL"
    ).fetchall()
    conn.executemany(
        "UPDATE nlp_results SET topic_id = ?, topic_label = ? WHERE document_id = ?",
        [(0 if "story" in t else -1, "Story" if "story" in t else "Outlier", d) for d, t in rows],
    )
    conn.execute("INSERT OR IGNORE INTO topics (topic_id, label, top_words, doc_count) "
                 "VALUES (0, 'Story', '[\"story\", \"characters\", \"quest\"]', 0)")
    conn.commit()
    conn.close()


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, str]]:
    """Row count and content hash of every table in the given databases."""
    out: dict[str, tuple[int, str]] = {}
    for path in paths:
        conn = sqlite3.connect(path)
        for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
            n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            out[f"{path.name}:{table}"] = (n, pipeline.table_signature(conn, table))
        conn.close()
    return out


def test_second_pipeline_run_changes_no_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.stats.N_BOOT", 50)
    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path / "labels_none")
    args = study_cli.build_parser().parse_args([
        "new", "--name", "mini", "--apps", ",".join(map(str, APPS)), "--cohort", "c1",
        "--names", "G101|G202|G303", "--out", str(tmp_path / "mini.toml"),
        "--data-dir", str(tmp_path / "data"), "--no-baseline",
    ])
    study = load_study(study_cli.new_study(args))
    _write_fixture_pulls(study.raw_dir)
    stages = [replace(s, run=_fake_topics) if s.name == "topics" else s for s in pipeline.STAGES]

    first = pipeline.run(study, full=True, stages=stages)
    # The fixture pages are already cached, so the pull stage has nothing to do.
    assert first == dict.fromkeys(first, "ran") | {"pull": "no-op"}, first
    before = _snapshot([study.full_db, study.seed_db])
    assert before[f"{study.seed_db.name}:documents"][0] == 360
    assert before[f"{study.seed_db.name}:theme_stats"][0] > 0

    second = pipeline.run(study, full=True, stages=stages)
    assert second == {
        "pull": "no-op", "load": "no-op", "seed": "skipped", "preprocess": "no-op",
        "sentiment": "no-op", "topics": "no-op", "aspects": "skipped", "stats": "skipped",
    }
    assert _snapshot([study.full_db, study.seed_db]) == before


def test_pipeline_refuses_the_april_baseline() -> None:
    with pytest.raises(ValueError, match="natural"):
        pipeline.run(load_study())


def test_load_applies_window_and_records_stratified_pulls(tmp_path: Path) -> None:
    """Reviews outside the window are not loaded; a stratified pull is flagged for the UI."""
    from dashboard.queries import has_stratified_pulls
    from src.db_setup import load_study_full

    args = study_cli.build_parser().parse_args([
        "new", "--name", "strat", "--apps", ",".join(map(str, APPS)), "--cohort", "c1",
        "--names", "A|B|C", "--out", str(tmp_path / "strat.toml"),
        "--data-dir", str(tmp_path / "data"), "--no-baseline", "--review-type", "negative",
        "--stratified", "--window-start", "2023-11-14", "--window-end", "2023-11-14T22:15:00",
    ])
    study = load_study(study_cli.new_study(args))
    _write_fixture_pulls(study.raw_dir)
    for app_id in APPS:
        path = study.raw_dir / f"{app_id}.pull.json"
        manifest = json.loads(path.read_text())
        manifest["stratified"] = True
        path.write_text(json.dumps(manifest))
    load_study_full(study)
    conn = sqlite3.connect(study.full_db)
    lo, hi = study.sampling.window_start, study.sampling.window_end
    n, outside = conn.execute(
        "SELECT COUNT(*), SUM(timestamp_created < ? OR timestamp_created > ?) FROM documents",
        (lo, hi),
    ).fetchone()
    assert (n, outside) == (100, 0)  # ids 1..100 fall at or before 22:15:00Z
    assert has_stratified_pulls(conn)
    conn.close()


def test_seed_sampling_applies_the_window_to_older_loads(tmp_path: Path) -> None:
    """A window narrowed after loading still limits the seed sample."""
    from src.db_setup import build_natural_seed, load_study_full
    from src.study import parse_timestamp

    args = study_cli.build_parser().parse_args([
        "new", "--name", "win", "--apps", ",".join(map(str, APPS)), "--cohort", "c1",
        "--names", "A|B|C", "--out", str(tmp_path / "win.toml"),
        "--data-dir", str(tmp_path / "data"), "--no-baseline",
    ])
    study = load_study(study_cli.new_study(args))
    _write_fixture_pulls(study.raw_dir)
    assert load_study_full(study) == 360
    lo, hi = parse_timestamp("2023-11-14T22:14:00"), parse_timestamp("2023-11-14T22:16:00")
    full = sqlite3.connect(study.full_db)
    try:
        counts = build_natural_seed(full, study.seed_db, 500, 7, (lo, hi))
    finally:
        full.close()
    seed = sqlite3.connect(study.seed_db)
    n, outside = seed.execute(
        "SELECT COUNT(*), SUM(timestamp_created < ? OR timestamp_created > ?) FROM documents",
        (lo, hi),
    ).fetchone()
    seed.close()
    # Review ids 40..160 fall in the window: 81 of app 101 and the first 40 of app 202.
    assert counts == {101: 81, 202: 40, 303: 0}
    assert (n, outside) == (121, 0)
