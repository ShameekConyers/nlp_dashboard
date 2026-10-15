"""Tests for the rtwp_turn_2026 phase B read, on synthetic data."""

import hashlib
import math
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src import rtwp_read as rr
from src.db_setup import create_study_tables
from src.replication import Finding
from src.stats import GameData, dev_test_split
from src.study import AppSpec, Sampling, Study


def _study(tmp_path: Path, apps: tuple[AppSpec, ...] | None = None) -> Study:
    """A synthetic study: four CRPGs, two in each subgroup, plus one strategy game."""
    sampling = Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 1)
    apps = apps or (
        AppSpec(1, "A", "CRPG", "crpg", "rtwp", ()),
        AppSpec(2, "B", "CRPG", "crpg", "rtwp", ()),
        AppSpec(3, "C", "CRPG", "crpg", "turn_based", ()),
        AppSpec(4, "D", "CRPG", "crpg", "turn_based", ()),
        AppSpec(5, "E", "Strategy", "strategy", "turn_based", ()),
    )
    return Study(
        name="p", title="t", seed_db=tmp_path / "seed.db", full_db=tmp_path / "full.db",
        raw_dir=tmp_path / "raw", language_filter=False, aspect_lexicon=None,
        sampling=sampling, baseline=None, apps=apps,
    )


def _full_db(study: Study, n: int = 400) -> list[tuple[str, int, int]]:
    """Fill the full database with synthetic reviews; return ``(id, app, recommended)``."""
    conn = sqlite3.connect(study.full_db)
    create_study_tables(conn)
    rows = [(str(70000 + i), 1 + i % 5, i % 2) for i in range(n)]
    conn.executemany(
        "INSERT INTO documents (steam_review_id, app_id, review_text, recommended, "
        "timestamp_created) VALUES (?, ?, 'text ' || ?, ?, 1700000000)",
        [(rid, app, rid, rec) for rid, app, rec in rows],
    )
    conn.commit()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------


def test_build_frame_keeps_test_half_minus_audit_ids(tmp_path: Path) -> None:
    """The frame is every test-half review outside the audit sets, sorted, IDs only."""
    study = _study(tmp_path)
    rows = _full_db(study)
    test = [(app, rid) for rid, app, _ in rows if dev_test_split(rid) == "test"]
    excluded = {test[0][1], test[5][1], next(r for r, _, _ in rows if dev_test_split(r) == "dev")}
    frame = rr.build_frame(study, excluded)
    assert frame == sorted((a, r) for a, r in test if r not in excluded)
    path = tmp_path / "frame.tsv"
    path.write_bytes(rr.frame_bytes(frame))
    assert rr.load_frame(path) == frame


def test_fetch_frame_reviews_reads_only_frame_ids(tmp_path: Path) -> None:
    """Text and recommend flags are read for the frame reviews and nothing else."""
    study = _study(tmp_path)
    rows = _full_db(study)
    frame = rr.build_frame(study, set())[::3]
    got, recommended = rr.fetch_frame_reviews(study, frame)
    assert sorted(r[0] for r in got) == sorted(rid for _, rid in frame)
    assert set(recommended) == {rid for _, rid in frame}
    flags = {rid: bool(rec) for rid, _, rec in rows}
    assert all(recommended[rid] == flags[rid] for rid in recommended)


def test_fetch_frame_reviews_refuses_a_review_in_another_game(tmp_path: Path) -> None:
    """A frame line whose game does not match the database stops the read."""
    study = _study(tmp_path)
    _full_db(study)
    app, rid = rr.build_frame(study, set())[0]
    with pytest.raises(RuntimeError, match="belongs to app"):
        rr.fetch_frame_reviews(study, [(app + 1, rid)])
    with pytest.raises(RuntimeError, match="missing"):
        rr.fetch_frame_reviews(study, [(app, "999999999999")])


# ---------------------------------------------------------------------------
# Gate and pins
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    """Run git in a test repository."""
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                   cwd=repo, check=True, capture_output=True)


def test_gate_refuses_dirty_tree_missing_tag_and_existing_results(tmp_path: Path) -> None:
    """Each gate condition is reported until it holds."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo, "init", "-q")
    protocol = repo / "docs" / "protocol.md"
    protocol.write_text("protocol\n")
    results = tmp_path / "out" / "results.json"
    problems = rr.gate_problems(repo, "proto-v1", protocol, results)
    assert any("not clean" in p for p in problems)
    assert any("not tracked" in p for p in problems)
    _git(repo, "add", "docs/protocol.md")
    _git(repo, "commit", "-q", "-m", "docs: add protocol")
    assert rr.gate_problems(repo, "proto-v1", protocol, results) == [
        "HEAD does not carry the tag proto-v1"]
    _git(repo, "tag", "proto-v1")
    assert rr.gate_problems(repo, "proto-v1", protocol, results) == []
    (repo / "stray.txt").write_text("x")
    assert any("not clean" in p for p in rr.gate_problems(repo, "proto-v1", protocol, results))
    (repo / "stray.txt").unlink()
    results.parent.mkdir()
    results.write_text("{}")
    assert any("already ran" in p for p in rr.gate_problems(repo, "proto-v1", protocol, results))
    results.unlink()
    (repo / "docs" / "more.md").write_text("y")
    _git(repo, "add", "docs/more.md")
    _git(repo, "commit", "-q", "-m", "docs: add more")
    assert rr.gate_problems(repo, "proto-v1", protocol, results) == [
        "HEAD does not carry the tag proto-v1"]


def _pins(tmp_path: Path) -> dict[str, tuple[Path, str | None]]:
    """One synthetic file per real pin, with its correct hash."""
    pins = {}
    for i, name in enumerate(rr.PINS):
        path = tmp_path / f"pin_{i}.bin"
        path.write_bytes(f"content {name}".encode())
        pins[name] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    return pins


def test_pin_problems_flag_each_mismatch(tmp_path: Path) -> None:
    """Every pinned file is checked: a change to any one of them is reported by name."""
    pins = _pins(tmp_path)
    assert rr.pin_problems(pins) == []
    assert set(rr.PINS) == {"candidates", "lexicon", "labels v2", "labels v3", "topic model",
                            "power", "frame"}
    for name, (path, _) in pins.items():
        original = path.read_bytes()
        path.write_bytes(original + b"!")
        assert rr.pin_problems(pins) == [f"{name} file {path} does not match its pinned SHA-256"]
        path.write_bytes(original)
    path, _ = pins["frame"]
    path.rename(tmp_path / "gone")
    assert "missing" in rr.pin_problems(pins)[0]
    assert "not pinned" in rr.pin_problems({"frame": (path, None)})[0]


def test_real_pins_are_filled() -> None:
    """Every pin carries a 64-character SHA-256."""
    assert all(want is not None and len(want) == 64 for _, want in rr.PINS.values())


def _no_outcomes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make any outcome or frame access fail the test; return the call log."""
    calls: list[str] = []

    def boom(*_: object, **__: object) -> None:
        calls.append("opened")
        raise AssertionError("outcomes opened before the gate passed")

    monkeypatch.setattr(rr, "fetch_frame_reviews", boom)
    monkeypatch.setattr(rr, "load_frame", boom)
    monkeypatch.setattr(rr, "_read_only", boom)
    monkeypatch.setattr(rr, "_run", boom)
    return calls


@pytest.mark.parametrize("problem", [
    "working tree is not clean (git status --porcelain is not empty)",
    "HEAD does not carry the tag rtwp-turn-protocol-v1",
    "results.json exists: the confirmatory read already ran",
])
def test_read_refuses_before_any_outcome_when_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """A closed gate stops the read before any review or recommend flag is read."""
    calls = _no_outcomes(monkeypatch)
    monkeypatch.setattr(rr, "gate_problems", lambda **_: [problem])
    monkeypatch.setattr(rr, "pin_problems", lambda: [])
    with pytest.raises(RuntimeError, match="refused"):
        rr.confirmatory_read(_study(tmp_path), tmp_path / "out")
    assert calls == []


def test_read_refuses_before_any_outcome_on_each_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed pinned file stops the read before any review or recommend flag is read."""
    calls = _no_outcomes(monkeypatch)
    monkeypatch.setattr(rr, "gate_problems", lambda **_: [])
    pins = _pins(tmp_path)
    monkeypatch.setattr(rr, "PINS", pins)
    for name, (path, _) in pins.items():
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with pytest.raises(RuntimeError, match=name):
            rr.confirmatory_read(_study(tmp_path), tmp_path / "out")
        path.write_bytes(original)
    assert calls == []


def test_read_processes_only_frame_reviews(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the gate open, the read passes exactly the frame reviews to the instruments."""
    study = _study(tmp_path)
    _full_db(study)
    frame = rr.build_frame(study, set())[::2]
    frame_path = tmp_path / "frame.tsv"
    frame_path.write_bytes(rr.frame_bytes(frame))
    monkeypatch.setattr(rr, "gate_problems", lambda **_: [])
    monkeypatch.setattr(rr, "pin_problems", lambda: [])
    monkeypatch.setattr(rr, "FRAME_PATH", frame_path)
    monkeypatch.setattr(rr, "FRAME_TOTAL", len(frame))
    seen: dict[str, object] = {}

    def capture(study_, rows, recommended, *_: object) -> dict:
        seen["rows"], seen["recommended"] = rows, recommended
        return {}

    monkeypatch.setattr(rr, "_run", capture)
    rr.confirmatory_read(study, tmp_path / "out", instruments=(None, None, None, None))
    assert sorted(r[0] for r in seen["rows"]) == sorted(rid for _, rid in frame)
    assert set(seen["recommended"]) == {rid for _, rid in frame}


def test_read_refuses_a_frame_of_the_wrong_size_or_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame must hold FRAME_TOTAL test-half reviews."""
    monkeypatch.setattr(rr, "gate_problems", lambda **_: [])
    monkeypatch.setattr(rr, "pin_problems", lambda: [])
    dev_id = next(str(i) for i in range(1000, 2000) if dev_test_split(str(i)) == "dev")
    monkeypatch.setattr(rr, "load_frame", lambda _: [(1, dev_id)])
    monkeypatch.setattr(rr, "FRAME_TOTAL", 2)
    with pytest.raises(RuntimeError, match="not 2"):
        rr.confirmatory_read(_study(tmp_path), tmp_path / "out")
    monkeypatch.setattr(rr, "FRAME_TOTAL", 1)
    with pytest.raises(RuntimeError, match="outside the test half"):
        rr.confirmatory_read(_study(tmp_path), tmp_path / "out")


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


def test_process_reviews_runs_instruments_on_kept_texts_only() -> None:
    """Dropped reviews reach no instrument; kept ones carry tags, topic and compound."""
    rows = [("r1", 1, 10, "keep one"), ("r2", 1, 11, "drop"), ("r3", 2, None, "keep two")]
    seen: list[list[str]] = []

    def clean(raws: list[str]) -> list[dict]:
        return [{"dropped": r == "drop", "cleaned_text": r.upper()} for r in raws]

    def record(value):
        def f(texts: list[str]) -> list:
            seen.append(list(texts))
            return [value(t) for t in texts]
        return f

    out = rr.process_reviews(rows, (clean, record(lambda t: {t[-3:]}), record(lambda t: 7),
                                    record(lambda t: 0.5)))
    assert seen == [["KEEP ONE", "KEEP TWO"]] * 3
    assert [r.kept for r in out] == [True, False, True]
    assert out[0].aspects == frozenset({"ONE"}) and out[0].topic_id == 7
    assert out[1].topic_id == -1 and math.isnan(out[1].compound)
    assert out[2].created is None and out[2].compound == 0.5


# ---------------------------------------------------------------------------
# Criterion and counts
# ---------------------------------------------------------------------------


def _game(app_id: int, n: int, shares: list[tuple[float, float]]) -> GameData:
    """A game with ``n`` recommended and ``n`` not-recommended reviews.

    Args:
        app_id: App ID.
        n: Reviews per side.
        shares: Per theme, the share of recommended and of not-recommended
            reviews that mention it.

    Returns:
        GameData.
    """
    rec = np.array([True] * n + [False] * n)
    themes = np.zeros((2 * n, len(shares)), dtype=bool)
    for j, (s_rec, s_not) in enumerate(shares):
        themes[: int(n * s_rec), j] = True
        themes[n : n + int(n * s_not), j] = True
    return GameData(app_id, rec, np.zeros(2 * n), themes)


def test_evaluate_and_summarize_on_a_toy_example(tmp_path: Path) -> None:
    """The criterion, eligibility on the phase B sample and the counts on known data."""
    study = _study(tmp_path)
    keys = [("aspect", "pause_micro"), ("aspect", "price_dlc"), ("aspect", "ui_clarity"),
            ("topic", "1"), ("aspect", "waiting_turns")]
    shares = [(0.05, 0.40), (0.20, 0.20), (0.30, 0.05), (0.10, 0.10), (0.30, 0.05)]
    games = {a: _game(a, 200, shares) for a in (1, 2, 3, 4)}
    games[5] = _game(5, 20, shares)  # too few kept reviews: not eligible in phase B
    cands = [
        Finding("pause_micro", "aspect", "pause_micro", "cohort", "crpg", "complaint"),
        Finding("price_dlc", "aspect", "price_dlc", "cohort", "crpg", "complaint"),
        Finding("ui_clarity", "aspect", "ui_clarity", "cohort", "crpg", "praise"),
        Finding("pause_micro", "aspect", "pause_micro", "subgroup", "rtwp", "complaint"),
        Finding("pause_micro", "aspect", "pause_micro", "subgroup", "turn_based", "complaint"),
        Finding("dlc_topic", "topic", "1", "cohort", "crpg", "complaint"),
        Finding("ui_clarity", "aspect", "ui_clarity", "cohort", "crpg", "complaint"),
        Finding("waiting_turns", "aspect", "waiting_turns", "cohort", "crpg", "praise"),
        Finding("waiting_turns", "aspect", "waiting_turns", "cohort", "crpg", "complaint"),
    ]
    dev = {f.key: {"lift": 2.0, "lift_ci": [1.5, 3.0]} for f in cands}
    passing = {"pause_micro", "price_dlc", "waiting_turns"}
    ev = rr.evaluate(study, games, keys, passing, dev, {1, 2, 3, 4, 5}, candidates=cands,
                     n_boot=200)
    by_key = {(r["key"], r["direction"]): r for r in ev["candidates"]}
    status = {k: r["status"] for k, r in by_key.items()}
    assert ev["eligible"] == [1, 2, 3, 4]
    assert status[("aspect:pause_micro@cohort:crpg", "complaint")] == "replicated"
    assert status[("aspect:price_dlc@cohort:crpg", "complaint")] == "not_replicated"
    assert status[("aspect:ui_clarity@cohort:crpg", "praise")] == "not_testable"  # failed audit
    assert status[("aspect:pause_micro@subgroup:rtwp", "complaint")] == "not_testable"  # 2 games
    # turn_based holds games 3, 4 and 5, all eligible in development, but 5 is not in phase B.
    row = by_key[("aspect:pause_micro@subgroup:turn_based", "complaint")]
    assert row["status"] == "not_testable" and row["dev_eligible_games"] == 3
    assert row["phase_b_games"] == [3, 4]
    assert status[("topic:1@cohort:crpg", "complaint")] == "not_replicated"
    assert status[("aspect:waiting_turns@cohort:crpg", "praise")] == "replicated"
    assert status[("aspect:waiting_turns@cohort:crpg", "complaint")] == "not_replicated"
    rep = by_key[("aspect:pause_micro@cohort:crpg", "complaint")]
    assert rep["lift"] == pytest.approx(8.0) and rep["ci"][0] > 1
    assert rep["combat"] is True and rep["dev_in_phase_b_ci"] is False
    assert by_key[("aspect:waiting_turns@cohort:crpg", "praise")]["ci"][1] < 1

    s = rr.summarize(ev["candidates"], count_threshold=2)
    assert (s["n_candidates"], s["n_testable"], s["n_replicated"], s["n_not_replicated"],
            s["n_not_testable"]) == (9, 5, 2, 3, 4)
    assert s["families"]["pause_micro"]["verdict"] == "replicated"
    assert s["families"]["waiting_turns"]["verdict"] == "not_replicated"  # 1 of 2 is not a majority
    assert s["families"]["price_dlc"]["verdict"] == "not_replicated"
    assert s["families"]["ui_clarity"]["verdict"] == "not_testable"
    assert s["families"]["dlc_topic"]["label"] == "Topic 1, DLC"
    assert s["study_verdict"] == "consistent_with_power"
    assert s["combat"] == {"candidates": 5, "replicated": 2, "not_testable": 2,
                           "expected_plug_in": 9.8, "expected_half_log_lift": 2.5,
                           "verdict": None}
    assert rr.summarize(ev["candidates"], count_threshold=3)["study_verdict"] == "below_power"
    assert rr.summarize(ev["candidates"], count_threshold=None)["study_verdict"] is None
    text = rr.render_report({"header": {"mode": "dry_run"}, "summary": s, "evaluation": ev,
                             "dual_mode": {}})
    assert "DRY RUN" in text and "## Deviations" in text


def test_protocol_constants_match_the_locked_decisions() -> None:
    """The pass line, expectations and candidate counts are the locked numbers."""
    assert (rr.COUNT_THRESHOLD, rr.EXPECTED_COUNT_PLUG_IN, rr.EXPECTED_COUNT_HALF) == (43, 60.0, 46.7)
    assert (rr.COMBAT_EXPECTED_PLUG_IN, rr.COMBAT_EXPECTED_HALF) == (9.8, 2.5)
    assert rr.FRAME_TOTAL == 119485
    combat = [f for f in rr.CANDIDATES if f.lens == "aspect" and f.theme in rr.COMBAT_ASPECTS]
    assert len(rr.CANDIDATES) == 66 and len(combat) == 14
