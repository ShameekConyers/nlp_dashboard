"""Tests for the confirmatory replication pipeline, on synthetic data."""

import json
import math
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src import replication as rep
from src.db_setup import create_study_tables
from src.stats import GameData, cluster_bootstrap_macro, macro_theme_metrics
from src.study import AppSpec, Sampling, Study


def _game(app_id: int, n_rec: int, n_not: int, cols: list[tuple[int, int]]) -> GameData:
    """Build a synthetic game with one column per ``(theme_rec, theme_not)`` pair.

    Args:
        app_id: App ID.
        n_rec: Recommended reviews.
        n_not: Not-recommended reviews.
        cols: Per theme, recommended and not-recommended reviews that mention it.

    Returns:
        GameData.
    """
    rec = np.array([True] * n_rec + [False] * n_not)
    m = np.zeros((n_rec + n_not, len(cols)), dtype=bool)
    for j, (t_rec, t_not) in enumerate(cols):
        m[:t_rec, j] = True
        m[n_rec : n_rec + t_not, j] = True
    return GameData(app_id, rec, np.zeros(n_rec + n_not), m)


def _study(tmp_path: Path, apps: list[AppSpec]) -> Study:
    """A minimal non-frozen study with seed and full databases under ``tmp_path``."""
    sampling = Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 500, 42)
    return Study(
        name="synthetic", title="Synthetic", seed_db=tmp_path / "seed.db",
        full_db=tmp_path / "full.db", raw_dir=tmp_path / "raw", language_filter=False,
        aspect_lexicon=None, sampling=sampling, baseline=None, apps=tuple(apps),
    )


def _fill(db: Path, docs: list[tuple[str, int, bool]]) -> None:
    """Create study tables and insert ``(steam_review_id, app_id, recommended)`` documents."""
    conn = sqlite3.connect(db)
    create_study_tables(conn)
    for app_id in sorted({d[1] for d in docs}):
        conn.execute("INSERT OR IGNORE INTO metadata (app_id, app_name, genre) VALUES (?, ?, ?)",
                     (app_id, f"Game {app_id}", "G"))
    conn.executemany(
        "INSERT INTO documents (steam_review_id, app_id, review_text, recommended, "
        "timestamp_created) VALUES (?, ?, ?, ?, ?)",
        [(rid, app, f"text of review {rid}", int(r), 1_700_000_000 + i)
         for i, (rid, app, r) in enumerate(docs)],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


def test_findings_are_24_in_8_families() -> None:
    """The protocol's grouping: 3, 1, 5, 4, 5, 3, 2 and 1 findings per family."""
    assert len(rep.FINDINGS) == 24
    assert len({f.key for f in rep.FINDINGS}) == 24
    counts = {fam: sum(f.family == fam for f in rep.FINDINGS) for fam in rep.FAMILIES}
    assert counts == {
        "crash_topic": 3, "fft_topic": 1, "bugs_performance": 5, "pacing": 4,
        "ui_clarity": 5, "price_dlc": 3, "ai_quality": 2, "preview_reliability": 1,
    }
    assert [f.direction for f in rep.FINDINGS].count("praise") == 1


# ---------------------------------------------------------------------------
# Frame and draw
# ---------------------------------------------------------------------------


def test_draw_sample_is_deterministic_capped_and_seed_dependent() -> None:
    """Same frame and seed give the same draw; the cap binds; another seed differs."""
    frame = {1: [str(i) for i in range(50)], 2: [str(i) for i in range(100, 105)], 3: []}
    a = rep.draw_sample(frame, cap=10, seed=7)
    b = rep.draw_sample({k: list(reversed(v)) for k, v in frame.items()}, cap=10, seed=7)
    assert a == b
    assert len(a[1]) == 10 and set(a[1]) <= set(frame[1])
    assert a[2] == sorted(frame[2]) and a[3] == []
    assert rep.draw_sample(frame, cap=10, seed=8)[1] != a[1]


def test_sample_digest_ignores_order() -> None:
    """The digest depends on the (app, id) pairs only."""
    assert rep.sample_digest({1: ["a", "b"], 2: ["c"]}) == rep.sample_digest({2: ["c"], 1: ["b", "a"]})
    assert rep.sample_digest({1: ["a"]}) != rep.sample_digest({2: ["a"]})


def test_build_frame_excludes_ids_and_never_reads_the_recommend_flag(tmp_path: Path) -> None:
    """The frame drops excluded IDs, and SQLite reports no read of ``recommended``."""
    db = tmp_path / "full.db"
    _fill(db, [("r1", 10, True), ("r2", 10, False), ("r3", 10, True), ("r4", 20, False)])
    conn = sqlite3.connect(db)
    reads: list[tuple[str, str]] = []

    def authorizer(action: int, arg1: str | None, arg2: str | None, *_: object) -> int:
        """Record every column read."""
        if action == sqlite3.SQLITE_READ:
            reads.append((arg1 or "", arg2 or ""))
        return sqlite3.SQLITE_OK

    conn.set_authorizer(authorizer)
    frame = rep.build_frame(conn, [10, 20, 30], {"r2", "r4"})
    conn.close()
    assert frame == {10: ["r1", "r3"], 20: [], 30: []}
    assert ("documents", "recommended") not in reads


def test_write_sample_excludes_seed_and_stores_no_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed reviews never enter the sample, and the sample table has no recommend column."""
    out = tmp_path / "outputs" / "replication_2026"
    monkeypatch.setattr(rep, "REPLICATION_DIR", out)
    monkeypatch.setattr(rep, "PROJECT_ROOT", tmp_path)
    apps = [AppSpec(10, "A", "G", "c", "", ()), AppSpec(20, "B", "G", "c", "", ())]
    study = _study(tmp_path, apps)
    full_docs = [(f"a{i}", 10, i % 3 == 0) for i in range(30)] + [("b0", 20, True)]
    _fill(study.full_db, full_docs)
    _fill(study.seed_db, [("a0", 10, False), ("a1", 10, True), ("b0", 20, True)])
    monkeypatch.setattr(rep, "SAMPLE_CAP_PER_GAME", 2000)
    manifest = rep.write_sample(study)
    conn = sqlite3.connect(out / rep.SAMPLE_DB_NAME)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(sample)")]
    ids = {r[0] for r in conn.execute("SELECT steam_review_id FROM sample")}
    conn.close()
    assert "recommended" not in cols
    assert ids == {f"a{i}" for i in range(2, 30)}
    assert manifest["not_resamplable"] == [20]
    assert rep.load_sample(out)[1] == manifest["sample_sha256"]
    # A second draw of the same frame is accepted; a changed frame is refused.
    assert rep.write_sample(study)["sample_sha256"] == manifest["sample_sha256"]
    _fill(study.full_db, [("a99", 10, True)])
    with pytest.raises(ValueError, match="Refusing to replace"):
        rep.write_sample(study)


def test_safe_out_refuses_paths_outside_the_replication_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writes may only land under the replication directory."""
    monkeypatch.setattr(rep, "REPLICATION_DIR", tmp_path / "rep")
    assert rep._safe_out(tmp_path / "rep" / "x.json") == (tmp_path / "rep" / "x.json").resolve()
    with pytest.raises(ValueError, match="outside"):
        rep._safe_out(tmp_path / "elsewhere.json")


# ---------------------------------------------------------------------------
# Fast bootstrap and power
# ---------------------------------------------------------------------------


def test_macro_lift_equals_stats_macro_lift() -> None:
    """The four-cell lift matches ``stats.macro_theme_metrics`` exactly."""
    games = [_game(1, 80, 20, [(8, 6)]), _game(2, 60, 40, [(3, 9)]), _game(3, 90, 10, [(9, 1)])]
    cells = np.array([rep.four_cells(g.rec, g.themes[:, 0]) for g in games])
    assert float(rep.macro_lift(cells)) == pytest.approx(float(macro_theme_metrics(games).lift[0]))


def test_cluster_lift_ci_agrees_with_stats_cluster_bootstrap() -> None:
    """The multinomial shortcut gives the same interval as the review-level bootstrap."""
    games = [_game(i, 150 + 10 * i, 40 + 5 * i, [(10 + i, 9 + 2 * i)]) for i in range(6)]
    cells = np.array([rep.four_cells(g.rec, g.themes[:, 0]) for g in games])
    lo, hi = rep.cluster_lift_ci(cells, 4000, np.random.default_rng(1))
    ref = cluster_bootstrap_macro(games, 4000, 2)["lift"]
    assert lo == pytest.approx(float(ref[0][0]), rel=0.06)
    assert hi == pytest.approx(float(ref[1][0]), rel=0.06)


def test_meets_criterion_needs_direction_and_interval() -> None:
    """Point and interval must both sit on the original side of 1."""
    assert rep.meets_criterion("complaint", 2.0, 1.1, 3.0)
    assert not rep.meets_criterion("complaint", 2.0, 0.9, 3.0)
    assert not rep.meets_criterion("complaint", 0.8, 1.1, 3.0)
    assert not rep.meets_criterion("complaint", math.nan, math.nan, math.nan)
    assert rep.meets_criterion("praise", 0.5, 0.3, 0.8)
    assert not rep.meets_criterion("praise", 0.5, 0.3, 1.2)


def test_shrink_members_halves_the_log_lift_on_average() -> None:
    """With shrink 0.5 a lift of 4 becomes about 2; shrink 0 leaves flags alone."""
    rec = np.array([True] * 20000 + [False] * 20000)
    member = np.zeros(40000, dtype=bool)
    member[:1000] = True          # praise share 0.05
    member[20000:24000] = True    # complaint share 0.20, lift 4
    rng = np.random.default_rng(3)
    assert rep.shrink_members(rec, member, 0.0, rng) is member
    out = rep.shrink_members(rec, member, 0.5, rng)
    cells = rep.four_cells(rec, out)
    lift = (cells[0] / 20000) / (cells[2] / 20000)
    assert lift == pytest.approx(2.0, rel=0.05)
    assert (out[:20000] == member[:20000]).all()


def test_simulate_power_separates_strong_and_null_effects() -> None:
    """A strong complaint theme has power near 1, a null theme near 0."""
    seed_games = {}
    for i in range(1, 6):
        g = _game(i, 200, 100, [(10, 30), (15, 8)])
        seed_games[i] = rep.SeedGame(i, 300, g.rec, g.themes)
    findings = [
        rep.Finding("f1", "aspect", "strong", "cohort", "c", "complaint"),
        rep.Finding("f2", "aspect", "null", "cohort", "c", "complaint"),
    ]
    keys = [("aspect", "strong"), ("aspect", "null")]
    scope = {f.key: [1, 2, 3, 4, 5] for f in findings}
    draws = {i: 600 for i in range(1, 6)}
    res = rep.simulate_power(seed_games, draws, scope, findings, keys, 0.0,
                             n_sim=40, n_boot=200, seed=1, min_kept=50, min_not=10)
    assert res["per_finding"][findings[0].key]["power"] > 0.9
    assert res["per_finding"][findings[1].key]["power"] < 0.2
    assert res["count_p05"] <= res["count_mean"] <= res["count_p95"]


# ---------------------------------------------------------------------------
# Processing and evaluation
# ---------------------------------------------------------------------------


def test_process_reviews_aligns_kept_and_dropped_rows() -> None:
    """Dropped reviews get no aspects and topic -1; kept ones carry instrument output."""
    rows = [("r1", 1, "the game crashes a lot"), ("r2", 1, "ok"), ("r3", 2, "bugs and crashes")]

    def clean(raws: list[str]) -> list[dict]:
        return [{"cleaned_text": r, "dropped": len(r) < 5} for r in raws]

    def tag(texts: list[str]) -> list[set[str]]:
        return [{"bugs_performance"} if "bugs" in t else set() for t in texts]

    def topics(texts: list[str]) -> list[int]:
        return [11 if "crash" in t else -1 for t in texts]

    out = rep.process_reviews(rows, clean, tag, topics)
    assert [p.kept for p in out] == [True, False, True]
    assert out[0].topic_id == 11 and out[0].aspects == frozenset()
    assert out[1].topic_id == -1 and out[1].aspects == frozenset()
    assert out[2].aspects == frozenset({"bugs_performance"}) and out[2].topic_id == 11


def test_evaluate_replicates_strong_theme_and_rejects_null_and_small_scope() -> None:
    """Strong theme replicates, null theme does not, a two-game scope is not testable."""
    games = {
        i: _game(i, 300, 120, [(15, 36), (30, 12)]) for i in range(1, 6)
    }
    keys = [("aspect", "strong"), ("aspect", "null")]
    pools = [[0, 1], []]
    findings = [
        rep.Finding("fa", "aspect", "strong", "cohort", "c", "complaint"),
        rep.Finding("fb", "aspect", "null", "cohort", "c", "complaint"),
        rep.Finding("fa", "aspect", "strong", "subgroup", "s", "complaint"),
    ]
    scope = {findings[0].key: [1, 2, 3, 4, 5], findings[1].key: [1, 2, 3, 4, 5],
             findings[2].key: [1, 2, 9]}
    original = {f.key: {"lift": 2.5, "lift_ci": [1.5, 4.0]} for f in findings}
    res = rep.evaluate(games, keys, pools, scope, original, findings, n_boot=300)
    assert [r["status"] for r in res] == ["replicated", "not_replicated", "not_testable"]
    assert res[0]["lift"] == pytest.approx((36 / 120) / (15 / 300))
    assert res[0]["full_rule_same_direction"] and res[0]["logo_stable"]
    assert res[2]["replication_games"] == [1, 2]


def test_evaluate_drops_games_that_fail_eligibility() -> None:
    """A game below the not-recommended floor leaves the scope."""
    games = {i: _game(i, 300, 120, [(15, 36)]) for i in range(1, 4)}
    games[4] = _game(4, 300, 10, [(15, 3)])
    f = rep.Finding("fa", "aspect", "strong", "cohort", "c", "complaint")
    res = rep.evaluate(games, [("aspect", "strong")], [[0], []], {f.key: [1, 2, 3, 4]},
                       {f.key: {"lift": 2.0, "lift_ci": [1.2, 3.0]}}, [f], n_boot=100)
    assert res[0]["replication_games"] == [1, 2, 3]


def test_summarize_counts_and_family_majority() -> None:
    """Families need more than half of their testable findings; every count is reported."""
    def r(family: str, status: str) -> dict:
        return {"family": family, "status": status, "full_rule_same_direction": status == "replicated",
                "original_in_replication_ci": False, "replication_in_original_ci": False}

    results = [
        r("pacing", "replicated"), r("pacing", "replicated"), r("pacing", "not_replicated"),
        r("ai_quality", "replicated"), r("ai_quality", "not_replicated"),
        r("fft_topic", "not_testable"),
    ]
    s = rep.summarize(results, count_threshold=3)
    assert (s["n_findings"], s["n_testable"], s["n_replicated"], s["n_not_replicated"],
            s["n_not_testable"]) == (6, 5, 3, 2, 1)
    assert s["families"]["pacing"]["verdict"] == "replicated"
    assert s["families"]["ai_quality"]["verdict"] == "not_replicated"
    assert s["families"]["fft_topic"]["verdict"] == "not_testable"
    assert s["study_verdict"] == "consistent_with_power"
    assert rep.summarize(results, count_threshold=4)["study_verdict"] == "below_power"
    assert rep.summarize(results, count_threshold=None)["study_verdict"] is None


def test_render_report_marks_dry_runs() -> None:
    """A dry-run report says it is not a replication result."""
    s = rep.summarize([], count_threshold=None)
    text = rep.render_report(s, [], {"mode": "dry_run", "not_resampled": []})
    assert "DRY RUN" in text and "not replication results" in text


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    """Run git in a test repository."""
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   cwd=repo, check=True, capture_output=True)


def test_gate_needs_clean_tree_tag_and_tracked_protocol(tmp_path: Path) -> None:
    """Each gate condition is reported until it holds."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    _git(repo, "init", "-q")
    protocol = repo / "docs" / "protocol.md"
    protocol.write_text("protocol\n")
    problems = rep.gate_problems(repo, "proto-v1", protocol)
    assert any("not clean" in p for p in problems)
    assert any("not tracked" in p for p in problems)
    _git(repo, "add", "docs/protocol.md")
    _git(repo, "commit", "-q", "-m", "docs: add protocol")
    assert rep.gate_problems(repo, "proto-v1", protocol) == ["HEAD does not carry the tag proto-v1"]
    _git(repo, "tag", "proto-v1")
    assert rep.gate_problems(repo, "proto-v1", protocol) == []
    (repo / "stray.txt").write_text("x")
    assert any("not clean" in p for p in rep.gate_problems(repo, "proto-v1", protocol))


def test_confirmatory_read_refuses_when_gate_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read stops before opening any outcome when a gate or the read-once check fails."""
    monkeypatch.setattr(rep, "gate_problems", lambda: ["HEAD does not carry the tag"])
    monkeypatch.setattr(rep, "load_sample", lambda: ([("r1", 1, "t")], "d"))
    monkeypatch.setattr(rep, "pin_problems", lambda study, digest: [])
    monkeypatch.setattr(rep, "excluded_review_ids", lambda study: set())

    def boom(*_: object) -> None:
        raise AssertionError("outcomes opened")

    monkeypatch.setattr(rep, "_read_only", boom)
    study = _study(tmp_path, [AppSpec(1, "A", "G", "c", "", ())])
    with pytest.raises(RuntimeError, match="tag"):
        rep.confirmatory_read(study, tmp_path / "confirmatory")
    monkeypatch.setattr(rep, "gate_problems", lambda: [])
    (tmp_path / "confirmatory").mkdir()
    (tmp_path / "confirmatory" / "results.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already ran"):
        rep.confirmatory_read(study, tmp_path / "confirmatory")


def test_pin_problems_flag_mismatched_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A sample digest that differs from the pinned one is reported."""
    study = _study(tmp_path, [AppSpec(1, "A", "G", "c", "", ())])
    monkeypatch.setattr(rep, "EXPECTED_SAMPLE_SHA256", "abc")
    problems = rep.pin_problems(study, "def")
    assert any("sample hash def does not match" in p for p in problems)


# ---------------------------------------------------------------------------
# Dry run on the committed seed with stand-in instruments
# ---------------------------------------------------------------------------


def test_dry_run_full_path_with_stub_instruments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry run reads the seed's development split, evaluates all 24 findings and
    writes a report marked as a dry run, without a study verdict."""
    from src.study import load_study

    monkeypatch.setattr(rep, "REPLICATION_DIR", tmp_path / "rep")

    def clean(raws: list[str]) -> list[dict]:
        return [{"cleaned_text": r or "", "dropped": len(r or "") < 10} for r in raws]

    def tag(texts: list[str]) -> list[set[str]]:
        return [{"bugs_performance"} if "bug" in t.lower() else set() for t in texts]

    def topics(texts: list[str]) -> list[int]:
        return [11 if "crash" in t.lower() else -1 for t in texts]

    study = load_study(rep.MARKET_STUDY_PATH)
    payload = rep.dry_run(study, per_game=80, n_boot=20, instruments=(clean, tag, topics))
    assert payload["header"]["mode"] == "dry_run"
    assert payload["summary"]["n_findings"] == 24
    assert payload["summary"]["study_verdict"] is None
    out = tmp_path / "rep" / rep.DRY_RUN_SUBDIR
    assert "DRY RUN" in (out / "report.md").read_text()
    assert json.loads((out / "results.json").read_text())["header"]["per_game"] == 80
