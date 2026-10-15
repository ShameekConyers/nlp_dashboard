"""Tests for the phase B power helpers, on synthetic data."""

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from src import rtwp_power
from src.db_setup import create_study_tables
from src.stats import dev_test_split
from src.study import AppSpec, Sampling, Study


def _study(tmp_path: Path) -> Study:
    sampling = Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 1)
    apps = (AppSpec(1, "A", "CRPG", "crpg", "rtwp", ()),
            AppSpec(2, "B", "CRPG", "crpg", "turn_based", ()))
    return Study(
        name="p", title="t", seed_db=tmp_path / "seed.db", full_db=tmp_path / "full.db",
        raw_dir=tmp_path / "raw", language_filter=False, aspect_lexicon=None,
        sampling=sampling, baseline=None, apps=apps,
    )


def test_frame_sizes_count_test_half_ids_minus_exclusions_and_cap(tmp_path: Path) -> None:
    study = _study(tmp_path)
    conn = sqlite3.connect(study.full_db)
    create_study_tables(conn)
    ids = [(str(9000 + i), 1 if i % 3 else 2) for i in range(300)]
    conn.executemany("INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                     "VALUES (?, ?, 'x', 1)", ids)
    conn.commit()
    conn.close()
    test_ids = [(rid, app) for rid, app in ids if dev_test_split(rid) == "test"]
    excluded = {test_ids[0][0], next(rid for rid, _ in ids if dev_test_split(rid) == "dev")}
    want = {a: sum(1 for rid, app in test_ids if app == a and rid not in excluded) for a in (1, 2)}
    assert rtwp_power.frame_sizes(study, excluded) == want
    assert rtwp_power.frame_sizes(study, excluded, cap=10) == {a: min(n, 10) for a, n in want.items()}


def _export(findings: list[dict], version: str = rtwp_power.CANDIDATES_LEXICON, **metrics) -> dict:
    return {"params": {"lexicon_version": version},
            "metrics": {"stats_split": "dev", "precision_assumed": True, "findings": findings,
                        **metrics}}


def _findings() -> list[dict]:
    return [
        {"lens": f.lens, "theme": f.theme, "scope_type": f.scope_type, "scope_id": f.scope_id,
         "finding": f.direction, "lift": 2.0, "lift_ci": [1.5, 3.0]}
        for f in rtwp_power.CANDIDATES
    ]


def test_check_candidates_accepts_the_export_and_rejects_changes(tmp_path: Path) -> None:
    findings = _findings()
    version = rtwp_power.CANDIDATES_LEXICON
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_export(findings)))
    assert len(rtwp_power.check_candidates(path, version)) == 66
    flipped = [{**f, "finding": "complaint"} if f["finding"] == "praise" else f for f in findings]
    path.write_text(json.dumps(_export(flipped)))
    with pytest.raises(ValueError, match="CANDIDATES"):
        rtwp_power.check_candidates(path, version)
    path.write_text(json.dumps(_export(findings[1:])))
    with pytest.raises(ValueError, match="CANDIDATES"):
        rtwp_power.check_candidates(path, version)
    path.write_text(json.dumps(_export(findings, stats_split=None)))
    with pytest.raises(ValueError, match="development-half"):
        rtwp_power.check_candidates(path, version)


def test_check_candidates_refuses_a_lexicon_version_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_export(_findings(), version="old_v")))
    with pytest.raises(ValueError, match="used lexicon old_v"):
        rtwp_power.check_candidates(path, rtwp_power.CANDIDATES_LEXICON)
    path.write_text(json.dumps(_export(_findings(), version="new_v")))
    with pytest.raises(ValueError, match="CANDIDATES come from lexicon"):
        rtwp_power.check_candidates(path, "new_v")


def test_candidates_file_takes_the_newest_run_of_the_study_lexicon(tmp_path: Path) -> None:
    for run_id, version in ((3, "v2"), (4, "v3"), (5, "v2"), (12, "v3"), (13, "v4")):
        (tmp_path / f"candidates_run_{run_id}.json").write_text(json.dumps(_export([], version)))
    (tmp_path / "candidates_run_x.json").write_text(json.dumps(_export([], "v3")))
    assert rtwp_power.candidates_file(tmp_path, "v3").name == "candidates_run_12.json"
    assert rtwp_power.candidates_file(tmp_path, "v2").name == "candidates_run_5.json"
    with pytest.raises(FileNotFoundError, match="v9"):
        rtwp_power.candidates_file(tmp_path, "v9")


def _write_export(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["audit_id", "steam_review_id", "aspect", "sentence"])
        w.writerows([i, rid, "a", "synthetic"] for i, rid in enumerate(ids))


def test_frame_excludes_every_audit_set_the_study_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from src.aspects import Lexicon

    monkeypatch.setattr("src.aspects.OUTPUTS_DIR", tmp_path / "out")
    study = replace(_study(tmp_path), aspect_labels=(("a", "lx_v3"), ("b", "lx_v2")))
    lexicon = Lexicon("lx_v3", (), revised=("a",), supersedes="lx_v2")
    assert rtwp_power.audit_versions(study, lexicon) == ["lx_v3", "lx_v2"]
    conn = sqlite3.connect(study.full_db)
    create_study_tables(conn)
    ids = [str(7000 + i) for i in range(400)]
    conn.executemany("INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                     "VALUES (?, 1, 'x', 1)", [(rid,) for rid in ids])
    conn.commit()
    conn.close()
    test_ids = [rid for rid in ids if dev_test_split(rid) == "test"]
    v2_path = tmp_path / "out" / "audit" / "p_aspect_audit_lx_v2_unlabeled.csv"
    v3_path = tmp_path / "out" / "audit" / "p_aspect_audit_lx_v3_unlabeled.csv"
    with pytest.raises(FileNotFoundError):
        rtwp_power.frame_exclusions(study, lexicon)
    _write_export(v2_path, test_ids[:5])
    _write_export(v3_path, test_ids[5:8])
    sets = rtwp_power.frame_exclusions(study, lexicon)
    assert sets == {v3_path.name: set(test_ids[5:8]), v2_path.name: set(test_ids[:5])}
    excluded = set().union(*sets.values())
    assert rtwp_power.frame_sizes(study, excluded) == {1: len(test_ids) - 8}
    with pytest.raises(ValueError, match="does not exclude every audit set"):
        rtwp_power.frame_exclusions(study, lexicon, [v3_path])
    with pytest.raises(ValueError, match="lx_v3"):
        rtwp_power.frame_exclusions(study, lexicon, [v2_path])
    assert rtwp_power.frame_exclusions(study, lexicon, [v2_path, v3_path]).keys() == sets.keys()


def test_testable_candidates_drop_failing_aspects_by_merged_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from src.aspects import Aspect, Lexicon
    from src.replication import Finding

    monkeypatch.setattr("src.aspects.DATA_DIR", tmp_path / "data")
    lexicon = Lexicon("lx_v3", tuple(Aspect(n, "design", n, (), ()) for n in ("a", "b")),
                      revised=("a",), supersedes="lx_v2")
    study = replace(_study(tmp_path), aspect_labels=(("a", "lx_v3"), ("b", "lx_v2")))
    cands = (Finding("a", "aspect", "a", "cohort", "crpg", "complaint"),
             Finding("b", "aspect", "b", "cohort", "crpg", "complaint"),
             Finding("t", "topic", "4", "cohort", "crpg", "complaint"))
    labels = tmp_path / "data" / "labels"
    labels.mkdir(parents=True)
    _labels(labels / "aspect_audit_lx_v3.csv", {"a": 20})
    with pytest.raises(FileNotFoundError, match="lx_v2"):
        rtwp_power.testable_candidates(study, lexicon, cands)
    _labels(labels / "aspect_audit_lx_v2.csv", {"a": 40, "b": 36})
    kept, precision = rtwp_power.testable_candidates(study, lexicon, cands)
    assert [f.theme for f in kept] == ["b", "4"]
    assert precision["a"]["label_version"] == "lx_v3" and not precision["a"]["passes"]
    assert precision["b"]["label_version"] == "lx_v2" and precision["b"]["passes"]


def _labels(path: Path, relevant: dict[str, int], n: int = 40) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["steam_review_id", "sentence_idx", "aspect", "label"])
        for aspect, rel in relevant.items():
            for i in range(n):
                w.writerow([f"{aspect}{i}", 0, aspect, "relevant" if i < rel else "not_relevant"])


def test_scope_games_list_every_study_game_in_the_scope() -> None:
    from src.study import load_study

    study = load_study(rtwp_power.STUDY_PATH)
    scopes = rtwp_power.scope_games(study)
    by = {a.app_id: a for a in study.apps}
    crpg = scopes["aspect:bugs_performance@cohort:crpg"]
    assert len(crpg) == 10 and all(by[a].cohort == "crpg" for a in crpg)
    sr = scopes["topic:11@cohort_subgroup:strategy/rtwp"]
    assert len(sr) == 5 and all((by[a].cohort, by[a].subgroup) == ("strategy", "rtwp") for a in sr)
    assert len(scopes["aspect:bugs_performance@cohort:dual_mode"]) == 6
