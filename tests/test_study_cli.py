"""Tests for study TOML validation and the study CLI (no network)."""

import json
from pathlib import Path

import pytest

from src import study_cli
from src.study import load_study, parse_timestamp

BASE = """
[study]
name = "t"
title = "T"
seed_db = "data/t_seed.db"
full_db = "data/t_full.db"
raw_dir = "data/raw/t"

[sampling]
filter = "recent"
language = "english"
review_type = "{review_type}"
purchase_type = "all"
num_per_page = 100
raw_cap_per_game = 100
seed_cap_per_game = 50
random_seed = 1
{extra}

[[apps]]
app_id = 10
name = "A"
genre = "G"
cohort = "{cohort}"

[[apps]]
app_id = {second_id}
name = "B"
genre = "G"
cohort = "c"
"""


def _write(tmp_path: Path, review_type: str = "all", extra: str = "", cohort: str = "c",
           second_id: int = 20) -> Path:
    path = tmp_path / "t.toml"
    path.write_text(BASE.format(review_type=review_type, extra=extra, cohort=cohort,
                                second_id=second_id), encoding="utf-8")
    return path


def test_valid_study_loads(tmp_path: Path) -> None:
    study = load_study(_write(tmp_path, extra='window_start = 2025-01-01\nwindow_end = "2025-06-30"'))
    assert [a.app_id for a in study.apps] == [10, 20]
    assert study.sampling.window_start == parse_timestamp("2025-01-01")
    assert study.sampling.in_window(parse_timestamp("2025-06-30T23:59:59"))
    assert not study.sampling.in_window(parse_timestamp("2025-07-01"))


def test_missing_cohort_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no cohort"):
        load_study(_write(tmp_path, cohort=""))


def test_duplicate_app_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        load_study(_write(tmp_path, second_id=10))


def test_stratified_review_type_needs_the_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stratify"):
        load_study(_write(tmp_path, review_type="negative"))
    study = load_study(_write(tmp_path, review_type="negative", extra="stratified = true"))
    assert study.sampling.stratified
    with pytest.raises(ValueError, match="needs review_type"):
        load_study(_write(tmp_path, extra="stratified = true"))


def _new_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["new", "--name", "mini", "--apps", "10,20,30", "--cohort", "c1",
            "--out", str(tmp_path / "mini.toml"), "--data-dir", str(tmp_path / "data"), *extra]


def test_cli_new_writes_a_loadable_study_with_every_sampling_flag(tmp_path: Path) -> None:
    args = study_cli.build_parser().parse_args(_new_args(
        tmp_path, "--names", "A|B|C", "--genre", "Puzzle", "--seed-cap", "300",
        "--raw-cap", "900", "--purchase-type", "steam", "--language", "german",
        "--window-start", "2024-01-01", "--window-end", "2024-12-31", "--random-seed", "7",
    ))
    study = load_study(study_cli.new_study(args))
    s = study.sampling
    assert (s.seed_cap_per_game, s.raw_cap_per_game, s.purchase_type, s.language, s.random_seed) \
        == (300, 900, "steam", "german", 7)
    assert s.window_start == parse_timestamp("2024-01-01")
    assert [a.name for a in study.apps] == ["A", "B", "C"]
    assert {a.cohort for a in study.apps} == {"c1"}
    assert study.aspect_lexicon is not None and study.aspect_lexicon.name == "aspects_generic_v1.toml"


def test_cli_rejects_stratified_review_type_without_flag(tmp_path: Path) -> None:
    args = study_cli.build_parser().parse_args(_new_args(tmp_path, "--review-type", "negative"))
    with pytest.raises(ValueError, match="stratify"):
        study_cli.new_study(args)
    assert not (tmp_path / "mini.toml").exists()
    args = study_cli.build_parser().parse_args(
        _new_args(tmp_path, "--review-type", "negative", "--stratified"))
    assert load_study(study_cli.new_study(args)).sampling.stratified


def test_cli_lookup_uses_cache_then_fetcher_and_add_appends_a_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "data" / "raw" / "mini" / "appdetails"
    cache.mkdir(parents=True)
    (cache / "10.json").write_text(json.dumps(
        {"10": {"success": True, "data": {"name": "Cached\x07 Game", "genres": [{"description": "RPG"}]}}}
    ))
    calls: list[int] = []

    def fake_fetch(app_id: int) -> dict:
        calls.append(app_id)
        return {str(app_id): {"success": app_id != 30,
                              "data": {"name": f"Live {app_id}", "genres": []}}}

    monkeypatch.setattr(study_cli, "APPDETAILS_DELAY", 0)
    # The pre-seeded cache makes the raw dir exist, which only --force accepts.
    args = study_cli.build_parser().parse_args(_new_args(tmp_path, "--lookup", "--force"))
    study = load_study(study_cli.new_study(args, fetch=fake_fetch))
    assert calls == [20, 30]
    assert [(a.name, a.genre) for a in study.apps] == [
        ("Cached  Game", "RPG"), ("Live 20", "Unknown"), ("App 30", "Unknown")]

    add = study_cli.build_parser().parse_args(
        ["add", "--study", str(tmp_path / "mini.toml"), "--apps", "40", "--cohort", "c2",
         "--names", "D"])
    study = load_study(study_cli.add_apps(add))
    assert [(a.app_id, a.cohort) for a in study.apps][-1] == (40, "c2")
    with pytest.raises(ValueError, match="already"):
        study_cli.add_apps(add)
