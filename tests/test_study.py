"""Tests for the study loader: per-app raw caps, the stats split and the stats source."""

from pathlib import Path

import pytest

from src.study import DEFAULT_STUDY_PATH, STUDIES_DIR, load_study

BASE = """
[study]
name = "s"
title = "t"
seed_db = "data/s_seed.db"
full_db = "data/s_full.db"
raw_dir = "data/raw/s"
{study_extra}

[sampling]
filter = "recent"
language = "english"
review_type = "all"
purchase_type = "all"
num_per_page = 100
raw_cap_per_game = 100
seed_cap_per_game = 10
random_seed = 1

[[apps]]
app_id = 1
name = "A"
genre = "G"
cohort = "c"
{app_extra}

[[apps]]
app_id = 2
name = "B"
genre = "G"
cohort = "c"
"""


def _write(tmp_path: Path, study_extra: str = "", app_extra: str = "") -> Path:
    path = tmp_path / "s.toml"
    path.write_text(BASE.format(study_extra=study_extra, app_extra=app_extra), encoding="utf-8")
    return path


def test_raw_cap_and_stats_split_are_read(tmp_path: Path) -> None:
    study = load_study(_write(tmp_path, 'stats_split = "dev"', "raw_cap = 500"))
    assert study.stats_split == "dev"
    assert [a.raw_cap for a in study.apps] == [500, None]


def test_defaults_leave_both_unset(tmp_path: Path) -> None:
    study = load_study(_write(tmp_path))
    assert study.stats_split is None
    assert all(a.raw_cap is None for a in study.apps)


@pytest.mark.parametrize(("study_extra", "app_extra", "match"), [
    ('stats_split = "train"', "", "stats_split"),
    ("", "raw_cap = 0", "raw_cap"),
    ('stats_split = "dev"\nstats_source = "both"', "", "stats_source"),
    ('stats_source = "full"', "", "needs a stats_split"),
])
def test_invalid_values_are_rejected(tmp_path: Path, study_extra: str, app_extra: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_study(_write(tmp_path, study_extra, app_extra))


def test_new_fields_stay_out_of_repr_so_shipped_fingerprints_hold() -> None:
    """The pipeline fingerprints ``repr(study)``; unset new fields must not change it."""
    for path in (DEFAULT_STUDY_PATH, STUDIES_DIR / "strategy_market_2026.toml"):
        text = repr(load_study(path))
        assert "raw_cap=" not in text and "stats_split" not in text
        assert "stats_source" not in text


def test_full_source_points_the_analysis_view_at_the_split_database(tmp_path: Path) -> None:
    study = load_study(_write(tmp_path, 'stats_split = "dev"\nstats_source = "full"'))
    assert study.stats_source == "full"
    assert study.analysis_db == study.full_db.with_name("s_full_dev.db")
    view = study.analysis_view()
    assert view.seed_db == study.analysis_db and view.full_db == study.full_db
    assert view.analysis_view() is view
    assert repr(view) != repr(study)  # stage fingerprints see the database they run on
    from src.study import study_files

    assert study.analysis_db.resolve() in study_files(study)


def test_seed_source_keeps_the_seed(tmp_path: Path) -> None:
    for extra in ('stats_split = "dev"', 'stats_split = "dev"\nstats_source = "seed"'):
        study = load_study(_write(tmp_path, extra))
        assert study.analysis_db == study.seed_db
        assert study.analysis_view() is study


def test_aspect_labels_map_each_aspect_to_its_governing_version(tmp_path: Path) -> None:
    path = _write(tmp_path)
    path.write_text(path.read_text() + '\n[aspect_labels]\nlx_v3 = ["a", "b"]\nlx_v2 = ["c"]\n')
    study = load_study(path)
    assert study.label_versions() == {"a": "lx_v3", "b": "lx_v3", "c": "lx_v2"}
    assert "aspect_labels" not in repr(study)
    assert load_study(_write(tmp_path)).aspect_labels == ()


@pytest.mark.parametrize(("table", "match"), [
    ('lx_v3 = ["a"]\nlx_v2 = ["a", "b"]', "more than one version"),
    ('lx_v3 = "a"', "must list aspect names"),
])
def test_bad_aspect_labels_are_rejected(tmp_path: Path, table: str, match: str) -> None:
    path = _write(tmp_path)
    path.write_text(path.read_text() + f"\n[aspect_labels]\n{table}\n")
    with pytest.raises(ValueError, match=match):
        load_study(path)
