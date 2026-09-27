"""Frozen (shipped) studies: pipeline and stage refusals, and study CLI path collisions."""

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from src import pipeline, study_cli
from src.study import DEFAULT_STUDY_PATH, STUDIES_DIR, FrozenStudyError, Study, guard_writable, load_study

MARKET_TOML = STUDIES_DIR / "strategy_market_2026.toml"
SENTINEL = b"shipped bytes"


def _new(name: str, out: Path, data: Path, *extra: str) -> Path:
    """Run ``study_cli new`` for a one-cohort, two-app study."""
    args = study_cli.build_parser().parse_args([
        "new", "--name", name, "--apps", "11,22", "--cohort", "c", "--names", "A|B",
        "--genre", "G", "--out", str(out), "--data-dir", str(data), "--no-baseline", *extra,
    ])
    return study_cli.new_study(args)


@pytest.fixture()
def studies_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary studies directory that the freeze checks scan instead of ``studies/``."""
    d = tmp_path / "studies"
    d.mkdir()
    monkeypatch.setattr("src.study.STUDIES_DIR", d)
    return d


@pytest.fixture()
def frozen(tmp_path: Path, studies_dir: Path) -> Study:
    """A frozen study whose seed and full DBs hold sentinel bytes."""
    path = _new("mini", studies_dir / "mini.toml", tmp_path / "data")
    path.write_text(path.read_text().replace("[study]\n", "[study]\nfrozen = true\n", 1))
    study = load_study(path)
    assert study.frozen
    for db in (study.seed_db, study.full_db):
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_bytes(SENTINEL)
    return study


def _unchanged(study: Study) -> bool:
    return all(p.read_bytes() == SENTINEL for p in (study.seed_db, study.full_db))


def test_pipeline_refuses_a_frozen_study(frozen: Study) -> None:
    for full in (False, True):
        with pytest.raises(FrozenStudyError, match="frozen"):
            pipeline.run(frozen, full=full)
    assert _unchanged(frozen)


def test_pipeline_refuses_a_study_that_reuses_frozen_files(frozen: Study) -> None:
    alias = replace(frozen, name="alias", frozen=False)
    with pytest.raises(FrozenStudyError, match="mini"):
        pipeline.run(alias, full=True)
    assert _unchanged(frozen)


def test_pipeline_cli_refuses_the_shipped_market_study(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        pipeline.main(["--study", str(MARKET_TOML), "--full"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "--unfreeze-i-know" in err


def test_shipped_studies_are_frozen() -> None:
    market, april = load_study(MARKET_TOML), load_study(DEFAULT_STUDY_PATH)
    assert market.frozen and april.frozen
    for path in (market.seed_db, april.seed_db, april.scores_path):
        with pytest.raises(FrozenStudyError):
            guard_writable([path])


def _writers() -> dict[str, Callable[[Study], object]]:
    from src import aspects, data_pull, db_setup, preprocessing, sentiment, stats, topics
    from src import sentiment_transformer as st

    return {
        "pull": data_pull.pull_study,
        "load": db_setup.load_study_full,
        "seed": lambda s: db_setup.build_natural_seed(sqlite3.connect(":memory:"), s.seed_db, 10, 1),
        "db_setup_full": lambda s: db_setup.main_study(s, full=True),
        "preprocess": lambda s: preprocessing.run(mode="seed", study=s),
        "sentiment": lambda s: sentiment.run(mode="seed", study=s),
        "topics": lambda s: topics.run(mode="seed", study=s),
        "aspects": aspects.run,
        "stats": lambda s: stats.run(s, n_boot=10),
        "coverage": stats.coverage_proxy,
        "reference": lambda s: stats.reference_checks(s, Path("missing.json")),
        "transformer_score": lambda s: st.score_study(s, None, "doc"),  # type: ignore[arg-type]
        "transformer_validate": lambda s: st.validate(s, s),
        "sentence_audit": lambda s: st.sentence_audit_metrics(s, "m"),
        "sentence_dev_select": st.sentence_dev_selection,
        "reference_sentiment": lambda s: st.reference_sentiment_check(s, Path("missing.json")),
    }


@pytest.mark.parametrize("stage", list(_writers()))
def test_every_writing_stage_refuses_a_frozen_study(stage: str, frozen: Study) -> None:
    with pytest.raises(FrozenStudyError, match="--unfreeze-i-know"):
        _writers()[stage](frozen)
    assert _unchanged(frozen)


def test_legacy_april_full_build_is_refused() -> None:
    from src import db_setup

    with pytest.raises(FrozenStudyError, match="april_2026_baseline"):
        db_setup.main("full")


def test_unfreeze_lets_a_stage_write(frozen: Study) -> None:
    from src.db_setup import build_natural_seed, create_study_tables

    full = sqlite3.connect(":memory:")
    create_study_tables(full)
    assert build_natural_seed(full, frozen.seed_db, 10, 1, unfreeze=True) == {}
    assert frozen.seed_db.read_bytes() != SENTINEL


def test_force_never_overwrites_a_frozen_toml(frozen: Study, studies_dir: Path, tmp_path: Path) -> None:
    toml = studies_dir / "mini.toml"
    before = toml.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        _new("other", toml, tmp_path / "data2", "--force")
    assert toml.read_bytes() == before


def test_new_never_reuses_frozen_paths_even_with_force(frozen: Study, tmp_path: Path) -> None:
    out = tmp_path / "elsewhere" / "mini.toml"
    with pytest.raises(ValueError, match="frozen study mini"):
        _new("mini", out, tmp_path / "data", "--force")
    assert not out.exists()


def test_new_refuses_the_shipped_market_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``--name market`` derives data/market_seed.db, the shipped seed."""
    out = tmp_path / "market.toml"
    with pytest.raises(SystemExit) as exc:
        study_cli.main([
            "new", "--name", "market", "--apps", "11", "--cohort", "c", "--names", "A",
            "--genre", "G", "--out", str(out), "--force",
        ])
    assert exc.value.code == 2
    assert "frozen study strategy_market_2026" in capsys.readouterr().err
    assert not out.exists()


def test_new_refuses_other_study_paths_and_existing_files_without_force(
    studies_dir: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    _new("a", studies_dir / "a.toml", data)
    with pytest.raises(ValueError, match="belongs to study a"):
        _new("a", studies_dir / "b.toml", data)
    assert load_study(_new("a", studies_dir / "b.toml", data, "--force")).name == "a"

    data.mkdir()
    (data / "c_seed.db").write_bytes(b"x")
    with pytest.raises(ValueError, match="already exists"):
        _new("c", studies_dir / "c.toml", data)
    assert _new("c", studies_dir / "c.toml", data, "--force").exists()


def test_add_refuses_a_frozen_study(frozen: Study, studies_dir: Path) -> None:
    toml = studies_dir / "mini.toml"
    before = toml.read_bytes()
    args = study_cli.build_parser().parse_args([
        "add", "--study", str(toml), "--apps", "33", "--cohort", "d", "--names", "C", "--genre", "G",
    ])
    with pytest.raises(ValueError, match="frozen"):
        study_cli.add_apps(args)
    assert toml.read_bytes() == before
