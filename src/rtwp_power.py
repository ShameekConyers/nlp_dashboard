"""Phase B power for the rtwp_turn_2026 study, from the development half only.

Phase A (``docs/rtwp_turn_study_design.md``) computed every outcome statistic
on the development half of the hash split. Its candidate findings
(``stats.py --candidates``, which assumes every aspect passes its audit) are
the hypotheses phase B tests on the test half (``docs/rtwp_turn_protocol.md``).
This module estimates how many of them phase B can be expected to meet:

- The phase B frame is every test-half review of the study's full database
  except the review IDs of every aspect audit export the study names (each
  version in ``[aspect_labels]``, the lexicon's own and the version it
  supersedes). Frame sizes are counted from review IDs alone. No test-half
  recommend flag, text or tag is read.
- The candidates come from the newest ``candidates_run_<id>.json`` whose
  lexicon version is the study's, and must list exactly ``CANDIDATES``.
- With ``--testable``, candidates of aspects that fail their audit are dropped
  first, by the merged precision of ``stats.study_precision`` (each aspect
  from the label file that governs it), so the threshold is computed on the
  testable candidates only. Every governing label file must exist.
- Each game's kept development-half reviews stand in for its truth. They come
  from the study's analysis database (``Study.analysis_view``): the
  development half of the full database for ``stats_source = "full"``.
  ``replication.simulate_power`` resamples them to the frame size, applies
  the eligibility rule to each synthetic sample and checks every candidate
  with the replication criterion (the 95% two-stage cluster-bootstrap lift
  interval excludes 1 on the candidate's side), once with the development
  lifts as they are and once with each game's log lift halved.
- Scopes list every game of the study in them, not only the games eligible on
  the development half, because phase B applies the eligibility rule to its
  own sample.

Results go to ``outputs/<study>/exploratory/power_<lexicon version>_cap<N>.json``,
with a ``_testable`` suffix under ``--testable`` (ignored by Git). The analysis and full databases are opened read-only.
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.replication import (
    POWER_BOOT,
    POWER_SEED,
    POWER_SIMULATIONS,
    SHRINK_SCENARIOS,
    Finding,
    SeedGame,
    simulate_power,
)
from src.stats import (
    _scopes,
    _split_ids,
    dev_test_split,
    load_aspect_sets,
    load_reviews,
    study_precision,
)
from src.study import Study, load_study
from src.utils import OUTPUTS_DIR, PROJECT_ROOT, get_logger

logger = get_logger(__name__)

STUDY_PATH: Path = PROJECT_ROOT / "studies" / "rtwp_turn_2026.toml"
"""The study this module serves."""

CANDIDATES_LEXICON: str = "rtwp_turn_v3"
"""Lexicon version of the candidate run ``CANDIDATES`` lists: run 4 of the
analysis database, the development half of the full database with lexicon
rtwp_turn_v3 and the full-dev BERTopic model."""


def _c(family: str, lens: str, theme: str, scope: str, direction: str = "complaint") -> Finding:
    """Build a candidate from a compact ``scope_type:scope_id`` string.

    Args:
        family: Theme family (the aspect name, or a short name for a topic).
        lens: ``aspect`` or ``topic``.
        theme: Aspect name or topic id.
        scope: ``scope_type:scope_id``.
        direction: ``complaint`` (lift above 1) or ``praise`` (lift below 1).

    Returns:
        The Finding.
    """
    scope_type, scope_id = scope.split(":", 1)
    return Finding(family, lens, theme, scope_type, scope_id, direction)


CANDIDATES: tuple[Finding, ...] = (
    _c("bugs_performance", "aspect", "bugs_performance", "cohort:crpg"),
    _c("bugs_performance", "aspect", "bugs_performance", "cohort:dual_mode"),
    _c("bugs_performance", "aspect", "bugs_performance", "cohort:strategy"),
    _c("bugs_performance", "aspect", "bugs_performance", "subgroup:rtwp"),
    _c("bugs_performance", "aspect", "bugs_performance", "subgroup:turn_based"),
    _c("bugs_performance", "aspect", "bugs_performance", "cohort_subgroup:crpg/rtwp"),
    _c("bugs_performance", "aspect", "bugs_performance", "cohort_subgroup:crpg/turn_based"),
    _c("bugs_performance", "aspect", "bugs_performance", "cohort_subgroup:strategy/rtwp"),
    _c("bugs_performance", "aspect", "bugs_performance", "cohort_subgroup:strategy/turn_based"),
    _c("combat_pacing", "aspect", "combat_pacing", "cohort:crpg"),
    _c("combat_pacing", "aspect", "combat_pacing", "subgroup:turn_based"),
    _c("combat_pacing", "aspect", "combat_pacing", "cohort_subgroup:crpg/turn_based"),
    _c("mode_turn_based", "aspect", "mode_turn_based", "subgroup:turn_based", "praise"),
    _c("mode_turn_based", "aspect", "mode_turn_based", "cohort_subgroup:strategy/turn_based", "praise"),
    _c("pause_micro", "aspect", "pause_micro", "cohort:dual_mode"),
    _c("pause_micro", "aspect", "pause_micro", "cohort:strategy"),
    _c("pause_micro", "aspect", "pause_micro", "subgroup:rtwp"),
    _c("pause_micro", "aspect", "pause_micro", "subgroup:turn_based"),
    _c("pause_micro", "aspect", "pause_micro", "cohort_subgroup:crpg/turn_based"),
    _c("pause_micro", "aspect", "pause_micro", "cohort_subgroup:strategy/rtwp"),
    _c("price_dlc", "aspect", "price_dlc", "cohort:dual_mode"),
    _c("price_dlc", "aspect", "price_dlc", "cohort:strategy"),
    _c("price_dlc", "aspect", "price_dlc", "subgroup:rtwp"),
    _c("price_dlc", "aspect", "price_dlc", "subgroup:turn_based"),
    _c("price_dlc", "aspect", "price_dlc", "cohort_subgroup:crpg/turn_based"),
    _c("price_dlc", "aspect", "price_dlc", "cohort_subgroup:strategy/rtwp"),
    _c("price_dlc", "aspect", "price_dlc", "cohort_subgroup:strategy/turn_based"),
    _c("realtime_chaos", "aspect", "realtime_chaos", "subgroup:turn_based"),
    _c("realtime_chaos", "aspect", "realtime_chaos", "cohort_subgroup:crpg/turn_based"),
    _c("ui_clarity", "aspect", "ui_clarity", "cohort:crpg"),
    _c("ui_clarity", "aspect", "ui_clarity", "cohort:dual_mode"),
    _c("ui_clarity", "aspect", "ui_clarity", "cohort:strategy"),
    _c("ui_clarity", "aspect", "ui_clarity", "subgroup:rtwp"),
    _c("ui_clarity", "aspect", "ui_clarity", "subgroup:turn_based"),
    _c("ui_clarity", "aspect", "ui_clarity", "cohort_subgroup:crpg/rtwp"),
    _c("ui_clarity", "aspect", "ui_clarity", "cohort_subgroup:crpg/turn_based"),
    _c("ui_clarity", "aspect", "ui_clarity", "cohort_subgroup:strategy/rtwp"),
    _c("ui_clarity", "aspect", "ui_clarity", "cohort_subgroup:strategy/turn_based"),
    _c("waiting_turns", "aspect", "waiting_turns", "subgroup:turn_based"),
    _c("dlc_topic", "topic", "1", "subgroup:turn_based"),
    _c("dlc_topic", "topic", "1", "cohort_subgroup:strategy/rtwp"),
    _c("party_combat_topic", "topic", "3", "cohort:crpg"),
    _c("party_combat_topic", "topic", "3", "cohort:dual_mode"),
    _c("party_combat_topic", "topic", "3", "cohort_subgroup:crpg/rtwp"),
    _c("party_combat_topic", "topic", "3", "cohort_subgroup:crpg/turn_based"),
    _c("crash_topic", "topic", "11", "cohort:crpg"),
    _c("crash_topic", "topic", "11", "cohort:dual_mode"),
    _c("crash_topic", "topic", "11", "cohort:strategy"),
    _c("crash_topic", "topic", "11", "subgroup:rtwp"),
    _c("crash_topic", "topic", "11", "subgroup:turn_based"),
    _c("crash_topic", "topic", "11", "cohort_subgroup:crpg/rtwp"),
    _c("crash_topic", "topic", "11", "cohort_subgroup:strategy/rtwp"),
    _c("crash_topic", "topic", "11", "cohort_subgroup:strategy/turn_based"),
    _c("ai_war_topic", "topic", "15", "cohort:strategy"),
    _c("ai_war_topic", "topic", "15", "subgroup:rtwp"),
    _c("ai_war_topic", "topic", "15", "cohort_subgroup:strategy/rtwp"),
    _c("ai_war_topic", "topic", "15", "cohort_subgroup:strategy/turn_based"),
    _c("multiplayer_topic", "topic", "19", "cohort:crpg"),
    _c("multiplayer_topic", "topic", "19", "cohort:strategy"),
    _c("multiplayer_topic", "topic", "19", "subgroup:rtwp"),
    _c("multiplayer_topic", "topic", "19", "subgroup:turn_based"),
    _c("multiplayer_topic", "topic", "19", "cohort_subgroup:crpg/rtwp"),
    _c("multiplayer_topic", "topic", "19", "cohort_subgroup:strategy/turn_based"),
    _c("paradox_dlc_topic", "topic", "38", "cohort:strategy"),
    _c("paradox_dlc_topic", "topic", "38", "subgroup:rtwp"),
    _c("paradox_dlc_topic", "topic", "38", "cohort_subgroup:strategy/rtwp"),
)
"""The 66 phase A candidates of run 4 (development half of the full database,
lexicon rtwp_turn_v3, every aspect assumed to pass its audit). Topic ids refer to the model in
``outputs/rtwp_turn_2026/bertopic_model_full_dev``: 1 DLC, 3 party and
enemies, 11 crashes, 15 AI and armies, 19 multiplayer and desync, 38 Paradox
DLC."""


def candidates_file(out_dir: Path, version: str) -> Path:
    """Find the newest candidate export computed with a lexicon version.

    Args:
        out_dir: Directory holding ``candidates_run_<id>.json`` exports.
        version: Lexicon version the run must have used.

    Returns:
        Path of the export with the highest run id among those whose
        ``params.lexicon_version`` is ``version``.

    Raises:
        FileNotFoundError: If no export used that version.
    """
    runs: list[tuple[int, Path]] = []
    for path in out_dir.glob("candidates_run_*.json"):
        m = re.fullmatch(r"candidates_run_(\d+)\.json", path.name)
        params = json.loads(path.read_text(encoding="utf-8")).get("params", {})
        if m and params.get("lexicon_version") == version:
            runs.append((int(m.group(1)), path))
    if not runs:
        raise FileNotFoundError(f"No candidate export in {out_dir} used lexicon {version}.")
    return max(runs)[1]


def check_candidates(path: Path, version: str) -> dict[str, dict[str, Any]]:
    """Read a candidate export and check it lists exactly ``CANDIDATES``.

    Args:
        path: ``candidates_run_<id>.json`` written by ``stats.py --candidates``.
        version: The study's lexicon version.

    Returns:
        ``{finding key: exported finding}``.

    Raises:
        ValueError: If the export used another lexicon version than the study,
            the study's lexicon is not the one ``CANDIDATES`` came from, the
            export is not a development-half candidate run, or its findings
            differ from ``CANDIDATES``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    used = data.get("params", {}).get("lexicon_version")
    if used != version:
        raise ValueError(f"{path} used lexicon {used}, the study uses {version}.")
    if version != CANDIDATES_LEXICON:
        raise ValueError(f"CANDIDATES come from lexicon {CANDIDATES_LEXICON}, the study uses "
                         f"{version}.")
    metrics = data["metrics"]
    if metrics.get("stats_split") != "dev" or not metrics.get("precision_assumed"):
        raise ValueError(f"{path} is not a development-half candidate run.")
    got = {
        f"{f['lens']}:{f['theme']}@{f['scope_type']}:{f['scope_id']}": f
        for f in metrics["findings"]
    }
    want = {f.key: f for f in CANDIDATES}
    if set(got) != set(want) or any(got[k]["finding"] != want[k].direction for k in want):
        raise ValueError("The candidate export's findings do not match CANDIDATES.")
    return got


def testable_candidates(
    study: Study, lexicon: Any, candidates: tuple[Finding, ...] = CANDIDATES
) -> tuple[tuple[Finding, ...], dict[str, dict[str, Any]]]:
    """Drop the candidates whose aspect fails its precision audit.

    Precision is the merged precision of ``stats.study_precision`` with every
    label file required: each aspect's from the label file that governs it.
    Topic candidates are always kept.

    Args:
        study: The study.
        lexicon: The study's ``aspects.Lexicon``.
        candidates: Candidates to filter.

    Returns:
        ``(kept candidates, precision by aspect)``.
    """
    precision = study_precision(study, lexicon, required=True)
    failing = {a for a, p in precision.items() if not p["passes"]}
    kept = tuple(f for f in candidates if not (f.lens == "aspect" and f.theme in failing))
    return kept, precision


def theme_keys(candidates: tuple[Finding, ...] = CANDIDATES) -> list[tuple[str, str]]:
    """Return the distinct ``(lens, theme)`` pairs the candidates use.

    Args:
        candidates: Candidates.

    Returns:
        Theme keys in candidate order.
    """
    return list(dict.fromkeys((f.lens, f.theme) for f in candidates))


def _read_only(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite file read-only.

    Args:
        path: Database path.

    Returns:
        Connection.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def dev_games(study: Study, keys: list[tuple[str, str]]) -> dict[int, SeedGame]:
    """Load each game's kept development-half reviews with theme membership.

    Args:
        study: The study (read through its analysis view).
        keys: ``(lens, theme)`` columns.

    Returns:
        ``{app_id: SeedGame}`` with ``n_docs`` the development-half document
        count, so the kept rate is development over development.
    """
    from src.aspects import load_lexicon

    assert study.aspect_lexicon is not None
    version = load_lexicon(study.aspect_lexicon).version
    conn = _read_only(study.analysis_view().seed_db)
    try:
        _split_ids(conn, "dev")
        rows = load_reviews(conn, in_split_only=True)
        tags = load_aspect_sets(conn, version)
        n_docs = dict(conn.execute(
            "SELECT d.app_id, COUNT(*) FROM documents d "
            "WHERE d.id IN (SELECT document_id FROM temp.stats_split_ids) GROUP BY d.app_id"
        ))
    finally:
        conn.close()
    by_app: dict[int, list] = {}
    for r in rows:
        by_app.setdefault(r.app_id, []).append(r)
    out: dict[int, SeedGame] = {}
    for app_id, rs in by_app.items():
        m = np.zeros((len(rs), len(keys)), dtype=bool)
        for i, r in enumerate(rs):
            doc_tags = tags.get(r.document_id, set())
            for j, (lens, theme) in enumerate(keys):
                m[i, j] = theme in doc_tags if lens == "aspect" else str(r.topic_id) == theme
        out[app_id] = SeedGame(app_id, int(n_docs[app_id]),
                               np.array([r.recommended for r in rs]), m)
    return out


def audit_versions(study: Study, lexicon: Any) -> list[str]:
    """Return every lexicon version whose audit set the study names.

    Args:
        study: The study.
        lexicon: The study's ``aspects.Lexicon``.

    Returns:
        The versions in ``[aspect_labels]``, the lexicon's own and the version
        it supersedes, without repeats.
    """
    versions = [v for _, v in study.aspect_labels] + [lexicon.version]
    if lexicon.supersedes:
        versions.append(lexicon.supersedes)
    return list(dict.fromkeys(versions))


def frame_exclusions(
    study: Study, lexicon: Any, audit_paths: list[Path] | None = None
) -> dict[str, set[str]]:
    """Collect the review IDs of every audit set the phase B frame excludes.

    Only the ``steam_review_id`` column of each export is read.

    Args:
        study: The study.
        lexicon: The study's ``aspects.Lexicon``.
        audit_paths: Audit exports to exclude (default the export of every
            version ``audit_versions`` names).

    Returns:
        ``{export file name: review IDs}``.

    Raises:
        ValueError: If ``audit_paths`` leaves out an audit set the study names.
        FileNotFoundError: If an export is missing.
    """
    from src.aspects import audit_export_path, audit_review_ids

    required = [audit_export_path(study, v) for v in audit_versions(study, lexicon)]
    paths = required if audit_paths is None else list(audit_paths)
    given = {p.resolve() for p in paths}
    missing = [p.name for p in required if p.resolve() not in given]
    if missing:
        raise ValueError(f"The frame does not exclude every audit set the study names: {missing}.")
    out: dict[str, set[str]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        out[path.name] = audit_review_ids(path)
    return out


def frame_sizes(study: Study, exclude: set[str], cap: int | None = None) -> dict[int, int]:
    """Count each game's phase B frame from review IDs alone.

    Args:
        study: The study.
        exclude: Review IDs kept out of phase B (the audit sentences' reviews).
        cap: Per-game cap on the sample drawn from the frame (None for none).

    Returns:
        ``{app_id: test-half reviews in the full database not in exclude}``,
        capped when ``cap`` is set.
    """
    conn = _read_only(study.full_db)
    try:
        counts: dict[int, int] = {}
        for app_id, rid in conn.execute("SELECT app_id, steam_review_id FROM documents"):
            if rid not in exclude and dev_test_split(rid) == "test":
                counts[app_id] = counts.get(app_id, 0) + 1
    finally:
        conn.close()
    return {a: (min(n, cap) if cap else n) for a, n in counts.items()}


def scope_games(
    study: Study, candidates: tuple[Finding, ...] = CANDIDATES
) -> dict[str, list[int]]:
    """Map each candidate to every study game in its scope.

    Args:
        study: The study.
        candidates: Candidates.

    Returns:
        ``{finding key: app IDs}``.
    """
    all_ids = {a.app_id for a in study.apps}
    scopes = {(t, s): ids for t, s, ids, _ in _scopes(study, all_ids, all_ids)}
    return {f.key: list(scopes[(f.scope_type, f.scope_id)]) for f in candidates}


def run(
    study_path: Path = STUDY_PATH,
    cap: int | None = None,
    n_sim: int = POWER_SIMULATIONS,
    n_boot: int = POWER_BOOT,
    out_dir: Path | None = None,
    audit_paths: list[Path] | None = None,
    testable_only: bool = False,
) -> Path:
    """Simulate phase B power for the candidates and write the result.

    Args:
        study_path: Study TOML.
        cap: Per-game cap on the phase B sample (None for the whole frame).
        n_sim: Synthetic samples per scenario.
        n_boot: Bootstrap resamples per check.
        out_dir: Output directory (default ``outputs/<study>/exploratory``).
        audit_paths: Audit exports whose review IDs are excluded (default
            every audit set the study names, see ``frame_exclusions``).
        testable_only: Drop candidates whose aspect fails its audit first.

    Returns:
        Path of the JSON written.
    """
    study = load_study(study_path)
    from src.aspects import load_lexicon

    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    out_dir = out_dir or OUTPUTS_DIR / study.name / "exploratory"
    source = candidates_file(out_dir, lexicon.version)
    exported = check_candidates(source, lexicon.version)
    exclusions = frame_exclusions(study, lexicon, audit_paths)
    candidates, precision = CANDIDATES, {}
    if testable_only:
        candidates, precision = testable_candidates(study, lexicon)
    excluded = set().union(*exclusions.values())
    draws = frame_sizes(study, excluded, cap)
    keys = theme_keys(candidates)
    games = dev_games(study, keys)
    scopes = scope_games(study, candidates)
    scenarios = {
        name: simulate_power(games, draws, scopes, candidates, keys, shrink,
                             n_sim=n_sim, n_boot=n_boot, seed=POWER_SEED)
        for name, shrink in SHRINK_SCENARIOS.items()
    }
    result = {
        "study": study.name,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"development half of {study.analysis_view().seed_db.name}",
        "lexicon_version": lexicon.version,
        "candidates_file": source.name,
        "testable_only": testable_only,
        "aspect_precision": {
            a: {k: p[k] for k in ("n", "relevant", "precision", "passes", "label_version")
                if k in p}
            for a, p in precision.items()
        },
        "dropped_candidates": [f.key for f in CANDIDATES if f not in candidates],
        "cap_per_game": cap,
        "excluded_audit_sets": {name: len(ids) for name, ids in exclusions.items()},
        "excluded_review_ids": len(excluded),
        "frame": {str(a): n for a, n in sorted(draws.items())},
        "frame_total": int(sum(draws.values())),
        "candidates": {
            f.key: {"family": f.family, "dev_lift": exported[f.key]["lift"],
                    "dev_lift_ci": exported[f.key]["lift_ci"], "games": scopes[f.key]}
            for f in candidates
        },
        "scenarios": scenarios,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_testable" if testable_only else ""
    path = out_dir / f"power_{lexicon.version}_cap{cap or 0}{suffix}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Phase B power from the development half.")
    parser.add_argument("--study", type=Path, default=STUDY_PATH)
    parser.add_argument("--cap", type=int, default=0,
                        help="Per-game cap on the phase B sample (0 for the whole frame).")
    parser.add_argument("--n-sim", type=int, default=POWER_SIMULATIONS)
    parser.add_argument("--n-boot", type=int, default=POWER_BOOT)
    parser.add_argument("--testable", action="store_true",
                        help="Drop candidates whose aspect fails its audit (every governing "
                             "label file must exist).")
    args = parser.parse_args()
    try:
        path = run(args.study, cap=args.cap or None, n_sim=args.n_sim, n_boot=args.n_boot,
                   testable_only=args.testable)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    result = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({
        "frame_total": result["frame_total"],
        "candidates": len(result["candidates"]),
        "dropped_candidates": len(result["dropped_candidates"]),
        **{name: {k: s[k] for k in ("count_mean", "count_p05", "count_p50", "count_p95")}
           for name, s in result["scenarios"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
