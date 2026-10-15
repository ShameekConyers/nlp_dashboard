"""Confirmatory replication of the market study's 24 lift findings.

The protocol is ``docs/replication_protocol.md``. It is committed and tagged
``PROTOCOL_TAG`` before the one confirmatory read, so anyone can check the
order in git. This module implements it:

- ``draw`` builds the sampling frame from ``data/market_full.db`` (every
  review not in the seed and not in any audit file), draws up to
  ``SAMPLE_CAP_PER_GAME`` reviews per game with a recorded seed and writes the
  review IDs and texts, never the recommend flag, to ``REPLICATION_DIR``.
- ``calibrate`` computes the numbers the protocol cites from non-replication
  data only: frame sizes, the games that cannot be resampled, a check of the
  fast bootstrap against the seed's stored intervals, the power simulation
  and the agreement between the saved BERTopic model's ``transform`` and its
  fitted assignments on the seed.
- ``dry-run`` exercises the whole read path on a development-split slice of
  the seed and never touches the replication sample's outcomes.
- ``read`` is the confirmatory read. It refuses to run unless the working
  tree is clean, HEAD carries ``PROTOCOL_TAG``, the sample, lexicon and
  BERTopic model match the pinned hashes, and no confirmatory result exists.

Every write goes under ``REPLICATION_DIR`` (``outputs/``, ignored by Git).
The seed, the full database, the lexicon and the saved model are opened
read-only and never changed.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.stats import (
    BOOT_SEED,
    MIN_GAMES_FOR_FINDING,
    MIN_KEPT_FOR_ELIGIBLE,
    MIN_NOT_REC_FOR_ELIGIBLE,
    N_BOOT,
    GameData,
    _scopes,
    classify_finding,
    cluster_bootstrap_macro,
    dev_test_split,
    leave_one_game_out,
    load_aspect_sets,
    load_audit_precision,
    load_reviews,
    macro_theme_metrics,
)
from src.study import STUDIES_DIR, Study, guard_writable, load_study
from src.utils import DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants (pinned by docs/replication_protocol.md)
# ---------------------------------------------------------------------------

PROTOCOL_TAG: str = "replication-protocol-v1"
"""Git tag that must point at HEAD before the confirmatory read."""

PROTOCOL_PATH: Path = PROJECT_ROOT / "docs" / "replication_protocol.md"
"""The committed protocol."""

MARKET_STUDY_PATH: Path = STUDIES_DIR / "strategy_market_2026.toml"
"""The study whose findings are replicated."""

REPLICATION_DIR: Path = OUTPUTS_DIR / "replication_2026"
"""Every file this module writes lives under this directory."""

SAMPLE_DB_NAME: str = "sample.db"
SAMPLE_MANIFEST_NAME: str = "sample_manifest.json"
CALIBRATION_NAME: str = "calibration.json"
DRY_RUN_SUBDIR: str = "dry_run"
CONFIRMATORY_SUBDIR: str = "confirmatory"

SAMPLE_CAP_PER_GAME: int = 2000
"""Maximum fresh reviews drawn per game."""

SAMPLE_SEED: int = 20261015
"""Seed for the per-game draw (``random.Random(f"{SAMPLE_SEED}:{app_id}")``)."""

SEED_STATS_RUN_ID: int = 10
"""The seed's ``analysis_runs`` stats run that produced the 24 findings."""

EXPECTED_SAMPLE_SHA256: str | None = (
    "5de0fbfeb7e4d5470ac18ab904cd24ad91585aa160a0ad4cfc532add5c2a480e"
)
"""SHA-256 of the drawn sample's sorted ``app_id<TAB>steam_review_id`` lines."""

EXPECTED_MODEL_SHA256: str | None = (
    "0692f55a0ea318b12f224fe6021047c20a9229fd8cc6c99b2e6379fbe402f417"
)
"""SHA-256 of ``outputs/strategy_market_2026/bertopic_model/model.pkl``."""

EXPECTED_LEXICON_SHA256: str | None = (
    "5ce3820e6189b11c498407d02fc778799649cc20c3f09ef57ad35a4fc656ff3a"
)
"""SHA-256 of ``studies/aspects_strategy_v1.toml``."""

POWER_SIMULATIONS: int = 500
"""Synthetic replications in the power simulation."""

POWER_BOOT: int = 1000
"""Bootstrap resamples per synthetic replication."""

POWER_SEED: int = 20261016
"""Seed for the power simulation."""

SHRINK_SCENARIOS: dict[str, float] = {"plug_in": 0.0, "half_log_lift": 0.5}
"""Power scenarios: the share of each game's log lift removed before simulating."""

COUNT_THRESHOLD: int | None = 12
"""Study-level threshold: replicated findings at or above it are consistent with power.

Set to the 5th percentile of the replicated count under ``half_log_lift``."""

EXPECTED_COUNT_PLUG_IN: float | None = 21.8
"""Mean replicated count in the ``plug_in`` power scenario."""

EXPECTED_COUNT_HALF: float | None = 14.8
"""Mean replicated count in the ``half_log_lift`` power scenario."""

DRY_RUN_PER_GAME: int = 150
"""Development-split seed reviews per game in the dry run."""

DRY_RUN_MIN_KEPT: int = 40
"""Dry-run eligibility (kept reviews); the read uses ``MIN_KEPT_FOR_ELIGIBLE``."""

DRY_RUN_MIN_NOT_REC: int = 8
"""Dry-run eligibility (not-recommended reviews); the read uses ``MIN_NOT_REC_FOR_ELIGIBLE``."""

DRY_RUN_BOOT: int = 200
"""Bootstrap resamples in the dry run."""


@dataclass(frozen=True)
class Finding:
    """One of the 24 original lift findings.

    Attributes:
        family: Theme family name (one of ``FAMILIES``).
        lens: ``aspect`` or ``topic``.
        theme: Aspect name or topic id (as a string).
        scope_type: ``cohort``, ``subgroup`` or ``subgroup_matched``.
        scope_id: Cohort or subgroup name.
        direction: ``complaint`` (lift above 1) or ``praise`` (lift below 1).
    """

    family: str
    lens: str
    theme: str
    scope_type: str
    scope_id: str
    direction: str

    @property
    def key(self) -> str:
        """Return a stable identifier such as ``aspect:pacing@subgroup:wego``."""
        return f"{self.lens}:{self.theme}@{self.scope_type}:{self.scope_id}"


FAMILIES: dict[str, str] = {
    "crash_topic": "Crash topic (topic 11, Crashes / Crash / Crashing / Screen)",
    "fft_topic": "Final Fantasy Tactics topic (topic 1, praise)",
    "bugs_performance": "Bugs and performance (aspect)",
    "pacing": "Pacing, grind and late game (aspect)",
    "ui_clarity": "UI and interface (aspect)",
    "price_dlc": "Price and DLC (aspect)",
    "ai_quality": "Enemy AI quality (aspect)",
    "preview_reliability": "Combat preview and hit chance (aspect)",
}
"""The eight theme families the 24 findings fall into."""


def _f(family: str, lens: str, theme: str, scope: str, direction: str = "complaint") -> Finding:
    """Build a Finding from a compact ``scope_type:scope_id`` string.

    Args:
        family: Family name.
        lens: ``aspect`` or ``topic``.
        theme: Theme name or topic id.
        scope: ``scope_type:scope_id``.
        direction: ``complaint`` or ``praise``.

    Returns:
        The Finding.
    """
    scope_type, scope_id = scope.split(":")
    return Finding(family, lens, theme, scope_type, scope_id, direction)


FINDINGS: tuple[Finding, ...] = (
    _f("crash_topic", "topic", "11", "cohort:turn_structure"),
    _f("crash_topic", "topic", "11", "subgroup:igo_ugo"),
    _f("crash_topic", "topic", "11", "subgroup:wego"),
    _f("fft_topic", "topic", "1", "subgroup_matched:alternating", "praise"),
    _f("bugs_performance", "aspect", "bugs_performance", "cohort:turn_structure"),
    _f("bugs_performance", "aspect", "bugs_performance", "subgroup:igo_ugo"),
    _f("bugs_performance", "aspect", "bugs_performance", "subgroup:alternating"),
    _f("bugs_performance", "aspect", "bugs_performance", "subgroup_matched:alternating"),
    _f("bugs_performance", "aspect", "bugs_performance", "cohort:anchor"),
    _f("pacing", "aspect", "pacing", "cohort:turn_structure"),
    _f("pacing", "aspect", "pacing", "subgroup:igo_ugo"),
    _f("pacing", "aspect", "pacing", "subgroup:alternating"),
    _f("pacing", "aspect", "pacing", "subgroup:wego"),
    _f("ui_clarity", "aspect", "ui_clarity", "cohort:turn_structure"),
    _f("ui_clarity", "aspect", "ui_clarity", "subgroup:igo_ugo"),
    _f("ui_clarity", "aspect", "ui_clarity", "subgroup_matched:igo_ugo"),
    _f("ui_clarity", "aspect", "ui_clarity", "subgroup:alternating"),
    _f("ui_clarity", "aspect", "ui_clarity", "subgroup:wego"),
    _f("price_dlc", "aspect", "price_dlc", "cohort:turn_structure"),
    _f("price_dlc", "aspect", "price_dlc", "subgroup:wego"),
    _f("price_dlc", "aspect", "price_dlc", "cohort:anchor"),
    _f("ai_quality", "aspect", "ai_quality", "cohort:turn_structure"),
    _f("ai_quality", "aspect", "ai_quality", "subgroup:alternating"),
    _f("preview_reliability", "aspect", "preview_reliability", "subgroup:alternating"),
)
"""The 24 findings of seed stats run ``SEED_STATS_RUN_ID``, grouped by family."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_only(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite file read-only.

    Args:
        path: Database file.

    Returns:
        A read-only connection.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_out(path: Path, root: Path | None = None) -> Path:
    """Return ``path`` after checking it lies under ``root`` and no frozen study owns it.

    Args:
        path: A path this module is about to write.
        root: The directory every write must stay under.

    Returns:
        The resolved path.

    Raises:
        ValueError: If the path escapes ``root``.
    """
    root = REPLICATION_DIR if root is None else root
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Refusing to write {resolved}: outside {root}.")
    guard_writable([resolved])
    return resolved


def _num(x: float) -> float | None:
    """Convert nan or inf to None for JSON.

    Args:
        x: A float.

    Returns:
        The float, or None when not finite.
    """
    return float(x) if x is not None and math.isfinite(x) else None


def _git(args: Sequence[str], repo: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command in ``repo``.

    Args:
        args: Arguments after ``git``.
        repo: Repository root.

    Returns:
        The completed process (never raises on a non-zero exit).
    """
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


# ---------------------------------------------------------------------------
# Sampling frame and draw
# ---------------------------------------------------------------------------


def audit_review_ids(labels_dir: Path = DATA_DIR / "labels") -> set[str]:
    """Return every Steam review ID that appears in a committed audit file.

    Covers the aspect precision audit (including its full-database rows) and
    the three sentence sentiment audits.

    Args:
        labels_dir: Directory of the committed label files.

    Returns:
        Steam review IDs.
    """
    ids: set[str] = set()
    for path in sorted(labels_dir.glob("*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "steam_review_id" in reader.fieldnames:
                ids.update(row["steam_review_id"] for row in reader)
    return ids


def excluded_review_ids(study: Study, labels_dir: Path = DATA_DIR / "labels") -> set[str]:
    """Return the reviews the replication frame excludes: the seed and every audit review.

    Args:
        study: The market study.
        labels_dir: Directory of the committed label files.

    Returns:
        Steam review IDs.
    """
    conn = _read_only(study.seed_db)
    try:
        seed_ids = {r[0] for r in conn.execute("SELECT steam_review_id FROM documents")}
    finally:
        conn.close()
    return seed_ids | audit_review_ids(labels_dir)


def build_frame(
    full_conn: sqlite3.Connection, app_ids: Sequence[int], excluded: set[str]
) -> dict[int, list[str]]:
    """List each game's fresh review IDs: in the full database, not excluded.

    Reads only review IDs and app IDs, never the recommend flag.

    Args:
        full_conn: Connection to the study's full database.
        app_ids: Games in the study.
        excluded: Review IDs to leave out.

    Returns:
        ``{app_id: sorted review IDs}``; a game with no fresh review maps to ``[]``.
    """
    frame: dict[int, list[str]] = {}
    for app_id in app_ids:
        ids = full_conn.execute(
            "SELECT steam_review_id FROM documents WHERE app_id = ?", (app_id,)
        ).fetchall()
        frame[app_id] = sorted(r[0] for r in ids if r[0] not in excluded)
    return frame


def draw_sample(
    frame: dict[int, list[str]], cap: int = SAMPLE_CAP_PER_GAME, seed: int = SAMPLE_SEED
) -> dict[int, list[str]]:
    """Draw up to ``cap`` reviews per game, uniformly and deterministically.

    Mirrors the seed draw (``db_setup.build_natural_seed``): sorted IDs, a
    per-game ``random.Random(f"{seed}:{app_id}")`` and ``sample``.

    Args:
        frame: ``{app_id: sorted fresh review IDs}``.
        cap: Maximum reviews per game.
        seed: Base seed.

    Returns:
        ``{app_id: sorted drawn IDs}``.
    """
    out: dict[int, list[str]] = {}
    for app_id, ids in frame.items():
        ordered = sorted(ids)
        rng = random.Random(f"{seed}:{app_id}")
        out[app_id] = sorted(rng.sample(ordered, min(len(ordered), cap)))
    return out


def sample_digest(sample: dict[int, list[str]]) -> str:
    """SHA-256 of the sorted ``app_id<TAB>steam_review_id`` lines of a sample.

    Args:
        sample: ``{app_id: review IDs}``.

    Returns:
        Hex digest, independent of dict and list order.
    """
    lines = sorted(f"{a}\t{rid}\n" for a, ids in sample.items() for rid in ids)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def write_sample(study: Study, out_dir: Path | None = None) -> dict[str, Any]:
    """Draw the replication sample and write its IDs and texts plus a manifest.

    The sample database has one table, ``sample``, with no recommend flag.
    A second call with an unchanged frame rewrites identical files. A call
    that would produce a different sample than an existing manifest refuses.

    Args:
        study: The market study.
        out_dir: Output directory under ``REPLICATION_DIR`` (default that directory).

    Returns:
        The manifest.

    Raises:
        ValueError: If an existing manifest records a different sample.
    """
    out_dir = _safe_out(REPLICATION_DIR if out_dir is None else out_dir)
    excluded = excluded_review_ids(study)
    full = _read_only(study.full_db)
    try:
        app_ids = [a.app_id for a in study.apps]
        frame = build_frame(full, app_ids, excluded)
        sample = draw_sample(frame)
        digest = sample_digest(sample)
        manifest_path = out_dir / SAMPLE_MANIFEST_NAME
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old.get("sample_sha256") != digest:
                raise ValueError(
                    f"{manifest_path} records sample {old.get('sample_sha256')}, this draw gives "
                    f"{digest}. Refusing to replace a drawn sample."
                )
        rows: list[tuple[str, int, int, str, int]] = []
        windows: dict[str, dict[str, Any]] = {}
        for app_id in app_ids:
            ids = sample[app_id]
            got = []
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                marks = ",".join("?" * len(chunk))
                got.extend(full.execute(
                    f"SELECT steam_review_id, app_id, timestamp_created, review_text "
                    f"FROM documents WHERE steam_review_id IN ({marks})",
                    chunk,
                ).fetchall())
            order = {rid: i for i, rid in enumerate(ids)}
            got.sort(key=lambda r: order[r[0]])
            rows.extend((r[0], r[1], r[2], r[3], order[r[0]]) for r in got)
            ts = [r[2] for r in got if r[2] is not None]
            windows[str(app_id)] = {
                "frame": len(frame[app_id]), "drawn": len(ids),
                "first_created": min(ts) if ts else None, "last_created": max(ts) if ts else None,
            }
    finally:
        full.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / SAMPLE_DB_NAME
    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(tmp)
    try:
        conn.execute(
            """
            CREATE TABLE sample (
                steam_review_id   TEXT PRIMARY KEY,
                app_id            INTEGER NOT NULL,
                timestamp_created INTEGER,
                review_text       TEXT NOT NULL,
                draw_order        INTEGER NOT NULL
            )
            """
        )
        conn.executemany("INSERT INTO sample VALUES (?, ?, ?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()
    tmp.replace(db_path)
    names = {a.app_id: a.name for a in study.apps}
    manifest = {
        "study": study.name,
        "full_db": str(study.full_db.relative_to(PROJECT_ROOT)),
        "cap_per_game": SAMPLE_CAP_PER_GAME,
        "sample_seed": SAMPLE_SEED,
        "draw_rule": "sorted fresh IDs, random.Random(f'{seed}:{app_id}').sample(ids, min(n, cap))",
        "n_excluded_ids": len(excluded),
        "n_sampled": len(rows),
        "sample_sha256": digest,
        "games": {k: {"name": names[int(k)], **v} for k, v in windows.items()},
        "not_resamplable": sorted(a for a in app_ids if not frame[a]),
        "drawn_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Drew %d reviews (sha256 %s) into %s", len(rows), digest, db_path)
    return manifest


def load_sample(out_dir: Path | None = None) -> tuple[list[tuple[str, int, str]], str]:
    """Read the drawn sample and recompute its digest.

    Args:
        out_dir: Directory holding ``sample.db`` (default ``REPLICATION_DIR``).

    Returns:
        ``([(steam_review_id, app_id, review_text), ...], digest)`` in draw order per game.
    """
    conn = _read_only((REPLICATION_DIR if out_dir is None else out_dir) / SAMPLE_DB_NAME)
    try:
        rows = conn.execute(
            "SELECT steam_review_id, app_id, review_text FROM sample ORDER BY app_id, draw_order"
        ).fetchall()
    finally:
        conn.close()
    by_app: dict[int, list[str]] = {}
    for rid, app_id, _ in rows:
        by_app.setdefault(app_id, []).append(rid)
    return [(r[0], int(r[1]), r[2]) for r in rows], sample_digest(by_app)


# ---------------------------------------------------------------------------
# Fast two-stage cluster bootstrap of one theme's macro lift
# ---------------------------------------------------------------------------


def four_cells(rec: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Count one game's reviews in the four theme by recommend cells.

    Args:
        rec: Recommend flag per review.
        member: Theme membership per review.

    Returns:
        ``[not & theme, not & no theme, rec & theme, rec & no theme]``.
    """
    rec = rec.astype(bool)
    member = member.astype(bool)
    return np.array([(~rec & member).sum(), (~rec & ~member).sum(),
                     (rec & member).sum(), (rec & ~member).sum()], dtype=np.int64)


def macro_lift(cells: np.ndarray) -> np.ndarray:
    """Macro complaint share over macro praise share, game-weighted.

    Equals ``stats.macro_theme_metrics(...).lift`` for one theme.

    Args:
        cells: ``(..., k_games, 4)`` counts from ``four_cells``.

    Returns:
        Lift over the leading axes (nan or inf where undefined).
    """
    cells = np.asarray(cells, dtype=float)
    n_not = cells[..., 0] + cells[..., 1]
    n_rec = cells[..., 2] + cells[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        complaint = cells[..., 0] / n_not
        praise = cells[..., 2] / n_rec
        c = np.nanmean(np.where(np.isfinite(complaint), complaint, np.nan), axis=-1)
        p = np.nanmean(np.where(np.isfinite(praise), praise, np.nan), axis=-1)
        return c / p


def cluster_lift_ci(
    cells: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Two-stage cluster bootstrap percentile interval of one theme's macro lift.

    Same resampling as ``stats.cluster_bootstrap_macro``: games with
    replacement, then reviews with replacement within each drawn game. A
    review resample only matters through the four cell counts, so it is a
    multinomial draw over the cells.

    Args:
        cells: ``(k_games, 4)`` counts.
        n_boot: Resamples.
        rng: Random generator.

    Returns:
        ``(low, high)`` 2.5th and 97.5th percentiles over finite resamples.
    """
    cells = np.asarray(cells, dtype=np.int64)
    k = cells.shape[0]
    n = cells.sum(axis=1)
    probs = cells / n[:, None]
    picks = rng.integers(0, k, size=(n_boot, k))
    draws = rng.multinomial(n[picks], probs[picks])
    lifts = macro_lift(draws)
    finite = lifts[np.isfinite(lifts)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


# ---------------------------------------------------------------------------
# Seed data for calibration (non-replication data only)
# ---------------------------------------------------------------------------


@dataclass
class SeedGame:
    """One game's kept seed reviews reduced to what the power simulation needs.

    Attributes:
        app_id: Steam application ID.
        n_docs: Seed documents (kept or not).
        rec: Recommend flag per kept review.
        themes: ``(n_kept, n_themes)`` membership over ``family_theme_keys()``.
    """

    app_id: int
    n_docs: int
    rec: np.ndarray
    themes: np.ndarray


def family_theme_keys() -> list[tuple[str, str]]:
    """Return the distinct ``(lens, theme)`` pairs the 24 findings use, in finding order.

    Returns:
        Theme keys.
    """
    return list(dict.fromkeys((f.lens, f.theme) for f in FINDINGS))


def load_seed_games(study: Study, theme_keys: Sequence[tuple[str, str]]) -> dict[int, SeedGame]:
    """Load the seed's kept reviews with membership in the given themes.

    Args:
        study: The market study.
        theme_keys: ``(lens, theme)`` pairs.

    Returns:
        ``{app_id: SeedGame}``.
    """
    conn = _read_only(study.seed_db)
    try:
        rows = load_reviews(conn)
        aspects = load_aspect_sets(conn, _lexicon_version(study))
        n_docs = dict(conn.execute("SELECT app_id, COUNT(*) FROM documents GROUP BY app_id"))
    finally:
        conn.close()
    by_app: dict[int, list] = {}
    for r in rows:
        by_app.setdefault(r.app_id, []).append(r)
    out: dict[int, SeedGame] = {}
    for app_id, rs in by_app.items():
        m = np.zeros((len(rs), len(theme_keys)), dtype=bool)
        for i, r in enumerate(rs):
            tags = aspects.get(r.document_id, set())
            for j, (lens, theme) in enumerate(theme_keys):
                m[i, j] = theme in tags if lens == "aspect" else str(r.topic_id) == theme
        out[app_id] = SeedGame(app_id, int(n_docs[app_id]),
                               np.array([r.recommended for r in rs]), m)
    return out


def _lexicon_version(study: Study) -> str:
    """Return the version of a study's aspect lexicon.

    Args:
        study: The study.

    Returns:
        Lexicon version, e.g. ``strategy_v1``.
    """
    from src.aspects import load_lexicon

    assert study.aspect_lexicon is not None
    return load_lexicon(study.aspect_lexicon).version


def load_original(study: Study, run_id: int = SEED_STATS_RUN_ID) -> dict[str, Any]:
    """Read the original findings and eligible games from the seed's stats run.

    Args:
        study: The market study.
        run_id: The stats run.

    Returns:
        ``{"eligible": set of app IDs, "findings": {key: {lift, lift_ci, finding}}}``.

    Raises:
        ValueError: If the run's findings differ from ``FINDINGS``.
    """
    conn = _read_only(study.seed_db)
    try:
        row = conn.execute(
            "SELECT metrics FROM analysis_runs WHERE run_id = ? AND stage = 'stats'", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"No stats run {run_id} in {study.seed_db}.")
    metrics = json.loads(row[0])
    got = {
        f"{f['lens']}:{f['theme']}@{f['scope_type']}:{f['scope_id']}": f
        for f in metrics["findings"]
    }
    want = {f.key: f for f in FINDINGS}
    if set(got) != set(want) or any(got[k]["finding"] != want[k].direction for k in want):
        raise ValueError("The seed stats run's findings do not match FINDINGS.")
    return {"eligible": set(metrics["eligible_app_ids"]), "findings": got}


def scope_app_ids(study: Study, finding: Finding, eligible: set[int]) -> list[int]:
    """Return the original eligible games of a finding's scope.

    Args:
        study: The market study.
        finding: The finding.
        eligible: The seed's eligible app IDs.

    Returns:
        App IDs in study order.
    """
    for scope_type, scope_id, _ids, elig in _scopes(study, eligible, eligible):
        if scope_type == finding.scope_type and scope_id == finding.scope_id:
            return list(elig)
    raise KeyError(finding.key)


# ---------------------------------------------------------------------------
# Power simulation
# ---------------------------------------------------------------------------


def shrink_members(
    rec: np.ndarray, member: np.ndarray, shrink: float, rng: np.random.Generator
) -> np.ndarray:
    """Thin one game's theme flags so its log lift moves ``shrink`` of the way to 0.

    With ``lift = complaint share / praise share`` above 1, not-recommended
    flags are kept with probability ``lift ** -shrink``; below 1,
    recommended flags are kept with probability ``lift ** shrink``. The
    game's expected lift becomes ``lift ** (1 - shrink)``.

    Args:
        rec: Recommend flags.
        member: Theme flags.
        shrink: Share of the log lift to remove (0 keeps the data as is).
        rng: Random generator.

    Returns:
        New theme flags.
    """
    if shrink <= 0:
        return member
    cells = four_cells(rec, member)
    n_not, n_rec = cells[0] + cells[1], cells[2] + cells[3]
    if min(cells[0], cells[2], n_not, n_rec) == 0:
        return member
    lift = (cells[0] / n_not) / (cells[2] / n_rec)
    out = member.copy()
    if lift > 1:
        side = ~rec.astype(bool) & member
        keep = lift ** -shrink
    else:
        side = rec.astype(bool) & member
        keep = lift ** shrink
    out[side] = rng.random(int(side.sum())) < keep
    return out


def simulate_power(
    seed_games: dict[int, SeedGame],
    draws: dict[int, int],
    scope_games: dict[str, list[int]],
    findings: Sequence[Finding],
    theme_keys: Sequence[tuple[str, str]],
    shrink: float,
    n_sim: int = POWER_SIMULATIONS,
    n_boot: int = POWER_BOOT,
    seed: int = POWER_SEED,
    min_kept: int = MIN_KEPT_FOR_ELIGIBLE,
    min_not: int = MIN_NOT_REC_FOR_ELIGIBLE,
) -> dict[str, Any]:
    """Simulate replications from the seed and count how many findings meet the criterion.

    Each synthetic replication draws, for every game, a kept count from
    ``Binomial(drawn, seed kept rate)`` and that many kept seed reviews with
    replacement, optionally shrinks each theme's per-game effect, applies the
    replication eligibility rule, and checks every finding with the primary
    criterion (direction match and a 95% cluster-bootstrap interval that
    excludes 1 on the original side). All findings share one synthetic
    replication, so the count distribution keeps their correlation.

    Args:
        seed_games: Seed data per game.
        draws: Reviews the replication draws per game.
        scope_games: ``{finding key: original eligible app IDs}``.
        findings: Findings to check.
        theme_keys: Columns of ``SeedGame.themes``.
        shrink: Share of each game's log lift removed before simulating.
        n_sim: Synthetic replications.
        n_boot: Bootstrap resamples per check.
        seed: RNG seed.
        min_kept: Eligibility, kept reviews.
        min_not: Eligibility, not-recommended reviews.

    Returns:
        Per-finding power and testable share, and the replicated count's mean
        and 5th, 50th and 95th percentiles.
    """
    rng = np.random.default_rng(seed)
    col = {k: j for j, k in enumerate(theme_keys)}
    success = {f.key: 0 for f in findings}
    testable = {f.key: 0 for f in findings}
    counts: list[int] = []
    for _ in range(n_sim):
        synth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for app_id, g in seed_games.items():
            n_draw = draws.get(app_id, 0)
            if n_draw == 0 or g.rec.size == 0:
                continue
            n_kept = int(rng.binomial(n_draw, g.rec.size / g.n_docs))
            idx = rng.integers(0, g.rec.size, size=n_kept)
            rec = g.rec[idx]
            themes = g.themes[idx]
            if shrink > 0:
                themes = np.column_stack([
                    shrink_members(rec, themes[:, j], shrink, rng) for j in range(themes.shape[1])
                ])
            if rec.size >= min_kept and (~rec).sum() >= min_not:
                synth[app_id] = (rec, themes)
        replicated = 0
        for f in findings:
            ids = [a for a in scope_games[f.key] if a in synth]
            if len(ids) < MIN_GAMES_FOR_FINDING:
                continue
            testable[f.key] += 1
            j = col[(f.lens, f.theme)]
            cells = np.array([four_cells(synth[a][0], synth[a][1][:, j]) for a in ids])
            point = float(macro_lift(cells))
            lo, hi = cluster_lift_ci(cells, n_boot, rng)
            if meets_criterion(f.direction, point, lo, hi):
                success[f.key] += 1
                replicated += 1
        counts.append(replicated)
    arr = np.array(counts)
    return {
        "shrink": shrink,
        "n_sim": n_sim,
        "n_boot": n_boot,
        "seed": seed,
        "per_finding": {
            k: {"power": success[k] / n_sim, "testable_share": testable[k] / n_sim}
            for k in success
        },
        "count_mean": float(arr.mean()),
        "count_p05": float(np.percentile(arr, 5)),
        "count_p50": float(np.percentile(arr, 50)),
        "count_p95": float(np.percentile(arr, 95)),
    }


def meets_criterion(direction: str, point: float, lo: float, hi: float) -> bool:
    """Primary replication criterion for one finding.

    The replication point lift lies on the original side of 1 and its 95%
    interval excludes 1 on that side.

    Args:
        direction: ``complaint`` or ``praise``.
        point: Replication point lift.
        lo: Interval lower bound.
        hi: Interval upper bound.

    Returns:
        True when the finding replicates.
    """
    if direction == "complaint":
        return math.isfinite(point) and point > 1 and math.isfinite(lo) and lo > 1
    return math.isfinite(point) and point < 1 and math.isfinite(hi) and hi < 1


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def topic_transform_agreement(study: Study, model_path: Path) -> dict[str, Any]:
    """Compare the saved model's ``transform`` with its fitted assignments on the seed.

    Args:
        study: The market study.
        model_path: Saved BERTopic model.

    Returns:
        For the whole training set at once (a reproduction check) and for
        two halves transformed separately (the path new reviews take):
        overall agreement, outlier shares and per-topic agreement for the
        topics that carry findings.
    """
    from bertopic import BERTopic

    from src.utils import MIN_TOKEN_COUNT

    conn = _read_only(study.seed_db)
    try:
        rows = conn.execute(
            """
            SELECT p.cleaned_text, COALESCE(n.topic_id, -1)
            FROM processed_documents p JOIN nlp_results n ON n.document_id = p.document_id
            WHERE p.token_count >= ? ORDER BY p.document_id
            """,
            (MIN_TOKEN_COUNT,),
        ).fetchall()
    finally:
        conn.close()
    model = BERTopic.load(str(model_path))
    texts = [r[0] for r in rows]
    fitted = np.array([r[1] for r in rows])
    # The whole training set hashes to UMAP's stored input, so umap-learn
    # returns the fitted embedding: a reproduction check only. Two halves
    # (even and odd positions) hash differently and force the new-data path.
    whole = np.array(model.transform(texts)[0])
    halves = np.empty_like(fitted)
    for start in (0, 1):
        halves[start::2] = model.transform(texts[start::2])[0]

    def compare(transformed: np.ndarray) -> dict[str, Any]:
        """Agreement of one transform with the fitted assignments."""
        res: dict[str, Any] = {
            "agreement": float((fitted == transformed).mean()),
            "outlier_share_fit": float((fitted == -1).mean()),
            "outlier_share_transform": float((transformed == -1).mean()),
            "topics": {},
        }
        for topic in sorted({int(f.theme) for f in FINDINGS if f.lens == "topic"}):
            in_fit = fitted == topic
            in_tr = transformed == topic
            res["topics"][str(topic)] = {
                "n_fit": int(in_fit.sum()),
                "n_transform": int(in_tr.sum()),
                "fit_kept_by_transform": float((in_fit & in_tr).sum() / max(in_fit.sum(), 1)),
                "transform_in_fit": float((in_fit & in_tr).sum() / max(in_tr.sum(), 1)),
            }
        return res

    return {"n": int(fitted.size), "whole_set": compare(whole), "two_halves": compare(halves)}


def calibrate(
    study: Study,
    out_dir: Path | None = None,
    with_transform: bool = True,
    n_sim: int = POWER_SIMULATIONS,
) -> dict[str, Any]:
    """Compute every number the protocol cites, from non-replication data only.

    Reads the sample manifest (review counts, no outcomes), the seed and its
    stats run, and the saved model. Writes ``calibration.json``.

    Args:
        study: The market study.
        out_dir: Directory with the sample manifest; output goes here too.
        with_transform: Also measure the saved model's transform agreement
            on the seed (a few minutes).
        n_sim: Synthetic replications per power scenario.

    Returns:
        The calibration dictionary.
    """
    out_dir = _safe_out(REPLICATION_DIR if out_dir is None else out_dir)
    manifest = json.loads((out_dir / SAMPLE_MANIFEST_NAME).read_text())
    original = load_original(study)
    eligible = original["eligible"]
    keys = family_theme_keys()
    seed_games = load_seed_games(study, keys)
    draws = {int(k): v["drawn"] for k, v in manifest["games"].items()}
    scope_games = {f.key: scope_app_ids(study, f, eligible) for f in FINDINGS}
    resamplable = {a for a, n in draws.items() if n > 0}

    # Check the fast bootstrap against the seed's stored intervals.
    rng = np.random.default_rng(BOOT_SEED)
    col = {k: j for j, k in enumerate(keys)}
    fast_check: dict[str, Any] = {}
    for f in FINDINGS:
        cells = np.array([
            four_cells(seed_games[a].rec, seed_games[a].themes[:, col[(f.lens, f.theme)]])
            for a in scope_games[f.key]
        ])
        lo, hi = cluster_lift_ci(cells, N_BOOT, rng)
        o = original["findings"][f.key]
        fast_check[f.key] = {
            "stored_lift": o["lift"], "fast_lift": float(macro_lift(cells)),
            "stored_ci": o["lift_ci"], "fast_ci": [lo, hi],
        }

    projected: dict[str, Any] = {}
    for a in study.apps:
        g = seed_games.get(a.app_id)
        kept_rate = g.rec.size / g.n_docs if g else math.nan
        rec_share = float(g.rec.mean()) if g else math.nan
        n = draws.get(a.app_id, 0)
        projected[str(a.app_id)] = {
            "name": a.name, "seed_eligible": a.app_id in eligible, "drawn": n,
            "seed_kept_rate": _num(kept_rate), "seed_kept_rec_share": _num(rec_share),
            "projected_kept": _num(n * kept_rate), "projected_not_rec": _num(n * kept_rate * (1 - rec_share)),
        }
    per_finding_games = {
        f.key: {
            "original_games": len(scope_games[f.key]),
            "resamplable_games": sum(a in resamplable for a in scope_games[f.key]),
        }
        for f in FINDINGS
    }
    power = {
        name: simulate_power(seed_games, draws, scope_games, FINDINGS, keys, shrink, n_sim=n_sim)
        for name, shrink in SHRINK_SCENARIOS.items()
    }
    result: dict[str, Any] = {
        "study": study.name,
        "seed_stats_run_id": SEED_STATS_RUN_ID,
        "sample_sha256": manifest["sample_sha256"],
        "n_sampled": manifest["n_sampled"],
        "not_resamplable": [
            {"app_id": a, "name": study.app(a).name, "seed_eligible": a in eligible}
            for a in manifest["not_resamplable"]
        ],
        "projected": projected,
        "per_finding_games": per_finding_games,
        "fast_bootstrap_check": fast_check,
        "power": power,
        "model_sha256": file_sha256(_model_path(study)),
        "lexicon_sha256": file_sha256(study.aspect_lexicon) if study.aspect_lexicon else None,
    }
    if with_transform:
        result["topic_transform_agreement"] = topic_transform_agreement(study, _model_path(study))
    (out_dir / CALIBRATION_NAME).write_text(json.dumps(result, indent=2))
    return result


def _model_path(study: Study) -> Path:
    """Return the saved BERTopic model of a study.

    Args:
        study: The study.

    Returns:
        ``outputs/<study>/bertopic_model/model.pkl``.
    """
    return OUTPUTS_DIR / study.name / "bertopic_model" / "model.pkl"


# ---------------------------------------------------------------------------
# Processing: the frozen instruments applied to new reviews
# ---------------------------------------------------------------------------


@dataclass
class ProcessedReview:
    """One review after cleaning, the language filter, aspect tagging and topic transform.

    Attributes:
        steam_review_id: Steam recommendation ID.
        app_id: Steam application ID.
        kept: Passed the length, token and language rules.
        aspects: Aspects tagged in any sentence (kept reviews only).
        topic_id: Topic from the saved model's ``transform`` (-1 outlier or dropped).
    """

    steam_review_id: str
    app_id: int
    kept: bool
    aspects: frozenset[str]
    topic_id: int


def process_reviews(
    rows: Sequence[tuple[str, int, str]],
    clean_and_filter: Callable[[list[str]], list[dict]],
    tag: Callable[[list[str]], list[set[str]]],
    assign_topics: Callable[[list[str]], list[int]],
) -> list[ProcessedReview]:
    """Apply the frozen instruments to raw reviews.

    Args:
        rows: ``(steam_review_id, app_id, raw_text)``.
        clean_and_filter: Raw texts to ``Preprocessor.process_batch`` results
            with the language filter applied.
        tag: Cleaned texts to the set of aspects tagged in each.
        assign_topics: Cleaned texts to topic IDs.

    Returns:
        One ProcessedReview per input row, in input order.
    """
    results = clean_and_filter([r[2] for r in rows])
    kept_idx = [i for i, res in enumerate(results) if not res["dropped"]]
    texts = [results[i]["cleaned_text"] for i in kept_idx]
    aspect_sets = tag(texts) if texts else []
    topic_ids = assign_topics(texts) if texts else []
    out = [ProcessedReview(r[0], r[1], False, frozenset(), -1) for r in rows]
    for i, aspects, topic in zip(kept_idx, aspect_sets, topic_ids):
        out[i] = ProcessedReview(rows[i][0], rows[i][1], True, frozenset(aspects), int(topic))
    return out


def frozen_instruments(study: Study) -> tuple[
    Callable[[list[str]], list[dict]],
    Callable[[list[str]], list[set[str]]],
    Callable[[list[str]], list[int]],
]:
    """Build the real instruments: preprocessing, frozen lexicon, saved BERTopic transform.

    Args:
        study: The market study.

    Returns:
        ``(clean_and_filter, tag, assign_topics)`` for ``process_reviews``.
    """
    from bertopic import BERTopic

    from src.aspects import SentenceSplitter, load_lexicon, tag_sentences
    from src.preprocessing import LanguageFilter, Preprocessor, apply_language_filter

    assert study.aspect_lexicon is not None
    pre = Preprocessor()
    lang = LanguageFilter() if study.language_filter else None
    lexicon = load_lexicon(study.aspect_lexicon)
    splitter = SentenceSplitter()
    model = BERTopic.load(str(_model_path(study)))

    def clean_and_filter(raws: list[str]) -> list[dict]:
        """Clean, tokenize and language-filter raw texts."""
        results = pre.process_batch(raws)
        if lang is not None:
            apply_language_filter(results, lang)
        return results

    def tag(texts: list[str]) -> list[set[str]]:
        """Tag each cleaned text with the aspects of the frozen lexicon."""
        return [
            {m.aspect for m in tag_sentences([s[2] for s in spans], lexicon)}
            for spans in splitter.split(texts)
        ]

    def assign_topics(texts: list[str]) -> list[int]:
        """Assign topics with the saved model's transform (never refit)."""
        return [int(t) for t in model.transform(texts)[0]]

    return clean_and_filter, tag, assign_topics


def write_processed(processed: Sequence[ProcessedReview], path: Path) -> None:
    """Write per-review instrument outputs (no recommend flag) to a CSV.

    Args:
        processed: Processed reviews.
        path: Output CSV under ``REPLICATION_DIR``.
    """
    path = _safe_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["steam_review_id", "app_id", "kept", "aspects", "topic_id"])
        for p in processed:
            w.writerow([p.steam_review_id, p.app_id, int(p.kept), "|".join(sorted(p.aspects)),
                        p.topic_id])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def analysis_theme_keys(study: Study) -> tuple[list[tuple[str, str]], list[list[int]]]:
    """Theme columns and rank pools as in the original stats run.

    Columns are the aspects that passed the precision audit and every
    non-outlier topic of the seed. The rank pools for the leave-one-game-out
    check are one per lens, as ``stats.run`` builds them.

    Args:
        study: The market study.

    Returns:
        ``(theme_keys, rank_pools)``.
    """
    from src.aspects import audit_paths, load_lexicon

    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    precision = load_audit_precision(audit_paths(study, lexicon)[1])
    passing = [a.name for a in lexicon.aspects if precision.get(a.name, {}).get("passes")]
    conn = _read_only(study.seed_db)
    try:
        topics = [str(r[0]) for r in conn.execute(
            "SELECT topic_id FROM topics WHERE topic_id != -1 ORDER BY topic_id")]
    finally:
        conn.close()
    keys = [("aspect", a) for a in passing] + [("topic", t) for t in topics]
    pools = [[j for j, k in enumerate(keys) if k[0] == lens] for lens in ("aspect", "topic")]
    return keys, pools


def build_games(
    processed: Sequence[ProcessedReview],
    recommended: dict[str, bool],
    theme_keys: Sequence[tuple[str, str]],
) -> dict[int, GameData]:
    """Build per-game arrays from kept processed reviews and their recommend flags.

    Args:
        processed: Processed reviews.
        recommended: ``{steam_review_id: recommend flag}``.
        theme_keys: Theme columns.

    Returns:
        ``{app_id: GameData}`` (VADER is not part of the finding rule and is zero).
    """
    by_app: dict[int, list[ProcessedReview]] = {}
    for p in processed:
        if p.kept:
            by_app.setdefault(p.app_id, []).append(p)
    games: dict[int, GameData] = {}
    for app_id, ps in by_app.items():
        m = np.zeros((len(ps), len(theme_keys)), dtype=bool)
        for i, p in enumerate(ps):
            for j, (lens, theme) in enumerate(theme_keys):
                m[i, j] = theme in p.aspects if lens == "aspect" else str(p.topic_id) == theme
        rec = np.array([recommended[p.steam_review_id] for p in ps], dtype=bool)
        games[app_id] = GameData(app_id, rec, np.zeros(len(ps)), m)
    return games


def evaluate(
    games: dict[int, GameData],
    theme_keys: Sequence[tuple[str, str]],
    rank_pools: Sequence[Sequence[int]],
    scope_games: dict[str, list[int]],
    original: dict[str, Any],
    findings: Sequence[Finding] = FINDINGS,
    n_boot: int = N_BOOT,
    min_kept: int = MIN_KEPT_FOR_ELIGIBLE,
    min_not: int = MIN_NOT_REC_FOR_ELIGIBLE,
) -> list[dict[str, Any]]:
    """Check every finding on new data with the frozen statistics code.

    For each finding, the replication games are the original eligible games
    of its scope that have replication data and meet the eligibility rule
    in it. Intervals come from ``stats.cluster_bootstrap_macro`` with
    ``BOOT_SEED``, stability from ``stats.leave_one_game_out`` with the
    original rank pools, and the full rule from ``stats.classify_finding``.

    Args:
        games: Replication data per game.
        theme_keys: Theme columns of ``games``.
        rank_pools: Rank pools for the stability check.
        scope_games: ``{finding key: original eligible app IDs}``.
        original: ``{finding key: {"lift", "lift_ci"}}``.
        findings: Findings to check.
        n_boot: Bootstrap resamples.
        min_kept: Eligibility, kept reviews.
        min_not: Eligibility, not-recommended reviews.

    Returns:
        One result dictionary per finding, in ``findings`` order.
    """
    col = {k: j for j, k in enumerate(theme_keys)}
    eligible = {a for a, g in games.items() if g.rec.size >= min_kept and (~g.rec).sum() >= min_not}
    cache: dict[tuple[int, ...], tuple[Any, Any, Any]] = {}
    out: list[dict[str, Any]] = []
    for f in findings:
        ids = tuple(a for a in scope_games[f.key] if a in eligible)
        orig = original[f.key]
        base = {
            "key": f.key, **asdict(f),
            "original_games": len(scope_games[f.key]), "replication_games": list(ids),
            "original_lift": orig["lift"], "original_ci": orig["lift_ci"],
        }
        if len(ids) < MIN_GAMES_FOR_FINDING:
            out.append({**base, "status": "not_testable"})
            continue
        if ids not in cache:
            sel = [games[a] for a in ids]
            cache[ids] = (
                macro_theme_metrics(sel),
                cluster_bootstrap_macro(sel, n_boot, BOOT_SEED),
                leave_one_game_out(sel, groups=rank_pools),
            )
        point, ci, logo = cache[ids]
        j = col[(f.lens, f.theme)]
        sel = [games[a] for a in ids]
        k_not = int(sum(g.themes[~g.rec, j].sum() for g in sel))
        k_rec = int(sum(g.themes[g.rec, j].sum() for g in sel))
        lift, lo, hi = float(point.lift[j]), float(ci["lift"][0][j]), float(ci["lift"][1][j])
        primary = meets_criterion(f.direction, lift, lo, hi)
        full_rule = classify_finding(lo, hi, len(ids), bool(logo[j]), True, k_not, k_rec)
        o_lo, o_hi = orig["lift_ci"]
        out.append({
            **base,
            "status": "replicated" if primary else "not_replicated",
            "lift": _num(lift), "ci": [_num(lo), _num(hi)],
            "complaint_share": _num(float(point.complaint[j])),
            "praise_share": _num(float(point.praise[j])),
            "k_not_recommended": k_not, "k_recommended": k_rec,
            "n_kept": int(sum(g.rec.size for g in sel)),
            "logo_stable": bool(logo[j]),
            "full_rule_same_direction": full_rule == f.direction,
            "original_in_replication_ci": bool(math.isfinite(lo) and math.isfinite(hi)
                                               and lo <= orig["lift"] <= hi),
            "replication_in_original_ci": bool(math.isfinite(lift) and o_lo <= lift <= o_hi),
            "log_lift_ratio": _num(math.log(lift) / math.log(orig["lift"]))
            if lift > 0 and math.isfinite(lift) else None,
        })
    return out


def summarize(
    results: Sequence[dict[str, Any]],
    count_threshold: int | None = COUNT_THRESHOLD,
) -> dict[str, Any]:
    """Count outcomes per finding and per family and apply the decision rule.

    A family replicates when more than half of its testable findings
    replicate. The study-level verdict compares the replicated count with
    ``count_threshold``.

    Args:
        results: Output of ``evaluate``.
        count_threshold: Pre-registered threshold for the replicated count.

    Returns:
        Counts, family verdicts and the study-level verdict.
    """
    def count(status: str, rows: Sequence[dict[str, Any]]) -> int:
        """Rows with the given status."""
        return sum(r["status"] == status for r in rows)

    testable = [r for r in results if r["status"] != "not_testable"]
    families: dict[str, Any] = {}
    for fam in FAMILIES:
        rows = [r for r in results if r["family"] == fam]
        t = [r for r in rows if r["status"] != "not_testable"]
        rep = count("replicated", rows)
        verdict = "not_testable" if not t else ("replicated" if rep * 2 > len(t) else "not_replicated")
        families[fam] = {
            "label": FAMILIES[fam], "findings": len(rows), "testable": len(t),
            "replicated": rep, "not_replicated": count("not_replicated", rows),
            "not_testable": count("not_testable", rows), "verdict": verdict,
        }
    n_rep = count("replicated", results)
    verdict = None
    if count_threshold is not None:
        verdict = "consistent_with_power" if n_rep >= count_threshold else "below_power"
    return {
        "n_findings": len(results),
        "n_testable": len(testable),
        "n_replicated": n_rep,
        "n_not_replicated": count("not_replicated", results),
        "n_not_testable": count("not_testable", results),
        "n_full_rule_same_direction": sum(bool(r.get("full_rule_same_direction")) for r in testable),
        "n_original_in_replication_ci": sum(bool(r.get("original_in_replication_ci")) for r in testable),
        "n_replication_in_original_ci": sum(bool(r.get("replication_in_original_ci")) for r in testable),
        "families_replicated": sum(v["verdict"] == "replicated" for v in families.values()),
        "families_not_replicated": sum(v["verdict"] == "not_replicated" for v in families.values()),
        "families_not_testable": sum(v["verdict"] == "not_testable" for v in families.values()),
        "families": families,
        "count_threshold": count_threshold,
        "expected_count_plug_in": EXPECTED_COUNT_PLUG_IN,
        "expected_count_half_log_lift": EXPECTED_COUNT_HALF,
        "study_verdict": verdict,
    }


def render_report(
    summary: dict[str, Any], results: Sequence[dict[str, Any]], header: dict[str, Any]
) -> str:
    """Fill the protocol's reporting template.

    Args:
        summary: Output of ``summarize``.
        results: Output of ``evaluate``.
        header: Run facts (mode, commit, tag, sample digest, games).

    Returns:
        Markdown text.
    """
    def fmt(x: float | None, nd: int = 2) -> str:
        """Format a number or a dash."""
        return "n/a" if x is None else f"{x:.{nd}f}"

    lines = [f"# Replication report ({header['mode']})", ""]
    if header["mode"] != "confirmatory":
        lines += ["DRY RUN on a development slice of the seed. These are not replication "
                  "results and must not be reported as such.", ""]
    lines += [
        f"- Commit: {header.get('commit')}, required tag: {header.get('tag')}",
        f"- Sample SHA-256: {header.get('sample_sha256')}",
        f"- Reviews read: {header.get('n_reviews')}, kept: {header.get('n_kept')}",
        f"- Games not resampled: {', '.join(header.get('not_resampled', [])) or 'none'}",
        "",
        "## Counts",
        "",
        f"- Findings: {summary['n_findings']}",
        f"- Testable: {summary['n_testable']}",
        f"- Replicated (primary criterion): {summary['n_replicated']}",
        f"- Not replicated: {summary['n_not_replicated']}",
        f"- Not testable: {summary['n_not_testable']}",
        f"- Full finding rule met in the same direction: {summary['n_full_rule_same_direction']}",
        f"- Original lift inside the replication interval: {summary['n_original_in_replication_ci']}",
        f"- Replication lift inside the original interval: {summary['n_replication_in_original_ci']}",
        f"- Families replicated / not replicated / not testable: "
        f"{summary['families_replicated']} / {summary['families_not_replicated']} / "
        f"{summary['families_not_testable']}",
        f"- Expected replicated count: {fmt(summary['expected_count_plug_in'], 1)} (plug-in), "
        f"{fmt(summary['expected_count_half_log_lift'], 1)} (half log lift)",
        f"- Threshold: {summary['count_threshold']}, study verdict: {summary['study_verdict']}",
        "",
        "## Families",
        "",
        "| Family | Findings | Testable | Replicated | Not replicated | Not testable | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in summary["families"].values():
        lines.append(f"| {v['label']} | {v['findings']} | {v['testable']} | {v['replicated']} | "
                     f"{v['not_replicated']} | {v['not_testable']} | {v['verdict']} |")
    lines += [
        "",
        "## Findings",
        "",
        "| Finding | Original lift (95% CI) | Games orig / rep | Replication lift (95% CI) | "
        "Status | Full rule | Stable |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        o_lo, o_hi = r["original_ci"]
        rep = "n/a"
        if r["status"] != "not_testable":
            rep = f"{fmt(r['lift'])} ({fmt(r['ci'][0])} to {fmt(r['ci'][1])})"
        lines.append(
            f"| {r['key']} | {r['original_lift']:.2f} ({o_lo:.2f} to {o_hi:.2f}) | "
            f"{r['original_games']} / {len(r['replication_games'])} | {rep} | {r['status']} | "
            f"{r.get('full_rule_same_direction', 'n/a')} | {r.get('logo_stable', 'n/a')} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Gate and the two run modes
# ---------------------------------------------------------------------------


def gate_problems(repo: Path = PROJECT_ROOT, tag: str = PROTOCOL_TAG,
                  protocol: Path = PROTOCOL_PATH) -> list[str]:
    """List the reasons the confirmatory read may not run yet.

    Args:
        repo: Repository root.
        tag: Required tag at HEAD.
        protocol: Protocol file that must be tracked.

    Returns:
        Problems; empty when the read may run.
    """
    problems: list[str] = []
    status = _git(["status", "--porcelain"], repo)
    if status.returncode != 0:
        return [f"git status failed: {status.stderr.strip()}"]
    if status.stdout.strip():
        problems.append("working tree is not clean (git status --porcelain is not empty)")
    tags = _git(["tag", "--points-at", "HEAD"], repo).stdout.split()
    if tag not in tags:
        problems.append(f"HEAD does not carry the tag {tag}")
    rel = protocol.resolve().relative_to(repo.resolve()) if protocol.resolve().is_relative_to(
        repo.resolve()) else protocol
    if _git(["ls-files", "--error-unmatch", str(rel)], repo).returncode != 0:
        problems.append(f"{rel} is not tracked")
    return problems


def pin_problems(study: Study, sample_digest_value: str | None) -> list[str]:
    """List mismatches between the frozen artifacts and the pinned hashes.

    Args:
        study: The market study.
        sample_digest_value: Digest of the sample on disk (None if missing).

    Returns:
        Problems; empty when every pin matches.
    """
    problems: list[str] = []
    pins = {
        "sample": (EXPECTED_SAMPLE_SHA256, sample_digest_value),
        "BERTopic model": (EXPECTED_MODEL_SHA256, file_sha256(_model_path(study))
                           if _model_path(study).exists() else None),
        "lexicon": (EXPECTED_LEXICON_SHA256, file_sha256(study.aspect_lexicon)
                    if study.aspect_lexicon and study.aspect_lexicon.exists() else None),
    }
    for name, (want, got) in pins.items():
        if want is None:
            problems.append(f"{name} hash is not pinned in src/replication.py")
        elif want != got:
            problems.append(f"{name} hash {got} does not match the pinned {want}")
    return problems


def _head() -> str:
    """Return the HEAD commit SHA.

    Returns:
        SHA string or ``unknown``.
    """
    r = _git(["rev-parse", "HEAD"], PROJECT_ROOT)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _run(
    study: Study,
    rows: Sequence[tuple[str, int, str]],
    recommended: dict[str, bool],
    out_dir: Path,
    header: dict[str, Any],
    instruments: tuple[Callable, Callable, Callable],
    n_boot: int,
    min_kept: int,
    min_not: int,
    count_threshold: int | None,
) -> dict[str, Any]:
    """Process reviews, evaluate the findings and write results and report.

    Args:
        study: The market study.
        rows: ``(steam_review_id, app_id, raw_text)``.
        recommended: Recommend flag per review ID.
        out_dir: Output directory under ``REPLICATION_DIR``.
        header: Run facts for the report.
        instruments: ``(clean_and_filter, tag, assign_topics)``.
        n_boot: Bootstrap resamples.
        min_kept: Eligibility, kept reviews.
        min_not: Eligibility, not-recommended reviews.
        count_threshold: Study-level threshold, or None for no verdict (dry run).

    Returns:
        ``{"summary", "results", "header"}``, also written to ``results.json``.
    """
    out_dir = _safe_out(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = process_reviews(rows, *instruments)
    write_processed(processed, out_dir / "processed_reviews.csv")
    keys, pools = analysis_theme_keys(study)
    original = load_original(study)
    scope_games = {f.key: scope_app_ids(study, f, original["eligible"]) for f in FINDINGS}
    games = build_games(processed, recommended, keys)
    results = evaluate(games, keys, pools, scope_games, original["findings"],
                       n_boot=n_boot, min_kept=min_kept, min_not=min_not)
    summary = summarize(results, count_threshold)
    header = {**header, "n_reviews": len(rows), "n_kept": sum(p.kept for p in processed),
              "n_boot": n_boot, "eligibility": {"min_kept": min_kept, "min_not_rec": min_not},
              "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    payload = {"header": header, "summary": summary, "results": results}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "report.md").write_text(render_report(summary, results, header))
    return payload


def dry_run(
    study: Study,
    out_dir: Path | None = None,
    per_game: int = DRY_RUN_PER_GAME,
    n_boot: int = DRY_RUN_BOOT,
    instruments: tuple[Callable, Callable, Callable] | None = None,
) -> dict[str, Any]:
    """Run the whole read path on development-split seed reviews.

    Uses up to ``per_game`` seed reviews per game from the development split,
    which the original study already analysed, with lowered eligibility
    thresholds so the slice exercises every step. Reports the gate and pin
    status without enforcing them. Never opens the replication sample's
    outcomes.

    Args:
        study: The market study.
        out_dir: Output directory under ``REPLICATION_DIR``.
        per_game: Development reviews per game.
        n_boot: Bootstrap resamples.
        instruments: Optional replacement instruments (tests).

    Returns:
        The written payload.
    """
    conn = _read_only(study.seed_db)
    try:
        docs = conn.execute(
            "SELECT steam_review_id, app_id, review_text, recommended FROM documents "
            "ORDER BY app_id, steam_review_id"
        ).fetchall()
    finally:
        conn.close()
    out_dir = REPLICATION_DIR / DRY_RUN_SUBDIR if out_dir is None else out_dir
    rows: list[tuple[str, int, str]] = []
    recommended: dict[str, bool] = {}
    taken: dict[int, int] = {}
    for rid, app_id, text, rec in docs:
        if dev_test_split(rid) != "dev" or taken.get(app_id, 0) >= per_game:
            continue
        taken[app_id] = taken.get(app_id, 0) + 1
        rows.append((rid, int(app_id), text))
        recommended[rid] = bool(rec)
    sample_path = REPLICATION_DIR / SAMPLE_DB_NAME
    digest = load_sample()[1] if sample_path.exists() else None
    header = {
        "mode": "dry_run", "commit": _head(), "tag": PROTOCOL_TAG,
        "sample_sha256": digest, "per_game": per_game,
        "gate_problems_now": gate_problems(), "pin_problems_now": pin_problems(study, digest),
        "not_resampled": [],
    }
    return _run(study, rows, recommended, out_dir, header,
                instruments or frozen_instruments(study), n_boot,
                DRY_RUN_MIN_KEPT, DRY_RUN_MIN_NOT_REC, None)


def confirmatory_read(study: Study, out_dir: Path | None = None) -> dict[str, Any]:
    """The single confirmatory read. Refuses unless every gate passes.

    Args:
        study: The market study.
        out_dir: Output directory under ``REPLICATION_DIR``.

    Returns:
        The written payload.

    Raises:
        RuntimeError: If the gate, a pin or the read-once check fails.
    """
    out_dir = REPLICATION_DIR / CONFIRMATORY_SUBDIR if out_dir is None else out_dir
    problems = gate_problems()
    sample_rows, digest = load_sample()
    problems += pin_problems(study, digest)
    if COUNT_THRESHOLD is None:
        problems.append("COUNT_THRESHOLD is not pinned in src/replication.py")
    if (out_dir / "results.json").exists():
        problems.append(f"{out_dir / 'results.json'} exists: the confirmatory read already ran")
    overlap = {r[0] for r in sample_rows} & excluded_review_ids(study)
    if overlap:
        problems.append(f"{len(overlap)} sampled reviews are in the seed or an audit file")
    if problems:
        raise RuntimeError("Confirmatory read refused: " + "; ".join(problems))
    full = _read_only(study.full_db)
    try:
        ids = [r[0] for r in sample_rows]
        recommended: dict[str, bool] = {}
        texts: dict[str, str] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for rid, rec, text in full.execute(
                f"SELECT steam_review_id, recommended, review_text FROM documents "
                f"WHERE steam_review_id IN ({marks})", chunk,
            ):
                recommended[rid] = bool(rec)
                texts[rid] = text
    finally:
        full.close()
    changed = [r[0] for r in sample_rows if texts.get(r[0]) != r[2]]
    if changed:
        raise RuntimeError(f"{len(changed)} sampled texts differ from the full database.")
    manifest = json.loads((REPLICATION_DIR / SAMPLE_MANIFEST_NAME).read_text())
    header = {
        "mode": "confirmatory", "commit": _head(), "tag": PROTOCOL_TAG, "sample_sha256": digest,
        "not_resampled": [study.app(a).name for a in manifest["not_resamplable"]],
    }
    return _run(study, sample_rows, recommended, out_dir, header, frozen_instruments(study),
                N_BOOT, MIN_KEPT_FOR_ELIGIBLE, MIN_NOT_REC_FOR_ELIGIBLE, COUNT_THRESHOLD)


def main() -> None:
    """CLI entry point: ``draw``, ``calibrate``, ``gate``, ``dry-run`` or ``read``."""
    parser = argparse.ArgumentParser(description="Confirmatory replication of the market study.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("draw", help="Draw the replication sample (IDs and texts, no outcomes).")
    cal = sub.add_parser("calibrate", help="Compute the protocol's numbers from non-replication data.")
    cal.add_argument("--skip-transform", action="store_true",
                     help="Skip the BERTopic transform agreement check on the seed.")
    cal.add_argument("--n-sim", type=int, default=POWER_SIMULATIONS)
    sub.add_parser("gate", help="Print whether the confirmatory read may run.")
    dry = sub.add_parser("dry-run", help="Run the read path on a development slice of the seed.")
    dry.add_argument("--per-game", type=int, default=DRY_RUN_PER_GAME)
    dry.add_argument("--n-boot", type=int, default=DRY_RUN_BOOT)
    sub.add_parser("read", help="The single confirmatory read (gated).")
    args = parser.parse_args()
    study = load_study(MARKET_STUDY_PATH)
    try:
        if args.command == "draw":
            m = write_sample(study)
            print(json.dumps({k: m[k] for k in ("n_sampled", "sample_sha256", "not_resamplable")}))
        elif args.command == "calibrate":
            c = calibrate(study, with_transform=not args.skip_transform, n_sim=args.n_sim)
            print(json.dumps({k: {kk: c["power"][k][kk] for kk in
                                  ("count_mean", "count_p05", "count_p50", "count_p95")}
                              for k in c["power"]}, indent=2))
        elif args.command == "gate":
            sample_path = REPLICATION_DIR / SAMPLE_DB_NAME
            digest = load_sample()[1] if sample_path.exists() else None
            problems = gate_problems() + pin_problems(study, digest)
            print("gate open" if not problems else "gate closed:\n- " + "\n- ".join(problems))
        elif args.command == "dry-run":
            p = dry_run(study, per_game=args.per_game, n_boot=args.n_boot)
            s = p["summary"]
            print(json.dumps({k: s[k] for k in ("n_findings", "n_testable", "n_replicated",
                                                "n_not_replicated", "n_not_testable")}))
        elif args.command == "read":
            p = confirmatory_read(study)
            print((REPLICATION_DIR / CONFIRMATORY_SUBDIR / "report.md").read_text())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
