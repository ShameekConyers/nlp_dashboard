"""Study configuration loader.

A study is a TOML file under ``studies/`` that names the games, cohorts,
sampling parameters, database paths and aspect lexicon for one analysis.
Every pipeline CLI accepts ``--study PATH`` and defaults to the April 2026
baseline, so the original commands keep their behavior.
"""

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path

from src.utils import PROJECT_ROOT

STUDIES_DIR: Path = PROJECT_ROOT / "studies"
DEFAULT_STUDY_PATH: Path = STUDIES_DIR / "april_2026_baseline.toml"
"""The April 2026 cross-genre study. Every CLI defaults to it."""

NATURAL_REVIEW_TYPES: frozenset[str] = frozenset({"all"})
"""Natural sampling never filters on the recommend flag."""

STATS_SPLITS: frozenset[str] = frozenset({"dev", "test"})
"""Values ``stats_split`` accepts (the halves of ``stats.dev_test_split``)."""

STATS_SOURCES: frozenset[str] = frozenset({"seed", "full"})
"""Values ``stats_source`` accepts: the seed, or one half of the full database."""

STRATIFIED_REVIEW_TYPES: frozenset[str] = frozenset({"positive", "negative"})
"""Review types allowed only with an explicit ``stratified = true``. They fix the
recommend rate by design, so it is no longer a market rate."""


@dataclass(frozen=True)
class AppSpec:
    """One game in a study.

    Attributes:
        app_id: Steam application ID.
        name: Display name.
        genre: Genre label written to ``metadata.genre``.
        cohort: Cohort the game belongs to.
        subgroup: Optional subgroup inside the cohort (empty string if none).
        tags: Free-form tags such as ``matched_basket`` or ``stand_in``.
        raw_cap: Per-game override of ``sampling.raw_cap_per_game`` (``raw_cap``
            in the app's TOML table), or None to use the study cap. Left out
            of ``repr`` so the fingerprints of studies without it stay as they were.
    """

    app_id: int
    name: str
    genre: str
    cohort: str
    subgroup: str
    tags: tuple[str, ...]
    raw_cap: int | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Sampling:
    """Steam appreviews parameters and sample sizes for a study.

    Attributes:
        method: ``natural`` (census pull, uniform seed) or
            ``stratified_legacy`` (the April 50/50 seed).
        filter: Steam ``filter`` parameter.
        language: Steam ``language`` parameter.
        review_type: Steam ``review_type`` parameter. Always ``all``.
        purchase_type: Steam ``purchase_type`` parameter.
        filter_offtopic_activity: Steam ``filter_offtopic_activity`` flag.
        num_per_page: Reviews per API page.
        raw_cap_per_game: Maximum reviews pulled per game.
        seed_cap_per_game: Maximum reviews per game in the committed seed.
        random_seed: Seed for the uniform seed sample.
        stratified: True only when ``review_type`` filters on the recommend
            flag; set explicitly (``--stratified``), recorded in ``pull_runs``.
        window_start: Earliest ``timestamp_created`` (Unix seconds, inclusive)
            kept at load, or None for no lower bound.
        window_end: Latest ``timestamp_created`` (Unix seconds, inclusive)
            kept at load, or None for no upper bound.
    """

    method: str
    filter: str
    language: str
    review_type: str
    purchase_type: str
    filter_offtopic_activity: int
    num_per_page: int
    raw_cap_per_game: int
    seed_cap_per_game: int
    random_seed: int
    stratified: bool = False
    window_start: int | None = None
    window_end: int | None = None

    def in_window(self, timestamp: int | None) -> bool:
        """Return whether a review creation time falls inside the study window.

        Args:
            timestamp: ``timestamp_created`` in Unix seconds, or None.

        Returns:
            True when no window is set, or the timestamp lies within it. A
            review without a timestamp is kept only when no window is set.
        """
        if self.window_start is None and self.window_end is None:
            return True
        if timestamp is None:
            return False
        if self.window_start is not None and timestamp < self.window_start:
            return False
        return not (self.window_end is not None and timestamp > self.window_end)


@dataclass(frozen=True)
class Baseline:
    """Parameters for the population check request (``num_per_page=0``).

    Attributes:
        language: Steam ``language`` parameter for the check.
        purchase_type: Steam ``purchase_type`` parameter for the check.
    """

    language: str
    purchase_type: str


@dataclass(frozen=True)
class Study:
    """A fully resolved study definition.

    Attributes:
        name: Machine name, also used for output folders.
        title: Human-readable title.
        seed_db: Path to the committed seed database.
        full_db: Path to the local full database.
        raw_dir: Directory holding cached raw API pages.
        language_filter: Whether preprocessing drops non-English reviews.
        aspect_lexicon: Path to the aspect lexicon TOML, or None.
        sampling: Sampling parameters.
        baseline: Population check parameters, or None.
        apps: Games in display order.
        topic_min_size: BERTopic minimum topic size from ``[topics]``, or None
            for the pipeline default.
        topic_nr_topics: Target topic count after merging from ``[topics]``,
            or None for no merging.
        topic_umap_preset: UMAP preset name from ``[topics]`` (see
            ``topics.UMAP_PRESETS``); ``legacy`` when absent.
        scores_db: Database for transformer sentiment scores. Defaults to
            ``seed_db``; the April study points at a sidecar so its frozen
            seed file never changes.
        frozen: True for a shipped study. Every stage that writes refuses its
            files unless called with ``--unfreeze-i-know``.
        stats_split: ``dev`` or ``test`` to compute the stats stage on one half
            of the hash split only (``stats_split`` in ``[study]``), or None for
            every kept review. Left out of ``repr`` so the fingerprints of
            studies without it stay as they were; the pipeline adds it to the
            stats fingerprint when it is set.
        stats_source: ``full`` to compute the processing stages and the
            statistics on the ``stats_split`` half of the full database, copied
            into its own analysis database (``analysis_db``), instead of on the
            seed. ``seed`` or None keeps the seed. Left out of ``repr`` like
            ``stats_split``.
        aspect_labels: ``(aspect, lexicon version)`` pairs from
            ``[aspect_labels]``: the version whose audit label file
            (``data/labels/aspect_audit_<version>.csv``) governs each aspect's
            precision. Empty for a study that reads only its current lexicon's
            labels. Left out of ``repr`` like ``stats_split``.
    """

    name: str
    title: str
    seed_db: Path
    full_db: Path
    raw_dir: Path
    language_filter: bool
    aspect_lexicon: Path | None
    sampling: Sampling
    baseline: Baseline | None
    apps: tuple[AppSpec, ...]
    topic_min_size: int | None = None
    topic_nr_topics: int | None = None
    topic_umap_preset: str = "legacy"
    scores_db: Path | None = None
    frozen: bool = False
    stats_split: str | None = field(default=None, repr=False)
    stats_source: str | None = field(default=None, repr=False)
    aspect_labels: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def label_versions(self) -> dict[str, str]:
        """Return ``{aspect: lexicon version}`` from ``aspect_labels``."""
        return dict(self.aspect_labels)

    @property
    def analysis_db(self) -> Path:
        """Return the database the processing stages and statistics use.

        The seed, unless ``stats_source`` is ``full``: then a database next to
        the full one that holds only the ``stats_split`` half of its reviews,
        for example ``<full stem>_dev.db``.
        """
        if self.stats_source != "full":
            return self.seed_db
        return self.full_db.with_name(f"{self.full_db.stem}_{self.stats_split}.db")

    def analysis_view(self) -> "Study":
        """Return this study with ``seed_db`` pointed at ``analysis_db``.

        Every stage reads and writes ``seed_db``, so passing the view runs it
        on the analysis database. A study without ``stats_source = "full"``
        is returned unchanged, and the view of a view is the view.
        """
        if self.stats_source != "full" or self.seed_db == self.analysis_db:
            return self
        return replace(self, seed_db=self.analysis_db)

    @property
    def scores_path(self) -> Path:
        """Return the database that holds this study's transformer scores."""
        return self.scores_db if self.scores_db is not None else self.seed_db

    def app(self, app_id: int) -> AppSpec:
        """Return the app spec for an app ID.

        Args:
            app_id: Steam application ID.

        Returns:
            The matching AppSpec.

        Raises:
            KeyError: If the app is not part of the study.
        """
        for spec in self.apps:
            if spec.app_id == app_id:
                return spec
        raise KeyError(f"App {app_id} is not in study {self.name}.")


def _resolve(path_str: str) -> Path:
    """Resolve a study path relative to the project root.

    Args:
        path_str: Path as written in the TOML file.

    Returns:
        Absolute path.
    """
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_timestamp(value: object, end_of_day: bool = False) -> int | None:
    """Convert a TOML or CLI window bound to Unix seconds (UTC).

    Args:
        value: None, an int (Unix seconds), a TOML date or datetime, or an
            ISO 8601 date or datetime string.
        end_of_day: For a bare date, use 23:59:59 instead of 00:00:00.

    Returns:
        Unix seconds, or None.

    Raises:
        ValueError: If the value cannot be read as a time.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"Not a time: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = datetime.fromisoformat(value) if "T" in value or " " in value else date.fromisoformat(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    if isinstance(value, date):
        t = (23, 59, 59) if end_of_day else (0, 0, 0)
        return int(datetime(value.year, value.month, value.day, *t, tzinfo=timezone.utc).timestamp())
    raise ValueError(f"Not a time: {value!r}")


def load_study(path: Path = DEFAULT_STUDY_PATH) -> Study:
    """Load and validate a study TOML file.

    Args:
        path: Path to the study file.

    Returns:
        The parsed Study.

    Raises:
        ValueError: If the file has no apps, duplicate app IDs, an app
            without a cohort, a ``raw_cap`` below 1, an unknown ``stats_split``
            or ``stats_source``, ``stats_source = "full"`` without a
            ``stats_split``, a ``review_type`` other than ``all`` without
            ``stratified = true``, ``stratified = true`` with ``review_type =
            "all"``, an unknown review type, a window that ends before it
            starts, or an ``[aspect_labels]`` entry that is not a list of
            aspect names or names an aspect under two versions.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    meta = raw["study"]
    samp = raw["sampling"]
    sampling = Sampling(
        method=samp.get("method", "natural"),
        filter=samp["filter"],
        language=samp["language"],
        review_type=samp["review_type"],
        purchase_type=samp["purchase_type"],
        filter_offtopic_activity=int(samp.get("filter_offtopic_activity", 1)),
        num_per_page=int(samp["num_per_page"]),
        raw_cap_per_game=int(samp["raw_cap_per_game"]),
        seed_cap_per_game=int(samp["seed_cap_per_game"]),
        random_seed=int(samp["random_seed"]),
        stratified=bool(samp.get("stratified", False)),
        window_start=parse_timestamp(samp.get("window_start")),
        window_end=parse_timestamp(samp.get("window_end"), end_of_day=True),
    )
    if sampling.review_type not in NATURAL_REVIEW_TYPES | STRATIFIED_REVIEW_TYPES:
        raise ValueError(f"Unknown review_type={sampling.review_type!r}.")
    if sampling.review_type in STRATIFIED_REVIEW_TYPES and not sampling.stratified:
        raise ValueError(
            f"review_type={sampling.review_type!r} would stratify the sample. Use 'all', "
            "or set stratified = true (study_cli --stratified) to accept a design-fixed "
            "recommend rate."
        )
    if sampling.stratified and sampling.review_type in NATURAL_REVIEW_TYPES:
        raise ValueError("stratified = true needs review_type 'positive' or 'negative'.")
    if (sampling.window_start is not None and sampling.window_end is not None
            and sampling.window_end < sampling.window_start):
        raise ValueError("window_end is before window_start.")

    baseline = None
    if "baseline" in raw:
        baseline = Baseline(
            language=raw["baseline"]["language"],
            purchase_type=raw["baseline"]["purchase_type"],
        )

    apps = tuple(
        AppSpec(
            app_id=int(a["app_id"]),
            name=a["name"],
            genre=a["genre"],
            cohort=a.get("cohort", ""),
            subgroup=a.get("subgroup", ""),
            tags=tuple(a.get("tags", [])),
            raw_cap=int(a["raw_cap"]) if "raw_cap" in a else None,
        )
        for a in raw.get("apps", [])
    )
    bad_caps = [a.app_id for a in apps if a.raw_cap is not None and a.raw_cap < 1]
    if bad_caps:
        raise ValueError(f"Study {path}: apps {bad_caps} have a raw_cap below 1.")
    if not apps:
        raise ValueError(f"Study {path} lists no apps.")
    missing = [a.app_id for a in apps if not a.cohort.strip()]
    if missing:
        raise ValueError(f"Study {path}: apps {missing} have no cohort.")
    ids = [a.app_id for a in apps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Study {path} has duplicate app IDs.")

    lexicon = meta.get("aspect_lexicon")
    topics = raw.get("topics", {})
    stats_split = meta.get("stats_split")
    if stats_split is not None and stats_split not in STATS_SPLITS:
        raise ValueError(f"Study {path}: stats_split must be one of {sorted(STATS_SPLITS)}.")
    stats_source = meta.get("stats_source")
    if stats_source is not None and stats_source not in STATS_SOURCES:
        raise ValueError(f"Study {path}: stats_source must be one of {sorted(STATS_SOURCES)}.")
    if stats_source == "full" and stats_split is None:
        raise ValueError(f"Study {path}: stats_source = \"full\" needs a stats_split.")
    aspect_labels: list[tuple[str, str]] = []
    for version, names in raw.get("aspect_labels", {}).items():
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"Study {path}: aspect_labels.{version} must list aspect names.")
        aspect_labels.extend((n, version) for n in names)
    seen = [a for a, _ in aspect_labels]
    twice = sorted({a for a in seen if seen.count(a) > 1})
    if twice:
        raise ValueError(f"Study {path}: aspect_labels names {twice} under more than one version.")
    return Study(
        name=meta["name"],
        title=meta["title"],
        seed_db=_resolve(meta["seed_db"]),
        full_db=_resolve(meta["full_db"]),
        raw_dir=_resolve(meta["raw_dir"]),
        language_filter=bool(meta.get("language_filter", False)),
        aspect_lexicon=_resolve(lexicon) if lexicon else None,
        sampling=sampling,
        baseline=baseline,
        apps=apps,
        topic_min_size=topics.get("min_topic_size"),
        topic_nr_topics=topics.get("nr_topics"),
        topic_umap_preset=topics.get("umap_preset", "legacy"),
        scores_db=_resolve(meta["scores_db"]) if "scores_db" in meta else None,
        frozen=bool(meta.get("frozen", False)),
        stats_split=stats_split,
        stats_source=stats_source,
        aspect_labels=tuple(aspect_labels),
    )


UNFREEZE_FLAG: str = "--unfreeze-i-know"
"""CLI flag that lets a stage write to a frozen (shipped) study's files."""

UNFREEZE_HELP: str = (
    "Allow writing to a frozen (shipped) study's files. Only for regenerating a shipped "
    "study on purpose; it changes committed databases."
)


class FrozenStudyError(ValueError):
    """A stage tried to write to a frozen study's files without the unfreeze flag."""


def study_files(study: Study) -> set[Path]:
    """Return the files and directories a study's stages write.

    Args:
        study: The study.

    Returns:
        Resolved seed, full, scores, analysis and raw paths.
    """
    return {p.resolve() for p in (study.seed_db, study.full_db, study.scores_path,
                                   study.analysis_db, study.raw_dir)}


def all_studies(studies_dir: Path | None = None) -> list[tuple[Path, Study]]:
    """Load every study file in a directory, skipping lexicons and unreadable files.

    Args:
        studies_dir: Directory to scan (default ``studies/``).

    Returns:
        ``(toml_path, study)`` pairs.
    """
    out = []
    for path in sorted((studies_dir or STUDIES_DIR).glob("*.toml")):
        if path.name.startswith("aspects_"):
            continue
        try:
            out.append((path, load_study(path)))
        except (ValueError, KeyError, OSError, tomllib.TOMLDecodeError):
            continue
    return out


def guard_writable(paths: Iterable[Path], unfreeze: bool = False) -> None:
    """Refuse to write to any path that belongs to a frozen study.

    Args:
        paths: Files or directories a stage is about to write.
        unfreeze: True only when the caller passed ``--unfreeze-i-know``.

    Raises:
        FrozenStudyError: If a path belongs to a frozen study and ``unfreeze``
            is False.
    """
    if unfreeze:
        return
    targets = {p.resolve() for p in paths}
    hits = sorted({s.name for _, s in all_studies() if s.frozen and study_files(s) & targets})
    if hits:
        raise FrozenStudyError(
            f"Refusing to write to files of frozen study {', '.join(hits)}. Shipped studies are "
            f"frozen so their committed databases stay reproducible. To regenerate one on "
            f"purpose, rerun with {UNFREEZE_FLAG}."
        )
