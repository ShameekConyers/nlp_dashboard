"""Study configuration loader.

A study is a TOML file under ``studies/`` that names the games, cohorts,
sampling parameters, database paths and aspect lexicon for one analysis.
Every pipeline CLI accepts ``--study PATH`` and defaults to the April 2026
baseline, so the original commands keep their behavior.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from src.utils import PROJECT_ROOT

STUDIES_DIR: Path = PROJECT_ROOT / "studies"
DEFAULT_STUDY_PATH: Path = STUDIES_DIR / "april_2026_baseline.toml"
"""The April 2026 cross-genre study. Every CLI defaults to it."""

ALLOWED_REVIEW_TYPES: frozenset[str] = frozenset({"all"})
"""Natural sampling never filters on the recommend flag."""


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
    """

    app_id: int
    name: str
    genre: str
    cohort: str
    subgroup: str
    tags: tuple[str, ...]


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


def load_study(path: Path = DEFAULT_STUDY_PATH) -> Study:
    """Load and validate a study TOML file.

    Args:
        path: Path to the study file.

    Returns:
        The parsed Study.

    Raises:
        ValueError: If the file has duplicate app IDs, no apps, or a
            ``review_type`` other than ``all``.
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
    )
    if sampling.review_type not in ALLOWED_REVIEW_TYPES:
        raise ValueError(
            f"review_type={sampling.review_type!r} would stratify the sample. Use 'all'."
        )

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
        )
        for a in raw.get("apps", [])
    )
    if not apps:
        raise ValueError(f"Study {path} lists no apps.")
    ids = [a.app_id for a in apps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Study {path} has duplicate app IDs.")

    lexicon = meta.get("aspect_lexicon")
    topics = raw.get("topics", {})
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
    )
