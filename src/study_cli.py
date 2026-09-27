"""Create or extend a study TOML from the command line.

``new`` writes a study file for a list of Steam app IDs in one cohort, with
every sampling parameter exposed as a flag. ``add`` appends another cohort of
apps to an existing study. Names and genres come from ``--names`` and
``--genre``, or from the public ``store.steampowered.com/api/appdetails``
endpoint with ``--lookup`` (responses cached under ``<raw_dir>/appdetails/``
and treated as untrusted text). Every written file is re-read with
``study.load_study``, so an invalid combination (for example a filtered
``review_type`` without ``--stratified``) is rejected before the file is kept.

Example::

    .venv/bin/python src/study_cli.py new --name roguelikes_2026 \\
        --title "Roguelike deckbuilders" --apps 646570,1102190 --cohort deckbuilder --lookup
    .venv/bin/python src/study_cli.py add --study studies/roguelikes_2026.toml \\
        --apps 1145360 --cohort action_roguelike --lookup
"""

import argparse
import json
import re
import sys
import time
import tomllib
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import study as study_mod
from src.study import STUDIES_DIR, Study, all_studies, load_study, study_files
from src.utils import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

APPDETAILS_URL: str = "https://store.steampowered.com/api/appdetails"
APPDETAILS_DELAY: float = 1.5
"""Seconds between uncached appdetails requests."""

GENERIC_LEXICON: str = "studies/aspects_generic_v1.toml"
NAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-z0-9_]+$")
_UNSAFE_TEXT: re.Pattern[str] = re.compile(r"[\x00-\x1f\x7f]")


# ---------------------------------------------------------------------------
# App name and genre lookup
# ---------------------------------------------------------------------------


def _clean(text: str, limit: int = 120) -> str:
    """Strip control characters and cap the length of store-supplied text.

    Args:
        text: Untrusted text from the store API.
        limit: Maximum length kept.

    Returns:
        The cleaned text.
    """
    return _UNSAFE_TEXT.sub(" ", text).strip()[:limit]


def _http_get(app_id: int) -> dict[str, Any] | None:
    """Fetch one appdetails response.

    Args:
        app_id: Steam application ID.

    Returns:
        Parsed JSON, or None on a network or HTTP error.
    """
    try:
        resp = requests.get(
            APPDETAILS_URL, params={"appids": str(app_id), "filters": "basic,genres"}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("appdetails %d failed: %s", app_id, exc)
        return None


def lookup_app(
    app_id: int, cache_dir: Path, fetch: Callable[[int], dict[str, Any] | None] = _http_get
) -> tuple[str, str] | None:
    """Return ``(name, genre)`` for an app from the cache or the store API.

    Args:
        app_id: Steam application ID.
        cache_dir: Directory for cached responses (``<raw_dir>/appdetails``).
        fetch: Network fetcher (replaced in tests).

    Returns:
        ``(name, first genre or "Unknown")``, or None if the store has no data.
    """
    path = cache_dir / f"{app_id}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = fetch(app_id)
        if data is None:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        time.sleep(APPDETAILS_DELAY)
    entry = (data or {}).get(str(app_id)) or {}
    if not entry.get("success") or not isinstance(entry.get("data"), dict):
        return None
    info = entry["data"]
    genres = info.get("genres") or []
    genre = _clean(str(genres[0].get("description", ""))) if genres else ""
    return _clean(str(info.get("name", ""))) or f"App {app_id}", genre or "Unknown"


# ---------------------------------------------------------------------------
# TOML rendering
# ---------------------------------------------------------------------------


def _toml_value(value: Any) -> str:
    """Render a scalar or list as TOML.

    Args:
        value: str, bool, int, float, date, datetime or a list of those.

    Returns:
        TOML text.
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Cannot render {type(value)} as TOML.")


def render_study_toml(doc: dict[str, Any]) -> str:
    """Render a study document as TOML: tables in order, then arrays of tables.

    Args:
        doc: For example ``{"study": {...}, "sampling": {...}, "baseline":
            {...} | None, "topics": {...}, "apps": [{...}, ...]}``. Empty
            tables and None values are omitted. Comments in a file read by
            ``add`` are not preserved.

    Returns:
        TOML text.
    """
    lines = ["# Written by src/study_cli.py. Edit freely; every stage reads this file.", ""]
    for table, values in doc.items():
        if isinstance(values, dict) and values:
            lines.append(f"[{table}]")
            lines += [f"{k} = {_toml_value(v)}" for k, v in values.items() if v is not None]
            lines.append("")
    for table, values in doc.items():
        if isinstance(values, list):
            for item in values:
                lines.append(f"[[{table}]]")
                lines += [f"{k} = {_toml_value(v)}" for k, v in item.items() if v is not None]
                lines.append("")
    return "\n".join(lines)


def _rel(path: Path) -> str:
    """Express a path relative to the project root when it lies inside it.

    Args:
        path: A path.

    Returns:
        A POSIX path string.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_validated(path: Path, doc: dict[str, Any]) -> Study:
    """Write a study file, keep it only if ``load_study`` accepts it.

    Args:
        path: Destination.
        doc: Study document for ``render_study_toml``.

    Returns:
        The loaded study.

    Raises:
        ValueError: If validation fails (the previous file, if any, is kept).
    """
    tmp = path.with_suffix(".toml.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(render_study_toml(doc), encoding="utf-8")
    try:
        study = load_study(tmp)
    except (ValueError, KeyError, tomllib.TOMLDecodeError):
        tmp.unlink()
        raise
    tmp.replace(path)
    return study


def _frozen_toml(path: Path) -> bool:
    """Return True when ``path`` is a readable study file marked ``frozen``.

    Args:
        path: Study TOML path.

    Returns:
        Whether the file exists and loads as a frozen study.
    """
    if not path.exists():
        return False
    try:
        return load_study(path).frozen
    except (ValueError, KeyError, OSError, tomllib.TOMLDecodeError):
        return False


def check_new_study_paths(out: Path, targets: Sequence[Path], force: bool) -> None:
    """Refuse a new study whose file or derived paths collide with existing ones.

    A frozen study's TOML or files are never reused, even with ``--force``.
    Another study's seed, full DB or raw dir, or an existing seed/full file or
    raw dir, is reused only with ``--force``. The study being replaced at
    ``out`` is not counted as another study.

    Args:
        out: Study TOML to write.
        targets: Derived seed DB, full DB and raw dir.
        force: True when ``--force`` was passed.

    Raises:
        ValueError: On a collision that ``force`` does not cover.
    """
    if _frozen_toml(out):
        raise ValueError(f"{out} is a frozen (shipped) study. --force never overwrites it.")
    if out.exists() and not force:
        raise ValueError(f"{out} exists. Use --force to overwrite.")
    wanted = {p.resolve() for p in targets}
    dirs = dict.fromkeys([study_mod.STUDIES_DIR.resolve(), out.parent.resolve()])
    others = [
        s for d in dirs for path, s in all_studies(d) if path.resolve() != out.resolve()
    ]
    for other in others:
        shared = sorted(str(p) for p in study_files(other) & wanted)
        if not shared:
            continue
        if other.frozen:
            raise ValueError(
                f"{', '.join(shared)} belongs to frozen study {other.name}. Pick another name; "
                "--force never reuses a frozen study's files."
            )
        if not force:
            raise ValueError(
                f"{', '.join(shared)} belongs to study {other.name}. Pick another name or use "
                "--force."
            )
    existing = sorted(str(p) for p in wanted if p.exists())
    if existing and not force:
        raise ValueError(f"{', '.join(existing)} already exists. Pick another name or use --force.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _parse_ids(text: str) -> list[int]:
    """Parse a comma-separated list of app IDs.

    Args:
        text: For example ``"646570, 1102190"``.

    Returns:
        App IDs in order.

    Raises:
        ValueError: If an entry is not a positive integer or repeats.
    """
    ids = [int(x) for x in text.replace(" ", "").split(",") if x]
    if not ids or any(i <= 0 for i in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"--apps needs distinct positive app IDs, got {text!r}.")
    return ids


def _app_entries(
    ids: Sequence[int],
    cohort: str,
    subgroup: str | None,
    names: Sequence[str] | None,
    genre: str | None,
    lookup_dir: Path | None,
    fetch: Callable[[int], dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Build ``[[apps]]`` entries for one cohort.

    Args:
        ids: App IDs.
        cohort: Cohort name for every app.
        subgroup: Optional subgroup for every app.
        names: Optional display names aligned with ``ids``.
        genre: Optional genre for every app (overrides the lookup).
        lookup_dir: Cache directory when ``--lookup`` is on, else None.
        fetch: Network fetcher for the lookup.

    Returns:
        App dictionaries.

    Raises:
        ValueError: If ``names`` has the wrong length.
    """
    if names is not None and len(names) != len(ids):
        raise ValueError("--names must list one name per app ID.")
    entries = []
    for i, app_id in enumerate(ids):
        found = lookup_app(app_id, lookup_dir, fetch) if lookup_dir is not None else None
        name = names[i] if names is not None else (found[0] if found else f"App {app_id}")
        entries.append({
            "app_id": app_id,
            "name": name,
            "genre": genre or (found[1] if found else "Unknown"),
            "cohort": cohort,
            "subgroup": subgroup or None,
        })
    return entries


def new_study(args: argparse.Namespace, fetch: Callable[[int], dict[str, Any] | None] = _http_get) -> Path:
    """Write a new study TOML from parsed CLI arguments.

    Args:
        args: Parsed ``new`` arguments.
        fetch: Network fetcher for ``--lookup``.

    Returns:
        Path of the written study file.

    Raises:
        ValueError: On an invalid name, a path collision (see
            ``check_new_study_paths``), or a study that ``load_study`` rejects.
    """
    if not NAME_PATTERN.match(args.name):
        raise ValueError("--name must use lowercase letters, digits and underscores.")
    out = args.out or STUDIES_DIR / f"{args.name}.toml"
    data = Path(args.data_dir)
    data = data if data.is_absolute() else PROJECT_ROOT / data
    raw_dir = data / "raw" / args.name
    check_new_study_paths(
        out, [data / f"{args.name}_seed.db", data / f"{args.name}_full.db", raw_dir], args.force
    )
    doc = {
        "study": {
            "name": args.name,
            "title": args.title or args.name.replace("_", " ").capitalize(),
            "seed_db": _rel(data / f"{args.name}_seed.db"),
            "full_db": _rel(data / f"{args.name}_full.db"),
            "raw_dir": _rel(raw_dir),
            "language_filter": not args.no_language_filter,
            "aspect_lexicon": args.lexicon or None,
        },
        "sampling": {
            "method": "natural",
            "filter": args.filter,
            "language": args.language,
            "review_type": args.review_type,
            "stratified": True if args.stratified else None,
            "purchase_type": args.purchase_type,
            "filter_offtopic_activity": args.offtopic,
            "num_per_page": args.num_per_page,
            "raw_cap_per_game": args.raw_cap,
            "seed_cap_per_game": args.seed_cap,
            "random_seed": args.random_seed,
            "window_start": args.window_start,
            "window_end": args.window_end,
        },
        "baseline": None if args.no_baseline else {
            "language": args.baseline_language,
            "purchase_type": args.baseline_purchase_type,
        },
        "topics": {"umap_preset": args.umap_preset, "min_topic_size": args.min_topic_size},
        "apps": _app_entries(
            _parse_ids(args.apps), args.cohort, args.subgroup,
            args.names.split("|") if args.names else None, args.genre,
            raw_dir / "appdetails" if args.lookup else None, fetch,
        ),
    }
    study = _write_validated(out, doc)
    logger.info("Wrote %s with %d apps in cohort %s.", out, len(study.apps), args.cohort)
    return out


def add_apps(args: argparse.Namespace, fetch: Callable[[int], dict[str, Any] | None] = _http_get) -> Path:
    """Append a cohort of apps to an existing study TOML.

    Args:
        args: Parsed ``add`` arguments.
        fetch: Network fetcher for ``--lookup``.

    Returns:
        Path of the updated study file.

    Raises:
        ValueError: If the study is frozen, an app is already in the study, or
            validation fails.
    """
    with open(args.study, "rb") as f:
        doc = tomllib.load(f)
    study = load_study(args.study)
    if study.frozen:
        raise ValueError(f"{args.study} is a frozen (shipped) study. Its apps cannot change.")
    ids = _parse_ids(args.apps)
    clash = sorted(set(ids) & {a.app_id for a in study.apps})
    if clash:
        raise ValueError(f"Apps {clash} are already in {args.study}.")
    doc["apps"] = list(doc.get("apps", [])) + _app_entries(
        ids, args.cohort, args.subgroup, args.names.split("|") if args.names else None,
        args.genre, study.raw_dir / "appdetails" if args.lookup else None, fetch,
    )
    _write_validated(Path(args.study), doc)
    logger.info("Added %d apps in cohort %s to %s.", len(ids), args.cohort, args.study)
    return Path(args.study)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``new``/``add`` argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description="Create or extend a study TOML.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def app_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--apps", required=True, help="Comma-separated Steam app IDs.")
        p.add_argument("--cohort", required=True, help="Cohort for these apps.")
        p.add_argument("--subgroup", default=None, help="Optional subgroup for these apps.")
        p.add_argument("--names", default=None, help="Display names separated by |.")
        p.add_argument("--genre", default=None, help="Genre for these apps.")
        p.add_argument("--lookup", action="store_true",
                       help="Fill names and genres from the public appdetails endpoint (cached).")

    new = sub.add_parser("new", help="Write a new study file.")
    new.add_argument("--name", required=True, help="Machine name, e.g. roguelikes_2026.")
    new.add_argument("--title", default=None)
    app_flags(new)
    new.add_argument("--out", type=Path, default=None, help="Default studies/<name>.toml.")
    new.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing study file or reuse another study's paths. Never applies "
        "to a frozen study.",
    )
    new.add_argument("--data-dir", default="data", help="Where the seed, full DB and raw cache go.")
    new.add_argument("--lexicon", default=GENERIC_LEXICON,
                     help=f"Aspect lexicon (default {GENERIC_LEXICON}); '' for none.")
    new.add_argument("--no-language-filter", action="store_true")
    new.add_argument("--filter", default="recent", choices=["recent", "updated", "all"])
    new.add_argument("--language", default="english")
    new.add_argument("--review-type", default="all", choices=["all", "positive", "negative"])
    new.add_argument("--stratified", action="store_true",
                     help="Required for --review-type positive/negative: the recommend rate "
                          "becomes design-fixed.")
    new.add_argument("--purchase-type", default="all", choices=["all", "steam", "non_steam_purchase"])
    new.add_argument("--offtopic", type=int, default=1, choices=[0, 1],
                     help="filter_offtopic_activity (1 excludes review-bomb windows).")
    new.add_argument("--num-per-page", type=int, default=100)
    new.add_argument("--raw-cap", type=int, default=20000, help="Reviews pulled per game, newest first.")
    new.add_argument("--seed-cap", type=int, default=500, help="Reviews per game in the seed.")
    new.add_argument("--random-seed", type=int, default=42)
    new.add_argument("--window-start", default=None,
                     help="Earliest review creation date kept at load (YYYY-MM-DD).")
    new.add_argument("--window-end", default=None,
                     help="Latest review creation date kept at load (YYYY-MM-DD, inclusive).")
    new.add_argument("--no-baseline", action="store_true", help="Skip the population summary request.")
    new.add_argument("--baseline-language", default="all")
    new.add_argument("--baseline-purchase-type", default="all")
    new.add_argument("--umap-preset", default="bertopic_default")
    new.add_argument("--min-topic-size", type=int, default=40)

    add = sub.add_parser("add", help="Append a cohort of apps to a study file.")
    add.add_argument("--study", type=Path, required=True)
    app_flags(add)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point. A rejected study prints ``error: ...`` and exits with status 2.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).
    """
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "new":
            new_study(args)
        else:
            add_apps(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
