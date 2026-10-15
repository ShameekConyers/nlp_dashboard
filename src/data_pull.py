"""Steam Reviews API ingestion. Pulls English reviews and caches raw JSON locally."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests

from src.study import (
    DEFAULT_STUDY_PATH,
    UNFREEZE_FLAG,
    UNFREEZE_HELP,
    AppSpec,
    Baseline,
    Sampling,
    Study,
    guard_writable,
    load_study,
)
from src.utils import GAME_IDS, RAW_DIR, get_logger

logger = get_logger(__name__)

STEAM_REVIEWS_URL = "https://store.steampowered.com/appreviews/{app_id}"
MAX_RETRIES: int = 3
BACKOFF_BASE: float = 2.0
PAGE_DELAY: float = 0.5
NUM_PER_PAGE: int = 100

CENSUS_PAGE_DELAY: float = 1.6
"""Seconds between pages in a natural-sampling census pull.

Steam returned HTTP 429 after about 250 requests in 6.5 minutes at 1.0 s
pacing, so the census paces at 1.6 s (about 140 requests per 5 minutes).
"""

RATE_LIMIT_WAITS: tuple[float, ...] = (30.0, 60.0, 120.0, 240.0, 300.0)
"""Waits after successive HTTP 429 responses that carry no Retry-After."""

CENSUS_MAX_RETRIES: int = len(RATE_LIMIT_WAITS) + 1
"""Attempts per census request before the run stops."""


class PullAbortedError(RuntimeError):
    """Raised when a census request keeps failing; the run stops to back off."""

RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
"""HTTP statuses that trigger a backoff and retry."""

AUTHOR_IDENTITY_FIELDS: tuple[str, ...] = (
    "personaname",
    "profile_url",
    "avatar",
    "persona_status",
)
"""Reviewer display fields dropped before a census page is cached."""

CLEAN_STOP_REASONS: frozenset[str] = frozenset(
    {"exhausted", "all_duplicate", "cursor_exhausted", "cap"}
)
"""Stop reasons that mean the pull finished and may be cached."""


def _sanitize_filename(name: str) -> str:
    """Convert a game name to a filesystem-safe string.

    Args:
        name: Human-readable game name.

    Returns:
        Lowercased name with non-alphanumeric characters replaced by underscores.
    """
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def _get_json_path(app_name: str, app_id: int) -> Path:
    """Return the expected JSON cache path for a game.

    Args:
        app_name: Human-readable game name.
        app_id: Steam application ID.

    Returns:
        Path like data/raw/the_witcher_3_292030.json.
    """
    return RAW_DIR / f"{_sanitize_filename(app_name)}_{app_id}.json"


def _fetch_page(app_id: int, cursor: str = "*") -> dict | None:
    """Fetch a single page of reviews from the Steam API.

    Args:
        app_id: Steam application ID.
        cursor: Pagination cursor. Use '*' for the first page.

    Returns:
        Parsed JSON response dict, or None on failure after retries.
    """
    params = {
        "json": "1",
        "language": "english",
        "filter": "recent",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": str(NUM_PER_PAGE),
        "cursor": cursor,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                STEAM_REVIEWS_URL.format(app_id=app_id),
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES:
                logger.warning(
                    "Attempt %d/%d failed for app %d: %s. Giving up.",
                    attempt,
                    MAX_RETRIES,
                    app_id,
                    exc,
                )
                break
            wait = BACKOFF_BASE**attempt
            logger.warning(
                "Attempt %d/%d failed for app %d: %s. Retrying in %.1fs.",
                attempt,
                MAX_RETRIES,
                app_id,
                exc,
                wait,
            )
            time.sleep(wait)
    return None


def pull_reviews_for_app(
    app_id: int,
    app_name: str,
    max_reviews: int = 5000,
) -> Path:
    """Pull English reviews for a single app. Return path to saved JSON file.

    Paginates through the Steam API until no new reviews are returned or
    max_reviews is reached. Deduplicates by Steam recommendationid within
    this pull.

    Args:
        app_id: Steam application ID.
        app_name: Human-readable game name (used for logging and filename).
        max_reviews: Stop after collecting this many unique reviews.

    Returns:
        Path to the JSON cache file in data/raw/. If no reviews could be
        collected, the file is not written and the path may not exist, so
        the next run retries instead of caching an empty result.
    """
    json_path = _get_json_path(app_name, app_id)

    if json_path.exists() and json_path.stat().st_size > 0:
        logger.info("SKIP %s (%d) — JSON already exists at %s", app_name, app_id, json_path)
        return json_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_reviews: dict[str, dict] = {}
    cursor = "*"
    page = 0

    while True:
        page += 1
        data = _fetch_page(app_id, cursor)

        if data is None:
            logger.warning("Giving up on %s (%d) after repeated failures.", app_name, app_id)
            break

        if data.get("success") != 1:
            logger.warning("API returned success!=1 for %s (%d). Stopping.", app_name, app_id)
            break

        reviews = data.get("reviews", [])
        if not reviews:
            logger.info("No more reviews for %s (%d). Total pages: %d", app_name, app_id, page)
            break

        new_count = 0
        for review in reviews:
            rid = review.get("recommendationid", "")
            if rid and rid not in all_reviews:
                all_reviews[rid] = review
                new_count += 1

        if len(all_reviews) >= max_reviews:
            logger.info("Reached max_reviews (%d) for %s. Stopping.", max_reviews, app_name)
            break

        if new_count == 0:
            logger.info("All reviews on page %d were duplicates for %s. Stopping.", page, app_name)
            break

        cursor = data.get("cursor", "")
        if not cursor:
            break

        if page % 10 == 0:
            logger.info("  %s: %d unique reviews so far (page %d)...", app_name, len(all_reviews), page)

        time.sleep(PAGE_DELAY)

    logger.info("Pulled %d reviews for %s (%d)", len(all_reviews), app_name, app_id)

    if not all_reviews:
        logger.warning(
            "No reviews collected for %s (%d). Skipping cache write so the next run retries.",
            app_name,
            app_id,
        )
        return json_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(all_reviews.values()), f, ensure_ascii=False)

    return json_path


def pull_all_reviews() -> list[Path]:
    """Pull reviews for every game in GAME_IDS. Skip games whose JSON already exists.

    Returns:
        List of paths to all saved JSON files (new and existing).
    """
    paths: list[Path] = []
    for app_name, app_id in GAME_IDS.items():
        path = pull_reviews_for_app(app_id, app_name)
        paths.append(path)
    return paths


def _main_legacy() -> None:
    """Pull the April catalog with the original fixed-count pull and log counts."""
    logger.info("Starting Steam review pull for %d games", len(GAME_IDS))
    paths = pull_all_reviews()

    total_reviews = 0
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            with open(path, encoding="utf-8") as f:
                reviews = json.load(f)
            count = len(reviews)
            total_reviews += count
            logger.info("  %s: %d reviews", path.stem, count)

    logger.info("Done. %d total reviews across %d games.", total_reviews, len(paths))


# ---------------------------------------------------------------------------
# Natural-sampling census pull (study-driven)
# ---------------------------------------------------------------------------


def build_review_params(sampling: Sampling, cursor: str) -> dict[str, str]:
    """Build appreviews query parameters for one census page.

    ``review_type`` is ``all`` for natural studies. ``positive`` or
    ``negative`` pass only for studies loaded with ``stratified = true``
    (``study.load_study`` enforces this), whose recommend rate is then fixed
    by design.

    Args:
        sampling: The study's sampling parameters.
        cursor: Pagination cursor (``*`` for the first page).

    Returns:
        Query parameters as strings.
    """
    return {
        "json": "1",
        "filter": sampling.filter,
        "language": sampling.language,
        "review_type": sampling.review_type,
        "purchase_type": sampling.purchase_type,
        "filter_offtopic_activity": str(sampling.filter_offtopic_activity),
        "num_per_page": str(sampling.num_per_page),
        "cursor": cursor,
    }


def build_baseline_params(baseline: Baseline) -> dict[str, str]:
    """Build the population check parameters (summary only, no reviews).

    Mirrors the research request ``language=all&purchase_type=all&num_per_page=0``
    with Steam's default ``filter``.

    Args:
        baseline: The study's baseline parameters.

    Returns:
        Query parameters as strings.
    """
    return {
        "json": "1",
        "language": baseline.language,
        "purchase_type": baseline.purchase_type,
        "num_per_page": "0",
    }


def _retry_wait(resp: requests.Response | None, attempt: int) -> float:
    """Return seconds to wait before the next attempt.

    Honours a numeric ``Retry-After`` header when present. A 429 without one
    waits ``RATE_LIMIT_WAITS``; other failures use exponential backoff.

    Args:
        resp: The failed response, or None for a network error.
        attempt: 1-based attempt number that just failed.

    Returns:
        Seconds to sleep.
    """
    if resp is not None:
        retry_after = resp.headers.get("Retry-After", "")
        if retry_after.isdigit():
            return float(retry_after)
        if resp.status_code == 429:
            return RATE_LIMIT_WAITS[min(attempt, len(RATE_LIMIT_WAITS)) - 1]
    return BACKOFF_BASE**attempt


def _request_json(app_id: int, params: dict[str, str]) -> dict | None:
    """GET one appreviews response with backoff on 429, 5xx and network errors.

    Args:
        app_id: Steam application ID.
        params: Query parameters.

    Returns:
        Parsed JSON, or None after ``CENSUS_MAX_RETRIES`` failed attempts.
    """
    for attempt in range(1, CENSUS_MAX_RETRIES + 1):
        resp: requests.Response | None = None
        try:
            resp = requests.get(
                STEAM_REVIEWS_URL.format(app_id=app_id), params=params, timeout=30
            )
            if resp.status_code in RETRY_STATUS_CODES:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == CENSUS_MAX_RETRIES:
                logger.warning("App %d: attempt %d failed (%s). Giving up.", app_id, attempt, exc)
                break
            wait = _retry_wait(resp, attempt)
            logger.warning(
                "App %d: attempt %d/%d failed (%s). Retrying in %.1fs.",
                app_id,
                attempt,
                CENSUS_MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    return None


def _strip_author_identity(review: dict) -> dict:
    """Return a copy of a review without reviewer display fields.

    Args:
        review: Raw review dict from the API.

    Returns:
        The review with ``AUTHOR_IDENTITY_FIELDS`` removed from ``author``.
    """
    out = dict(review)
    author = dict(out.get("author") or {})
    for field in AUTHOR_IDENTITY_FIELDS:
        author.pop(field, None)
    out["author"] = author
    return out


def census_paths(study: Study, app_id: int) -> tuple[Path, Path]:
    """Return the raw review file and manifest paths for one app.

    Args:
        study: The study.
        app_id: Steam application ID.

    Returns:
        ``(raw_json_path, manifest_path)``.
    """
    return study.raw_dir / f"{app_id}.json", study.raw_dir / f"{app_id}.pull.json"


def census_pull(app: AppSpec, study: Study) -> dict[str, Any] | None:
    """Pull every review for one app, newest first, up to its cap.

    The cap is the app's ``raw_cap`` when set, otherwise the study's
    ``raw_cap_per_game``.

    Stops when the API returns no reviews, a page adds no new review IDs,
    the cursor stops advancing, or the cap is reached. Page 1's
    ``query_summary`` is recorded, and one extra ``num_per_page=0`` request
    records the population summary with the study's baseline parameters.

    Cached games are skipped. A pull is cached only when it stops cleanly
    and collected at least one review. If requests keep failing, progress is
    saved to ``<app_id>.partial.json`` and the next run resumes from it.

    Args:
        app: The app to pull.
        study: The study supplying sampling parameters and the raw directory.

    Returns:
        The manifest dict (also written to disk), or None if the API
        reported an error or the pull collected nothing.

    Raises:
        PullAbortedError: If a request fails after every retry.
    """
    raw_path, manifest_path = census_paths(study, app.app_id)
    if raw_path.exists() and manifest_path.exists():
        logger.info("SKIP %s (%d): cached at %s", app.name, app.app_id, raw_path)
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    sampling = study.sampling
    cap = app.raw_cap if app.raw_cap is not None else sampling.raw_cap_per_game
    partial_path = study.raw_dir / f"{app.app_id}.partial.json"
    started_at = datetime.now(timezone.utc).isoformat()
    reviews: dict[str, dict] = {}
    query_summary: dict | None = None
    cursor = "*"
    seen_cursors: set[str] = set()
    pages = 0
    stop_reason = ""
    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as f:
            state = json.load(f)
        started_at = state["started_at"]
        reviews = {str(r["recommendationid"]): r for r in state["reviews"]}
        query_summary = state["query_summary"]
        cursor = state["cursor"]
        seen_cursors = set(state["seen_cursors"])
        pages = state["pages"]
        logger.info(
            "RESUME %s (%d) from page %d with %d reviews", app.name, app.app_id, pages, len(reviews)
        )

    while True:
        pages += 1
        data = _request_json(app.app_id, build_review_params(sampling, cursor))
        if data is None:
            stop_reason = "request_failed"
            break
        if data.get("success") != 1:
            stop_reason = "api_error"
            break
        if pages == 1:
            query_summary = data.get("query_summary")

        page_reviews = data.get("reviews", [])
        if not page_reviews:
            stop_reason = "exhausted"
            break

        new_count = 0
        for review in page_reviews:
            rid = str(review.get("recommendationid", ""))
            if rid and rid not in reviews:
                reviews[rid] = _strip_author_identity(review)
                new_count += 1
                if len(reviews) >= cap:
                    break

        if len(reviews) >= cap:
            stop_reason = "cap"
            break
        if new_count == 0:
            stop_reason = "all_duplicate"
            break

        seen_cursors.add(cursor)
        cursor = data.get("cursor", "")
        if not cursor or cursor in seen_cursors:
            stop_reason = "cursor_exhausted"
            break

        if pages % 20 == 0:
            logger.info("  %s: %d reviews after %d pages", app.name, len(reviews), pages)
        time.sleep(CENSUS_PAGE_DELAY)

    if stop_reason == "request_failed":
        study.raw_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "started_at": started_at,
            "reviews": list(reviews.values()),
            "query_summary": query_summary,
            "cursor": cursor,
            "seen_cursors": sorted(seen_cursors),
            "pages": pages - 1,
        }
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        raise PullAbortedError(
            f"{app.name} ({app.app_id}): requests kept failing at page {pages} with "
            f"{len(reviews)} reviews. Progress saved to {partial_path.name}; rerun to resume."
        )
    if stop_reason not in CLEAN_STOP_REASONS or not reviews:
        logger.warning(
            "%s (%d): pull stopped (%s) with %d reviews. Not caching; rerun to retry.",
            app.name,
            app.app_id,
            stop_reason,
            len(reviews),
        )
        return None

    baseline_summary: dict | None = None
    if study.baseline is not None:
        time.sleep(CENSUS_PAGE_DELAY)
        base = _request_json(app.app_id, build_baseline_params(study.baseline))
        if base is not None and base.get("success") == 1:
            baseline_summary = base.get("query_summary")
        else:
            logger.warning("%s (%d): baseline summary request failed.", app.name, app.app_id)

    timestamps = [r["timestamp_created"] for r in reviews.values() if r.get("timestamp_created")]
    manifest: dict[str, Any] = {
        "app_id": app.app_id,
        "app_name": app.name,
        "study": study.name,
        "params": build_review_params(sampling, "*"),
        "baseline_params": build_baseline_params(study.baseline) if study.baseline else None,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "pages": pages,
        "reviews_collected": len(reviews),
        "cap": cap,
        "cap_hit": stop_reason == "cap",
        "stratified": sampling.stratified,
        "stop_reason": stop_reason,
        "effective_window_start": min(timestamps) if timestamps else None,
        "effective_window_end": max(timestamps) if timestamps else None,
        "query_summary": query_summary,
        "baseline_summary": baseline_summary,
    }

    study.raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(list(reviews.values()), f, ensure_ascii=False)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    partial_path.unlink(missing_ok=True)

    total = (query_summary or {}).get("total_reviews")
    logger.info(
        "Pulled %d reviews for %s (%d) in %d pages, stop=%s, English total=%s",
        len(reviews),
        app.name,
        app.app_id,
        pages,
        stop_reason,
        total,
    )
    return manifest


def pull_study(study: Study, unfreeze: bool = False) -> list[dict[str, Any]]:
    """Run the census pull for every app in a study, skipping cached apps.

    The run stops at the first app whose requests keep failing, so a rate
    limit is not spread across the remaining apps.

    Args:
        study: The study to pull.
        unfreeze: Allow writing a frozen study's raw directory.

    Returns:
        Manifests for the apps that are cached after this run.

    Raises:
        FrozenStudyError: If a target path belongs to a frozen study and ``unfreeze`` is False.
    """
    guard_writable([study.raw_dir], unfreeze)
    manifests: list[dict[str, Any]] = []
    for app in study.apps:
        try:
            manifest = census_pull(app, study)
        except PullAbortedError as exc:
            logger.error("Stopping the run to back off: %s", exc)
            break
        if manifest is not None:
            manifests.append(manifest)
    logger.info("Study %s: %d/%d apps cached.", study.name, len(manifests), len(study.apps))
    return manifests


def main() -> None:
    """Entry point. Pulls the April catalog or a natural-sampling study."""
    parser = argparse.ArgumentParser(description="Pull Steam reviews for a study.")
    parser.add_argument(
        "--study",
        type=Path,
        default=DEFAULT_STUDY_PATH,
        help="Study TOML file (default: the April 2026 baseline).",
    )
    parser.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    args = parser.parse_args()
    study = load_study(args.study)
    try:
        if study.sampling.method != "natural":
            guard_writable([study.raw_dir], args.unfreeze)
            _main_legacy()
            return
        pull_study(study, unfreeze=args.unfreeze)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
