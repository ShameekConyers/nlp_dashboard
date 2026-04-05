"""Steam Reviews API ingestion. Pulls English reviews and caches raw JSON locally."""

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests

from src.utils import GAME_IDS, RAW_DIR, get_logger

logger = get_logger(__name__)

STEAM_REVIEWS_URL = "https://store.steampowered.com/appreviews/{app_id}"
MAX_RETRIES: int = 3
BACKOFF_BASE: float = 2.0
PAGE_DELAY: float = 0.5
NUM_PER_PAGE: int = 100


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
        Path to the saved JSON file in data/raw/.
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


def main() -> None:
    """Entry point. Calls pull_all_reviews and logs summary stats."""
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


if __name__ == "__main__":
    main()
