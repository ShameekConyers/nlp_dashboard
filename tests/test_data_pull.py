"""Tests for the Steam Reviews data pull module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_pull import (
    _fetch_page,
    _get_json_path,
    _sanitize_filename,
    pull_reviews_for_app,
)
from src.study import Study


def test_sanitize_filename_basic() -> None:
    """Verify special characters are replaced with underscores."""
    assert _sanitize_filename("The Witcher 3") == "the_witcher_3"


def test_sanitize_filename_strips_trailing_underscores() -> None:
    """Verify leading/trailing underscores from special chars are stripped."""
    assert _sanitize_filename("  Hollow Knight!  ") == "hollow_knight"


def test_get_json_path_format() -> None:
    """Verify the JSON path follows the expected naming pattern."""
    path = _get_json_path("Hollow Knight", 367520)

    assert path.name == "hollow_knight_367520.json"
    assert path.parent.name == "raw"


@patch("src.data_pull.requests.get")
def test_fetch_page_success(mock_get: MagicMock) -> None:
    """Verify _fetch_page returns parsed JSON on a successful response."""
    expected = {"success": 1, "reviews": [{"recommendationid": "1"}]}
    mock_response = MagicMock()
    mock_response.json.return_value = expected
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = _fetch_page(367520, "*")

    assert result == expected
    mock_get.assert_called_once()


@patch("src.data_pull.time.sleep")
@patch("src.data_pull.requests.get")
def test_fetch_page_retries_on_failure(
    mock_get: MagicMock, mock_sleep: MagicMock
) -> None:
    """Verify _fetch_page retries and returns None after max failures."""
    mock_get.side_effect = requests.ConnectionError("network down")

    result = _fetch_page(367520, "*")

    assert result is None
    assert mock_get.call_count == 3


@patch("src.data_pull.time.sleep")
@patch("src.data_pull._fetch_page")
def test_pull_reviews_for_app_skips_existing(
    mock_fetch: MagicMock, mock_sleep: MagicMock, tmp_path: Path
) -> None:
    """Verify pull_reviews_for_app skips download when JSON already exists."""
    json_path = tmp_path / "hollow_knight_367520.json"
    json_path.write_text(json.dumps([{"review": "test"}]), encoding="utf-8")

    with patch("src.data_pull._get_json_path", return_value=json_path):
        result = pull_reviews_for_app(367520, "Hollow Knight")

    assert result == json_path
    mock_fetch.assert_not_called()


@patch("src.data_pull.time.sleep")
@patch("src.data_pull._fetch_page")
def test_pull_reviews_for_app_deduplicates(
    mock_fetch: MagicMock, mock_sleep: MagicMock, tmp_path: Path
) -> None:
    """Verify pull_reviews_for_app deduplicates reviews by recommendationid."""
    json_path = tmp_path / "test_game_123.json"
    if json_path.exists():
        json_path.unlink()

    # Page 1: two reviews, one duplicate on page 2
    page1 = {
        "success": 1,
        "reviews": [
            {"recommendationid": "aaa", "review": "Good"},
            {"recommendationid": "bbb", "review": "Bad"},
        ],
        "cursor": "page2",
    }
    # Page 2: one duplicate (aaa), one new (ccc), then no more
    page2 = {
        "success": 1,
        "reviews": [
            {"recommendationid": "aaa", "review": "Good duplicate"},
            {"recommendationid": "ccc", "review": "Okay"},
        ],
        "cursor": "page3",
    }
    page3: dict = {"success": 1, "reviews": [], "cursor": ""}

    mock_fetch.side_effect = [page1, page2, page3]

    with (
        patch("src.data_pull._get_json_path", return_value=json_path),
        patch("src.data_pull.RAW_DIR", tmp_path),
    ):
        result = pull_reviews_for_app(123, "Test Game")

    assert result == json_path

    with open(json_path, encoding="utf-8") as f:
        saved = json.load(f)

    assert len(saved) == 3, f"Expected 3 unique reviews, got {len(saved)}"
    saved_ids = {r["recommendationid"] for r in saved}
    assert saved_ids == {"aaa", "bbb", "ccc"}


@patch("src.data_pull.time.sleep")
@patch("src.data_pull.requests.get")
def test_fetch_page_no_sleep_after_final_attempt(
    mock_get: MagicMock, mock_sleep: MagicMock
) -> None:
    """Verify a fully failed fetch sleeps between attempts but not after the last."""
    from src.data_pull import MAX_RETRIES

    mock_get.side_effect = requests.ConnectionError("network down")

    assert _fetch_page(367520, "*") is None
    assert mock_sleep.call_count == MAX_RETRIES - 1


# ---------------------------------------------------------------------------
# Natural-sampling census pull
# ---------------------------------------------------------------------------


def _make_study(tmp_path: Path, cap: int = 1000) -> Study:
    """Build a one-app natural study whose raw dir lives under tmp_path.

    Args:
        tmp_path: pytest temporary directory.
        cap: raw_cap_per_game for the study.

    Returns:
        A Study instance.
    """
    from src.study import AppSpec, Baseline, Sampling, Study

    return Study(
        name="test_study",
        title="Test",
        seed_db=tmp_path / "seed.db",
        full_db=tmp_path / "full.db",
        raw_dir=tmp_path / "raw",
        language_filter=True,
        aspect_lexicon=None,
        sampling=Sampling(
            method="natural",
            filter="recent",
            language="english",
            review_type="all",
            purchase_type="all",
            filter_offtopic_activity=1,
            num_per_page=100,
            raw_cap_per_game=cap,
            seed_cap_per_game=100,
            random_seed=42,
        ),
        baseline=Baseline(language="all", purchase_type="all"),
        apps=(AppSpec(1, "Test Game", "Tactics", "c", "", ()),),
    )


def _page(ids: list[int], cursor: str) -> dict:
    """Build a synthetic appreviews page.

    Args:
        ids: Review IDs on the page; each ID doubles as its timestamp.
        cursor: Cursor returned with the page.

    Returns:
        A dict shaped like the Steam response.
    """
    return {
        "success": 1,
        "query_summary": {"total_positive": 8, "total_negative": 2, "total_reviews": 10},
        "reviews": [
            {"recommendationid": str(i), "review": "text", "timestamp_created": 1000 + i}
            for i in ids
        ],
        "cursor": cursor,
    }


def test_build_review_params_never_stratifies() -> None:
    """Verify params carry the study's language and purchase type and review_type=all."""
    from src.data_pull import build_review_params

    study = _make_study(Path("/tmp"))
    params = build_review_params(study.sampling, "abc")

    assert params["review_type"] == "all"
    assert params["language"] == "english"
    assert params["purchase_type"] == "all"
    assert params["filter"] == "recent"
    assert params["cursor"] == "abc"


@patch("src.data_pull.time.sleep")
@patch("src.data_pull._request_json")
def test_census_pull_respects_cap(
    mock_req: MagicMock, mock_sleep: MagicMock, tmp_path: Path
) -> None:
    """Verify the census stops at the cap and records cap_hit and the window."""
    from src.data_pull import census_pull

    study = _make_study(tmp_path, cap=5)
    baseline = {"success": 1, "query_summary": {"total_reviews": 99}}
    mock_req.side_effect = [_page([1, 2, 3], "c2"), _page([4, 5, 6], "c3"), baseline]

    manifest = census_pull(study.apps[0], study)

    assert manifest is not None
    assert manifest["reviews_collected"] == 5
    assert manifest["cap_hit"] is True
    assert manifest["effective_window_start"] == 1001
    assert manifest["effective_window_end"] == 1005
    assert manifest["query_summary"]["total_reviews"] == 10
    assert manifest["baseline_summary"]["total_reviews"] == 99
    saved = json.loads((tmp_path / "raw" / "1.json").read_text(encoding="utf-8"))
    assert len(saved) == 5


@patch("src.data_pull.time.sleep")
@patch("src.data_pull._request_json")
def test_census_pull_stops_on_all_duplicate_page(
    mock_req: MagicMock, mock_sleep: MagicMock, tmp_path: Path
) -> None:
    """Verify an all-duplicate page ends the pull without hitting the cap."""
    from src.data_pull import census_pull

    study = _make_study(tmp_path, cap=1000)
    baseline = {"success": 1, "query_summary": {"total_reviews": 3}}
    mock_req.side_effect = [_page([1, 2], "c2"), _page([1, 2], "c3"), baseline]

    manifest = census_pull(study.apps[0], study)

    assert manifest is not None
    assert manifest["stop_reason"] == "all_duplicate"
    assert manifest["cap_hit"] is False
    assert manifest["reviews_collected"] == 2


@patch("src.data_pull.time.sleep")
@patch("src.data_pull._request_json")
def test_census_pull_saves_partial_and_resumes(
    mock_req: MagicMock, mock_sleep: MagicMock, tmp_path: Path
) -> None:
    """A failed pull caches nothing final, saves progress, and the rerun resumes."""
    from src.data_pull import PullAbortedError, census_pull

    study = _make_study(tmp_path)
    mock_req.side_effect = [_page([1, 2], "c2"), None]

    with pytest.raises(PullAbortedError):
        census_pull(study.apps[0], study)
    assert not (tmp_path / "raw" / "1.json").exists()
    assert (tmp_path / "raw" / "1.partial.json").exists()

    baseline = {"success": 1, "query_summary": {"total_reviews": 3}}
    mock_req.side_effect = [_page([3], "c3"), _page([], ""), baseline]
    manifest = census_pull(study.apps[0], study)

    assert manifest is not None
    assert manifest["reviews_collected"] == 3
    assert manifest["query_summary"]["total_reviews"] == 10
    assert mock_req.call_args_list[2].args[1]["cursor"] == "c2"
    assert not (tmp_path / "raw" / "1.partial.json").exists()
