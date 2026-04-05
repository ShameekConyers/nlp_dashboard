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
