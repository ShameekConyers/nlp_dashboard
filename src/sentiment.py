"""VADER sentiment scoring and ground-truth validation for Steam reviews.

Scores every processed review in ``processed_documents`` with VADER sentiment
and writes the results to ``nlp_results``. Validates VADER's compound score
against the Steam recommendation flag (thumbs up/down) as ground truth.

The module is safe to run repeatedly: ``INSERT OR IGNORE`` keyed on
``document_id`` makes it idempotent, and ``--rebuild`` wipes ``nlp_results``
before re-scoring.
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.db_setup import create_tables
from src.utils import (
    MIN_TOKEN_COUNT,
    VADER_POSITIVE_THRESHOLD,
    get_db_path,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class SentimentScorer:
    """Stateful scorer that loads VADER once and reuses it across documents.

    Attributes:
        analyzer: NLTK VADER SentimentIntensityAnalyzer instance.
    """

    analyzer: SentimentIntensityAnalyzer

    def __init__(self) -> None:
        """Load the VADER lexicon."""
        self.analyzer = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> dict[str, float]:
        """Return VADER polarity scores for a single text.

        For empty or whitespace-only strings, returns all zeros (neutral
        baseline). This handles sentinel rows from Phase 2.

        Args:
            text: The cleaned review text to score.

        Returns:
            Dict with keys ``compound``, ``pos``, ``neg``, ``neu``.
        """
        if not text or not text.strip():
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
        scores = self.analyzer.polarity_scores(text)
        return {
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
        }

    def score_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Score multiple texts. Returns list aligned with input.

        Args:
            texts: List of cleaned review texts.

        Returns:
            List of score dicts, one per input text.
        """
        return [self.score_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def compute_validation_metrics(
    compounds: list[float],
    recommended: list[bool],
    threshold: float = VADER_POSITIVE_THRESHOLD,
) -> dict[str, float | int]:
    """Compare VADER compound scores against the recommendation flag.

    Converts compound scores to binary predictions using the given threshold:
    compound >= threshold maps to predicted positive (matches recommended=True).

    Args:
        compounds: List of VADER compound scores.
        recommended: List of ground-truth recommendation flags, aligned with
            compounds.
        threshold: Compound score cutoff for positive classification.

    Returns:
        Dict with keys: ``accuracy``, ``true_pos``, ``false_pos``,
        ``true_neg``, ``false_neg``, ``precision_pos``, ``recall_pos``,
        ``precision_neg``, ``recall_neg``, ``total``.
    """
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for comp, rec in zip(compounds, recommended):
        predicted_positive = comp >= threshold
        if predicted_positive and rec:
            true_pos += 1
        elif predicted_positive and not rec:
            false_pos += 1
        elif not predicted_positive and not rec:
            true_neg += 1
        else:
            false_neg += 1

    total = len(compounds)
    accuracy = (true_pos + true_neg) / total if total > 0 else 0.0

    precision_pos = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall_pos = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    precision_neg = true_neg / (true_neg + false_neg) if (true_neg + false_neg) > 0 else 0.0
    recall_neg = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "true_pos": true_pos,
        "false_pos": false_pos,
        "true_neg": true_neg,
        "false_neg": false_neg,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "precision_neg": precision_neg,
        "recall_neg": recall_neg,
        "total": total,
    }


def find_optimal_threshold(
    compounds: list[float],
    recommended: list[bool],
    start: float = -0.3,
    stop: float = 0.3,
    step: float = 0.05,
) -> tuple[float, float]:
    """Sweep thresholds and return (best_threshold, best_accuracy).

    Tests every threshold from start to stop (inclusive) in step increments.
    Ties go to the threshold closest to 0.05 (VADER's standard default).

    Args:
        compounds: List of VADER compound scores.
        recommended: List of ground-truth recommendation flags.
        start: Lower bound of threshold sweep.
        stop: Upper bound of threshold sweep.
        step: Increment between tested thresholds.

    Returns:
        Tuple of (best_threshold, best_accuracy).
    """
    best_threshold = VADER_POSITIVE_THRESHOLD
    best_accuracy = 0.0

    threshold = start
    while threshold <= stop + 1e-9:
        metrics = compute_validation_metrics(compounds, recommended, threshold)
        accuracy = metrics["accuracy"]
        if accuracy > best_accuracy or (
            accuracy == best_accuracy
            and abs(threshold - VADER_POSITIVE_THRESHOLD)
            < abs(best_threshold - VADER_POSITIVE_THRESHOLD)
        ):
            best_accuracy = accuracy
            best_threshold = threshold
        threshold = round(threshold + step, 10)

    return (round(best_threshold, 4), best_accuracy)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def fetch_unscored_documents(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[tuple[int, str, bool, int]]:
    """Return rows needing sentiment scoring.

    Joins ``processed_documents`` with ``documents`` (for the recommended
    flag) and LEFT JOINs ``nlp_results`` to find rows where
    ``nlp_results.document_id IS NULL``.

    Args:
        conn: Active SQLite connection.
        limit: Optional cap on rows returned. None means no limit.

    Returns:
        List of (document_id, cleaned_text, recommended, token_count) tuples.
    """
    sql = """
        SELECT p.document_id, p.cleaned_text, d.recommended, p.token_count
        FROM processed_documents p
        JOIN documents d ON p.document_id = d.id
        LEFT JOIN nlp_results n ON p.document_id = n.document_id
        WHERE n.document_id IS NULL
        ORDER BY p.document_id
    """
    if limit is not None:
        sql += " LIMIT ?"
        return conn.execute(sql, (limit,)).fetchall()
    return conn.execute(sql).fetchall()


def write_sentiment_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[int, float, float, float, float]],
) -> int:
    """Bulk-insert sentiment scores into nlp_results.

    Uses ``INSERT OR IGNORE`` keyed on ``document_id``. Returns count
    actually inserted via count-before/count-after pattern.

    Args:
        conn: Active SQLite connection.
        rows: Tuples of (document_id, compound, pos, neg, neu).

    Returns:
        Number of rows actually inserted.
    """
    if not rows:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    count_before = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO nlp_results
            (document_id, sentiment_compound, sentiment_pos,
             sentiment_neg, sentiment_neu, processed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(doc_id, comp, pos, neg, neu, now) for doc_id, comp, pos, neg, neu in rows],
    )
    count_after = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
    return count_after - count_before


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run(
    mode: str = "seed",
    rebuild: bool = False,
    batch_size: int = 500,
) -> None:
    """End-to-end sentiment scoring on the chosen DB.

    Steps:
        1. Open the seed.db or full.db connection.
        2. Ensure schema exists via ``db_setup.create_tables``.
        3. If rebuild, ``DELETE FROM nlp_results`` first.
        4. Fetch unscored rows, score with SentimentScorer, write to
           nlp_results.
        5. For kept reviews (token_count >= MIN_TOKEN_COUNT), run validation
           and log metrics.

    Args:
        mode: 'seed' or 'full'.
        rebuild: If True, wipe nlp_results before re-scoring.
        batch_size: Number of rows per batch for scoring and writing.

    Raises:
        FileNotFoundError: If the target database does not exist.
    """
    db_path = get_db_path(mode)
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. Run db_setup.py --full first."
        )

    logger.info("Sentiment scoring mode=%s db=%s rebuild=%s", mode, db_path, rebuild)

    conn = sqlite3.connect(db_path)
    try:
        create_tables(conn)

        if rebuild:
            deleted = conn.execute("DELETE FROM nlp_results").rowcount
            conn.commit()
            logger.info("Rebuild: deleted %d existing nlp_results rows.", deleted)

        to_score = fetch_unscored_documents(conn)
        total = len(to_score)
        logger.info("Scoring %d unscored documents from %s", total, db_path.name)

        if total == 0:
            logger.info("Nothing to do — all documents already scored.")
            return

        scorer = SentimentScorer()

        scored_total = 0
        kept_count = 0
        sentinel_count = 0

        for start in range(0, total, batch_size):
            batch = to_score[start : start + batch_size]
            texts = [row[1] for row in batch]
            scores = scorer.score_batch(texts)

            rows_to_write: list[tuple[int, float, float, float, float]] = []
            for (doc_id, _text, _rec, token_count), score in zip(batch, scores):
                rows_to_write.append((
                    doc_id,
                    score["compound"],
                    score["pos"],
                    score["neg"],
                    score["neu"],
                ))
                if token_count >= MIN_TOKEN_COUNT:
                    kept_count += 1
                else:
                    sentinel_count += 1

            conn.execute("BEGIN")
            write_sentiment_rows(conn, rows_to_write)
            conn.commit()
            scored_total += len(batch)
            logger.info("  progress: %d/%d scored", scored_total, total)

        logger.info(
            "Scored %d documents (%d kept, %d sentinels)",
            scored_total,
            kept_count,
            sentinel_count,
        )

        # Validation on kept reviews only
        validation_rows = conn.execute(
            """
            SELECT n.sentiment_compound, d.recommended
            FROM nlp_results n
            JOIN processed_documents p ON n.document_id = p.document_id
            JOIN documents d ON n.document_id = d.id
            WHERE p.token_count >= ?
            """,
            (MIN_TOKEN_COUNT,),
        ).fetchall()

        if validation_rows:
            compounds = [row[0] for row in validation_rows]
            rec_flags = [bool(row[1]) for row in validation_rows]

            metrics = compute_validation_metrics(compounds, rec_flags)
            logger.info(
                "Ground truth validation (threshold=%.2f): "
                "accuracy=%.1f%%, TP=%d, FP=%d, TN=%d, FN=%d",
                VADER_POSITIVE_THRESHOLD,
                metrics["accuracy"] * 100,
                metrics["true_pos"],
                metrics["false_pos"],
                metrics["true_neg"],
                metrics["false_neg"],
            )

            best_thresh, best_acc = find_optimal_threshold(compounds, rec_flags)
            logger.info(
                "Optimal threshold=%.2f: accuracy=%.1f%%",
                best_thresh,
                best_acc * 100,
            )

            if metrics["accuracy"] < 0.55:
                logger.warning(
                    "Accuracy %.1f%% is below 55%% — something may be wrong.",
                    metrics["accuracy"] * 100,
                )
    finally:
        conn.close()


def main() -> None:
    """CLI entry point. Parses ``--full`` and ``--rebuild`` flags, calls run()."""
    parser = argparse.ArgumentParser(
        description="Score Steam reviews with VADER sentiment analysis."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run against full.db. Default is seed.db.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete all nlp_results rows before re-scoring.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for scoring and writing (default: 500).",
    )
    args = parser.parse_args()
    run(
        mode="full" if args.full else "seed",
        rebuild=args.rebuild,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
