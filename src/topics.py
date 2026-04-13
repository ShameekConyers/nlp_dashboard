"""BERTopic topic modeling for Steam reviews.

Fits a BERTopic model on kept reviews from ``processed_documents``, assigns
each review a topic, and writes results to ``nlp_results`` (updating
``topic_id`` and ``topic_label`` columns that Phase 3 left NULL). A ``topics``
table stores per-topic metadata (label, representative words, document count)
so the dashboard can query topic information via SQL joins.

The module is safe to run repeatedly: rows already assigned a topic are
skipped. Use ``--rebuild`` to clear topic assignments and re-fit.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from bertopic import BERTopic
from umap import UMAP

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.db_setup import create_tables
from src.utils import (
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_MIN_TOPIC_SIZE,
    BERTOPIC_RANDOM_STATE,
    MIN_TOKEN_COUNT,
    OUTPUTS_DIR,
    get_db_path,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------


def format_topic_label(raw_label: str) -> str:
    """Clean BERTopic's auto-generated label into a human-readable string.

    Strips the numeric prefix (e.g., ``'0_combat_fight_enemy'``) and converts
    to ``'Combat / Fight / Enemy'``. Outlier topic (-1) returns ``'Outlier'``.

    Args:
        raw_label: BERTopic's raw topic label string.

    Returns:
        Cleaned, capitalized label.
    """
    if raw_label.startswith("-1"):
        return "Outlier"
    # Strip numeric prefix: "0_combat_fight_enemy" -> "combat_fight_enemy"
    parts = raw_label.split("_", 1)
    if len(parts) == 2 and parts[0].lstrip("-").isdigit():
        words_str = parts[1]
    else:
        words_str = raw_label
    words = [w.capitalize() for w in words_str.split("_") if w]
    return " / ".join(words) if words else raw_label


# ---------------------------------------------------------------------------
# TopicModeler
# ---------------------------------------------------------------------------


class TopicModeler:
    """Stateful modeler that fits BERTopic and stores the model for reuse.

    Attributes:
        model: Fitted BERTopic instance (None before fitting).
        embedding_model: Name of the sentence-transformer model.
        min_topic_size: Minimum cluster size for HDBSCAN.
        random_state: UMAP random seed for reproducibility.
    """

    model: BERTopic | None
    embedding_model: str
    min_topic_size: int
    random_state: int

    def __init__(
        self,
        embedding_model: str = BERTOPIC_EMBEDDING_MODEL,
        min_topic_size: int = BERTOPIC_MIN_TOPIC_SIZE,
        random_state: int = BERTOPIC_RANDOM_STATE,
    ) -> None:
        """Initialize the topic modeler with configuration.

        Args:
            embedding_model: Sentence-transformer model name.
            min_topic_size: Minimum number of documents to form a topic.
            random_state: Random seed for UMAP dimensionality reduction.
        """
        self.model = None
        self.embedding_model = embedding_model
        self.min_topic_size = min_topic_size
        self.random_state = random_state

    def fit(self, texts: list[str]) -> list[int]:
        """Fit BERTopic on the corpus and return topic assignments.

        Args:
            texts: List of cleaned review texts.

        Returns:
            List of topic IDs aligned with input. -1 means outlier.
        """
        umap_model = UMAP(random_state=self.random_state)
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            min_topic_size=self.min_topic_size,
            verbose=True,
        )
        topics, _probs = self.model.fit_transform(texts)
        return list(topics)

    def reduce_topics(self, texts: list[str], nr_topics: int) -> list[int]:
        """Merge topics down to the target count. Must call fit() first.

        Calls ``model.reduce_topics(texts, nr_topics=nr_topics)`` which
        modifies ``model.topics_`` in place.

        Args:
            texts: Same texts passed to fit().
            nr_topics: Target number of topics (excluding outliers).

        Returns:
            Updated list of topic IDs from ``self.model.topics_``.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.model is None:
            raise ValueError("Must call fit() before reduce_topics().")
        self.model.reduce_topics(texts, nr_topics=nr_topics)
        return list(self.model.topics_)

    def get_topic_info(self) -> list[dict]:
        """Return per-topic metadata from the fitted model.

        Returns:
            List of dicts with keys: topic_id, label, top_words, doc_count.
            Includes the outlier topic (topic_id=-1).

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.model is None:
            raise ValueError("Must call fit() before get_topic_info().")
        info_df = self.model.get_topic_info()
        result: list[dict] = []
        for _, row in info_df.iterrows():
            topic_id = int(row["Topic"])
            raw_name = str(row["Name"])
            # Representation is a list of top words
            representation = row.get("Representation", [])
            if isinstance(representation, str):
                top_words = [representation]
            else:
                top_words = list(representation) if representation is not None else []
            result.append({
                "topic_id": topic_id,
                "label": format_topic_label(raw_name),
                "top_words": top_words,
                "doc_count": int(row["Count"]),
            })
        return result

    def save_model(self, path: Path) -> None:
        """Save the fitted BERTopic model to disk.

        Args:
            path: Directory to save the model into.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        if self.model is None:
            raise ValueError("Must call fit() before save_model().")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path, serialization="pickle")
        logger.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def fetch_untopiced_documents(
    conn: sqlite3.Connection,
) -> list[tuple[int, str]]:
    """Return (document_id, cleaned_text) for kept reviews needing topic assignment.

    Joins nlp_results with processed_documents to find rows where
    topic_id IS NULL and token_count >= MIN_TOKEN_COUNT (excludes sentinels).

    Args:
        conn: Active SQLite connection.

    Returns:
        List of (document_id, cleaned_text) tuples, ordered by document_id.
    """
    return conn.execute(
        """
        SELECT p.document_id, p.cleaned_text
        FROM processed_documents p
        JOIN nlp_results n ON p.document_id = n.document_id
        WHERE n.topic_id IS NULL
          AND p.token_count >= ?
        ORDER BY p.document_id
        """,
        (MIN_TOKEN_COUNT,),
    ).fetchall()


def write_topic_assignments(
    conn: sqlite3.Connection,
    assignments: list[tuple[int, int, str]],
) -> int:
    """Update nlp_results rows with topic assignments.

    Uses ``executemany`` with a single ``UPDATE`` statement inside a
    transaction. Uses UPDATE (not INSERT) since Phase 3 already created
    the rows.

    Args:
        conn: Active SQLite connection.
        assignments: List of (document_id, topic_id, topic_label) tuples.

    Returns:
        Number of rows updated.
    """
    if not assignments:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        UPDATE nlp_results
        SET topic_id = ?, topic_label = ?, processed_at = ?
        WHERE document_id = ?
        """,
        [(tid, label, now, doc_id) for doc_id, tid, label in assignments],
    )
    return len(assignments)


def write_topics_table(
    conn: sqlite3.Connection,
    topic_info: list[dict],
) -> int:
    """Populate the topics table with per-topic metadata.

    Uses ``INSERT OR REPLACE`` so re-runs update existing rows.

    Args:
        conn: Active SQLite connection.
        topic_info: List of dicts from TopicModeler.get_topic_info().

    Returns:
        Number of rows written.
    """
    if not topic_info:
        return 0

    conn.executemany(
        """
        INSERT OR REPLACE INTO topics (topic_id, label, top_words, doc_count)
        VALUES (?, ?, ?, ?)
        """,
        [
            (t["topic_id"], t["label"], json.dumps(t["top_words"]), t["doc_count"])
            for t in topic_info
        ],
    )
    return len(topic_info)


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run(
    mode: str = "seed",
    rebuild: bool = False,
    nr_topics: int | None = None,
    min_topic_size: int | None = None,
) -> None:
    """End-to-end topic modeling on the chosen DB.

    Steps:
        1. Open the seed.db or full.db connection.
        2. Ensure schema exists via db_setup.create_tables.
        3. If rebuild, clear topic columns in nlp_results and delete topics
           table rows.
        4. Fetch untopiced documents (kept reviews only).
        5. Fit BERTopic, optionally reduce topics to nr_topics.
        6. Update nlp_results with topic assignments.
        7. Populate topics table with per-topic metadata.
        8. Save model to outputs/bertopic_model/.

    Args:
        mode: 'seed' or 'full'.
        rebuild: If True, clear topic assignments before re-fitting.
        nr_topics: Target number of topics after merging. None means no
            merging.
        min_topic_size: Override BERTOPIC_MIN_TOPIC_SIZE. None uses the
            default (10 for seed, 50 for full).

    Raises:
        FileNotFoundError: If the target database does not exist.
    """
    db_path = get_db_path(mode)
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. Run db_setup.py --full first."
        )

    logger.info(
        "Topic modeling mode=%s db=%s rebuild=%s", mode, db_path, rebuild
    )

    conn = sqlite3.connect(db_path)
    try:
        create_tables(conn)

        # Pre-check: Phase 3 must have run first
        nlp_count = conn.execute("SELECT COUNT(*) FROM nlp_results").fetchone()[0]
        if nlp_count == 0:
            logger.error(
                "No nlp_results rows found. Run sentiment.py first."
            )
            return

        if rebuild:
            updated = conn.execute(
                "UPDATE nlp_results SET topic_id = NULL, topic_label = NULL "
                "WHERE topic_id IS NOT NULL"
            ).rowcount
            deleted = conn.execute("DELETE FROM topics").rowcount
            conn.commit()
            logger.info(
                "Rebuild: cleared %d topic assignments, deleted %d topics rows.",
                updated,
                deleted,
            )

        docs = fetch_untopiced_documents(conn)
        if not docs:
            logger.info("Nothing to do — 0 untopiced documents.")
            return

        doc_ids = [d[0] for d in docs]
        texts = [d[1] for d in docs]
        logger.info("Found %d documents to topic-model.", len(texts))

        # Determine min_topic_size
        effective_min_topic_size = min_topic_size
        if effective_min_topic_size is None:
            effective_min_topic_size = 10 if mode == "seed" else 50

        modeler = TopicModeler(min_topic_size=effective_min_topic_size)
        topic_ids = modeler.fit(texts)

        # Count topics and outliers
        non_outlier_topics = {t for t in topic_ids if t != -1}
        outlier_count = sum(1 for t in topic_ids if t == -1)
        logger.info(
            "BERTopic found %d topics (%d outlier documents out of %d total)",
            len(non_outlier_topics),
            outlier_count,
            len(topic_ids),
        )

        if outlier_count > len(topic_ids) * 0.5:
            logger.warning(
                "High outlier ratio (%.0f%%). Consider lowering min_topic_size.",
                outlier_count / len(topic_ids) * 100,
            )

        if len(non_outlier_topics) < 3:
            logger.warning(
                "Very few topics found. Corpus may be too small or homogeneous."
            )

        # Optional topic reduction
        if nr_topics is not None and len(non_outlier_topics) > nr_topics:
            topic_ids = modeler.reduce_topics(texts, nr_topics)
            non_outlier_topics = {t for t in topic_ids if t != -1}
            logger.info("Reduced to %d topics.", len(non_outlier_topics))

        # Get topic metadata and build label lookup
        topic_info = modeler.get_topic_info()
        label_lookup: dict[int, str] = {
            t["topic_id"]: t["label"] for t in topic_info
        }

        # Build assignments
        assignments: list[tuple[int, int, str]] = []
        for doc_id, tid in zip(doc_ids, topic_ids):
            label = label_lookup.get(tid, f"Topic {tid}")
            assignments.append((doc_id, tid, label))

        # Write to DB
        conn.execute("BEGIN")
        n_updated = write_topic_assignments(conn, assignments)
        n_topics = write_topics_table(conn, topic_info)
        conn.commit()

        logger.info("Updated %d nlp_results rows with topic assignments.", n_updated)
        logger.info("Wrote %d rows to topics table.", n_topics)

        # Save model
        model_path = OUTPUTS_DIR / "bertopic_model" / "model.pkl"
        modeler.save_model(model_path)

    finally:
        conn.close()


def main() -> None:
    """CLI entry point. Parses --full, --rebuild, --nr-topics, --min-topic-size."""
    parser = argparse.ArgumentParser(
        description="Fit BERTopic on Steam reviews and write topic assignments."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run against full.db. Default is seed.db.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Clear topic assignments before re-fitting.",
    )
    parser.add_argument(
        "--nr-topics",
        type=int,
        default=None,
        help="Target number of topics after merging. None means no merging.",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=None,
        help="Override minimum cluster size (default: 10 for seed, 50 for full).",
    )
    args = parser.parse_args()
    run(
        mode="full" if args.full else "seed",
        rebuild=args.rebuild,
        nr_topics=args.nr_topics,
        min_topic_size=args.min_topic_size,
    )


if __name__ == "__main__":
    main()
