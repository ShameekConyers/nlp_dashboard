"""Text preprocessing pipeline for Steam reviews.

Reads raw rows from the `documents` table, runs a deterministic cleaning and
tokenization pipeline, and writes the results to `processed_documents`.

Two representations are stored per document:

    * `cleaned_text` — light cleaning only (HTML/BBCode/URLs/emails stripped,
      mojibake repaired, whitespace collapsed). Caps, punctuation, and emoji
      are preserved so VADER can read them in Phase 3.
    * `tokens` — JSON-encoded list of lowercased lemmas with stopwords,
      punctuation, digits, short tokens, and project-specific stoplist
      entries removed. Feeds the word cloud and topic-term view.

The module is safe to run repeatedly: `INSERT OR IGNORE` keyed on
`document_id` makes it idempotent, and `--rebuild` wipes `processed_documents`
before reprocessing when the cleaning logic itself changes.
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import ftfy
import spacy

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.db_setup import create_study_tables, create_tables
from src.study import (
    DEFAULT_STUDY_PATH,
    UNFREEZE_FLAG,
    UNFREEZE_HELP,
    Study,
    guard_writable,
    load_study,
)
from src.utils import (
    DETECTOR_LANGUAGES,
    LANGUAGE_MIN_CONFIDENCE,
    MIN_CLEAN_TEXT_CHARS,
    MIN_TOKEN_COUNT,
    PROJECT_STOPWORDS,
    SPACY_MODEL_NAME,
    get_db_path,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

HTML_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>")
"""Match raw HTML tags like `<br>` or `<p>` that sneak into reviews."""

URL_RE: re.Pattern[str] = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)
"""Match http/https URLs and bare `www.` URLs."""

EMAIL_RE: re.Pattern[str] = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
"""Match standard email addresses."""

WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")
"""Match runs of whitespace including newlines, tabs, and non-breaking spaces."""

STEAM_BBCODE_RE: re.Pattern[str] = re.compile(r"\[/?[a-zA-Z*][^\]]*\]")
"""Match any Steam BBCode tag.

Catches `[b]`, `[/b]`, `[url=...]`, `[/url]`, `[spoiler]`, `[h1]`, `[list]`,
`[*]`, `[code]`, `[quote]`, `[strike]`, `[noparse]`, `[table]`, and any future
tag Steam adds. The `*` in the character class matters: Steam's list-item
marker is `[*]` and a letter-only regex would miss it.
"""


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------


class Preprocessor:
    """Stateful text preprocessor that loads spaCy once and reuses it.

    The spaCy model is loaded in `__init__` with the heavy `parser` and `ner`
    components disabled — we only need the tokenizer, POS tagger, and
    lemmatizer. Tokenization batches go through `nlp.pipe` for the ~3x
    throughput win documented in spaCy's performance guide.

    Attributes:
        nlp: Loaded spaCy language pipeline.
        project_stopwords: Project-specific stoplist from
            `utils.PROJECT_STOPWORDS`, layered on top of spaCy's
            `token.is_stop` defaults.

    Example:
        >>> pre = Preprocessor()
        >>> pre.clean_text("<br>GREAT game!!! http://x.com")
        'GREAT game!!!'
    """

    nlp: spacy.language.Language
    project_stopwords: frozenset[str]

    def __init__(self, model_name: str = SPACY_MODEL_NAME) -> None:
        """Load the spaCy model with heavy components disabled.

        Args:
            model_name: spaCy model identifier. Defaults to
                `utils.SPACY_MODEL_NAME` (`en_core_web_sm`).

        Raises:
            OSError: If the requested spaCy model is not installed.
        """
        self.nlp = spacy.load(model_name, disable=["parser", "ner"])
        self.project_stopwords = PROJECT_STOPWORDS

    def clean_text(self, raw: str) -> str:
        """Strip HTML, BBCode, URLs, emails, repair encoding, collapse whitespace.

        Preserves capitalization, punctuation, and emoji so VADER can read
        them downstream. Order of operations matters: ftfy runs first so the
        regexes see well-formed UTF-8; BBCode runs before HTML because BBCode
        tags do not share structure with HTML tags and removing them first
        avoids partial matches.

        Args:
            raw: Raw review text as pulled from the Steam API.

        Returns:
            Cleaned text with caps and punctuation intact. May be empty if
            the input was nothing but tags, URLs, or whitespace.
        """
        if not raw:
            return ""

        text = ftfy.fix_text(raw)
        text = STEAM_BBCODE_RE.sub(" ", text)
        text = HTML_TAG_RE.sub(" ", text)
        text = URL_RE.sub(" ", text)
        text = EMAIL_RE.sub(" ", text)
        text = WHITESPACE_RE.sub(" ", text)
        return text.strip()

    def tokenize_batch(self, cleaned_texts: list[str]) -> list[list[str]]:
        """Tokenize and lemmatize many cleaned texts in one spaCy pass.

        For each input document, returns a list of lowercased lemmas with
        the following dropped:

            * stopwords (`token.is_stop`)
            * punctuation (`token.is_punct`)
            * whitespace-only tokens (`token.is_space`)
            * digits and numeric tokens (`token.is_digit`, `token.like_num`)
            * lemmas shorter than 3 characters
            * lemmas in `self.project_stopwords`

        Note:
            The `like_num` filter drops "ten", "one hundred", and "10/10".
            That is acceptable because `cleaned_text` still preserves them
            for VADER. Revisit only if Phase 3 BERTopic surfaces a weak
            rating-language topic.

        Args:
            cleaned_texts: Already-cleaned review strings.

        Returns:
            One token list per input, aligned positionally with the input.
        """
        if not cleaned_texts:
            return []

        token_lists: list[list[str]] = []
        for doc in self.nlp.pipe(cleaned_texts, batch_size=200):
            tokens: list[str] = []
            for token in doc:
                if token.is_stop or token.is_punct or token.is_space:
                    continue
                if token.is_digit or token.like_num:
                    continue
                lemma = token.lemma_.lower().strip()
                if len(lemma) < 3:
                    continue
                if lemma in self.project_stopwords:
                    continue
                tokens.append(lemma)
            token_lists.append(tokens)
        return token_lists

    def process_batch(self, raws: list[str]) -> list[dict]:
        """Clean and tokenize a batch of raw reviews, applying filter rules.

        A review is dropped if any of the following are true after cleaning
        and tokenization:

            * ``len(cleaned_text) < MIN_CLEAN_TEXT_CHARS``
            * ``len(tokens) < MIN_TOKEN_COUNT``

        Dropped reviews still produce a result dict so that the caller can
        distinguish kept from dropped without re-deriving the reason, and
        can write sentinel rows to maintain idempotency.

        Args:
            raws: Raw review texts.

        Returns:
            List aligned with ``raws``. Each entry is a dict with keys
            ``cleaned_text``, ``tokens``, ``char_count``, ``token_count``,
            ``dropped`` (bool), and ``drop_reason`` (str or None).
        """
        cleaned: list[str] = [self.clean_text(r) for r in raws]

        # Mark slots that are already too short so we don't waste spaCy on them.
        keep_indices: list[int] = [
            i for i, c in enumerate(cleaned) if len(c) >= MIN_CLEAN_TEXT_CHARS
        ]
        texts_to_tokenize: list[str] = [cleaned[i] for i in keep_indices]
        tokens_by_kept: list[list[str]] = self.tokenize_batch(texts_to_tokenize)

        # Pre-fill every slot as dropped-short; override kept slots below.
        results: list[dict] = [
            {
                "cleaned_text": "",
                "tokens": [],
                "char_count": 0,
                "token_count": 0,
                "dropped": True,
                "drop_reason": "too_short",
            }
            for _ in raws
        ]

        for kept_idx, orig_idx in enumerate(keep_indices):
            tokens = tokens_by_kept[kept_idx]
            cleaned_text = cleaned[orig_idx]
            if len(tokens) < MIN_TOKEN_COUNT:
                results[orig_idx] = {
                    "cleaned_text": cleaned_text,
                    "tokens": tokens,
                    "char_count": len(cleaned_text),
                    "token_count": len(tokens),
                    "dropped": True,
                    "drop_reason": "too_few_tokens",
                }
                continue
            results[orig_idx] = {
                "cleaned_text": cleaned_text,
                "tokens": tokens,
                "char_count": len(cleaned_text),
                "token_count": len(tokens),
                "dropped": False,
                "drop_reason": None,
            }
        return results


class LanguageFilter:
    """Detect review language with lingua, restricted to Steam's languages.

    Steam's ``language`` field is declared by the reviewer, so English pulls
    still contain other languages. Lingua is deterministic and handles short
    text; it is imported lazily so the dashboard never loads it.

    Attributes:
        detector: Lingua language detector.
        min_confidence: Confidence at which a non-English detection drops a review.
    """

    min_confidence: float

    def __init__(self, min_confidence: float = LANGUAGE_MIN_CONFIDENCE) -> None:
        """Build the detector over ``utils.DETECTOR_LANGUAGES``.

        Args:
            min_confidence: Confidence at which a non-English detection drops
                a review.
        """
        from lingua import Language, LanguageDetectorBuilder

        languages = [getattr(Language, name) for name in DETECTOR_LANGUAGES]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
        self.min_confidence = min_confidence

    def detect(self, text: str) -> tuple[str, float]:
        """Return the most likely language and its confidence.

        Args:
            text: Cleaned review text.

        Returns:
            ``(language_code, confidence)`` where the code is a lowercase
            lingua name such as ``english``. Empty text returns
            ``("unknown", 0.0)``.
        """
        values = self.detector.compute_language_confidence_values(text) if text else []
        if not values:
            return "unknown", 0.0
        top = values[0]
        return top.language.name.lower(), float(top.value)

    def is_english(self, lang: str, confidence: float) -> bool:
        """Return whether a review counts as English.

        A review is non-English only when its top language is not English
        and that detection reaches ``min_confidence``. Requiring English
        itself to reach the threshold dropped short English reviews: lingua
        spreads confidence across 26 languages on text like "Good strategy
        game. I love it." (see the Phase 7.1 language-filter note).

        Args:
            lang: Detected (top) language code.
            confidence: Confidence of that detection.

        Returns:
            False only for a confident non-English detection.
        """
        return lang == "english" or confidence < self.min_confidence


def apply_language_filter(
    results: list[dict],
    language_filter: LanguageFilter,
) -> list[tuple[str, float] | None]:
    """Turn kept non-English results into ``non_english`` sentinels in place.

    Only reviews that passed the length and token checks are examined, so the
    existing drop reasons keep their counts. A non-English review keeps its
    ``cleaned_text`` but gets empty tokens and ``token_count`` 0, which every
    query already excludes.

    Args:
        results: Output of ``Preprocessor.process_batch`` (mutated).
        language_filter: The detector.

    Returns:
        Per-result detection ``(lang, confidence)``, or None where the review
        was already dropped.
    """
    detections: list[tuple[str, float] | None] = []
    for result in results:
        if result["dropped"]:
            detections.append(None)
            continue
        lang, confidence = language_filter.detect(result["cleaned_text"])
        detections.append((lang, confidence))
        if not language_filter.is_english(lang, confidence):
            result.update(
                {"tokens": [], "token_count": 0, "dropped": True, "drop_reason": "non_english"}
            )
    return detections


def write_language_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, float]],
) -> None:
    """Upsert detected language and confidence into review_language.

    Args:
        conn: Active SQLite connection with study tables created.
        rows: ``(document_id, detected_lang, confidence)`` tuples.
    """
    conn.executemany(
        """
        INSERT INTO review_language (document_id, detected_lang, confidence)
        VALUES (?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            detected_lang = excluded.detected_lang,
            confidence = excluded.confidence
        """,
        rows,
    )


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def fetch_unprocessed_documents(
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> list[tuple[int, str]]:
    """Return `(document_id, review_text)` rows that have no processed row yet.

    Args:
        conn: Active SQLite connection.
        limit: Optional cap on number of rows returned. `None` means no limit.

    Returns:
        List of `(document_id, review_text)` tuples ordered by document id.
    """
    sql = """
        SELECT d.id, d.review_text
        FROM documents d
        LEFT JOIN processed_documents p ON p.document_id = d.id
        WHERE p.document_id IS NULL
        ORDER BY d.id
    """
    if limit is not None:
        sql += " LIMIT ?"
        return [(row[0], row[1]) for row in conn.execute(sql, (limit,)).fetchall()]
    return [(row[0], row[1]) for row in conn.execute(sql).fetchall()]


def write_processed_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, str, int, int]],
) -> int:
    """Bulk-insert processed rows and return the number actually inserted.

    The caller must serialize tokens via `json.dumps` before passing rows in
    — this function performs no serialization, only the SQL write. Uses
    `INSERT OR IGNORE` so re-runs against an already-populated table are
    safe.

    Args:
        conn: Active SQLite connection.
        rows: Tuples of `(document_id, cleaned_text, tokens_json,
            token_count, char_count)`.

    Returns:
        Number of rows actually inserted, counted via count-before /
        count-after (same pattern as `db_setup.insert_documents`).
    """
    if not rows:
        return 0

    count_before = conn.execute(
        "SELECT COUNT(*) FROM processed_documents"
    ).fetchone()[0]
    conn.executemany(
        """
        INSERT OR IGNORE INTO processed_documents
            (document_id, cleaned_text, tokens, token_count, char_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    count_after = conn.execute(
        "SELECT COUNT(*) FROM processed_documents"
    ).fetchone()[0]
    return count_after - count_before


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run(
    mode: str = "seed",
    rebuild: bool = False,
    batch_size: int = 200,
    study: Study | None = None,
    unfreeze: bool = False,
) -> None:
    """Run the full preprocessing pipeline against the chosen database.

    Steps:
        1. Open the seed.db or full.db connection.
        2. Ensure schema exists via `db_setup.create_tables` (single source
           of truth — this module never defines its own DDL).
        3. If `rebuild`, `DELETE FROM processed_documents` first.
        4. Fetch unprocessed rows, run `Preprocessor.process_batch` in
           batches, serialize tokens via `json.dumps`, and call
           `write_processed_rows`.
        5. Log kept/dropped counts and the reason each drop occurred.

    Args:
        mode: Either 'seed' or 'full'.
        rebuild: If True, wipe `processed_documents` before reprocessing.
            Combines freely with `mode`; `mode='full', rebuild=True` is a
            valid combo that wipes and reprocesses `full.db`.
        batch_size: Batch size for spaCy `nlp.pipe` and the fetch/write loop.
        study: Study whose databases to use. None means the April baseline.
            When the study sets ``language_filter``, non-English reviews
            become ``non_english`` sentinels and detections are written to
            ``review_language``.
        unfreeze: Allow writing a frozen (shipped) study's database.

    Raises:
        FrozenStudyError: If the database belongs to a frozen study and
            ``unfreeze`` is False.
        ValueError: If `mode` is not 'seed' or 'full'.
        FileNotFoundError: If the target database does not exist.
    """
    db_path = get_db_path(mode) if study is None else get_db_path(mode, study)
    guard_writable([db_path], unfreeze)
    use_language_filter = study is not None and study.language_filter
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. Run db_setup.py --full first."
        )

    logger.info("Preprocessing mode=%s db=%s rebuild=%s", mode, db_path, rebuild)

    conn = sqlite3.connect(db_path)
    try:
        if use_language_filter:
            create_study_tables(conn)
        else:
            create_tables(conn)

        if rebuild:
            deleted = conn.execute("DELETE FROM processed_documents").rowcount
            conn.commit()
            logger.info("Rebuild: deleted %d existing processed rows.", deleted)

        to_process = fetch_unprocessed_documents(conn)
        total = len(to_process)
        logger.info("Processing %d unprocessed documents from %s", total, db_path.name)

        if total == 0:
            logger.info("Nothing to do — all documents already processed.")
            return

        pre = Preprocessor()
        lang_filter = LanguageFilter() if use_language_filter else None

        kept = 0
        dropped_short_text = 0
        dropped_few_tokens = 0
        dropped_non_english = 0
        processed_so_far = 0

        for start in range(0, total, batch_size):
            batch = to_process[start : start + batch_size]
            raws = [row[1] for row in batch]
            ids = [row[0] for row in batch]

            results = pre.process_batch(raws)
            language_rows: list[tuple[int, str, float]] = []
            if lang_filter is not None:
                detections = apply_language_filter(results, lang_filter)
                language_rows = [
                    (doc_id, det[0], det[1])
                    for doc_id, det in zip(ids, detections)
                    if det is not None
                ]

            rows_to_write: list[tuple[int, str, str, int, int]] = []
            batch_kept = 0
            for doc_id, result in zip(ids, results):
                if result["dropped"]:
                    if result["drop_reason"] == "too_short":
                        dropped_short_text += 1
                    elif result["drop_reason"] == "non_english":
                        dropped_non_english += 1
                    else:
                        dropped_few_tokens += 1
                else:
                    batch_kept += 1
                rows_to_write.append((
                    doc_id,
                    result["cleaned_text"],
                    json.dumps(result["tokens"], ensure_ascii=False),
                    result["token_count"],
                    result["char_count"],
                ))

            conn.execute("BEGIN")
            write_processed_rows(conn, rows_to_write)
            if language_rows:
                write_language_rows(conn, language_rows)
            conn.commit()
            kept += batch_kept

            processed_so_far += len(batch)
            logger.info(
                "  batch %d/%d: kept=%d dropped=%d",
                processed_so_far,
                total,
                batch_kept,
                len(batch) - batch_kept,
            )

        logger.info(
            "Processed: kept %d, dropped %d (%d too short, %d too few tokens, %d non-English)",
            kept,
            dropped_short_text + dropped_few_tokens + dropped_non_english,
            dropped_short_text,
            dropped_few_tokens,
            dropped_non_english,
        )
        total_kept = conn.execute(
            "SELECT COUNT(*) FROM processed_documents WHERE token_count >= ?",
            (MIN_TOKEN_COUNT,),
        ).fetchone()[0]
        if total_kept == 0:
            logger.warning(
                "Zero reviews kept — preprocessing pipeline may be broken."
            )
    finally:
        conn.close()


def main() -> None:
    """CLI entry point. Parses `--full`, `--rebuild` and `--study` and calls `run`.

    All four combinations of the two flags are valid: there is no
    mutual-exclusion check.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess Steam reviews into processed_documents."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run against full.db. Default is seed.db.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete all processed_documents rows before reprocessing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="spaCy pipe batch size (default: 200).",
    )
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
        run(
            mode="full" if args.full else "seed",
            rebuild=args.rebuild,
            batch_size=args.batch_size,
            study=None if study.name == "april_2026_baseline" else study,
            unfreeze=args.unfreeze,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
