"""Phrase-lexicon aspect tagging for design themes.

Splits each kept English review into sentences with spaCy's rule-based
sentencizer, tags sentences against a versioned regex lexicon
(``studies/aspects_*.toml``) and writes matches to ``aspect_mentions``.
Topic modeling is the study's primary lens; aspects are a supplementary,
audited lens for named design questions.

Audit workflow: ``--export-audit`` writes an unlabeled CSV of test-split
matched sentences to ``outputs/audit/``. A separate labeler fills the
``label`` column. ``--import-labels`` then writes the committed label file
(``data/labels/aspect_audit_<version>.csv``) holding ids, aspect and label
only, since the sentence text already lives in the seed database.
"""

import argparse
import csv
import random
import re
import sqlite3
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import spacy

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.db_setup import create_study_tables
from src.stats import dev_test_split
from src.study import UNFREEZE_FLAG, UNFREEZE_HELP, Study, guard_writable, load_study
from src.utils import DATA_DIR, MIN_TOKEN_COUNT, OUTPUTS_DIR, get_logger

logger = get_logger(__name__)

AUDIT_PER_ASPECT: int = 40
"""Maximum test-split sentences exported per aspect for the precision audit."""

AUDIT_RANDOM_SEED: int = 20260926
"""Seed for drawing audit sentences."""

FULL_DB_AUDIT_ASPECTS: frozenset[str] = frozenset(
    {"turn_waiting", "first_strike", "initiative_clarity"}
)
"""Aspects whose audit sentences come from the full database's test split.

The seed's test split holds fewer than 30 matched sentences for these three
(7, 3 and 18 for strategy_v1), below the audit minimum. The orchestrator
approved drawing them from the full DB test split (2026-09-26) with the same
kept-review rules, tagging and sampling. The full DB is local, so re-drawing
these rows needs the cached pull.
"""

VALID_LABELS: frozenset[str] = frozenset({"relevant", "not_relevant"})
"""Labels the audit accepts."""

TURN_STRUCTURE_ASPECTS: tuple[str, ...] = (
    "turn_waiting",
    "first_strike",
    "initiative_clarity",
    "preview_reliability",
)
"""Aspects that describe the turn system itself."""

ANY_TURN_STRUCTURE: str = "turn_structure_any"
"""Derived theme: a review mentions at least one turn-structure aspect."""


@dataclass(frozen=True)
class Aspect:
    """One lexicon aspect.

    Attributes:
        name: Machine name, e.g. ``turn_waiting``.
        kind: ``design`` or ``control``.
        label: Human-readable label.
        include: Compiled include patterns (any match tags the sentence).
        exclude: Compiled exclude patterns (any match vetoes the sentence).
    """

    name: str
    kind: str
    label: str
    include: tuple[re.Pattern[str], ...]
    exclude: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class Lexicon:
    """A versioned set of aspects.

    Attributes:
        version: Lexicon version written with every mention.
        aspects: Aspects in file order.
    """

    version: str
    aspects: tuple[Aspect, ...]


@dataclass(frozen=True)
class AspectMatch:
    """One aspect tagged in one sentence.

    Attributes:
        sentence_idx: Index of the sentence in its review.
        aspect: Aspect name.
        matched_term: Text matched by the first include pattern that fired.
    """

    sentence_idx: int
    aspect: str
    matched_term: str


def load_lexicon(path: Path) -> Lexicon:
    """Load and compile an aspect lexicon TOML file.

    Args:
        path: Lexicon file.

    Returns:
        The compiled Lexicon.
    """
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    aspects = tuple(
        Aspect(
            name=a["name"],
            kind=a["kind"],
            label=a["label"],
            include=tuple(re.compile(p, re.IGNORECASE) for p in a["include"]),
            exclude=tuple(re.compile(p, re.IGNORECASE) for p in a.get("exclude", [])),
        )
        for a in raw["aspects"]
    )
    return Lexicon(version=raw["lexicon"]["version"], aspects=aspects)


def tag_sentences(sentences: list[str], lexicon: Lexicon) -> list[AspectMatch]:
    """Tag each sentence with every aspect whose patterns fire.

    A sentence gets an aspect when an include pattern matches and no exclude
    pattern of that aspect matches anywhere in the sentence.

    Args:
        sentences: Sentences of one review, in order.
        lexicon: The compiled lexicon.

    Returns:
        Matches ordered by sentence, then lexicon order.
    """
    matches: list[AspectMatch] = []
    for idx, sentence in enumerate(sentences):
        for aspect in lexicon.aspects:
            if any(p.search(sentence) for p in aspect.exclude):
                continue
            for pattern in aspect.include:
                found = pattern.search(sentence)
                if found:
                    matches.append(AspectMatch(idx, aspect.name, found.group(0)[:80]))
                    break
    return matches


class SentenceSplitter:
    """Rule-based sentence splitter (spaCy blank English plus sentencizer).

    Deterministic and fast; the parser is never loaded.

    Attributes:
        nlp: spaCy pipeline with only the sentencizer.
    """

    nlp: spacy.language.Language

    def __init__(self) -> None:
        """Build the blank pipeline with the sentencizer."""
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("sentencizer")

    def split(self, texts: list[str]) -> list[list[tuple[int, int, str]]]:
        """Split texts into sentences with character offsets.

        Args:
            texts: Cleaned review texts.

        Returns:
            One list per text of ``(char_start, char_end, sentence_text)``.
        """
        out: list[list[tuple[int, int, str]]] = []
        for doc in self.nlp.pipe(texts, batch_size=500):
            out.append([(s.start_char, s.end_char, s.text) for s in doc.sents])
        return out


def tagged_share(texts: list[str], lexicon: Lexicon, aspects: set[str]) -> tuple[int, int]:
    """Count texts tagged with at least one of the given aspects.

    Args:
        texts: Free texts (each split into sentences before tagging).
        lexicon: The compiled lexicon.
        aspects: Aspect names that count as a hit.

    Returns:
        ``(hits, total)``.
    """
    splitter = SentenceSplitter()
    hits = 0
    for spans in splitter.split(texts):
        found = tag_sentences([s[2] for s in spans], lexicon)
        if any(m.aspect in aspects for m in found):
            hits += 1
    return hits, len(texts)


# ---------------------------------------------------------------------------
# Database stage
# ---------------------------------------------------------------------------


def fetch_kept_documents(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """Return kept English reviews (sentinels, including non-English, excluded).

    Args:
        conn: Connection to a study seed database.

    Returns:
        ``(document_id, steam_review_id, cleaned_text)`` ordered by id.
    """
    return conn.execute(
        """
        SELECT p.document_id, d.steam_review_id, p.cleaned_text
        FROM processed_documents p
        JOIN documents d ON d.id = p.document_id
        WHERE p.token_count >= ?
        ORDER BY p.document_id
        """,
        (MIN_TOKEN_COUNT,),
    ).fetchall()


def run(study: Study, unfreeze: bool = False) -> int:
    """Tag every kept English review in the study seed and write aspect_mentions.

    Existing rows for the lexicon version are replaced, so reruns are
    idempotent.

    Args:
        study: A study with an ``aspect_lexicon``.
        unfreeze: Allow writing a frozen study's seed database.

    Returns:
        Number of aspect_mentions rows written.

    Raises:
        ValueError: If the study has no aspect lexicon.
        FrozenStudyError: If the study is frozen and ``unfreeze`` is False.
    """
    if study.aspect_lexicon is None:
        raise ValueError(f"Study {study.name} has no aspect_lexicon.")
    guard_writable([study.seed_db], unfreeze)
    lexicon = load_lexicon(study.aspect_lexicon)
    conn = sqlite3.connect(study.seed_db)
    try:
        create_study_tables(conn)
        docs = fetch_kept_documents(conn)
        splitter = SentenceSplitter()
        rows: list[tuple[int, str, int, str, int, int, str]] = []
        for (doc_id, _rid, _text), spans in zip(docs, splitter.split([d[2] for d in docs])):
            for m in tag_sentences([s[2] for s in spans], lexicon):
                start, end, _ = spans[m.sentence_idx]
                rows.append(
                    (doc_id, m.aspect, m.sentence_idx, m.matched_term, start, end, lexicon.version)
                )
        conn.execute("BEGIN")
        conn.execute("DELETE FROM aspect_mentions WHERE lexicon_version = ?", (lexicon.version,))
        conn.executemany(
            """
            INSERT INTO aspect_mentions
                (document_id, aspect, sentence_idx, matched_term, char_start, char_end,
                 lexicon_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        logger.info(
            "Tagged %d documents: %d aspect mentions (lexicon %s).",
            len(docs),
            len(rows),
            lexicon.version,
        )
        return len(rows)
    finally:
        conn.close()


def audit_paths(study: Study, lexicon: Lexicon) -> tuple[Path, Path]:
    """Return the unlabeled audit CSV path and the committed label path.

    Args:
        study: The study.
        lexicon: The lexicon being audited.

    Returns:
        ``(unlabeled_csv, committed_labels_csv)``.
    """
    return (
        OUTPUTS_DIR / "audit" / f"{study.name}_aspect_audit_{lexicon.version}_unlabeled.csv",
        DATA_DIR / "labels" / f"aspect_audit_{lexicon.version}.csv",
    )


def full_db_mentions(
    study: Study, lexicon: Lexicon, aspects: frozenset[str]
) -> list[tuple[str, str, int, str]]:
    """Tag the given aspects in kept English test-split reviews of the full DB.

    Applies the seed pipeline's rules to every full-DB review in the test
    split: the same cleaning, length and token checks and language filter
    (``preprocessing``), the same sentencizer and lexicon. Reviews whose
    cleaned text matches no include pattern of the aspects are skipped before
    the costly steps, which cannot change the result because a sentence match
    implies a match in the whole text.

    Args:
        study: The study (its ``full_db`` must exist).
        lexicon: The frozen lexicon.
        aspects: Aspect names to tag.

    Returns:
        ``(steam_review_id, aspect, sentence_idx, sentence)`` rows ordered by
        review ID, aspect and sentence index.
    """
    from src.preprocessing import LanguageFilter, Preprocessor, apply_language_filter

    sub = Lexicon(lexicon.version, tuple(a for a in lexicon.aspects if a.name in aspects))
    pre = Preprocessor()
    conn = sqlite3.connect(study.full_db)
    try:
        docs = conn.execute("SELECT steam_review_id, review_text FROM documents").fetchall()
    finally:
        conn.close()
    candidates: list[tuple[str, str]] = []
    for rid, raw in docs:
        if dev_test_split(rid) != "test":
            continue
        cleaned = pre.clean_text(raw)
        if any(p.search(cleaned) for a in sub.aspects for p in a.include):
            candidates.append((rid, raw))
    results = pre.process_batch([raw for _, raw in candidates])
    apply_language_filter(results, LanguageFilter())
    kept = [(rid, r["cleaned_text"]) for (rid, _), r in zip(candidates, results) if not r["dropped"]]
    rows: list[tuple[str, str, int, str]] = []
    splitter = SentenceSplitter()
    for (rid, _text), spans in zip(kept, splitter.split([t for _, t in kept])):
        for m in tag_sentences([sp[2] for sp in spans], sub):
            rows.append((rid, m.aspect, m.sentence_idx, spans[m.sentence_idx][2]))
    logger.info(
        "Full DB: %d test-split candidates, %d kept, %d mentions of %s.",
        len(candidates), len(kept), len(rows), sorted(aspects),
    )
    return sorted(rows)


def export_audit(study: Study, per_aspect: int = AUDIT_PER_ASPECT) -> Path:
    """Write test-split matched sentences for the precision audit.

    Draws up to ``per_aspect`` sentences per aspect from the test split,
    at most one sentence per review per aspect, in a seeded random order
    (one RNG per aspect, seeded from ``AUDIT_RANDOM_SEED`` and the aspect
    name). Aspects in ``FULL_DB_AUDIT_ASPECTS`` draw from the full database's
    test split; the rest from the seed's.

    Args:
        study: A study whose aspect stage has run.
        per_aspect: Maximum sentences per aspect.

    Returns:
        Path to the unlabeled CSV.
    """
    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    out_path, _ = audit_paths(study, lexicon)
    conn = sqlite3.connect(study.seed_db)
    try:
        rows = conn.execute(
            """
            SELECT d.steam_review_id, a.aspect, a.sentence_idx,
                   SUBSTR(p.cleaned_text, a.char_start + 1, a.char_end - a.char_start)
            FROM aspect_mentions a
            JOIN documents d ON d.id = a.document_id
            JOIN processed_documents p ON p.document_id = a.document_id
            WHERE a.lexicon_version = ?
            ORDER BY d.steam_review_id, a.aspect, a.sentence_idx
            """,
            (lexicon.version,),
        ).fetchall()
    finally:
        conn.close()
    rows = [r for r in rows if r[1] not in FULL_DB_AUDIT_ASPECTS]
    rows += full_db_mentions(study, lexicon, FULL_DB_AUDIT_ASPECTS)

    chosen: list[tuple[str, str, int, str]] = []
    for aspect in lexicon.aspects:
        seen: set[str] = set()
        pool = []
        for rid, name, idx, text in rows:
            if name == aspect.name and dev_test_split(rid) == "test" and rid not in seen:
                seen.add(rid)
                pool.append((rid, name, idx, text))
        random.Random(f"{AUDIT_RANDOM_SEED}:{aspect.name}").shuffle(pool)
        chosen.extend(pool[:per_aspect])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {a.name: a.label for a in lexicon.aspects}
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["audit_id", "steam_review_id", "sentence_idx", "aspect", "aspect_label",
             "sentence", "label"]
        )
        for i, (rid, name, idx, text) in enumerate(chosen, start=1):
            writer.writerow([i, rid, idx, name, labels[name], text, ""])
    logger.info("Wrote %d audit sentences to %s", len(chosen), out_path)
    return out_path


def import_labels(study: Study, labeled_csv: Path) -> Path:
    """Validate a labeled audit CSV and write the committed label file.

    The committed file holds ``steam_review_id, sentence_idx, aspect, label``
    only. Every row must carry a label in ``VALID_LABELS``.

    Args:
        study: The study.
        labeled_csv: The audit CSV with its ``label`` column filled.

    Returns:
        Path to the committed label file.

    Raises:
        ValueError: If any row is unlabeled or carries an unknown label.
    """
    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    _, out_path = audit_paths(study, lexicon)
    with open(labeled_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["audit_id"] for r in rows if r["label"].strip() not in VALID_LABELS]
    if bad:
        raise ValueError(f"{len(bad)} rows lack a valid label, e.g. audit_id {bad[:5]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["steam_review_id", "sentence_idx", "aspect", "label"])
        for r in rows:
            writer.writerow([r["steam_review_id"], r["sentence_idx"], r["aspect"],
                             r["label"].strip()])
    logger.info("Wrote %d labels to %s", len(rows), out_path)
    return out_path


def main() -> None:
    """CLI entry point: tag, export the audit set, or import labels."""
    parser = argparse.ArgumentParser(description="Tag design aspects in study reviews.")
    parser.add_argument("--study", type=Path, required=True, help="Study TOML file.")
    parser.add_argument(
        "--export-audit", action="store_true", help="Write the unlabeled test-split audit CSV."
    )
    parser.add_argument(
        "--import-labels", type=Path, default=None, help="Labeled audit CSV to import."
    )
    parser.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    args = parser.parse_args()
    study = load_study(args.study)
    try:
        if args.import_labels is not None:
            import_labels(study, args.import_labels)
        elif args.export_audit:
            export_audit(study)
        else:
            run(study, unfreeze=args.unfreeze)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
