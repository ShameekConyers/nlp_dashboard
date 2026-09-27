"""Transformer sentiment scoring and its held-out validation (Decision 11).

Scores kept English reviews (``unit="doc"``) and aspect sentences
(``unit="sentence"``) with pinned Hugging Face sentiment models and writes
the positive-class probability to ``sentiment_scores``. Every row carries a
``model_id`` that joins to the model name and revision in ``sentiment_models``. Long inputs keep their first ``HEAD_TOKENS`` and
last ``TAIL_TOKENS`` tokens, since strategy reviews often put the verdict at
the end.

Validation follows the pre-registered rule. On the market seed's dev split,
VADER's compound threshold is tuned for balanced accuracy and the candidate
with the higher balanced accuracy is selected. The test split is then read
once for every metric, and the selected candidate is adopted only if the
paired-bootstrap 95% interval of its balanced-accuracy gain over VADER lies
above 0 at both VADER thresholds. The April seed is scored for continuity
with the same thresholds and never used for tuning.

``torch`` and ``transformers`` are imported only when a model loads, so the
dashboard and the tests never import them.
"""

import argparse
import csv
import json
import random
import sqlite3
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.db_setup import create_scores_tables
from src.stats import BOOT_SEED, N_BOOT, _git_sha, _num, balanced_accuracy, dev_test_split
from src.study import Study, load_study
from src.utils import (
    DATA_DIR,
    HEAD_TOKENS,
    MIN_TOKEN_COUNT,
    TAIL_TOKENS,
    TRANSFORMER_CANDIDATES,
    TRANSFORMER_POSITIVE_CUTOFF,
    VADER_POSITIVE_THRESHOLD,
    get_logger,
)
from src.validation import Predictor, bootstrap_validation, macro_f1, macro_f1_multiclass, tune_threshold

logger = get_logger(__name__)

DOC_SENTENCE_IDX: int = -1
"""``sentence_idx`` stored for document-level rows."""

POSITIVE_CUTOFF: float = TRANSFORMER_POSITIVE_CUTOFF
"""A transformer predicts positive when its positive probability is at least this (argmax)."""

WRITE_CHUNK: int = 512
"""Inputs scored and committed per chunk, so an interrupted run resumes."""

SHORT_NAMES: dict[str, str] = {
    "siebert/sentiment-roberta-large-english": "roberta_large",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english": "distilbert_sst2",
}
"""Short keys used in metrics JSON."""

VADER_DEFAULT_KEY: str = "vader_0.05"
VADER_TUNED_KEY: str = "vader_dev_tuned"

ADOPTION_RULE: str = (
    "Select the candidate with the higher dev-split balanced accuracy. Adopt it only if the "
    "review-level paired-bootstrap 95% interval of (candidate - VADER) balanced accuracy on "
    "the market test split lies above 0 against VADER at 0.05 and at the dev-tuned threshold."
)

SENTENCE_AUDIT_VERSION: str = "v1"
SENTENCE_AUDIT_SIZE: int = 100
SENTENCE_AUDIT_SEED: int = 20260927
SENTENCE_LABELS: tuple[str, ...] = ("positive", "negative", "neither")
SENTENCE_GATE_MACRO_F1: float = 0.70
"""Sentence sentiment is shown in the dashboard only at or above this audit macro-F1."""

SENTENCE_NEITHER_BAND: tuple[float, float] = (0.2, 0.8)
"""Fixed before any sentence label existed: a sentence scores ``neither`` when its
positive probability lies strictly inside this band, ``positive`` at or above the
upper bound and ``negative`` at or below the lower bound. A binary model needs a
band to predict ``neither`` at all."""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def head_tail(ids: Sequence[int], head: int = HEAD_TOKENS, tail: int = TAIL_TOKENS) -> list[int]:
    """Keep the first ``head`` and last ``tail`` token ids of a long sequence.

    Args:
        ids: Token ids without special tokens.
        head: Tokens kept from the start.
        tail: Tokens kept from the end.

    Returns:
        ``ids`` unchanged when it fits in ``head + tail``, else head plus tail.
    """
    if len(ids) <= head + tail:
        return list(ids)
    return list(ids[:head]) + list(ids[len(ids) - tail:])


def pick_device() -> str:
    """Return ``mps`` when Apple's GPU backend is available, else ``cpu``.

    Returns:
        A torch device string.
    """
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class TransformerScorer:
    """Positive-class probability from a pinned sequence-classification model.

    The model loads on first use. ``score`` sorts inputs by length to batch
    similar lengths together, then returns results in input order.

    Attributes:
        model_name: Hugging Face model id.
        revision: Pinned commit hash.
        batch_size: Inputs per forward pass.
        device: Torch device, chosen on load when None.
    """

    def __init__(
        self, model_name: str, revision: str, batch_size: int = 16, device: str | None = None
    ) -> None:
        """Store the model identity; nothing is downloaded until first use.

        Args:
            model_name: Hugging Face model id.
            revision: Pinned commit hash.
            batch_size: Inputs per forward pass.
            device: Torch device, or None to pick MPS or CPU.
        """
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._pos_index: int = 1

    def _load(self) -> None:
        """Download (or reuse the cache) and load the tokenizer and model."""
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or pick_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self._tokenizer.model_max_length = 10**9  # truncation is done by head_tail
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, revision=self.revision
        )
        self._model = model.to(self.device).eval()
        labels = {v.upper(): int(k) for k, v in model.config.id2label.items()}
        self._pos_index = labels["POSITIVE"]
        tok = self._tokenizer
        probe = "A probe sentence."
        plain = tok(probe, add_special_tokens=False)["input_ids"]
        if tok(probe)["input_ids"] != [tok.cls_token_id, *plain, tok.sep_token_id]:
            raise ValueError(f"{self.model_name}: unexpected special-token layout.")
        logger.info("Loaded %s@%s on %s", self.model_name, self.revision[:8], self.device)

    def encode(self, text: str) -> list[int]:
        """Tokenize one text with head+tail truncation and special tokens.

        Args:
            text: Input text.

        Returns:
            Model input ids, at most ``HEAD_TOKENS + TAIL_TOKENS`` plus specials.
        """
        self._load()
        tok = self._tokenizer
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return [tok.cls_token_id, *head_tail(ids), tok.sep_token_id]

    def _forward(self, batch: list[list[int]]) -> list[float]:
        """Run one padded batch and return positive probabilities.

        Args:
            batch: Encoded inputs.

        Returns:
            Positive-class probability per input.
        """
        import torch

        padded = self._tokenizer.pad({"input_ids": batch}, return_tensors="pt")
        padded = {k: v.to(self.device) for k, v in padded.items()}
        with torch.inference_mode():
            logits = self._model(**padded).logits
        probs = torch.softmax(logits.float(), dim=-1)[:, self._pos_index]
        return probs.cpu().tolist()

    def score(self, texts: Sequence[str]) -> list[float]:
        """Score texts, batching by length, and return results in input order.

        Args:
            texts: Input texts.

        Returns:
            Positive-class probability per text, aligned with ``texts``.
        """
        encoded = [self.encode(t) for t in texts]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]), reverse=True)
        out = [0.0] * len(encoded)
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            for i, p in zip(chunk, self._forward([encoded[i] for i in chunk])):
                out[i] = float(p)
        return out


# ---------------------------------------------------------------------------
# Database stage
# ---------------------------------------------------------------------------


def _read_only(path: Path) -> sqlite3.Connection:
    """Open a SQLite file read-only, so a frozen seed cannot change.

    Args:
        path: Database file.

    Returns:
        A read-only connection.
    """
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def model_id(conn: sqlite3.Connection, model_name: str, revision: str) -> int:
    """Return the ``sentiment_models`` id for a model and revision, adding it if new.

    Args:
        conn: Connection to a scores database.
        model_name: Hugging Face model id.
        revision: Pinned commit hash.

    Returns:
        The integer model id.
    """
    conn.execute(
        "INSERT OR IGNORE INTO sentiment_models (model_name, model_revision) VALUES (?, ?)",
        (model_name, revision),
    )
    return int(conn.execute(
        "SELECT model_id FROM sentiment_models WHERE model_name = ? AND model_revision = ?",
        (model_name, revision),
    ).fetchone()[0])


def model_scores(
    conn: sqlite3.Connection, model_name: str, revision: str, unit: str
) -> dict[tuple[int, int], float]:
    """Return one model's stored probabilities for a unit.

    Args:
        conn: Connection to a scores database.
        model_name: Hugging Face model id.
        revision: Pinned commit hash.
        unit: ``doc`` or ``sentence``.

    Returns:
        ``{(document_id, sentence_idx): positive_prob}``.
    """
    rows = conn.execute(
        """
        SELECT s.document_id, s.sentence_idx, s.positive_prob
        FROM sentiment_scores s
        JOIN sentiment_models m ON m.model_id = s.model_id
        WHERE m.model_name = ? AND m.model_revision = ? AND s.unit = ?
        """,
        (model_name, revision, unit),
    ).fetchall()
    return {(d, i): p for d, i, p in rows}


def fetch_doc_inputs(conn: sqlite3.Connection) -> list[tuple[int, int, str]]:
    """Return kept English reviews as scoring inputs.

    Args:
        conn: Study seed connection.

    Returns:
        ``(document_id, DOC_SENTENCE_IDX, cleaned_text)`` ordered by id.
    """
    rows = conn.execute(
        "SELECT document_id, cleaned_text FROM processed_documents WHERE token_count >= ? "
        "ORDER BY document_id",
        (MIN_TOKEN_COUNT,),
    ).fetchall()
    return [(d, DOC_SENTENCE_IDX, t) for d, t in rows]


def fetch_sentence_inputs(conn: sqlite3.Connection, lexicon_version: str) -> list[tuple[int, int, str]]:
    """Return each distinct aspect-tagged sentence once.

    Args:
        conn: Study seed connection.
        lexicon_version: Lexicon version of the mentions.

    Returns:
        ``(document_id, sentence_idx, sentence_text)`` ordered by id and index.
    """
    return conn.execute(
        """
        SELECT DISTINCT a.document_id, a.sentence_idx,
               SUBSTR(p.cleaned_text, a.char_start + 1, a.char_end - a.char_start)
        FROM aspect_mentions a
        JOIN processed_documents p ON p.document_id = a.document_id
        WHERE a.lexicon_version = ?
        ORDER BY a.document_id, a.sentence_idx
        """,
        (lexicon_version,),
    ).fetchall()


def score_study(
    study: Study,
    scorer: TransformerScorer,
    unit: str,
    lexicon_version: str | None = None,
) -> dict[str, Any]:
    """Score one unit of a study with one model and store the probabilities.

    Rows already stored for the same model, revision and unit are skipped, so
    reruns resume. Each chunk is committed on its own.

    Args:
        study: The study; inputs come from ``seed_db`` (opened read-only) and
            scores go to ``study.scores_path``.
        scorer: The model.
        unit: ``doc`` or ``sentence``.
        lexicon_version: Required for ``sentence``.

    Returns:
        Timing and counts, also stored as an ``analysis_runs`` row.
    """
    src = _read_only(study.seed_db)
    try:
        inputs = (
            fetch_doc_inputs(src) if unit == "doc"
            else fetch_sentence_inputs(src, lexicon_version or "")
        )
    finally:
        src.close()
    out = sqlite3.connect(study.scores_path)
    try:
        create_scores_tables(out)
        mid = model_id(out, scorer.model_name, scorer.revision)
        out.commit()
        done = set(model_scores(out, scorer.model_name, scorer.revision, unit))
        todo = [row for row in inputs if (row[0], row[1]) not in done]
        logger.info("%s %s %s: %d inputs, %d to score", study.name, scorer.model_name, unit,
                    len(inputs), len(todo))
        t0 = time.perf_counter()
        for start in range(0, len(todo), WRITE_CHUNK):
            chunk = todo[start:start + WRITE_CHUNK]
            probs = scorer.score([r[2] for r in chunk])
            out.executemany(
                "INSERT OR REPLACE INTO sentiment_scores (document_id, model_id, unit,"
                " sentence_idx, positive_prob) VALUES (?, ?, ?, ?, ?)",
                [(r[0], mid, unit, r[1], p) for r, p in zip(chunk, probs)],
            )
            out.commit()
            logger.info("  %d/%d", min(start + WRITE_CHUNK, len(todo)), len(todo))
        seconds = time.perf_counter() - t0
        metrics = {
            "n_inputs": len(inputs),
            "n_scored": len(todo),
            "n_skipped": len(inputs) - len(todo),
            "seconds": round(seconds, 1),
            "inputs_per_second": round(len(todo) / seconds, 2) if todo and seconds else None,
        }
        params = {
            "study": study.name, "model_name": scorer.model_name, "model_revision": scorer.revision,
            "unit": unit, "device": scorer.device, "batch_size": scorer.batch_size,
            "head_tokens": HEAD_TOKENS, "tail_tokens": TAIL_TOKENS,
            "lexicon_version": lexicon_version,
        }
        if todo:
            out.execute(
                "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
                ("transformer_scoring", json.dumps(params), _git_sha(), json.dumps(metrics)),
            )
            out.commit()
        return metrics
    finally:
        out.close()


# ---------------------------------------------------------------------------
# Validation (Decision 11)
# ---------------------------------------------------------------------------


def load_validation_frame(study: Study) -> dict[str, np.ndarray]:
    """Load kept reviews with labels, VADER compound and every candidate's probability.

    Args:
        study: The study.

    Returns:
        Arrays keyed ``document_id``, ``steam_review_id``, ``app_id``,
        ``genre``, ``recommended``, ``compound`` and one per candidate short name.

    Raises:
        ValueError: If a kept review lacks a candidate's document score.
    """
    src = _read_only(study.seed_db)
    try:
        rows = src.execute(
            """
            SELECT d.id, d.steam_review_id, d.app_id, m.genre, d.recommended, n.sentiment_compound
            FROM documents d
            JOIN processed_documents p ON p.document_id = d.id
            JOIN nlp_results n ON n.document_id = d.id
            JOIN metadata m ON m.app_id = d.app_id
            WHERE p.token_count >= ?
            ORDER BY d.id
            """,
            (MIN_TOKEN_COUNT,),
        ).fetchall()
    finally:
        src.close()
    frame: dict[str, np.ndarray] = {
        "document_id": np.array([r[0] for r in rows]),
        "steam_review_id": np.array([r[1] for r in rows]),
        "app_id": np.array([r[2] for r in rows]),
        "genre": np.array([r[3] for r in rows]),
        "recommended": np.array([bool(r[4]) for r in rows]),
        "compound": np.array([float(r[5]) for r in rows]),
    }
    scores = _read_only(study.scores_path)
    try:
        for name, revision in TRANSFORMER_CANDIDATES.items():
            got = {d: p for (d, _), p in model_scores(scores, name, revision, "doc").items()}
            missing = [int(d) for d in frame["document_id"] if int(d) not in got]
            if missing:
                raise ValueError(f"{study.name}: {len(missing)} kept reviews lack {name} scores.")
            frame[SHORT_NAMES[name]] = np.array([got[int(d)] for d in frame["document_id"]])
    finally:
        scores.close()
    return frame


def _predictors(frame: dict[str, np.ndarray], mask: np.ndarray, tuned: float) -> dict[str, Predictor]:
    """Build VADER (two thresholds) and candidate predictors for a subset.

    Args:
        frame: Output of ``load_validation_frame``.
        mask: Rows to include.
        tuned: Dev-tuned VADER threshold.

    Returns:
        ``{name: Predictor}``.
    """
    comp = frame["compound"][mask]
    preds = {
        VADER_DEFAULT_KEY: Predictor(comp >= VADER_POSITIVE_THRESHOLD, comp),
        VADER_TUNED_KEY: Predictor(comp >= tuned, comp),
    }
    for short in SHORT_NAMES.values():
        prob = frame[short][mask]
        preds[short] = Predictor(prob >= POSITIVE_CUTOFF, prob)
    return preds


def validate(market: Study, april: Study, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict[str, Any]:
    """Run the Decision 11 selection, test read and adoption verdict.

    Dev-split work (VADER threshold, candidate selection) finishes before any
    test-split value is computed. Results are stored as a
    ``sentiment_validation`` run in the market seed and the April scores DB.

    Args:
        market: The market study (selection and verdict).
        april: The April baseline (continuity only).
        n_boot: Bootstrap resamples.
        seed: Random seed.

    Returns:
        The metrics dictionary that was stored.
    """
    mk = load_validation_frame(market)
    split = np.array([dev_test_split(r) for r in mk["steam_review_id"]])
    dev, test = split == "dev", split == "test"
    y_dev = mk["recommended"][dev]

    tuned, tuned_dev_ba = tune_threshold(y_dev, mk["compound"][dev])
    dev_ba = {VADER_DEFAULT_KEY: balanced_accuracy(y_dev, mk["compound"][dev] >= VADER_POSITIVE_THRESHOLD),
              VADER_TUNED_KEY: tuned_dev_ba}
    for short in SHORT_NAMES.values():
        dev_ba[short] = balanced_accuracy(y_dev, mk[short][dev] >= POSITIVE_CUTOFF)
    order = [SHORT_NAMES[n] for n in TRANSFORMER_CANDIDATES]
    selected = max(order, key=lambda k: (dev_ba[k], -order.index(k)))
    other = [k for k in order if k != selected][0]
    logger.info("Dev: VADER tuned threshold %.2f; dev BA %s; selected %s", tuned, dev_ba, selected)

    pairs = [(selected, VADER_TUNED_KEY), (selected, VADER_DEFAULT_KEY),
             (other, VADER_TUNED_KEY), (other, VADER_DEFAULT_KEY), (selected, other)]
    y_test = mk["recommended"][test]
    market_test = bootstrap_validation(y_test, _predictors(mk, test, tuned), pairs, n_boot, seed)
    market_cluster = bootstrap_validation(
        y_test, _predictors(mk, test, tuned), pairs, n_boot, seed, clusters=mk["app_id"][test]
    )["differences"]
    lows = [market_test["differences"][f"{selected}_minus_{v}"]["ci"][0]
            for v in (VADER_TUNED_KEY, VADER_DEFAULT_KEY)]
    adopted = all(lo is not None and lo > 0 for lo in lows)

    ap = load_validation_frame(april)
    everything = np.ones(ap["recommended"].size, dtype=bool)
    strategy = ap["genre"] == "Strategy"
    april_all = bootstrap_validation(ap["recommended"], _predictors(ap, everything, tuned), pairs,
                                     n_boot, seed)
    april_strategy = bootstrap_validation(ap["recommended"][strategy],
                                          _predictors(ap, strategy, tuned), pairs, n_boot, seed)

    metrics = {
        "adoption_rule": ADOPTION_RULE,
        "models": {SHORT_NAMES[n]: {"model_name": n, "revision": r}
                   for n, r in TRANSFORMER_CANDIDATES.items()},
        "dev": {"n": int(dev.sum()), "vader_tuned_threshold": tuned, "balanced_accuracy": dev_ba},
        "selected": selected,
        "adopted": adopted,
        "recommended_scorer": selected if adopted else "vader",
        "market_test": market_test,
        "market_test_cluster_differences": market_cluster,
        "april_all": april_all,
        "april_strategy": april_strategy,
    }
    params = {"market_study": market.name, "april_study": april.name, "n_boot": n_boot,
              "seed": seed, "positive_cutoff": POSITIVE_CUTOFF,
              "vader_default_threshold": VADER_POSITIVE_THRESHOLD,
              "bootstrap": "review-level paired percentile; game-cluster two-stage as sensitivity",
              "april_scope": "all kept reviews of the balanced April seed; Strategy genre subset"}
    for path in dict.fromkeys([market.scores_path, april.scores_path]):
        conn = sqlite3.connect(path)
        try:
            create_scores_tables(conn)
            conn.execute(
                "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
                ("sentiment_validation", json.dumps(params), _git_sha(),
                 json.dumps(metrics, default=_json_default)),
            )
            conn.commit()
        finally:
            conn.close()
    return metrics


def _json_default(x: Any) -> Any:
    """Serialize numpy scalars and non-finite floats for JSON.

    Args:
        x: Value json cannot encode natively.

    Returns:
        A JSON-safe value.
    """
    if isinstance(x, np.generic):
        return _num(float(x)) if isinstance(x, np.floating) else x.item()
    raise TypeError(f"Not JSON serializable: {type(x)}")


def latest_validation(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """Return the metrics of the newest ``sentiment_validation`` run.

    Args:
        conn: Connection to a scores database.

    Returns:
        The metrics dictionary, or None if no run exists.
    """
    row = conn.execute(
        "SELECT metrics FROM analysis_runs WHERE stage = 'sentiment_validation' "
        "ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    return json.loads(row[0]) if row else None


# ---------------------------------------------------------------------------
# Sentence audit
# ---------------------------------------------------------------------------


def sentence_audit_paths() -> tuple[Path, Path]:
    """Return the unlabeled export and the committed label file paths.

    Returns:
        ``(unlabeled_csv, labeled_csv)`` under ``data/labels/``.
    """
    base = DATA_DIR / "labels"
    return (base / f"sentence_sentiment_audit_{SENTENCE_AUDIT_VERSION}_unlabeled.csv",
            base / f"sentence_sentiment_audit_{SENTENCE_AUDIT_VERSION}.csv")


def _passing_aspects(study: Study) -> list[str]:
    """Return the aspects that passed the precision audit, sorted by name.

    Args:
        study: A study with an aspect lexicon and imported audit labels.

    Returns:
        Aspect names.
    """
    from src.aspects import audit_paths, load_lexicon
    from src.stats import load_audit_precision

    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    precision = load_audit_precision(audit_paths(study, lexicon)[1])
    return sorted(a for a, p in precision.items() if p["passes"])


def allocate(total: int, strata: Sequence[str]) -> dict[str, int]:
    """Split ``total`` as evenly as possible, extra units to the first strata.

    Args:
        total: Items to allocate.
        strata: Strata in priority order.

    Returns:
        ``{stratum: count}``.
    """
    base, extra = divmod(total, len(strata))
    return {s: base + (1 if i < extra else 0) for i, s in enumerate(strata)}


def export_sentence_audit(
    study: Study, n: int = SENTENCE_AUDIT_SIZE, seed: int = SENTENCE_AUDIT_SEED
) -> Path:
    """Write the unlabeled sentence-sentiment audit set.

    Draws ``n`` distinct test-split aspect sentences, stratified evenly across
    aspects that passed the precision audit, each sentence used once. No
    scores appear in the file.

    Args:
        study: The market study.
        n: Sentences to draw.
        seed: Sampling seed.

    Returns:
        Path of the unlabeled CSV.

    Raises:
        ValueError: If an aspect has too few test-split sentences.
    """
    from src.aspects import load_lexicon

    assert study.aspect_lexicon is not None
    version = load_lexicon(study.aspect_lexicon).version
    aspects = _passing_aspects(study)
    quotas = allocate(n, aspects)
    conn = _read_only(study.seed_db)
    try:
        rows = conn.execute(
            """
            SELECT d.steam_review_id, a.aspect, a.sentence_idx, a.document_id,
                   SUBSTR(p.cleaned_text, a.char_start + 1, a.char_end - a.char_start)
            FROM aspect_mentions a
            JOIN documents d ON d.id = a.document_id
            JOIN processed_documents p ON p.document_id = a.document_id
            WHERE a.lexicon_version = ?
            ORDER BY d.steam_review_id, a.aspect, a.sentence_idx
            """,
            (version,),
        ).fetchall()
    finally:
        conn.close()
    by_aspect: dict[str, list[tuple[str, int, str]]] = {}
    for rid, aspect, idx, _doc, text in rows:
        if aspect in quotas and dev_test_split(rid) == "test":
            by_aspect.setdefault(aspect, []).append((rid, idx, text))
    used: set[tuple[str, int]] = set()
    picked: list[tuple[str, int, str, str]] = []
    for aspect in aspects:
        pool = by_aspect.get(aspect, [])
        random.Random(f"{seed}:{aspect}").shuffle(pool)
        take = [r for r in pool if (r[0], r[1]) not in used][: quotas[aspect]]
        if len(take) < quotas[aspect]:
            raise ValueError(f"{aspect}: {len(take)} test sentences, need {quotas[aspect]}.")
        used.update((r[0], r[1]) for r in take)
        picked.extend((r[0], r[1], aspect, r[2]) for r in take)
    random.Random(seed).shuffle(picked)
    out, _ = sentence_audit_paths()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "steam_review_id", "sentence_idx", "aspect", "sentence", "label"])
        for i, (rid, idx, aspect, text) in enumerate(picked, 1):
            w.writerow([i, rid, idx, aspect, text, ""])
    logger.info("Wrote %d audit sentences to %s (%s)", len(picked), out, quotas)
    return out


def import_sentence_labels(labeled_csv: Path) -> Path:
    """Validate a labeled audit file and write the committed label file.

    The committed file keeps ids, aspect and label; sentence text stays in
    the seed database and the unlabeled export.

    Args:
        labeled_csv: The labeler's filled copy of the export.

    Returns:
        Path of the committed label file.

    Raises:
        ValueError: If ids differ from the export or a label is invalid.
    """
    unlabeled, committed = sentence_audit_paths()
    with open(unlabeled, encoding="utf-8", newline="") as f:
        expected = {r["audit_id"]: r for r in csv.DictReader(f)}
    with open(labeled_csv, encoding="utf-8", newline="") as f:
        got = list(csv.DictReader(f))
    if {r["audit_id"] for r in got} != set(expected) or len(got) != len(expected):
        raise ValueError("Labeled file rows do not match the export.")
    for r in got:
        e = expected[r["audit_id"]]
        if (r["steam_review_id"], r["sentence_idx"]) != (e["steam_review_id"], e["sentence_idx"]):
            raise ValueError(f"audit_id {r['audit_id']} ids changed.")
        if r["label"].strip().lower() not in SENTENCE_LABELS:
            raise ValueError(f"audit_id {r['audit_id']}: invalid label {r['label']!r}.")
    with open(committed, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "steam_review_id", "sentence_idx", "aspect", "label"])
        for r in sorted(got, key=lambda r: int(r["audit_id"])):
            w.writerow([r["audit_id"], r["steam_review_id"], r["sentence_idx"], r["aspect"],
                        r["label"].strip().lower()])
    return committed


def sentence_class(prob: float, band: tuple[float, float] = SENTENCE_NEITHER_BAND) -> str:
    """Map a positive probability to positive, negative or neither.

    Args:
        prob: Positive-class probability.
        band: ``(low, high)``; strictly inside is ``neither``.

    Returns:
        The class name.
    """
    if prob >= band[1]:
        return "positive"
    if prob <= band[0]:
        return "negative"
    return "neither"


def sentence_audit_metrics(study: Study, model_name: str) -> dict[str, Any]:
    """Score the committed sentence labels against one model's sentence scores.

    Args:
        study: The market study.
        model_name: Hugging Face id of the scored model.

    Returns:
        Macro-F1 over three classes (the gate), binary macro-F1 on sentences
        labelled positive or negative, the confusion counts and whether the
        gate passed. Also stored as a ``sentence_audit`` run.
    """
    _, committed = sentence_audit_paths()
    with open(committed, encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    conn = sqlite3.connect(study.scores_path)
    try:
        ids = dict(conn.execute("SELECT steam_review_id, id FROM documents").fetchall())
        revision = TRANSFORMER_CANDIDATES[model_name]
        prob = model_scores(conn, model_name, revision, "sentence")
        y_true = [r["label"] for r in labels]
        probs = [prob[(ids[r["steam_review_id"]], int(r["sentence_idx"]))] for r in labels]
        y_pred = [sentence_class(p) for p in probs]
        binary = [(t == "positive", p >= POSITIVE_CUTOFF) for t, p in zip(y_true, probs)
                  if t != "neither"]
        confusion = {t: {p: sum(1 for a, b in zip(y_true, y_pred) if a == t and b == p)
                         for p in SENTENCE_LABELS} for t in SENTENCE_LABELS}
        f1 = macro_f1_multiclass(y_true, y_pred, SENTENCE_LABELS)
        metrics = {
            "model_name": model_name, "model_revision": revision, "n": len(labels),
            "label_counts": {c: y_true.count(c) for c in SENTENCE_LABELS},
            "neither_band": list(SENTENCE_NEITHER_BAND),
            "macro_f1_3class": f1,
            "binary_n": len(binary),
            "binary_macro_f1": macro_f1([b[0] for b in binary], [b[1] for b in binary])
            if binary else None,
            "confusion": confusion,
            "gate": SENTENCE_GATE_MACRO_F1,
            "passes": f1 >= SENTENCE_GATE_MACRO_F1,
        }
        conn.execute(
            "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
            ("sentence_audit", json.dumps({"study": study.name, "labels": committed.name}),
             _git_sha(), json.dumps(metrics)),
        )
        conn.commit()
        return metrics
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reference check (Decision 14)
# ---------------------------------------------------------------------------


def reference_sentiment_check(
    study: Study,
    reference_path: Path,
    scorer_factory: Callable[[str, str], TransformerScorer] = TransformerScorer,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Score hand-coded praise and complaint mentions and store derived numbers only.

    Reads the private research file at run time and never copies its text.
    Mixed and neutral mentions are excluded.

    Args:
        study: The market study (its apps select the mentions).
        reference_path: Private research JSON.
        scorer_factory: Builds a scorer from ``(model_name, revision)``.
        n_boot: Bootstrap resamples for the balanced-accuracy interval.

    Returns:
        Per-scorer balanced accuracy with interval and n, stored as a
        ``reference_sentiment_check`` run.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    with open(reference_path, encoding="utf-8") as f:
        table = json.load(f)["result"]["table"]
    study_ids = {a.app_id for a in study.apps}
    quotes: list[str] = []
    praise: list[bool] = []
    for row in table:
        app = (row.get("metrics") or {}).get("steamAppId")
        if not app or int(app) not in study_ids:
            continue
        for m in row.get("mentions", []):
            if m.get("sentiment") in ("praise", "complaint") and m.get("quote"):
                quotes.append(m["quote"])
                praise.append(m["sentiment"] == "praise")
    y = np.array(praise)
    vader = SentimentIntensityAnalyzer()
    comp = np.array([vader.polarity_scores(q)["compound"] for q in quotes])
    predictors = {VADER_DEFAULT_KEY: Predictor(comp >= VADER_POSITIVE_THRESHOLD, comp)}
    for name, rev in TRANSFORMER_CANDIDATES.items():
        prob = np.array(scorer_factory(name, rev).score(quotes))
        predictors[SHORT_NAMES[name]] = Predictor(prob >= POSITIVE_CUTOFF, prob)
    result = bootstrap_validation(y, predictors, [], n_boot, BOOT_SEED)
    metrics = {
        "n": int(y.size), "n_praise": int(y.sum()), "n_complaint": int((~y).sum()),
        "balanced_accuracy": {k: v["balanced_accuracy"] for k, v in result["scorers"].items()},
        "note": "praise=positive, complaint=negative; mixed and neutral excluded; "
                "each mention scored as one text",
    }
    conn = sqlite3.connect(study.scores_path)
    try:
        conn.execute(
            "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
            ("reference_sentiment_check", json.dumps({"study": study.name}), _git_sha(),
             json.dumps(metrics)),
        )
        conn.commit()
    finally:
        conn.close()
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point with subcommands score, validate, export-audit, import-labels,
    audit-metrics and reference-check."""
    parser = argparse.ArgumentParser(description="Transformer sentiment scoring and validation.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score", help="Score kept reviews and aspect sentences.")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--units", default="doc,sentence")
    p.add_argument("--models", default="all", help="Comma-separated model ids, or all.")
    p.add_argument("--batch-size", type=int, default=16)
    p = sub.add_parser("validate", help="Decision 11 selection, test read and verdict.")
    p.add_argument("--market", type=Path, required=True)
    p.add_argument("--april", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p = sub.add_parser("export-audit", help="Write the unlabeled sentence audit set.")
    p.add_argument("--study", type=Path, required=True)
    p = sub.add_parser("import-labels", help="Validate and commit sentence audit labels.")
    p.add_argument("labeled_csv", type=Path)
    p = sub.add_parser("audit-metrics", help="Score the sentence audit.")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--model", default=None, help="Model id; default the selected candidate.")
    p = sub.add_parser("reference-check", help="Decision 14 sentence-sentiment check.")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "score":
        study = load_study(args.study)
        names = list(TRANSFORMER_CANDIDATES) if args.models == "all" else args.models.split(",")
        version = None
        if study.aspect_lexicon is not None:
            from src.aspects import load_lexicon

            version = load_lexicon(study.aspect_lexicon).version
        for name in names:
            scorer = TransformerScorer(name, TRANSFORMER_CANDIDATES[name], args.batch_size)
            for unit in args.units.split(","):
                if unit == "sentence" and version is None:
                    continue
                logger.info("Result: %s", score_study(study, scorer, unit, version))
    elif args.cmd == "validate":
        m = validate(load_study(args.market), load_study(args.april), n_boot=args.n_boot)
        logger.info("Selected %s, adopted %s", m["selected"], m["adopted"])
    elif args.cmd == "export-audit":
        export_sentence_audit(load_study(args.study))
    elif args.cmd == "import-labels":
        logger.info("Wrote %s", import_sentence_labels(args.labeled_csv))
    elif args.cmd == "audit-metrics":
        study = load_study(args.study)
        model = args.model
        if model is None:
            conn = _read_only(study.scores_path)
            try:
                v = latest_validation(conn)
            finally:
                conn.close()
            assert v is not None, "Run validate first."
            model = v["models"][v["selected"]]["model_name"]
        logger.info("Sentence audit: %s", sentence_audit_metrics(study, model))
    elif args.cmd == "reference-check":
        logger.info("Reference check: %s",
                    reference_sentiment_check(load_study(args.study), args.reference))


if __name__ == "__main__":
    main()
