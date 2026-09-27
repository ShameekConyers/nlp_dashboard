"""Transformer sentiment scoring and its held-out validation (Decision 11).

Scores kept English reviews (``unit="doc"``) and aspect sentences
(``unit="sentence"``) with pinned Hugging Face sentiment models and writes
the positive-class probability to ``sentiment_scores`` (plus the neutral
probability for a three-class model). Every row carries a ``model_id`` that
joins to the model name and revision in ``sentiment_models``. Long inputs keep
their first ``HEAD_TOKENS`` and last ``TAIL_TOKENS`` tokens, since strategy
reviews often put the verdict at the end.

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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.db_setup import create_scores_tables
from src.stats import BOOT_SEED, N_BOOT, _git_sha, _num, balanced_accuracy, dev_test_split
from src.study import UNFREEZE_FLAG, UNFREEZE_HELP, Study, guard_writable, load_study
from src.utils import (
    DATA_DIR,
    HEAD_TOKENS,
    MIN_TOKEN_COUNT,
    SENTENCE_TRANSFORMER_CANDIDATES,
    TAIL_TOKENS,
    TRANSFORMER_CANDIDATES,
    TRANSFORMER_POSITIVE_CUTOFF,
    VADER_POSITIVE_THRESHOLD,
    get_logger,
)
from src.validation import (
    Predictor,
    bootstrap_validation,
    classify_sentence,
    macro_f1,
    macro_f1_multiclass,
    tune_band,
    tune_threshold,
)

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
    "cardiffnlp/twitter-roberta-base-sentiment-latest": "roberta_base_3class",
}
"""Short keys used in metrics JSON."""

ALL_MODELS: dict[str, str] = {**TRANSFORMER_CANDIDATES, **SENTENCE_TRANSFORMER_CANDIDATES}
"""Every pinned model this module may score: name -> revision."""

VADER_DEFAULT_KEY: str = "vader_0.05"
VADER_TUNED_KEY: str = "vader_dev_tuned"

ADOPTION_RULE: str = (
    "Select the candidate with the higher dev-split balanced accuracy. Adopt it only if the "
    "review-level paired-bootstrap 95% interval of (candidate - VADER) balanced accuracy on "
    "the market test split lies above 0 against VADER at 0.05 and at the dev-tuned threshold."
)

SENTENCE_AUDIT_VERSIONS: dict[str, int] = {"v1": 100, "v2": 200}
"""Sentence audit sets in order: version -> size. Each later set excludes the
sentences of every earlier one."""

SENTENCE_AUDIT_VERSION: str = "v1"
"""The first audit set: the 7.3 test read, re-used as the development set for
the v2 candidate selection (plan decision 14)."""

SENTENCE_TEST_VERSION: str = "v2"
"""The fresh test set read once for the three-class gate."""

SENTENCE_AUDIT_SEED: int = 20260927
SENTENCE_LABELS: tuple[str, ...] = ("positive", "negative", "neither")
SENTENCE_GATE_MACRO_F1: float = 0.70
"""Sentence sentiment is shown in the dashboard only at or above this audit macro-F1."""

SENTENCE_NEITHER_BAND: tuple[float, float] = (0.2, 0.8)
"""Fixed before any sentence label existed: a sentence scores ``neither`` when its
positive probability lies strictly inside this band, ``positive`` at or above the
upper bound and ``negative`` at or below the lower bound. A binary model needs a
band to predict ``neither`` at all."""

NEITHER_BAND_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 10))
"""Symmetric bands ``(c, 1 - c)`` tried on the development set for the binary
candidate: c from 0.05 to 0.45 in steps of 0.05, ties to the c nearest 0.20."""

NEITHER_BAND_ANCHOR: float = SENTENCE_NEITHER_BAND[0]

SELECTION_RULE: str = (
    "Candidates: each three-class model by argmax over negative, neutral and positive; the "
    "adopted binary model with a symmetric neither band tuned on the development labels. "
    "Select the highest development three-class macro-F1, ties to the three-class model. "
    "Read the fresh test set only if the selected candidate reaches the gate on development."
)


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
    """Class probabilities from a pinned sequence-classification model.

    The model loads on first use. ``score_probs`` sorts inputs by length to
    batch similar lengths together, then returns results in input order. A
    binary model yields ``(P(positive), None)``; a model with a ``neutral``
    head yields ``(P(positive), P(neutral))``.

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
        self._neu_index: int | None = None

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
        self._neu_index = labels.get("NEUTRAL")
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

    def _forward(self, batch: list[list[int]]) -> list[tuple[float, float | None]]:
        """Run one padded batch and return positive (and neutral) probabilities.

        Args:
            batch: Encoded inputs.

        Returns:
            ``(P(positive), P(neutral) or None)`` per input.
        """
        import torch

        padded = self._tokenizer.pad({"input_ids": batch}, return_tensors="pt")
        padded = {k: v.to(self.device) for k, v in padded.items()}
        with torch.inference_mode():
            logits = self._model(**padded).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        pos = probs[:, self._pos_index].tolist()
        if self._neu_index is None:
            return [(float(p), None) for p in pos]
        neu = probs[:, self._neu_index].tolist()
        return [(float(p), float(n)) for p, n in zip(pos, neu)]

    def score_probs(self, texts: Sequence[str]) -> list[tuple[float, float | None]]:
        """Score texts, batching by length, and return results in input order.

        Args:
            texts: Input texts.

        Returns:
            ``(P(positive), P(neutral) or None)`` per text, aligned with ``texts``.
        """
        encoded = [self.encode(t) for t in texts]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]), reverse=True)
        out: list[tuple[float, float | None]] = [(0.0, None)] * len(encoded)
        for start in range(0, len(order), self.batch_size):
            chunk = order[start:start + self.batch_size]
            for i, pn in zip(chunk, self._forward([encoded[i] for i in chunk])):
                out[i] = pn
        return out

    def score(self, texts: Sequence[str]) -> list[float]:
        """Return the positive-class probability per text, in input order.

        Args:
            texts: Input texts.

        Returns:
            P(positive) per text, aligned with ``texts``.
        """
        return [p for p, _ in self.score_probs(texts)]


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


def model_class_probs(
    conn: sqlite3.Connection, model_name: str, revision: str, unit: str
) -> dict[tuple[int, int], tuple[float, float | None]]:
    """Return one model's stored positive and neutral probabilities for a unit.

    Args:
        conn: Connection to a scores database.
        model_name: Hugging Face model id.
        revision: Pinned commit hash.
        unit: ``doc`` or ``sentence``.

    Returns:
        ``{(document_id, sentence_idx): (positive_prob, neutral_prob or None)}``.
    """
    rows = conn.execute(
        """
        SELECT s.document_id, s.sentence_idx, s.positive_prob, s.neutral_prob
        FROM sentiment_scores s
        JOIN sentiment_models m ON m.model_id = s.model_id
        WHERE m.model_name = ? AND m.model_revision = ? AND s.unit = ?
        """,
        (model_name, revision, unit),
    ).fetchall()
    return {(d, i): (p, n) for d, i, p, n in rows}


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
    unfreeze: bool = False,
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
        unfreeze: Allow writing a frozen (shipped) study's scores database.

    Returns:
        Timing and counts, also stored as an ``analysis_runs`` row.

    Raises:
        FrozenStudyError: If a target belongs to a frozen study and ``unfreeze``
            is False.
    """
    guard_writable([study.scores_path], unfreeze)
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
            probs = scorer.score_probs([r[2] for r in chunk])
            out.executemany(
                "INSERT OR REPLACE INTO sentiment_scores (document_id, model_id, unit,"
                " sentence_idx, positive_prob, neutral_prob) VALUES (?, ?, ?, ?, ?, ?)",
                [(r[0], mid, unit, r[1], p, n) for r, (p, n) in zip(chunk, probs)],
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


def validate(
    market: Study,
    april: Study,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    unfreeze: bool = False,
) -> dict[str, Any]:
    """Run the Decision 11 selection, test read and adoption verdict.

    Dev-split work (VADER threshold, candidate selection) finishes before any
    test-split value is computed. Results are stored as a
    ``sentiment_validation`` run in the market seed and the April scores DB.

    Args:
        market: The market study (selection and verdict).
        april: The April baseline (continuity only).
        n_boot: Bootstrap resamples.
        seed: Random seed.
        unfreeze: Allow writing a frozen (shipped) study's scores databases.

    Returns:
        The metrics dictionary that was stored.

    Raises:
        FrozenStudyError: If either scores database belongs to a frozen study
            and ``unfreeze`` is False.
    """
    guard_writable([market.scores_path, april.scores_path], unfreeze)
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


def sentence_audit_paths(version: str = SENTENCE_AUDIT_VERSION) -> tuple[Path, Path]:
    """Return the unlabeled export and the committed label file paths of one audit set.

    Args:
        version: A key of ``SENTENCE_AUDIT_VERSIONS``.

    Returns:
        ``(unlabeled_csv, labeled_csv)`` under ``data/labels/``.
    """
    base = DATA_DIR / "labels"
    return (base / f"sentence_sentiment_audit_{version}_unlabeled.csv",
            base / f"sentence_sentiment_audit_{version}.csv")


def load_sentence_labels(version: str) -> list[dict[str, str]]:
    """Read one committed sentence label file.

    Args:
        version: A key of ``SENTENCE_AUDIT_VERSIONS``.

    Returns:
        Rows with ``audit_id``, ``steam_review_id``, ``sentence_idx``, ``aspect``, ``label``.
    """
    with open(sentence_audit_paths(version)[1], encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def earlier_versions(version: str) -> list[str]:
    """Return the audit versions that precede ``version``.

    Args:
        version: A key of ``SENTENCE_AUDIT_VERSIONS``.

    Returns:
        Earlier keys in order.
    """
    order = list(SENTENCE_AUDIT_VERSIONS)
    return order[: order.index(version)]


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


def allocate(
    total: int, strata: Sequence[str], capacity: Mapping[str, int] | None = None
) -> dict[str, int]:
    """Split ``total`` as evenly as possible, extra units to the first strata.

    With ``capacity``, a stratum never gets more than it can supply; its
    shortfall is spread evenly over the strata that still have room.

    Args:
        total: Items to allocate.
        strata: Strata in priority order.
        capacity: Optional ``{stratum: available items}``.

    Returns:
        ``{stratum: count}``.

    Raises:
        ValueError: If the strata cannot supply ``total`` between them.
    """
    if capacity is not None:
        available = sum(capacity.get(s, 0) for s in strata)
        if available < total:
            raise ValueError(f"{total} sentences needed, {available} available.")
    quotas: dict[str, int] = {}
    open_strata = list(strata)
    remaining = total
    while open_strata:
        base, extra = divmod(remaining, len(open_strata))
        share = {s: base + (1 if i < extra else 0) for i, s in enumerate(open_strata)}
        capped = [s for s in open_strata if capacity is not None and share[s] > capacity.get(s, 0)]
        if not capped:
            quotas.update(share)
            break
        for s in capped:
            quotas[s] = capacity[s]  # type: ignore[index]
            remaining -= quotas[s]
        open_strata = [s for s in open_strata if s not in capped]
    return quotas


def _pick(
    by_aspect: Mapping[str, Sequence[tuple[str, int, str]]],
    quotas: Mapping[str, int],
    used: set[tuple[str, int]],
    key: Callable[[str], str],
) -> list[tuple[str, int, str, str]]:
    """Draw each aspect's quota of unused sentences in a seeded shuffle order.

    Args:
        by_aspect: ``{aspect: [(steam_review_id, sentence_idx, text), ...]}``.
        quotas: Sentences per aspect, in draw order.
        used: ``(steam_review_id, sentence_idx)`` pairs never drawn; grows with the picks.
        key: Maps an aspect to its shuffle seed.

    Returns:
        ``(steam_review_id, sentence_idx, aspect, text)`` per pick.

    Raises:
        ValueError: If an aspect has fewer unused sentences than its quota.
    """
    picked: list[tuple[str, int, str, str]] = []
    for aspect, quota in quotas.items():
        pool = list(by_aspect.get(aspect, []))
        random.Random(key(aspect)).shuffle(pool)
        take = [r for r in pool if (r[0], r[1]) not in used][:quota]
        if len(take) < quota:
            raise ValueError(f"{aspect}: {len(take)} test sentences, need {quota}.")
        used.update((r[0], r[1]) for r in take)
        picked.extend((r[0], r[1], aspect, r[2]) for r in take)
    return picked


def export_sentence_audit(
    study: Study,
    version: str = SENTENCE_TEST_VERSION,
    n: int | None = None,
    seed: int = SENTENCE_AUDIT_SEED,
) -> Path:
    """Write the unlabeled sentence-sentiment audit set of one version.

    Draws ``n`` distinct test-split aspect sentences, none of them in an
    earlier version's label file, spread across the aspects that passed the
    precision audit as evenly as their unused test sentences allow. Each
    sentence is used once and no scores appear in the file. ``v1`` keeps its
    original shuffle keys, so its committed export stays reproducible.

    Args:
        study: The market study.
        version: A key of ``SENTENCE_AUDIT_VERSIONS``.
        n: Sentences to draw; default the version's size.
        seed: Sampling seed.

    Returns:
        Path of the unlabeled CSV.

    Raises:
        ValueError: If the passing aspects cannot supply the sentences.
    """
    from src.aspects import load_lexicon

    assert study.aspect_lexicon is not None
    n = SENTENCE_AUDIT_VERSIONS[version] if n is None else n
    lexicon_version = load_lexicon(study.aspect_lexicon).version
    aspects = _passing_aspects(study)
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
            (lexicon_version,),
        ).fetchall()
    finally:
        conn.close()
    used: set[tuple[str, int]] = {
        (r["steam_review_id"], int(r["sentence_idx"]))
        for v in earlier_versions(version) for r in load_sentence_labels(v)
    }
    by_aspect: dict[str, list[tuple[str, int, str]]] = {}
    for rid, aspect, idx, _doc, text in rows:
        if aspect in aspects and dev_test_split(rid) == "test":
            by_aspect.setdefault(aspect, []).append((rid, idx, text))
    capacity = {a: sum(1 for r in by_aspect.get(a, []) if (r[0], r[1]) not in used)
                for a in aspects}
    quotas = allocate(n, aspects, capacity)
    tag = "" if version == SENTENCE_AUDIT_VERSION else f":{version}"
    picked = _pick(by_aspect, quotas, used, lambda a: f"{seed}{tag}:{a}")
    random.Random(f"{seed}{tag}" if tag else seed).shuffle(picked)
    out, _ = sentence_audit_paths(version)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "steam_review_id", "sentence_idx", "aspect", "sentence", "label"])
        for i, (rid, idx, aspect, text) in enumerate(picked, 1):
            w.writerow([i, rid, idx, aspect, text, ""])
    logger.info("Wrote %d audit sentences to %s (%s)", len(picked), out, quotas)
    return out


def import_sentence_labels(labeled_csv: Path, version: str = SENTENCE_TEST_VERSION) -> Path:
    """Validate a labeled audit file and write the committed label file.

    The committed file keeps ids, aspect and label; sentence text stays in
    the seed database and the unlabeled export.

    Args:
        labeled_csv: The labeler's filled copy of the export.
        version: A key of ``SENTENCE_AUDIT_VERSIONS``.

    Returns:
        Path of the committed label file.

    Raises:
        ValueError: If ids differ from the export or a label is invalid.
    """
    unlabeled, committed = sentence_audit_paths(version)
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


def _audit_predictions(
    study: Study,
    labels: Sequence[Mapping[str, str]],
    model_name: str,
    band: tuple[float, float],
) -> tuple[list[str], list[str], list[tuple[float, float | None]]]:
    """Return labels, predicted classes and stored probabilities for audit rows.

    A three-class model is classified by argmax, a binary one by ``band``.

    Args:
        study: The market study; its scores database is opened read-only.
        labels: Rows of a committed label file.
        model_name: Hugging Face id of a scored model.
        band: Neither band for a binary model.

    Returns:
        ``(y_true, y_pred, [(positive_prob, neutral_prob or None), ...])``.
    """
    conn = _read_only(study.scores_path)
    try:
        ids = dict(conn.execute("SELECT steam_review_id, id FROM documents").fetchall())
        stored = model_class_probs(conn, model_name, ALL_MODELS[model_name], "sentence")
    finally:
        conn.close()
    y_true = [r["label"] for r in labels]
    probs = [stored[(ids[r["steam_review_id"]], int(r["sentence_idx"]))] for r in labels]
    y_pred = [classify_sentence(p, n, band) for p, n in probs]
    return y_true, y_pred, probs


def _binary_pred(positive: float, neutral: float | None) -> bool:
    """Return the positive-versus-negative call, ignoring the neutral head.

    Args:
        positive: P(positive).
        neutral: P(neutral) or None for a binary model.

    Returns:
        True when positive beats negative.
    """
    if neutral is None:
        return positive >= POSITIVE_CUTOFF
    return positive >= 1.0 - positive - neutral


def _store_run(study: Study, stage: str, params: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Append one ``analysis_runs`` row to the study's scores database.

    Args:
        study: The study.
        stage: Run stage name.
        params: Run parameters.
        metrics: Run metrics.
    """
    conn = sqlite3.connect(study.scores_path)
    try:
        conn.execute(
            "INSERT INTO analysis_runs (stage, params, git_sha, metrics) VALUES (?, ?, ?, ?)",
            (stage, json.dumps(params), _git_sha(), json.dumps(metrics, default=_json_default)),
        )
        conn.commit()
    finally:
        conn.close()


def sentence_dev_selection(
    study: Study, dev_version: str = SENTENCE_AUDIT_VERSION, unfreeze: bool = False
) -> dict[str, Any]:
    """Select the sentence-sentiment candidate on the development labels (plan decision 14).

    Every three-class candidate is classified by argmax. The adopted binary
    model gets a symmetric neither band tuned on the same labels. The
    candidate with the highest development macro-F1 is selected, ties to a
    three-class model, and ``proceed`` says whether it reaches the gate, which
    is the condition for reading the fresh test set.

    Args:
        study: The market study.
        dev_version: Label set used for selection and calibration.
        unfreeze: Allow writing a frozen (shipped) study's scores database.

    Returns:
        The metrics dictionary, also stored as a ``sentence_dev_selection`` run.

    Raises:
        FrozenStudyError: If a target belongs to a frozen study and ``unfreeze``
            is False.
        ValueError: If no transformer was adopted for the study.
    """
    guard_writable([study.scores_path], unfreeze)
    labels = load_sentence_labels(dev_version)
    y_true = [r["label"] for r in labels]
    conn = _read_only(study.scores_path)
    try:
        validation = latest_validation(conn)
    finally:
        conn.close()
    if validation is None or not validation["adopted"]:
        raise ValueError(f"{study.name}: no adopted transformer; run validate first.")
    binary_name = validation["models"][validation["selected"]]["model_name"]
    candidates: dict[str, dict[str, Any]] = {}
    for name in SENTENCE_TRANSFORMER_CANDIDATES:
        _, y_pred, _ = _audit_predictions(study, labels, name, SENTENCE_NEITHER_BAND)
        candidates[name] = {
            "classification": "argmax", "neither_band": None,
            "macro_f1_3class": macro_f1_multiclass(y_true, y_pred, SENTENCE_LABELS),
        }
    _, fixed_pred, probs = _audit_predictions(study, labels, binary_name, SENTENCE_NEITHER_BAND)
    band, band_f1 = tune_band([*y_true], [p for p, _ in probs], NEITHER_BAND_GRID,
                              SENTENCE_LABELS, NEITHER_BAND_ANCHOR)
    candidates[binary_name] = {
        "classification": "band", "neither_band": list(band), "macro_f1_3class": band_f1,
        "fixed_band": list(SENTENCE_NEITHER_BAND),
        "fixed_band_macro_f1": macro_f1_multiclass(y_true, fixed_pred, SENTENCE_LABELS),
    }
    order = [*SENTENCE_TRANSFORMER_CANDIDATES, binary_name]
    selected = max(order, key=lambda k: (candidates[k]["macro_f1_3class"], -order.index(k)))
    metrics = {
        "selection_rule": SELECTION_RULE,
        "dev_version": dev_version, "n": len(labels),
        "label_counts": {c: y_true.count(c) for c in SENTENCE_LABELS},
        "band_grid": list(NEITHER_BAND_GRID),
        "candidates": candidates,
        "selected": selected,
        "selected_revision": ALL_MODELS[selected],
        "gate": SENTENCE_GATE_MACRO_F1,
        "proceed": candidates[selected]["macro_f1_3class"] >= SENTENCE_GATE_MACRO_F1,
    }
    _store_run(study, "sentence_dev_selection",
               {"study": study.name, "labels": sentence_audit_paths(dev_version)[1].name}, metrics)
    return metrics


def sentence_audit_metrics(
    study: Study,
    model_name: str,
    version: str = SENTENCE_AUDIT_VERSION,
    band: tuple[float, float] = SENTENCE_NEITHER_BAND,
    unfreeze: bool = False,
) -> dict[str, Any]:
    """Score one version's committed sentence labels against one model's sentence scores.

    Args:
        study: The market study.
        model_name: Hugging Face id of the scored model.
        version: A key of ``SENTENCE_AUDIT_VERSIONS``.
        band: Neither band for a binary model; ignored for a three-class model.
        unfreeze: Allow writing a frozen (shipped) study's scores database.

    Returns:
        Macro-F1 over three classes (the gate), binary macro-F1 on sentences
        labelled positive or negative, the confusion counts and whether the
        gate passed. Also stored as a ``sentence_audit`` run.

    Raises:
        FrozenStudyError: If a target belongs to a frozen study and ``unfreeze``
            is False.
    """
    guard_writable([study.scores_path], unfreeze)
    labels = load_sentence_labels(version)
    y_true, y_pred, probs = _audit_predictions(study, labels, model_name, band)
    three_class = bool(probs) and probs[0][1] is not None
    binary = [(t == "positive", _binary_pred(p, n)) for t, (p, n) in zip(y_true, probs)
              if t != "neither"]
    confusion = {t: {p: sum(1 for a, b in zip(y_true, y_pred) if a == t and b == p)
                     for p in SENTENCE_LABELS} for t in SENTENCE_LABELS}
    f1 = macro_f1_multiclass(y_true, y_pred, SENTENCE_LABELS)
    metrics = {
        "audit_version": version,
        "model_name": model_name, "model_revision": ALL_MODELS[model_name], "n": len(labels),
        "label_counts": {c: y_true.count(c) for c in SENTENCE_LABELS},
        "classification": "argmax" if three_class else "band",
        "neither_band": None if three_class else list(band),
        "macro_f1_3class": f1,
        "binary_n": len(binary),
        "binary_macro_f1": macro_f1([b[0] for b in binary], [b[1] for b in binary])
        if binary else None,
        "confusion": confusion,
        "gate": SENTENCE_GATE_MACRO_F1,
        "passes": f1 >= SENTENCE_GATE_MACRO_F1,
    }
    _store_run(study, "sentence_audit",
               {"study": study.name, "labels": sentence_audit_paths(version)[1].name,
                "audit_version": version}, metrics)
    return metrics


# ---------------------------------------------------------------------------
# Reference check (Decision 14)
# ---------------------------------------------------------------------------


def reference_sentiment_check(
    study: Study,
    reference_path: Path,
    scorer_factory: Callable[[str, str], TransformerScorer] = TransformerScorer,
    n_boot: int = N_BOOT,
    unfreeze: bool = False,
) -> dict[str, Any]:
    """Score hand-coded praise and complaint mentions and store derived numbers only.

    Reads the private research file at run time and never copies its text.
    Mixed and neutral mentions are excluded.

    Args:
        study: The market study (its apps select the mentions).
        reference_path: Private research JSON.
        scorer_factory: Builds a scorer from ``(model_name, revision)``.
        n_boot: Bootstrap resamples for the balanced-accuracy interval.
        unfreeze: Allow writing a frozen (shipped) study's scores database.

    Returns:
        Per-scorer balanced accuracy with interval and n, stored as a
        ``reference_sentiment_check`` run.

    Raises:
        FrozenStudyError: If a target belongs to a frozen study and ``unfreeze``
            is False.
    """
    guard_writable([study.scores_path], unfreeze)
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
    dev-select, audit-metrics and reference-check."""
    parser = argparse.ArgumentParser(description="Transformer sentiment scoring and validation.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    versions = list(SENTENCE_AUDIT_VERSIONS)
    p = sub.add_parser("score", help="Score kept reviews and aspect sentences.")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--units", default="doc,sentence")
    p.add_argument("--models", default="all",
                   help="Comma-separated model ids, or all (the review-level candidates).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    p = sub.add_parser("validate", help="Decision 11 selection, test read and verdict.")
    p.add_argument("--market", type=Path, required=True)
    p.add_argument("--april", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    p = sub.add_parser("export-audit", help="Write an unlabeled sentence audit set.")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--version", choices=versions, default=SENTENCE_TEST_VERSION)
    p = sub.add_parser("import-labels", help="Validate and commit sentence audit labels.")
    p.add_argument("labeled_csv", type=Path)
    p.add_argument("--version", choices=versions, default=SENTENCE_TEST_VERSION)
    p = sub.add_parser("dev-select", help="Select the sentence candidate on development labels.")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--dev-version", choices=versions, default=SENTENCE_AUDIT_VERSION)
    p.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    p = sub.add_parser("audit-metrics", help="Score one sentence audit set (the test read).")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--version", choices=versions, default=SENTENCE_TEST_VERSION)
    p.add_argument("--model", default=None,
                   help="Model id; default the model selected for the version (validate for "
                        "v1, dev-select for later versions).")
    p.add_argument("--band", default=None, help="Neither band low,high for a binary model.")
    p.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    p = sub.add_parser("reference-check", help="Decision 14 sentence-sentiment check.")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument(UNFREEZE_FLAG, dest="unfreeze", action="store_true", help=UNFREEZE_HELP)
    args = parser.parse_args()
    try:
        _dispatch(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


def _selected_sentence_model(
    study: Study, version: str, band: tuple[float, float]
) -> tuple[str, tuple[float, float]]:
    """Return the model and band the pre-registration selected for an audit version.

    ``v1`` uses the Decision 11 adopted model with the fixed band. Later
    versions use the newest ``sentence_dev_selection`` run, which must have
    reached the gate on development.

    Args:
        study: The market study.
        version: A key of ``SENTENCE_AUDIT_VERSIONS``.
        band: Fallback band for a binary model.

    Returns:
        ``(model_name, band)``.

    Raises:
        ValueError: If the selecting run is missing or did not clear the gate.
    """
    conn = _read_only(study.scores_path)
    try:
        if version == SENTENCE_AUDIT_VERSION:
            v = latest_validation(conn)
            if v is None:
                raise ValueError("Run validate first.")
            return v["models"][v["selected"]]["model_name"], band
        row = conn.execute(
            "SELECT metrics FROM analysis_runs WHERE stage = 'sentence_dev_selection' "
            "ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError("Run dev-select first.")
    sel = json.loads(row[0])
    if not sel["proceed"]:
        raise ValueError(
            f"Development selection did not reach the gate ({sel['gate']}); no test read."
        )
    chosen = sel["candidates"][sel["selected"]]
    chosen_band = tuple(chosen["neither_band"]) if chosen["neither_band"] else band
    return sel["selected"], chosen_band  # type: ignore[return-value]


def _dispatch(args: argparse.Namespace) -> None:
    """Run the parsed subcommand.

    Args:
        args: Parsed CLI arguments.
    """

    if args.cmd == "score":
        study = load_study(args.study)
        guard_writable([study.scores_path], args.unfreeze)  # before any model loads
        names = list(TRANSFORMER_CANDIDATES) if args.models == "all" else args.models.split(",")
        version = None
        if study.aspect_lexicon is not None:
            from src.aspects import load_lexicon

            version = load_lexicon(study.aspect_lexicon).version
        for name in names:
            if name not in ALL_MODELS:
                raise ValueError(f"Unknown model {name!r}; known: {', '.join(ALL_MODELS)}.")
            scorer = TransformerScorer(name, ALL_MODELS[name], args.batch_size)
            for unit in args.units.split(","):
                if unit == "sentence" and version is None:
                    continue
                logger.info("Result: %s", score_study(study, scorer, unit, version, args.unfreeze))
    elif args.cmd == "validate":
        m = validate(
            load_study(args.market), load_study(args.april), n_boot=args.n_boot,
            unfreeze=args.unfreeze,
        )
        logger.info("Selected %s, adopted %s", m["selected"], m["adopted"])
    elif args.cmd == "export-audit":
        export_sentence_audit(load_study(args.study), args.version)
    elif args.cmd == "import-labels":
        logger.info("Wrote %s", import_sentence_labels(args.labeled_csv, args.version))
    elif args.cmd == "dev-select":
        m = sentence_dev_selection(load_study(args.study), args.dev_version, args.unfreeze)
        logger.info("Dev selection: %s", m)
    elif args.cmd == "audit-metrics":
        study = load_study(args.study)
        model, band = args.model, SENTENCE_NEITHER_BAND
        if args.band:
            low, high = (float(x) for x in args.band.split(","))
            band = (low, high)
        if model is None:
            model, band = _selected_sentence_model(study, args.version, band)
        logger.info("Sentence audit: %s",
                    sentence_audit_metrics(study, model, args.version, band, args.unfreeze))
    elif args.cmd == "reference-check":
        logger.info("Reference check: %s",
                    reference_sentiment_check(
            load_study(args.study), args.reference, unfreeze=args.unfreeze))


if __name__ == "__main__":
    main()
