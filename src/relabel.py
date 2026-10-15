"""Blind owner relabel of the audits that carry findings near their floors.

``export`` writes two blind files for the owner, with no model label, score
or ID column, in a seeded shuffled order:

- the 160 aspect audit sentences of pacing, bugs and performance, UI and
  price (40 each, the committed audit rows of ``aspect_audit_strategy_v1``);
- the 200 sentences of the v3 sentence sentiment test set.

A separate key maps each row ID to the review, the independent model
labeler's label and, for the sentiment set, the selected sentence rule's
class. Blind files go to ``RELABEL_DIR`` and the key to ``KEY_DIR``, both
under ``outputs/`` (ignored by Git, since the blind files carry review text).

``score`` reads the owner's filled files and reports Cohen's kappa and
Gwet's AC1 against the model labeler with bootstrap intervals, each
aspect's precision on the owner's labels against the 0.80 floor, and the
selected sentence rule's macro-F1 on the owner's labels against the 0.70
gate. Instructions for the owner: ``docs/relabel_instructions.md``.
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is on sys.path so `src.*` imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agreement import BOOT_SEED, N_BOOT, agreement_report, bootstrap_interval
from src.stats import MIN_AUDIT_N, MIN_PRECISION, wilson_interval
from src.study import STUDIES_DIR, Study, load_study
from src.utils import OUTPUTS_DIR, get_logger
from src.validation import macro_f1_multiclass

logger = get_logger(__name__)

RELABEL_DIR: Path = OUTPUTS_DIR / "relabel_2026"
"""Blind files for the owner and, after scoring, the results."""

KEY_DIR: Path = OUTPUTS_DIR / "relabel_2026_key"
"""Key files. The owner does not open this directory until labelling is done."""

ASPECT_BLIND: str = "aspect_relevance_blind.csv"
SENTIMENT_BLIND: str = "sentence_sentiment_blind.csv"
ASPECT_KEY: str = "aspect_relevance_key.csv"
SENTIMENT_KEY: str = "sentence_sentiment_key.csv"
KEY_MANIFEST: str = "key_manifest.json"
RESULTS: str = "relabel_scores.json"

RELABEL_ASPECTS: tuple[str, ...] = ("pacing", "bugs_performance", "ui_clarity", "price_dlc")
"""The four passing aspects that carry findings near the 0.80 floor."""

SENTIMENT_VERSION: str = "v3"
"""The sentence sentiment test set that passed its gate."""

SHUFFLE_SEED: int = 20261017
"""Seed for the row order of both blind files."""

ASPECT_LABELS: tuple[str, ...] = ("relevant", "not_relevant")
SENTIMENT_LABELS: tuple[str, ...] = ("positive", "negative", "neither")

MARKET_STUDY_PATH: Path = STUDIES_DIR / "strategy_market_2026.toml"


def _safe(path: Path) -> Path:
    """Return ``path`` after checking it lies under ``RELABEL_DIR`` or ``KEY_DIR``.

    Args:
        path: A path about to be written.

    Returns:
        The resolved path.

    Raises:
        ValueError: If the path is elsewhere.
    """
    resolved = path.resolve()
    if not any(resolved.is_relative_to(d.resolve()) for d in (RELABEL_DIR, KEY_DIR)):
        raise ValueError(f"Refusing to write {resolved}: outside {RELABEL_DIR} and {KEY_DIR}.")
    return resolved


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dictionaries.

    Args:
        path: CSV file.

    Returns:
        Rows.
    """
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Write a CSV under the relabel directories.

    Args:
        path: Output file.
        header: Column names.
        rows: Rows.
    """
    path = _safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def shuffled(items: Sequence[Any], seed: int, tag: str) -> list[Any]:
    """Return a seeded shuffle of ``items``.

    Args:
        items: Items in source order.
        seed: Base seed.
        tag: Per-file tag so the two files get different orders.

    Returns:
        A new shuffled list.
    """
    out = list(items)
    random.Random(f"{seed}:{tag}").shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def aspect_rows(study: Study, labels_path: Path | None = None) -> list[dict[str, Any]]:
    """Collect the audit rows of ``RELABEL_ASPECTS`` with their sentence text.

    Text comes from the seed (``aspect_mentions`` offsets into the cleaned
    review), exactly as the audit export drew it.

    Args:
        study: The market study.
        labels_path: Committed aspect label file (default ``aspect_audit_strategy_v1.csv``).

    Returns:
        Rows with ``steam_review_id``, ``sentence_idx``, ``aspect``,
        ``aspect_label``, ``sentence`` and ``model_label``.

    Raises:
        ValueError: If a labelled sentence is missing from the seed.
    """
    from src.aspects import audit_paths, load_lexicon
    from src.replication import _read_only

    assert study.aspect_lexicon is not None
    lexicon = load_lexicon(study.aspect_lexicon)
    labels_path = labels_path or audit_paths(study, lexicon)[1]
    names = {a.name: a.label for a in lexicon.aspects}
    wanted = [r for r in _read_csv(labels_path) if r["aspect"] in RELABEL_ASPECTS]
    conn = _read_only(study.seed_db)
    try:
        out: list[dict[str, Any]] = []
        for r in wanted:
            hit = conn.execute(
                """
                SELECT SUBSTR(p.cleaned_text, a.char_start + 1, a.char_end - a.char_start)
                FROM aspect_mentions a
                JOIN documents d ON d.id = a.document_id
                JOIN processed_documents p ON p.document_id = a.document_id
                WHERE d.steam_review_id = ? AND a.sentence_idx = ? AND a.aspect = ?
                  AND a.lexicon_version = ?
                """,
                (r["steam_review_id"], int(r["sentence_idx"]), r["aspect"], lexicon.version),
            ).fetchone()
            if hit is None:
                raise ValueError(f"{r['steam_review_id']}:{r['sentence_idx']} {r['aspect']} not in seed")
            out.append({
                "steam_review_id": r["steam_review_id"], "sentence_idx": int(r["sentence_idx"]),
                "aspect": r["aspect"], "aspect_label": names[r["aspect"]], "sentence": hit[0],
                "model_label": r["label"],
            })
    finally:
        conn.close()
    return out


def sentiment_rows(study: Study) -> list[dict[str, Any]]:
    """Collect the v3 test sentences with the labeler's label and the selected rule's class.

    Args:
        study: The market study.

    Returns:
        Rows with ``audit_id``, ``steam_review_id``, ``sentence_idx``,
        ``aspect``, ``sentence``, ``model_label`` and ``model_pred``.

    Raises:
        ValueError: If the unlabeled export and the label file disagree.
    """
    from src.sentiment_transformer import (
        SENTENCE_NEITHER_BAND,
        _audit_predictions,
        _selected_sentence_model,
        load_sentence_labels,
        sentence_audit_paths,
    )

    labels = load_sentence_labels(SENTIMENT_VERSION)
    texts = {r["audit_id"]: r for r in _read_csv(sentence_audit_paths(SENTIMENT_VERSION)[0])}
    model, band, threshold = _selected_sentence_model(study, SENTIMENT_VERSION, SENTENCE_NEITHER_BAND)
    _, preds, _ = _audit_predictions(study, labels, model, band, threshold)
    out: list[dict[str, Any]] = []
    for r, pred in zip(labels, preds):
        t = texts.get(r["audit_id"])
        ids = (r["steam_review_id"], r["sentence_idx"])
        if t is None or (t["steam_review_id"], t["sentence_idx"]) != ids:
            raise ValueError(f"v3 audit_id {r['audit_id']}: export and labels disagree.")
        out.append({
            "audit_id": r["audit_id"], "steam_review_id": r["steam_review_id"],
            "sentence_idx": int(r["sentence_idx"]), "aspect": r["aspect"], "sentence": t["sentence"],
            "model_label": r["label"], "model_pred": pred,
        })
    return out


def _has_owner_labels(path: Path) -> bool:
    """Return whether a blind file exists and already holds any label.

    Args:
        path: Blind CSV.

    Returns:
        True when at least one ``label`` cell is filled.
    """
    return path.exists() and any(r.get("label", "").strip() for r in _read_csv(path))


def write_exports(
    aspects: Sequence[dict[str, Any]],
    sentiments: Sequence[dict[str, Any]],
    blind_dir: Path = RELABEL_DIR,
    key_dir: Path = KEY_DIR,
    seed: int = SHUFFLE_SEED,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the two blind files and the key.

    Args:
        aspects: Output of ``aspect_rows``.
        sentiments: Output of ``sentiment_rows``.
        blind_dir: Directory for the blind files.
        key_dir: Directory for the key.
        seed: Shuffle seed.
        meta: Extra facts for the key manifest.

    Returns:
        The key manifest.

    Raises:
        ValueError: If a blind file already holds owner labels.
    """
    for name in (ASPECT_BLIND, SENTIMENT_BLIND):
        if _has_owner_labels(blind_dir / name):
            raise ValueError(f"{blind_dir / name} already holds labels. Refusing to overwrite.")
    a_rows = shuffled(aspects, seed, "aspect")
    s_rows = shuffled(sentiments, seed, "sentiment")
    a_ids = [f"A{i:03d}" for i in range(1, len(a_rows) + 1)]
    s_ids = [f"S{i:03d}" for i in range(1, len(s_rows) + 1)]
    _write_csv(blind_dir / ASPECT_BLIND, ["row_id", "aspect", "aspect_label", "sentence", "label"],
               [[i, r["aspect"], r["aspect_label"], r["sentence"], ""] for i, r in zip(a_ids, a_rows)])
    _write_csv(blind_dir / SENTIMENT_BLIND, ["row_id", "sentence", "label"],
               [[i, r["sentence"], ""] for i, r in zip(s_ids, s_rows)])
    _write_csv(key_dir / ASPECT_KEY,
               ["row_id", "steam_review_id", "sentence_idx", "aspect", "model_label"],
               [[i, r["steam_review_id"], r["sentence_idx"], r["aspect"], r["model_label"]]
                for i, r in zip(a_ids, a_rows)])
    _write_csv(key_dir / SENTIMENT_KEY,
               ["row_id", "audit_id", "steam_review_id", "sentence_idx", "aspect", "model_label",
                "model_pred"],
               [[i, r["audit_id"], r["steam_review_id"], r["sentence_idx"], r["aspect"],
                 r["model_label"], r["model_pred"]] for i, r in zip(s_ids, s_rows)])
    manifest = {
        "shuffle_seed": seed,
        "n_aspect": len(a_rows),
        "n_sentiment": len(s_rows),
        "aspect_counts": {a: sum(r["aspect"] == a for r in a_rows) for a in RELABEL_ASPECTS},
        "blind_sha256": {
            name: hashlib.sha256((blind_dir / name).read_bytes()).hexdigest()
            for name in (ASPECT_BLIND, SENTIMENT_BLIND)
        },
        **(meta or {}),
    }
    (_safe(key_dir / KEY_MANIFEST)).write_text(json.dumps(manifest, indent=2))
    return manifest


def export(study: Study) -> dict[str, Any]:
    """Build both relabel sets from the committed audits and write them.

    Args:
        study: The market study.

    Returns:
        The key manifest.
    """
    from src.sentiment_transformer import SENTENCE_GATE_MACRO_F1

    aspects = aspect_rows(study)
    sentiments = sentiment_rows(study)
    check = macro_f1_multiclass([r["model_label"] for r in sentiments],
                                [r["model_pred"] for r in sentiments], SENTIMENT_LABELS)
    meta = {
        "aspect_source": "data/labels/aspect_audit_strategy_v1.csv, text from data/market_seed.db",
        "sentiment_source": "data/labels/sentence_sentiment_audit_v3.csv and _unlabeled.csv",
        "model_label": "independent model labeler (Claude Opus), as committed",
        "model_pred": "selected sentence rule (newest sentence_dev_selection run)",
        "selected_rule_macro_f1_on_model_labels": check,
        "gate": SENTENCE_GATE_MACRO_F1,
    }
    manifest = write_exports(aspects, sentiments, meta=meta)
    logger.info("Wrote %d aspect and %d sentiment rows (rule macro-F1 on model labels %.3f).",
                manifest["n_aspect"], manifest["n_sentiment"], check)
    return manifest


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def read_owner_labels(path: Path, allowed: Sequence[str]) -> dict[str, str]:
    """Read and validate a filled blind file.

    Args:
        path: The owner's labelled copy.
        allowed: Valid labels.

    Returns:
        ``{row_id: label}``.

    Raises:
        ValueError: If a row is unlabeled or carries an invalid label.
    """
    rows = _read_csv(path)
    labels = {r["row_id"]: r["label"].strip().lower() for r in rows}
    bad = [rid for rid, lab in labels.items() if lab not in allowed]
    if bad:
        raise ValueError(f"{path}: {len(bad)} rows lack a valid label, e.g. {bad[:5]}")
    return labels


def _join(owner: dict[str, str], key: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Attach owner labels to key rows.

    Args:
        owner: ``{row_id: label}``.
        key: Key rows.

    Returns:
        Key rows with an ``owner_label`` field.

    Raises:
        ValueError: If the row IDs differ.
    """
    if set(owner) != {r["row_id"] for r in key}:
        raise ValueError("Owner file row IDs do not match the key.")
    return [{**r, "owner_label": owner[r["row_id"]]} for r in key]


def score_aspects(rows: Sequence[dict[str, str]], n_boot: int = N_BOOT) -> dict[str, Any]:
    """Agreement and owner-label precision for the aspect relabel.

    Args:
        rows: Joined rows with ``aspect``, ``model_label`` and ``owner_label``.
        n_boot: Bootstrap resamples.

    Returns:
        Overall and per-aspect agreement, and per-aspect precision on both
        label sets with the pass rule of ``stats.load_audit_precision``.
    """
    out: dict[str, Any] = {
        "overall": agreement_report([r["owner_label"] for r in rows],
                                    [r["model_label"] for r in rows], ASPECT_LABELS, n_boot),
        "aspects": {},
    }
    for aspect in RELABEL_ASPECTS:
        sub = [r for r in rows if r["aspect"] == aspect]
        if not sub:
            continue
        entry: dict[str, Any] = {
            "agreement": agreement_report([r["owner_label"] for r in sub],
                                          [r["model_label"] for r in sub], ASPECT_LABELS, n_boot),
        }
        for who in ("owner", "model"):
            k = sum(r[f"{who}_label"] == "relevant" for r in sub)
            lo, hi = wilson_interval(k, len(sub))
            entry[f"{who}_precision"] = {
                "relevant": k, "n": len(sub), "precision": k / len(sub), "ci": [lo, hi],
                "passes": len(sub) >= MIN_AUDIT_N and k / len(sub) >= MIN_PRECISION,
            }
        out["aspects"][aspect] = entry
    return out


def score_sentiment(rows: Sequence[dict[str, str]], n_boot: int = N_BOOT) -> dict[str, Any]:
    """Agreement and the sentence gate recomputed on the owner's labels.

    Args:
        rows: Joined rows with ``model_label``, ``model_pred`` and ``owner_label``.
        n_boot: Bootstrap resamples.

    Returns:
        Agreement (owner against model labeler) and the selected rule's
        three-class macro-F1 on each label set, with a bootstrap interval
        for the owner-label score and the gate verdict.
    """
    from src.sentiment_transformer import SENTENCE_GATE_MACRO_F1

    owner = np.array([r["owner_label"] for r in rows])
    pred = np.array([r["model_pred"] for r in rows])
    f1 = macro_f1_multiclass(list(owner), list(pred), SENTIMENT_LABELS)
    ci = bootstrap_interval(
        lambda i: macro_f1_multiclass(list(owner[i]), list(pred[i]), SENTIMENT_LABELS),
        len(rows), n_boot, BOOT_SEED,
    )
    return {
        "agreement": agreement_report(list(owner), [r["model_label"] for r in rows],
                                      SENTIMENT_LABELS, n_boot),
        "owner_label_counts": {c: int((owner == c).sum()) for c in SENTIMENT_LABELS},
        "rule_macro_f1_owner": f1,
        "rule_macro_f1_owner_ci": list(ci),
        "rule_macro_f1_model_labels": macro_f1_multiclass(
            [r["model_label"] for r in rows], list(pred), SENTIMENT_LABELS),
        "gate": SENTENCE_GATE_MACRO_F1,
        "passes_on_owner_labels": f1 >= SENTENCE_GATE_MACRO_F1,
    }


def score(
    aspect_file: Path = RELABEL_DIR / ASPECT_BLIND,
    sentiment_file: Path = RELABEL_DIR / SENTIMENT_BLIND,
    key_dir: Path = KEY_DIR,
    out_path: Path = RELABEL_DIR / RESULTS,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Score the owner's filled files against the key and write the results.

    Args:
        aspect_file: Owner-labelled aspect file.
        sentiment_file: Owner-labelled sentiment file.
        key_dir: Directory with the key files.
        out_path: Results JSON.
        n_boot: Bootstrap resamples.

    Returns:
        ``{"aspects", "sentiment", "affected_findings"}``.
    """
    from src.replication import FINDINGS

    a = _join(read_owner_labels(aspect_file, ASPECT_LABELS), _read_csv(key_dir / ASPECT_KEY))
    s = _join(read_owner_labels(sentiment_file, SENTIMENT_LABELS), _read_csv(key_dir / SENTIMENT_KEY))
    aspects = score_aspects(a, n_boot)
    failing = [k for k, v in aspects["aspects"].items() if not v["owner_precision"]["passes"]]
    result = {
        "aspects": aspects,
        "sentiment": score_sentiment(s, n_boot),
        "aspects_failing_on_owner_labels": failing,
        "affected_findings": [f.key for f in FINDINGS if f.lens == "aspect" and f.theme in failing],
    }
    _safe(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    """CLI entry point: ``export`` or ``score``."""
    parser = argparse.ArgumentParser(description="Blind owner relabel: export and scoring.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export", help="Write the blind files and the key.")
    sc = sub.add_parser("score", help="Score the owner's filled files.")
    sc.add_argument("--aspect-file", type=Path, default=RELABEL_DIR / ASPECT_BLIND)
    sc.add_argument("--sentiment-file", type=Path, default=RELABEL_DIR / SENTIMENT_BLIND)
    args = parser.parse_args()
    try:
        if args.command == "export":
            m = export(load_study(MARKET_STUDY_PATH))
            print(json.dumps({k: m[k] for k in ("n_aspect", "n_sentiment", "aspect_counts")}))
        else:
            r = score(args.aspect_file, args.sentiment_file)
            print(json.dumps({"failing": r["aspects_failing_on_owner_labels"],
                              "gate_on_owner": r["sentiment"]["passes_on_owner_labels"]}))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
