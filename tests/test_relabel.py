"""Tests for the blind relabel export and its scoring, on synthetic rows."""

import csv
from pathlib import Path

import pytest

from src import relabel


@pytest.fixture()
def dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Point the blind and key directories at a temporary location."""
    blind, key = tmp_path / "relabel", tmp_path / "relabel_key"
    monkeypatch.setattr(relabel, "RELABEL_DIR", blind)
    monkeypatch.setattr(relabel, "KEY_DIR", key)
    return blind, key


def _aspect_rows() -> list[dict]:
    """Ten synthetic rows per relabel aspect; the model labeler marks 8 of 10 relevant."""
    rows = []
    for aspect in relabel.RELABEL_ASPECTS:
        for i in range(10):
            rows.append({
                "steam_review_id": f"{aspect}-{i}", "sentence_idx": i, "aspect": aspect,
                "aspect_label": aspect.title(), "sentence": f"Sentence {i} about {aspect}.",
                "model_label": "relevant" if i < 8 else "not_relevant",
            })
    return rows


def _sentiment_rows() -> list[dict]:
    """Thirty synthetic sentiment rows with labeler labels and rule predictions."""
    labels = ["positive"] * 10 + ["negative"] * 10 + ["neither"] * 10
    preds = ["positive"] * 9 + ["neither"] + ["negative"] * 8 + ["neither"] * 2 + \
        ["neither"] * 6 + ["negative"] * 4
    return [
        {"audit_id": str(i + 1), "steam_review_id": f"s{i}", "sentence_idx": 0,
         "aspect": "pacing", "sentence": f"Sentiment sentence {i}.", "model_label": lab,
         "model_pred": pred}
        for i, (lab, pred) in enumerate(zip(labels, preds))
    ]


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fill(path: Path, labels: dict[str, str]) -> None:
    """Write owner labels into a blind file in place."""
    rows = _read(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "label": labels[r["row_id"]]})


def test_blind_files_carry_no_model_columns_and_key_maps_back(dirs: tuple[Path, Path]) -> None:
    """Blind headers hold only row ID, the question and an empty label; the key restores each row."""
    blind, key = dirs
    aspects, sentiments = _aspect_rows(), _sentiment_rows()
    relabel.write_exports(aspects, sentiments, blind, key, seed=5)
    a_blind, s_blind = _read(blind / relabel.ASPECT_BLIND), _read(blind / relabel.SENTIMENT_BLIND)
    assert list(a_blind[0]) == ["row_id", "aspect", "aspect_label", "sentence", "label"]
    assert list(s_blind[0]) == ["row_id", "sentence", "label"]
    assert all(r["label"] == "" for r in a_blind + s_blind)
    assert [r["sentence"] for r in a_blind] != [r["sentence"] for r in aspects]
    a_key = {r["row_id"]: r for r in _read(key / relabel.ASPECT_KEY)}
    by_sentence = {r["sentence"]: r for r in aspects}
    for r in a_blind:
        src = by_sentence[r["sentence"]]
        assert a_key[r["row_id"]]["steam_review_id"] == src["steam_review_id"]
        assert a_key[r["row_id"]]["model_label"] == src["model_label"]
    s_key = {r["row_id"]: r for r in _read(key / relabel.SENTIMENT_KEY)}
    by_s = {r["sentence"]: r for r in sentiments}
    assert all(s_key[r["row_id"]]["model_pred"] == by_s[r["sentence"]]["model_pred"] for r in s_blind)


def test_export_refuses_to_overwrite_owner_labels(dirs: tuple[Path, Path]) -> None:
    """Once the owner has labelled a row, a new export is refused."""
    blind, key = dirs
    relabel.write_exports(_aspect_rows(), _sentiment_rows(), blind, key)
    relabel.write_exports(_aspect_rows(), _sentiment_rows(), blind, key)  # unlabelled: fine
    rows = _read(blind / relabel.ASPECT_BLIND)
    _fill(blind / relabel.ASPECT_BLIND, {r["row_id"]: "" for r in rows} | {rows[0]["row_id"]: "relevant"})
    with pytest.raises(ValueError, match="already holds labels"):
        relabel.write_exports(_aspect_rows(), _sentiment_rows(), blind, key)


def test_score_recomputes_precision_agreement_and_gate(dirs: tuple[Path, Path]) -> None:
    """Owner labels change pacing's precision to 7/10 (fails), others stay at 8/10."""
    blind, key = dirs
    relabel.write_exports(_aspect_rows(), _sentiment_rows(), blind, key, seed=1)
    a_key = _read(key / relabel.ASPECT_KEY)
    owner_a = {r["row_id"]: r["model_label"] for r in a_key}
    flip = next(r["row_id"] for r in a_key if r["aspect"] == "pacing" and r["model_label"] == "relevant")
    owner_a[flip] = "not_relevant"
    _fill(blind / relabel.ASPECT_BLIND, owner_a)
    s_key = _read(key / relabel.SENTIMENT_KEY)
    _fill(blind / relabel.SENTIMENT_BLIND, {r["row_id"]: r["model_label"] for r in s_key})

    res = relabel.score(blind / relabel.ASPECT_BLIND, blind / relabel.SENTIMENT_BLIND, key,
                        blind / relabel.RESULTS, n_boot=200)
    pacing = res["aspects"]["aspects"]["pacing"]
    assert pacing["owner_precision"]["relevant"] == 7 and not pacing["owner_precision"]["passes"]
    assert pacing["model_precision"]["precision"] == pytest.approx(0.8)
    assert res["aspects"]["overall"]["percent_agreement"] == pytest.approx(39 / 40)
    # n = 10 per aspect is below the 30-sentence minimum, so no aspect passes here.
    assert set(res["aspects_failing_on_owner_labels"]) == set(relabel.RELABEL_ASPECTS)
    assert "aspect:pacing@subgroup:wego" in res["affected_findings"]
    sent = res["sentiment"]
    assert sent["agreement"]["kappa"] == pytest.approx(1.0)
    assert sent["rule_macro_f1_owner"] == pytest.approx(sent["rule_macro_f1_model_labels"])
    assert (blind / relabel.RESULTS).exists()


def test_read_owner_labels_rejects_blank_rows(dirs: tuple[Path, Path]) -> None:
    """Every row needs a valid label before scoring."""
    blind, key = dirs
    relabel.write_exports(_aspect_rows(), _sentiment_rows(), blind, key)
    with pytest.raises(ValueError, match="lack a valid label"):
        relabel.read_owner_labels(blind / relabel.ASPECT_BLIND, relabel.ASPECT_LABELS)


def test_writes_outside_relabel_dirs_are_refused(dirs: tuple[Path, Path], tmp_path: Path) -> None:
    """The writer only touches the blind and key directories."""
    with pytest.raises(ValueError, match="outside"):
        relabel._safe(tmp_path / "elsewhere.csv")
