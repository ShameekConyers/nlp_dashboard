"""Tests for the phrase-lexicon aspect tagger."""

from pathlib import Path

import pytest

from src.aspects import Lexicon, SentenceSplitter, load_lexicon, tag_sentences
from src.utils import PROJECT_ROOT

LEXICON_PATH: Path = PROJECT_ROOT / "studies" / "aspects_strategy_v1.toml"


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    """Load the frozen strategy lexicon once.

    Returns:
        The compiled lexicon.
    """
    return load_lexicon(LEXICON_PATH)


def _aspects(sentence: str, lexicon: Lexicon) -> set[str]:
    """Return the aspect names tagged in one sentence.

    Args:
        sentence: Sentence text.
        lexicon: Compiled lexicon.

    Returns:
        Set of aspect names.
    """
    return {m.aspect for m in tag_sentences([sentence], lexicon)}


@pytest.mark.parametrize(
    "sentence",
    [
        "The store page uses AI art everywhere.",
        "Half the portraits are AI generated.",
        "The AI voice acting is flat.",
        "This is AI slop and the devs should be ashamed.",
    ],
)
def test_generative_ai_does_not_tag_ai_quality(sentence: str, lexicon: Lexicon) -> None:
    """Generative-AI complaints are excluded from enemy AI quality."""
    assert "ai_quality" not in _aspects(sentence, lexicon)


@pytest.mark.parametrize(
    "sentence",
    [
        "Every turn feels meaningful.",
        "I am waiting for a sale before buying the DLC.",
    ],
)
def test_turn_alone_and_sale_waiting_do_not_tag_turn_waiting(
    sentence: str, lexicon: Lexicon
) -> None:
    """The word 'turn' alone and waiting for a sale are not turn waiting."""
    assert "turn_waiting" not in _aspects(sentence, lexicon)


@pytest.mark.parametrize(
    ("sentence", "aspect"),
    [
        ("The enemy turns take forever on large maps.", "turn_waiting"),
        ("Whoever gets the alpha strike wins the fight.", "first_strike"),
        ("The turn order display makes initiative easy to read.", "initiative_clarity"),
        ("I missed three 95% shots in a row.", "preview_reliability"),
        ("The enemy AI is surprisingly clever at flanking.", "ai_quality"),
        ("The maps have great chokepoints and varied terrain.", "map_design"),
        ("The late game becomes a tedious grind.", "pacing"),
    ],
)
def test_each_design_aspect_has_a_positive(sentence: str, aspect: str, lexicon: Lexicon) -> None:
    """One representative sentence per design aspect is tagged."""
    assert aspect in _aspects(sentence, lexicon)


def test_splitter_offsets_recover_sentences() -> None:
    """Sentence offsets slice the original text back to each sentence."""
    text = "Great maps. The AI is dumb! Would buy again."
    spans = SentenceSplitter().split([text])[0]
    assert [text[s:e] for s, e, _ in spans] == [t for _, _, t in spans]
    assert len(spans) == 3


def test_full_db_mentions_uses_test_split_and_kept_rules(tmp_path: Path) -> None:
    """Full-DB audit draws only test-split, kept English reviews, tagged per sentence."""
    import sqlite3

    from src.aspects import full_db_mentions
    from src.db_setup import create_study_tables
    from src.stats import dev_test_split
    from src.study import AppSpec, Sampling, Study

    test_ids = [str(i) for i in range(100) if dev_test_split(str(i)) == "test"][:3]
    dev_id = next(str(i) for i in range(100) if dev_test_split(str(i)) == "dev")
    reviews = [
        (test_ids[0], "Great maps. The enemy turns take forever on large maps."),
        (test_ids[1], "Die Züge der Gegner dauern ewig, enemy turns take forever und das nervt sehr."),
        (test_ids[2], "Solid tactics, nothing about waiting here at all."),
        (dev_id, "The enemy turns take forever on large maps."),
    ]
    study = Study(
        name="t", title="t", seed_db=tmp_path / "seed.db", full_db=tmp_path / "full.db",
        raw_dir=tmp_path, language_filter=True, aspect_lexicon=LEXICON_PATH,
        sampling=Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 42),
        baseline=None, apps=(AppSpec(1, "G", "Tactics", "c", "", ()),),
    )
    conn = sqlite3.connect(study.full_db)
    create_study_tables(conn)
    conn.executemany(
        "INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
        "VALUES (?, 1, ?, 1)", reviews,
    )
    conn.commit()
    conn.close()

    rows = full_db_mentions(study, load_lexicon(LEXICON_PATH), frozenset({"turn_waiting"}))

    assert rows == [(test_ids[0], "turn_waiting", 1, "The enemy turns take forever on large maps.")]
