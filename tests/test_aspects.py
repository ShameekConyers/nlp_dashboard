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


RTWP_PATH: Path = PROJECT_ROOT / "studies" / "aspects_rtwp_turn_v1.toml"


@pytest.fixture(scope="module")
def rtwp_lexicon() -> Lexicon:
    """Load the frozen RTwP versus turn-based lexicon once."""
    return load_lexicon(RTWP_PATH)


@pytest.mark.parametrize(
    ("sentence", "aspect"),
    [
        ("I prefer the turn-based mode over the old combat.", "mode_turn_based"),
        ("Playing in WeGo is relaxing.", "mode_turn_based"),
        ("Real-time with pause combat is super nice.", "mode_rtwp"),
        ("I bounced off the RTwP approach.", "mode_rtwp"),
        ("I found myself pausing combat a lot.", "pause_micro"),
        ("It needs constant micromanagement of every unit.", "pause_micro"),
        ("The combat is a chaotic mess.", "realtime_chaos"),
        ("Fights feel cluttered and chaotic with spells everywhere.", "realtime_chaos"),
        ("You wait endlessly for AI turns in the late game.", "waiting_turns"),
        ("Late game turns take forever.", "waiting_turns"),
        ("The combat is slow and repetitive.", "combat_pacing"),
        ("Way too many trash fights.", "combat_pacing"),
    ],
)
def test_rtwp_design_aspects_have_positives(sentence: str, aspect: str, rtwp_lexicon: Lexicon) -> None:
    assert aspect in _aspects(sentence, rtwp_lexicon)


@pytest.mark.parametrize(
    ("sentence", "aspect"),
    [
        ("The Chaos Space Marines are the best faction.", "realtime_chaos"),
        ("My chaotic good rogue stole everything.", "realtime_chaos"),
        ("You decide the battle between order and chaos.", "realtime_chaos"),
        ("I can't wait for their next turn-based game.", "waiting_turns"),
        ("The dragon fights were epic.", "combat_pacing"),
        ("There are no micro-transactions at all.", "pause_micro"),
        ("The pause menu has a nice soundtrack.", "pause_micro"),
    ],
)
def test_rtwp_false_friends_do_not_tag(sentence: str, aspect: str, rtwp_lexicon: Lexicon) -> None:
    assert aspect not in _aspects(sentence, rtwp_lexicon)


RTWP_V2_PATH: Path = PROJECT_ROOT / "studies" / "aspects_rtwp_turn_v2.toml"


@pytest.mark.parametrize(
    "sentence",
    [
        "Greate RPG with real time pause, very different locations.",
        "Combat comes in both turn-based and pause-and-play variants.",
        "I hate realtime crpg combat.",
        "It can be played in real-time, but combat is easier in turn-based mode.",
    ],
)
def test_rtwp_v2_adds_mode_rtwp_phrasings_v1_missed(sentence: str, rtwp_lexicon: Lexicon) -> None:
    v2 = load_lexicon(RTWP_V2_PATH)
    assert v2.version == "rtwp_turn_v2"
    assert "mode_rtwp" in _aspects(sentence, v2)
    assert "mode_rtwp" not in _aspects(sentence, rtwp_lexicon)


def test_rtwp_v2_changes_only_mode_rtwp(rtwp_lexicon: Lexicon) -> None:
    v2 = {a.name: a for a in load_lexicon(RTWP_V2_PATH).aspects}
    v1 = {a.name: a for a in rtwp_lexicon.aspects}
    assert set(v1) == set(v2)
    changed = {n for n in v1 if v1[n] != v2[n]}
    assert changed == {"mode_rtwp"}
    assert set(p.pattern for p in v1["mode_rtwp"].include) < set(p.pattern for p in v2["mode_rtwp"].include)


def test_full_db_audit_is_read_and_validated(tmp_path: Path, rtwp_lexicon: Lexicon) -> None:
    assert rtwp_lexicon.full_db_audit == ("realtime_chaos", "waiting_turns", "combat_pacing")
    assert load_lexicon(LEXICON_PATH).full_db_audit == ()
    bad = tmp_path / "bad.toml"
    bad.write_text(RTWP_PATH.read_text().replace('"combat_pacing"]', '"nope"]', 1))
    with pytest.raises(ValueError, match="nope"):
        load_lexicon(bad)


def test_export_audit_draws_full_db_audit_aspects_from_the_full_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An aspect named in full_db_audit comes from the full DB test split, not the seed."""
    import csv
    import sqlite3

    from src.aspects import export_audit
    from src.aspects import run as run_aspects
    from src.db_setup import create_study_tables
    from src.stats import dev_test_split
    from src.study import AppSpec, Sampling, Study

    monkeypatch.setattr("src.aspects.OUTPUTS_DIR", tmp_path / "out")
    test_ids = [str(i) for i in range(200) if dev_test_split(str(i)) == "test"]
    study = Study(
        name="t", title="t", seed_db=tmp_path / "seed.db", full_db=tmp_path / "full.db",
        raw_dir=tmp_path, language_filter=False, aspect_lexicon=RTWP_PATH,
        sampling=Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 42),
        baseline=None, apps=(AppSpec(1, "G", "CRPG", "c", "", ()),),
    )
    seed_text = "The turn-based mode is great. Late game turns take forever."
    full_text = "Great story overall. Late game turns take forever."
    seed = sqlite3.connect(study.seed_db)
    create_study_tables(seed)
    doc = seed.execute("INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                       "VALUES (?, 1, ?, 1)", (test_ids[0], seed_text)).lastrowid
    seed.execute("INSERT INTO processed_documents VALUES (?, ?, '[]', 5, ?, NULL)",
                 (doc, seed_text, len(seed_text)))
    seed.commit()
    seed.close()
    full = sqlite3.connect(study.full_db)
    create_study_tables(full)
    full.execute("INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                 "VALUES (?, 1, ?, 1)", (test_ids[1], full_text))
    full.commit()
    full.close()

    run_aspects(study)
    with open(export_audit(study), newline="", encoding="utf-8") as f:
        rows = {(r["aspect"], r["steam_review_id"]) for r in csv.DictReader(f)}

    assert ("mode_turn_based", test_ids[0]) in rows
    assert ("waiting_turns", test_ids[1]) in rows
    assert ("waiting_turns", test_ids[0]) not in rows


def test_export_audit_for_a_full_source_study_draws_every_aspect_from_the_full_test_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With stats_source = "full" the seed is not read: all aspects come from the full DB."""
    import csv
    import sqlite3

    from src.aspects import export_audit
    from src.db_setup import create_study_tables
    from src.stats import dev_test_split
    from src.study import AppSpec, Sampling, Study

    monkeypatch.setattr("src.aspects.OUTPUTS_DIR", tmp_path / "out")
    ids = [str(i) for i in range(200)]
    test_ids = [i for i in ids if dev_test_split(i) == "test"]
    dev_ids = [i for i in ids if dev_test_split(i) == "dev"]
    study = Study(
        name="t", title="t", seed_db=tmp_path / "missing_seed.db", full_db=tmp_path / "full.db",
        raw_dir=tmp_path, language_filter=False, aspect_lexicon=RTWP_PATH,
        sampling=Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 42),
        baseline=None, apps=(AppSpec(1, "G", "CRPG", "c", "", ()),),
        stats_split="dev", stats_source="full",
    )
    text = "The turn-based mode is great but the UI is a mess and it crashes."
    full = sqlite3.connect(study.full_db)
    create_study_tables(full)
    full.executemany("INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                     "VALUES (?, 1, ?, 1)", [(i, text) for i in test_ids[:3] + dev_ids[:3]])
    full.commit()
    full.close()

    with open(export_audit(study), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["aspect"] for r in rows} >= {"mode_turn_based", "ui_clarity", "bugs_performance"}
    assert {r["steam_review_id"] for r in rows} == set(test_ids[:3])
    assert not study.seed_db.exists()


RTWP_V3_PATH: Path = PROJECT_ROOT / "studies" / "aspects_rtwp_turn_v3.toml"


def test_rtwp_v3_revises_only_the_listed_aspects() -> None:
    v2 = {a.name: a for a in load_lexicon(RTWP_V2_PATH).aspects}
    v3_lex = load_lexicon(RTWP_V3_PATH)
    v3 = {a.name: a for a in v3_lex.aspects}
    assert v3_lex.version == "rtwp_turn_v3" and v3_lex.supersedes == "rtwp_turn_v2"
    assert list(v2) == list(v3)
    changed = {n for n in v2 if v2[n] != v3[n]}
    assert changed == set(v3_lex.revised)
    assert changed.isdisjoint({"mode_rtwp", "realtime_chaos", "bugs_performance"})


@pytest.mark.parametrize(
    ("sentence", "aspect"),
    [
        ("Age of Wonders 4 is a 4X turn based strategy game.", "mode_turn_based"),
        ("If you like turn-based games you will love it.", "mode_turn_based"),
        ("The real-time with pause combat is fine.", "pause_micro"),
        ("It is waiting for your buildings to finish for 10 turns.", "waiting_turns"),
        ("The game got stuck on the enemy turn forever.", "waiting_turns"),
        ("The combat is tedious and the decisions do not matter.", "combat_pacing"),
        ("The menu theme is my favourite song.", "ui_clarity"),
        ("I bought this on sale.", "price_dlc"),
        ("Gold buys build points and they are expensive (in resources) later.", "price_dlc"),
    ],
)
def test_rtwp_v3_drops_v2_false_matches(sentence: str, aspect: str) -> None:
    v2, v3 = load_lexicon(RTWP_V2_PATH), load_lexicon(RTWP_V3_PATH)
    assert aspect in _aspects(sentence, v2)
    assert aspect not in _aspects(sentence, v3)


@pytest.mark.parametrize(
    ("sentence", "aspect"),
    [
        ("I played it in turn-based mode and loved it.", "mode_turn_based"),
        ("You constantly have to pause to issue orders.", "pause_micro"),
        ("The micromanagement of armies is exhausting.", "pause_micro"),
        ("Late game AI turns take forever.", "waiting_turns"),
        ("You spend more time waiting for your turn than playing.", "waiting_turns"),
        ("Fights drag on and there are too many trash fights.", "combat_pacing"),
        ("The UI is cluttered and hard to read.", "ui_clarity"),
        ("The DLC is overpriced for what it adds.", "price_dlc"),
        ("Wait for a sale, it is not worth full price.", "price_dlc"),
    ],
)
def test_rtwp_v3_keeps_true_matches(sentence: str, aspect: str) -> None:
    assert aspect in _aspects(sentence, load_lexicon(RTWP_V3_PATH))


def test_revised_and_supersedes_are_validated(tmp_path: Path) -> None:
    text = RTWP_V3_PATH.read_text()
    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace('"price_dlc"]', '"nope"]', 1))
    with pytest.raises(ValueError, match="nope"):
        load_lexicon(bad)
    bad.write_text(text.replace('supersedes = "rtwp_turn_v2"\n', "", 1))
    with pytest.raises(ValueError, match="together"):
        load_lexicon(bad)
    assert load_lexicon(RTWP_V2_PATH).revised == () and load_lexicon(RTWP_V2_PATH).supersedes is None


def test_audit_review_ids_reads_only_the_id_column(tmp_path: Path) -> None:
    from src.aspects import audit_review_ids

    path = tmp_path / "a.csv"
    path.write_text('audit_id,steam_review_id,sentence_idx,aspect,aspect_label,sentence,label\n'
                    '1,111,0,ui_clarity,UI,"multi\nline, text",\n2,222,1,price_dlc,P,x,\n')
    assert audit_review_ids(path) == {"111", "222"}


def test_export_audit_of_a_retry_covers_revised_aspects_and_skips_prior_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lexicon with revised and supersedes exports only the revised aspects, from
    test-half reviews absent from the superseded export, and needs that export."""
    import csv
    import sqlite3

    from src.aspects import export_audit
    from src.db_setup import create_study_tables
    from src.stats import dev_test_split
    from src.study import AppSpec, Sampling, Study

    monkeypatch.setattr("src.aspects.OUTPUTS_DIR", tmp_path / "out")
    ids = [str(i) for i in range(200)]
    test_ids = [i for i in ids if dev_test_split(i) == "test"]
    dev_ids = [i for i in ids if dev_test_split(i) == "dev"]
    study = Study(
        name="t", title="t", seed_db=tmp_path / "missing_seed.db", full_db=tmp_path / "full.db",
        raw_dir=tmp_path, language_filter=False, aspect_lexicon=RTWP_V3_PATH,
        sampling=Sampling("natural", "recent", "english", "all", "all", 1, 100, 100, 100, 42),
        baseline=None, apps=(AppSpec(1, "G", "CRPG", "c", "", ()),),
        stats_split="dev", stats_source="full",
    )
    text = "I play in turn-based mode but the UI is a mess and it crashes."
    full = sqlite3.connect(study.full_db)
    create_study_tables(full)
    full.executemany("INSERT INTO documents (steam_review_id, app_id, review_text, recommended) "
                     "VALUES (?, 1, ?, 1)", [(i, text) for i in test_ids[:4] + dev_ids[:3]])
    full.commit()
    full.close()

    with pytest.raises(FileNotFoundError):
        export_audit(study)
    prior = tmp_path / "out" / "audit" / "t_aspect_audit_rtwp_turn_v2_unlabeled.csv"
    prior.parent.mkdir(parents=True)
    prior.write_text("audit_id,steam_review_id,sentence_idx,aspect,aspect_label,sentence,label\n"
                     f"1,{test_ids[0]},0,bugs_performance,B,s,\n")
    out = export_audit(study)
    assert out.name == "t_aspect_audit_rtwp_turn_v3_unlabeled.csv"
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["aspect"] for r in rows} == {"mode_turn_based", "ui_clarity"}
    assert {r["steam_review_id"] for r in rows} == set(test_ids[1:4])
