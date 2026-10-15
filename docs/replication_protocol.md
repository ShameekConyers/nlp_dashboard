# Replication protocol v1: Steam market study

Written October 15, 2026, before any statistic was computed on the replication sample. The protocol, the code that runs it (`src/replication.py`) and the pinned hashes are committed together and tagged `replication-protocol-v1`. The confirmatory read runs only at a clean checkout of that tag, once. Anyone can check the order with `git log` and `git show replication-protocol-v1`.

## Summary

- **What is tested:** the 24 lift findings of the market study (seed stats run 10 in `data/market_seed.db`), grouped into 8 theme families.
- **New data:** 36,774 reviews from `data/market_full.db` that are in neither the seed nor any audit file, up to 2,000 per game, drawn with seed 20261015. The draw is already done and pinned by its SHA-256. It read review IDs, creation times and texts, never the recommend flag.
- **Criterion per finding:** the replication lift lies on the original side of 1 and its 95% two-stage cluster-bootstrap interval excludes 1 on that side.
- **Expected count from power:** 21.8 of 24 if the original effects are the truth, 14.8 if each game's log lift is half as large. The 5th percentile of the second scenario is 12.
- **Decision rule:** 12 or more replicated findings is "consistent with power". Fewer is "below power", which says the original set overstated its effects. Every count is published, including failures and findings that cannot be tested.
- **Games that cannot be resampled:** 8 games have no review left outside the seed. 4 of them were eligible in the original study (Frozen Synapse 2, Combat Mission: Battle for Normandy, Three Kingdoms: The Last Warlord, Romance of the Three Kingdoms 8 Remake). The plan's count of four is confirmed.

## 1. Findings under test

The 24 findings are the `findings` list of `analysis_runs` row 10 (stage `stats`) in `data/market_seed.db`, the same rows as the README's topic and design-theme finding tables. `src/replication.py` refuses to run if that list differs from its own `FINDINGS`. Lifts and intervals below are copied from that run.

| Family | Scope | Side | Original lift (95% CI) | Eligible games, original / resamplable | Power, plug-in | Power, half log lift |
|---|---|---|---|---|---|---|
| Crash topic (topic 11) | Turn-structure cohort | complaint | 12.84 (7.25 to 29.33) | 18 / 16 | 1.00 | 1.00 |
| Crash topic (topic 11) | IGO-UGO | complaint | 12.08 (4.95 to 52.08) | 5 / 5 | 1.00 | 0.96 |
| Crash topic (topic 11) | WEGO | complaint | 11.14 (4.92 to 41.68) | 7 / 5 | 0.99 | 0.76 |
| Final Fantasy Tactics topic (topic 1) | Alternating, matched basket | praise | 0.47 (0.24 to 0.74) | 3 / 3 | 1.00 | 1.00 |
| Bugs and performance | Turn-structure cohort | complaint | 2.53 (1.94 to 3.51) | 18 / 16 | 1.00 | 1.00 |
| Bugs and performance | IGO-UGO | complaint | 2.66 (1.12 to 5.70) | 5 / 5 | 0.82 | 0.27 |
| Bugs and performance | Alternating | complaint | 3.11 (1.90 to 4.66) | 6 / 6 | 1.00 | 0.95 |
| Bugs and performance | Alternating, matched basket | complaint | 2.59 (1.18 to 4.64) | 3 / 3 | 0.93 | 0.37 |
| Bugs and performance | Anchors | complaint | 4.86 (3.32 to 8.43) | 3 / 3 | 1.00 | 0.99 |
| Pacing | Turn-structure cohort | complaint | 2.82 (1.90 to 3.96) | 18 / 16 | 1.00 | 0.99 |
| Pacing | IGO-UGO | complaint | 2.61 (1.45 to 4.62) | 5 / 5 | 0.99 | 0.53 |
| Pacing | Alternating | complaint | 3.27 (1.66 to 6.20) | 6 / 6 | 1.00 | 0.96 |
| Pacing | WEGO | complaint | 2.44 (1.16 to 4.27) | 7 / 5 | 0.87 | 0.49 |
| UI and interface | Turn-structure cohort | complaint | 2.29 (1.63 to 3.15) | 18 / 16 | 1.00 | 0.99 |
| UI and interface | IGO-UGO | complaint | 1.91 (1.06 to 2.96) | 5 / 5 | 0.84 | 0.19 |
| UI and interface | IGO-UGO, matched basket | complaint | 1.97 (1.14 to 3.16) | 4 / 4 | 0.91 | 0.26 |
| UI and interface | Alternating | complaint | 2.01 (1.02 to 4.29) | 6 / 6 | 0.89 | 0.24 |
| UI and interface | WEGO | complaint | 2.57 (1.51 to 4.13) | 7 / 5 | 1.00 | 0.80 |
| Price and DLC | Turn-structure cohort | complaint | 1.42 (1.09 to 1.77) | 18 / 16 | 0.44 | 0.16 |
| Price and DLC | WEGO | complaint | 1.71 (1.25 to 2.43) | 7 / 5 | 0.57 | 0.07 |
| Price and DLC | Anchors | complaint | 1.88 (1.43 to 2.43) | 3 / 3 | 1.00 | 0.97 |
| Enemy AI quality | Turn-structure cohort | complaint | 1.91 (1.05 to 3.37) | 18 / 16 | 0.73 | 0.03 |
| Enemy AI quality | Alternating | complaint | 4.94 (1.79 to 19.30) | 6 / 6 | 1.00 | 0.71 |
| Combat preview and hit chance | Alternating | complaint | 5.77 (1.16 to 13.24) | 6 / 6 | 0.88 | 0.10 |

Family sizes: crash topic 3, Final Fantasy Tactics topic 1, bugs and performance 5, pacing 4, UI 5, price and DLC 3, enemy AI 2, combat preview 1. Power columns are explained in section 6.

## 2. Replication criterion

After the Open Science Collaboration (2015, "Estimating the reproducibility of psychological science", Science 349), which counted a study as replicated when the replication effect was significant in the original direction, and also reported whether the original estimate fell inside the replication interval.

**Primary criterion (decides the count).** A complaint finding replicates when the replication point lift is above 1 and the lower bound of its 95% interval is above 1. The praise finding replicates when the point lift is below 1 and the upper bound is below 1. An undefined lift or interval does not replicate.

**Reported, not deciding:**
- whether the full finding rule (`stats.FINDING_RULE`: at least 3 eligible games, interval off 1, leave-one-game-out stable, at least 5 mentions on each side) holds in the same direction,
- whether the original point lift lies inside the replication interval,
- whether the replication point lift lies inside the original interval,
- the ratio of replication to original log lift (how much of the effect survives).

**Uncertainty.** The replication lift and interval come from the unchanged study code: `stats.macro_theme_metrics` (each eligible game weighted equally) and `stats.cluster_bootstrap_macro` with 2,000 resamples and seed 42, games then reviews. Stability comes from `stats.leave_one_game_out` with the original rank pools (passing aspects among passing aspects, topics among topics).

**Multiplicity.** The 24 tests are fixed in advance with their direction, so each is a one-sided test at 2.5%. If no finding were real, the expected number of chance replications is 24 x 0.025 = 0.6. The permutation null (plan item N4) is separate and not part of this read.

## 3. Scopes and games

Each finding keeps its original scope. Its replication games are the games that were eligible for that scope in the original run, that have reviews in the replication sample, and that meet the original eligibility rule in the replication sample: at least 100 kept reviews and at least 30 not-recommended kept reviews (`stats.MIN_KEPT_FOR_ELIGIBLE`, `stats.MIN_NOT_REC_FOR_ELIGIBLE`). No game joins a scope it was not eligible for originally. A finding with fewer than 3 replication games is "not testable". It is reported with its count and counts neither as a success nor as a failure.

**Games that cannot be resampled** (no review in `market_full.db` outside the seed, from `outputs/replication_2026/sample_manifest.json`):

| Game | Reviews in the full database | Eligible originally |
|---|---|---|
| Frozen Synapse 2 | 213 | yes (WEGO) |
| Combat Mission: Battle for Normandy | 213 | yes (WEGO) |
| Three Kingdoms: The Last Warlord | 371 | yes (historical peers) |
| Romance of the Three Kingdoms 8 Remake | 393 | yes (historical peers) |
| Romance of the Three Kingdoms X with Power Up Kit | 80 | no |
| Flashpoint Campaigns: Southern Storm | 89 | no |
| Nobunaga's Ambition: Sphere of Influence - Ascension | 167 | no |
| Gary Grigsby's War in the East 2 | 289 | no |

These match the plan's eight counts (80, 89, 167, 213, 213, 289, 371, 393) and its four eligible games. No finding sits in the historical-peer cohort, so its loss removes no finding. WEGO loses 2 of its 7 eligible games.

**At risk:** Combat Mission: Shock Force 2 (WEGO) has 165 fresh reviews. At its seed kept rate and recommend share it projects to about 135 kept and 28 not-recommended reviews, under the 30 floor, so it will probably drop at read time and leave WEGO with 4 games. The power simulation applies the same rule in every synthetic replication.

## 4. Sampling frame and draw

- **Source:** `data/market_full.db`, the 208,380 loaded reviews of the September 26, 2026 pull. The seed was a uniform sample of the same rows, so the replication covers the same windows: full history for the 22 uncapped games, and the same recent window for the 7 capped games (XCOM 2 from December 2021, BattleTech from October 2018, Divinity: Original Sin 2 from January 2023, Baldur's Gate 3 from February 2026, Civilization VI from August 2024, Stellaris from February 2024, Total War: WARHAMMER III from March 2025). Per-game first and last creation dates of the drawn reviews are in the manifest.
- **Excluded:** 12,424 review IDs, the 12,315 seed reviews and 109 further reviews that appear only in the aspect audit's full-database rows. Every committed file in `data/labels/` with a `steam_review_id` column is read for the exclusion list, which covers the aspect audit and the three sentence sentiment audits.
- **Draw:** per game, the sorted fresh IDs, then `random.Random(f"20261015:{app_id}").sample(ids, min(n, 2000))`. This mirrors the seed's own draw. Games with fewer than 2,000 fresh reviews contribute all of them.
- **Drawn:** 36,774 reviews from 21 games, 2,000 each from 15 games and every fresh review from 6 (Panzer Corps 2 1,640, Unity of Command II 741, Order of Battle 1,432, Combat Mission: Shock Force 2 165, Frozen Synapse 1,443, Dominions 5 1,353). At seed kept rates about 28,200 will pass the text filters.
- **Stored:** `outputs/replication_2026/sample.db` (review ID, app ID, creation time, raw text, draw order, and no recommend flag) and `sample_manifest.json`. Git ignores both because they hold review text.
- **Pinned:** SHA-256 of the sorted `app_id<TAB>steam_review_id` lines, `5de0fbfeb7e4d5470ac18ab904cd24ad91585aa160a0ad4cfc532add5c2a480e` (`EXPECTED_SAMPLE_SHA256`). The read recomputes it and refuses on a mismatch. It also checks that no sampled ID is in the seed or an audit file, and that every sampled text still equals the full database's text.

## 5. Frozen instruments

Nothing is refit or retuned on the replication sample.

- **Text rules:** `preprocessing.Preprocessor` and `LanguageFilter` with the study's settings (cleaning, at least 10 characters, at least 3 content tokens, lingua non-English at confidence 0.50 or higher is dropped).
- **Lexicon:** `studies/aspects_strategy_v1.toml`, version `strategy_v1`, SHA-256 `5ce3820e6189b11c498407d02fc778799649cc20c3f09ef57ad35a4fc656ff3a` (`EXPECTED_LEXICON_SHA256`), applied per sentence with the same sentencizer. Only the seven aspects that passed the precision audit enter the analysis, as in the original run.
- **Topics:** the saved model `outputs/strategy_market_2026/bertopic_model/model.pkl`, SHA-256 `0692f55a0ea318b12f224fe6021047c20a9229fd8cc6c99b2e6379fbe402f417` (`EXPECTED_MODEL_SHA256`), applied with `BERTopic.transform`. It is the model behind the seed: its stored assignments equal the seed's topic column for all 9,622 kept reviews.
- **What transform does to new text** (measured on the seed, `calibration.json`, `topic_transform_agreement`): passing all 9,622 seed reviews at once returns the fitted assignments exactly, because umap-learn recognises its training input. That is a reproduction check only. Passing the seed in two halves forces the path new reviews take: 93.6% of reviews keep their fitted topic, the outlier share rises from 42.0% to 44.1%, the crash topic keeps 85.6% of its 146 reviews and 92.6% of its transform assignments are fitted crash reviews, and the Final Fantasy Tactics topic keeps 97.3%. New reviews cannot do better than this, so topic shares in the replication are a slightly noisier instrument than in the original. Lift is a ratio within one instrument, so this shift moves both sides of it.
- **Statistics:** `src/stats.py` unchanged (section 2). VADER and transformer sentiment are not part of the finding rule and are not computed on the replication sample.

## 6. Power

Computed by `src/replication.py calibrate` from the seed only (results in `outputs/replication_2026/calibration.json`, reproducible with the same command and seed 20261016).

- **Method:** 500 synthetic replications per scenario. Each one draws, for every resamplable game, a kept count from Binomial(drawn reviews, seed kept rate) and that many kept seed reviews with replacement, applies the eligibility rule, and checks all 24 findings with the primary criterion and a 1,000-resample cluster bootstrap. Findings share each synthetic replication, so the count keeps their correlation.
- **Bootstrap check:** the fast bootstrap used in the simulation reproduces every stored point lift exactly and every stored interval within Monte Carlo noise (for example bugs and performance in the turn-structure cohort: stored 1.94 to 3.51, fast 1.95 to 3.56).
- **Plug-in scenario:** each game's seed data is the truth. Expected count 21.8, 5th to 95th percentile 19 to 24.
- **Half log lift scenario:** each game's theme flags are thinned on the dominant side so its log lift is halved, which reflects the drop in effect size the Open Science Collaboration found and the winner's curse on findings selected by an interval rule. Expected count 14.8, 5th to 95th percentile 12 to 18.
- **Weakest findings:** enemy AI in the turn-structure cohort (power 0.73 plug-in, 0.03 halved), price and DLC in the turn-structure cohort (0.44, 0.16) and in WEGO (0.57, 0.07), combat preview in alternating (0.88, 0.10), UI in IGO-UGO (0.84, 0.19). These include the lower bounds near 1 the plan named as risks.

## 7. Decision rule

1. **Per finding:** "replicated", "not replicated" or "not testable" by sections 2 and 3.
2. **Per family:** "replicated" when more than half of its testable findings replicate, "not replicated" otherwise, "not testable" when none is testable. A family of two needs both.
3. **Study level:** 12 or more replicated findings (`COUNT_THRESHOLD`, the 5th percentile of the half log lift scenario) is "consistent with power". Fewer than 12 is "below power": the original findings as a set overstated their effects.
4. **What changes in the README:** every finding keeps its row and gains its replication status and replication lift with interval. Replicated findings may be called confirmatory. Findings that did not replicate stay as exploratory, marked "did not replicate". No finding is removed and none is reworded to fit the result.
5. **One read.** The read refuses to run if `outputs/replication_2026/confirmatory/results.json` exists. If it fails with an error before that file is written, the fix is committed, a new tag `replication-protocol-v1.1` is made (with `PROTOCOL_TAG` changed to match in the same commit), and the report lists the failure as a deviation. Criterion, sample, thresholds and instruments may not change in that commit.

## 8. Reporting template

`src/replication.py read` writes `outputs/replication_2026/confirmatory/results.json` and `report.md`. The published report copies them in full, in this shape:

- Header: commit, tag, sample SHA-256, reviews read and kept, games not resampled.
- Counts: findings (24), testable, replicated, not replicated, not testable, full rule met in the same direction, original lift inside the replication interval, replication lift inside the original interval, families replicated / not replicated / not testable, expected counts (21.8 and 14.8), threshold (12) and the study verdict.
- Family table: findings, testable, replicated, not replicated, not testable and verdict for each of the 8 families.
- Finding table: all 24 rows with original lift and interval, original and replication game counts, replication lift and interval, status, full-rule result and stability.
- Deviations from this protocol, or "none".

Every count is published whatever it shows, as the README already does for the two failed sentence audits.

## 9. Blind relabel (plan item N3)

Run separately from the replication and committed under the same tag, so its rules are also fixed before any owner label exists.

- **Sets:** the 160 committed aspect audit sentences of pacing (32 of 40 relevant by the model labeler, exactly at the 0.80 floor), bugs and performance (34), UI (37) and price and DLC (33), and the 200 sentences of the v3 sentence sentiment test set.
- **Blind files:** `outputs/relabel_2026/aspect_relevance_blind.csv` (`row_id, aspect, aspect_label, sentence, label`) and `outputs/relabel_2026/sentence_sentiment_blind.csv` (`row_id, sentence, label`), shuffled with seed 20261017 so rows are not grouped by aspect, and with no model label, score or review ID. The key, `outputs/relabel_2026_key/`, maps each row to its review, the model labeler's label and the selected sentence rule's class. Git ignores both directories. Instructions: `docs/relabel_instructions.md`.
- **Scores** (`src/relabel.py score`): Cohen's kappa and Gwet's AC1 between the owner and the model labeler with 2,000-resample bootstrap intervals, overall and per aspect. Each aspect's precision on the owner's labels with a Wilson interval against the 0.80 floor on at least 30 sentences. The selected sentence rule's three-class macro-F1 on the owner's labels against the 0.70 gate, with a bootstrap interval. On the model labels the rule scores 0.707, the committed v3 result.
- **Consequences:** an aspect whose precision falls below 0.80 on the owner's labels fails its audit, and its findings become descriptive whatever the replication shows (pacing carries 4 findings). A sentence macro-F1 below 0.70 on the owner's labels returns the sentence view to unvalidated. Both outcomes are reported.
- **Independence:** the owner wrote the lexicon. His labels are independent of the model labeler, not of the lexicon's design.

## 10. Limits stated in advance

- The replication uses the same games. It tests whether the findings hold on other reviews of these games, not whether they generalise to other games.
- The reviews come from the same windows as the seed, so this is not an out-of-time test.
- The plug-in power uses the seed's estimates, which are selected and so likely inflated. The half log lift scenario is the planning case.
- Topic membership in the replication comes from `transform`, which agrees with the fitted assignments on 93.6% of seed reviews when forced through the new-data path (section 5).

## 11. Commands

```bash
# Already run before this protocol was committed (no outcome read):
.venv/bin/python src/replication.py draw
.venv/bin/python src/replication.py calibrate
.venv/bin/python src/replication.py dry-run
.venv/bin/python src/relabel.py export

# After the commit and tag:
.venv/bin/python src/replication.py gate      # must print "gate open"
.venv/bin/python src/replication.py read      # the single confirmatory read
.venv/bin/python src/relabel.py score         # after the owner fills both blind files
```
