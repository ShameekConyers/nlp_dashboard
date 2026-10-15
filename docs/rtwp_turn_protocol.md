# Phase B protocol: real-time-with-pause versus turn-based combat

Written October 15, 2026 and revised the same day: after phase A was rerun on the development half of the full database, after the first aspect audit failed six aspects and lexicon v3 was built as the single retry, and after the v3 audit passed all six (section 13). No statistic has been computed on the test half. The audit exports read test-half text, and their labels are the only test-half judgments made before the tag. Every other test-half step runs inside the read, after its gate passes (section 11). Phase A, the exploratory half, is `docs/rtwp_turn_study_design.md`. This protocol is committed together with the code that runs the read and tagged `rtwp-turn-protocol-v1`, and the read runs once, at a clean checkout of that tag.

## Summary

- **What is tested:** the 66 candidate findings of phase A (`outputs/rtwp_turn_2026/exploratory/candidates_run_4.json`, run 4 of the analysis database `rtwp_turn_2026_full_dev.db`, lexicon `rtwp_turn_v3`), in 14 families. 14 are combat-model aspects: `combat_pacing` (3), `pause_micro` (6), `mode_turn_based` (2, praise side), `realtime_chaos` (2) and `waiting_turns` (1). The other 52 are bugs and performance (9), UI clarity (9), price and DLC (7) and six topics (27). 64 are complaint findings and 2 praise findings. The v2 candidate set (run 3, 70 candidates) is superseded (section 1).
- **Aspect audit, two attempts (section 13):** the v2 audit passed `mode_rtwp` (0.900), `realtime_chaos` (0.825) and `bugs_performance` (0.925) and failed the other six. Lexicon v3 revised those six as the single retry, and the v3 audit passed all six: `mode_turn_based` 1.000, `pause_micro` 0.950, `waiting_turns` 0.950, `combat_pacing` 0.800, `ui_clarity` 0.850 and `price_dlc` 0.825, each on 40 sentences. All 66 candidates are testable by the audit rule. The result is final. A lexicon change and a labeler change are confounded across the two audits.
- **New data:** every test-half review of the full database except the 359 reviews of the v2 audit set and the 240 reviews of the v3 audit set, 119,485 in all, pinned as a list of review IDs (`data/frames/rtwp_turn_2026_phase_b_frame.tsv`, section 4). No recommend flag, tag or text of a test-half review has been read for an outcome.
- **Criterion per finding:** the phase B lift lies on the candidate's side of 1 and its 95% two-stage cluster-bootstrap interval excludes 1 on that side.
- **Expected count from power:** 60.0 of 66 if the development-half effects are the truth, 46.7 if each game's log lift is half as large (5th to 95th percentile 43 to 51).
- **Decision rule:** one count of the 66. A count of 43 or more, the 5th percentile of the half log lift scenario, is "consistent with power". Fewer is "below power". Every count is published.
- **Weak spot:** the combat-model candidates. 9.8 of the 14 are expected to replicate at full effect and 2.5 at half, against 50.2 and 44.3 of the other 52. Their own count is reported beside the main count, with no verdict.

## 1. Findings under test

The 66 candidates come from `stats.py --candidates` on a copy of the analysis database, the development half of the full database, with lexicon `rtwp_turn_v3` and every aspect assumed to pass its audit (run 4). Lifts and intervals are copied from that run. "Eligible in phase A" counts development-half eligibility. Phase B applies the rule to its own sample over every game in the scope (section 3). Power columns are the share of the 500 synthetic samples in which the candidate replicates, from the power run on the v3 candidates (section 6).

Against the v2 set (run 3, `candidates_run_3.json`), v3 drops 6 candidates and adds 2. Dropped: `combat_pacing` in `dual_mode`, `strategy`, `crpg/rtwp` and `strategy/turn_based`, `mode_turn_based` (praise) in `strategy`, and `pause_micro` in `crpg`. Added: price and DLC in `crpg/turn_based` and `strategy/turn_based`. The other 64 candidates are in both sets. The lifts of revised aspects moved with their narrower patterns.

| Family | Scope | Side | Development lift (95% CI) | Eligible in phase A / games in scope | Power, plug-in | Power, half log lift |
|---|---|---|---|---|---|---|
| `combat_pacing` | `crpg` | complaint | 4.61 (2.00 to 9.71) | 10 / 10 | 0.99 | 0.11 |
| `combat_pacing` | `turn_based` (pooled) | complaint | 4.43 (2.07 to 10.13) | 10 / 10 | 1.00 | 0.42 |
| `combat_pacing` | `crpg/turn_based` | complaint | 5.89 (2.79 to 14.75) | 5 / 5 | 1.00 | 0.39 |
| `pause_micro` | `dual_mode` | complaint | 1.90 (1.23 to 2.63) | 5 / 6 | 0.61 | 0.10 |
| `pause_micro` | `strategy` | complaint | 1.90 (1.16 to 2.97) | 10 / 10 | 0.78 | 0.17 |
| `pause_micro` | `rtwp` (pooled) | complaint | 1.60 (1.03 to 2.30) | 10 / 10 | 0.44 | 0.04 |
| `pause_micro` | `turn_based` (pooled) | complaint | 2.33 (1.21 to 4.18) | 10 / 10 | 0.70 | 0.15 |
| `pause_micro` | `crpg/turn_based` | complaint | 5.21 (2.08 to 15.90) | 5 / 5 | 0.82 | 0.13 |
| `pause_micro` | `strategy/rtwp` | complaint | 1.78 (1.29 to 2.81) | 5 / 5 | 0.77 | 0.09 |
| `mode_turn_based` | `turn_based` (pooled) | praise | 0.63 (0.39 to 0.99) | 10 / 10 | 0.47 | 0.27 |
| `mode_turn_based` | `strategy/turn_based` | praise | 0.47 (0.25 to 0.74) | 5 / 5 | 0.85 | 0.52 |
| `realtime_chaos` | `turn_based` (pooled) | complaint | 4.76 (1.78 to 13.67) | 10 / 10 | 0.56 | 0.05 |
| `realtime_chaos` | `crpg/turn_based` | complaint | 4.69 (1.23 to 17.92) | 5 / 5 | 0.34 | 0.01 |
| `waiting_turns` | `turn_based` (pooled) | complaint | 2.96 (1.26 to 6.42) | 10 / 10 | 0.51 | 0.03 |
| Bugs and performance | `crpg` | complaint | 2.87 (2.36 to 3.59) | 10 / 10 | 1.00 | 1.00 |
| Bugs and performance | `dual_mode` | complaint | 2.21 (1.68 to 3.14) | 5 / 6 | 1.00 | 0.99 |
| Bugs and performance | `strategy` | complaint | 3.89 (3.13 to 4.91) | 10 / 10 | 1.00 | 1.00 |
| Bugs and performance | `rtwp` (pooled) | complaint | 3.37 (2.56 to 4.56) | 10 / 10 | 1.00 | 1.00 |
| Bugs and performance | `turn_based` (pooled) | complaint | 3.17 (2.71 to 3.86) | 10 / 10 | 1.00 | 1.00 |
| Bugs and performance | `crpg/rtwp` | complaint | 2.90 (2.10 to 4.24) | 5 / 5 | 1.00 | 1.00 |
| Bugs and performance | `crpg/turn_based` | complaint | 2.84 (2.47 to 3.37) | 5 / 5 | 1.00 | 1.00 |
| Bugs and performance | `strategy/rtwp` | complaint | 4.24 (3.40 to 5.90) | 5 / 5 | 1.00 | 1.00 |
| Bugs and performance | `strategy/turn_based` | complaint | 3.62 (2.69 to 5.05) | 5 / 5 | 1.00 | 1.00 |
| UI clarity | `crpg` | complaint | 2.48 (1.69 to 3.83) | 10 / 10 | 1.00 | 0.86 |
| UI clarity | `dual_mode` | complaint | 2.37 (1.72 to 3.28) | 5 / 6 | 0.99 | 0.53 |
| UI clarity | `strategy` | complaint | 2.65 (2.02 to 3.67) | 10 / 10 | 1.00 | 1.00 |
| UI clarity | `rtwp` (pooled) | complaint | 2.53 (1.74 to 3.68) | 10 / 10 | 1.00 | 0.87 |
| UI clarity | `turn_based` (pooled) | complaint | 2.59 (1.93 to 3.83) | 10 / 10 | 1.00 | 1.00 |
| UI clarity | `crpg/rtwp` | complaint | 2.07 (1.28 to 3.24) | 5 / 5 | 0.82 | 0.27 |
| UI clarity | `crpg/turn_based` | complaint | 3.45 (2.10 to 7.11) | 5 / 5 | 1.00 | 0.98 |
| UI clarity | `strategy/rtwp` | complaint | 3.86 (2.88 to 5.40) | 5 / 5 | 1.00 | 1.00 |
| UI clarity | `strategy/turn_based` | complaint | 2.11 (1.48 to 2.95) | 5 / 5 | 1.00 | 0.90 |
| Price and DLC | `dual_mode` | complaint | 1.64 (1.23 to 2.11) | 5 / 6 | 0.96 | 0.40 |
| Price and DLC | `strategy` | complaint | 1.90 (1.52 to 2.28) | 10 / 10 | 1.00 | 1.00 |
| Price and DLC | `rtwp` (pooled) | complaint | 1.94 (1.33 to 2.47) | 10 / 10 | 1.00 | 0.98 |
| Price and DLC | `turn_based` (pooled) | complaint | 1.41 (1.13 to 1.65) | 10 / 10 | 0.97 | 0.72 |
| Price and DLC | `crpg/turn_based` | complaint | 1.49 (1.06 to 1.82) | 5 / 5 | 0.49 | 0.16 |
| Price and DLC | `strategy/rtwp` | complaint | 2.33 (2.00 to 2.72) | 5 / 5 | 1.00 | 1.00 |
| Price and DLC | `strategy/turn_based` | complaint | 1.35 (1.02 to 1.65) | 5 / 5 | 0.52 | 0.15 |
| Topic 1, DLC | `turn_based` (pooled) | complaint | 2.02 (1.27 to 3.27) | 10 / 10 | 0.94 | 0.39 |
| Topic 1, DLC | `strategy/rtwp` | complaint | 1.54 (1.20 to 1.92) | 5 / 5 | 0.98 | 0.76 |
| Topic 3, party and enemies | `crpg` | complaint | 3.05 (1.88 to 4.64) | 10 / 10 | 1.00 | 0.88 |
| Topic 3, party and enemies | `dual_mode` | complaint | 3.55 (2.62 to 4.44) | 5 / 6 | 1.00 | 1.00 |
| Topic 3, party and enemies | `crpg/rtwp` | complaint | 2.25 (1.09 to 4.44) | 5 / 5 | 0.53 | 0.11 |
| Topic 3, party and enemies | `crpg/turn_based` | complaint | 4.29 (3.24 to 5.55) | 5 / 5 | 1.00 | 0.99 |
| Topic 11, crashes | `crpg` | complaint | 24.41 (14.00 to 45.01) | 10 / 10 | 1.00 | 0.96 |
| Topic 11, crashes | `dual_mode` | complaint | 17.27 (7.26 to 52.95) | 5 / 6 | 1.00 | 0.79 |
| Topic 11, crashes | `strategy` | complaint | 10.33 (7.28 to 15.69) | 10 / 10 | 1.00 | 1.00 |
| Topic 11, crashes | `rtwp` (pooled) | complaint | 11.82 (7.28 to 21.11) | 10 / 10 | 1.00 | 1.00 |
| Topic 11, crashes | `turn_based` (pooled) | complaint | 14.90 (9.56 to 31.04) | 10 / 10 | 1.00 | 1.00 |
| Topic 11, crashes | `crpg/rtwp` | complaint | 23.46 (9.89 to 66.31) | 5 / 5 | 0.99 | 0.58 |
| Topic 11, crashes | `strategy/rtwp` | complaint | 9.03 (5.30 to 14.18) | 5 / 5 | 1.00 | 1.00 |
| Topic 11, crashes | `strategy/turn_based` | complaint | 12.23 (7.94 to 29.12) | 5 / 5 | 1.00 | 0.99 |
| Topic 15, AI and armies | `strategy` | complaint | 6.23 (4.63 to 8.99) | 10 / 10 | 1.00 | 1.00 |
| Topic 15, AI and armies | `rtwp` (pooled) | complaint | 8.58 (4.31 to 14.72) | 10 / 10 | 1.00 | 0.79 |
| Topic 15, AI and armies | `strategy/rtwp` | complaint | 9.50 (4.65 to 17.38) | 5 / 5 | 1.00 | 0.95 |
| Topic 15, AI and armies | `strategy/turn_based` | complaint | 5.30 (3.90 to 7.33) | 5 / 5 | 1.00 | 1.00 |
| Topic 19, multiplayer and desync | `crpg` | complaint | 11.31 (6.46 to 18.00) | 10 / 10 | 1.00 | 0.85 |
| Topic 19, multiplayer and desync | `strategy` | complaint | 9.08 (5.97 to 12.37) | 10 / 10 | 1.00 | 1.00 |
| Topic 19, multiplayer and desync | `rtwp` (pooled) | complaint | 12.10 (7.26 to 18.38) | 10 / 10 | 1.00 | 0.95 |
| Topic 19, multiplayer and desync | `turn_based` (pooled) | complaint | 8.42 (5.66 to 11.48) | 10 / 10 | 1.00 | 1.00 |
| Topic 19, multiplayer and desync | `crpg/rtwp` | complaint | 15.01 (7.77 to 34.55) | 5 / 5 | 1.00 | 0.53 |
| Topic 19, multiplayer and desync | `strategy/turn_based` | complaint | 9.02 (5.85 to 14.04) | 5 / 5 | 1.00 | 0.98 |
| Topic 38, Paradox DLC | `strategy` | complaint | 5.22 (3.49 to 7.83) | 10 / 10 | 1.00 | 1.00 |
| Topic 38, Paradox DLC | `rtwp` (pooled) | complaint | 5.15 (3.29 to 8.27) | 10 / 10 | 1.00 | 1.00 |
| Topic 38, Paradox DLC | `strategy/rtwp` | complaint | 5.18 (3.39 to 7.89) | 5 / 5 | 1.00 | 1.00 |

The pooled `rtwp` and `turn_based` scopes mix CRPG and strategy games. Topic numbers are those of `outputs/rtwp_turn_2026/bertopic_model_full_dev/model.pkl`. BERTopic labels them "Dlc / Dlcs / Base / Buy" (1), "Party / Just / Enemies / Level" (3), "Crashes / Crash / Crashing / Crashed" (11), "Ai / Units / War / Army" (15), "Multiplayer / Desync / Desyncs / Unplayable" (19) and "Paradox / Dlc / Dlcs / Base" (38).

The candidate run also has 3 ranking findings (bugs and performance as the most-cited complaint aspect in `crpg`, `crpg/rtwp` and pooled `turn_based`). The third is new in run 4. Rankings compare aspects with each other, so narrowing the revised aspects can change one although bugs and performance itself is unchanged. They are not candidates, the power simulation does not cover them, and phase B reports them without a verdict (decision 6, section 8).

## 2. Criterion

**Primary criterion (decides the count).** A candidate replicates when its phase B point lift is on the candidate's side of 1 (above 1 for a complaint candidate, below 1 for a praise candidate) and its 95% interval excludes 1 on that side. An undefined lift or interval does not replicate. This is the market study's replication criterion (`docs/replication_protocol.md`, section 2).

**Reported, not deciding:**
- whether the full finding rule (`stats.FINDING_RULE`) holds in the same direction, including leave-one-game-out stability and at least 5 mentions on each side,
- whether the development point lift lies inside the phase B interval,
- the ratio of phase B to development log lift.

**Uncertainty.** `stats.macro_theme_metrics` (each eligible game weighted equally) and `stats.cluster_bootstrap_macro` with 2,000 resamples and seed 42, games then reviews.

**Multiplicity.** The 66 tests are fixed in advance with their direction, so each is a one-sided test at 2.5%. If no candidate were real, the expected number of chance replications is 66 x 0.025 = 1.65. The tests are not independent. Topic 11 overlaps bugs and performance, topics 1 and 38 overlap price and DLC, and every pooled subgroup scope shares its games with the cohort scopes. The power simulation keeps that correlation because all candidates are checked on the same synthetic sample.

## 3. Scopes, games and eligibility

Each candidate keeps its scope. Eligibility for the read is computed on the phase B sample itself (both halves share the full database). Development eligibility is reported beside it. A candidate's phase B games are every study game in its scope that meets the eligibility rule in the phase B sample: at least 100 kept reviews and at least 30 not-recommended kept reviews. A candidate with fewer than 3 eligible games is "not testable" and counts neither as a success nor as a failure.

This is the same rule on a comparable sample. Phase A and phase B are the two halves of one hash split of the same full database, pulled once with the same windows, so each game contributes about the same number of reviews to each half. 25 of 26 games are eligible in phase A. The one that is not, Combat Mission: Battle for Normandy, has 94 reviews in the phase B frame and cannot reach 100 kept reviews, so it will be ineligible in phase B too. The smallest eligible games have phase B frames of 298 (Combat Mission: Shock Force 2) and 320 (Black Geyser) reviews. The power simulation applies the rule in every synthetic sample.

## 4. Sampling frame

- **Source:** `data/rtwp_turn_2026/rtwp_turn_2026_full.db`, the 239,850 reviews loaded from the October 15, 2026 pull. Windows are those of the pull (design doc, section 3): full history for uncapped games, the most recent 10,000 (20,000 for the dual-mode titles) for capped ones.
- **Frame:** every review whose ID falls in the test half of `stats.dev_test_split` (120,084), minus the 359 reviews that supply the 360 sentences of the v2 audit and the 240 reviews that supply the 240 sentences of the v3 audit. The two sets share no review. 119,485 reviews, 94 to 9,986 per game. No draw is made: phase B reads the whole frame (decision 1 in section 12).
- **Excluded:** every development-half review (119,766, all read in phase A) and the reviews of both audit sets (599).
- **Pin:** `data/frames/rtwp_turn_2026_phase_b_frame.tsv` holds the frame's sorted `app_id<TAB>steam_review_id` lines and nothing else: 119,485 lines, 2,040,389 bytes, SHA-256 `4cf9fa3a44107836c96571096fd3ce9831dbc45071459b500e8456a42826965d`. It is committed with this protocol and is far below the repository's 25 MB file budget. `src/rtwp_read.py frame` wrote it from three inputs only: the review and game IDs of the full database, the hash split, and the `steam_review_id` column of the two audit exports. The read refuses a frame file whose hash differs, that does not hold 119,485 lines or that holds a development-half ID, and it reads text, dates and recommend flags for the frame's IDs and no others.

## 5. Frozen instruments

Nothing is refit or retuned on the test half. Every test-half step below (cleaning, the language and token filters, VADER, the topic transform and aspect tagging) runs inside `src/rtwp_read.py read` after its gate passes, so nothing about the test half beyond review IDs is computed before the tag. The audit labels of section 13 are the one exception, and they are judgments of single sentences, not statistics.

- **Text rules:** `preprocessing.Preprocessor` and `LanguageFilter` with the study's settings, and the 3-token minimum, as in phase A.
- **Lexicon:** `studies/aspects_rtwp_turn_v3.toml`, version `rtwp_turn_v3`, frozen October 15, 2026, at the tagged commit. Its `mode_rtwp`, `realtime_chaos` and `bugs_performance` patterns are those of v2. Only aspects that pass the audit (section 13) enter the analysis. All nine pass.
- **Topics:** every kept test-half review gets its topic from `BERTopic.transform` of `outputs/rtwp_turn_2026/bertopic_model_full_dev/model.pkl`, SHA-256 `dd037032288fa2df3c26df517c1fe0ea1517ffd5f173c6a2a11fd3ce9999e9d0`, fitted on the 89,953 kept development-half reviews. The first pass's seed model (`outputs/rtwp_turn_2026/bertopic_model/model.pkl`) is not used.
- **Topic transform agreement:** measured on the development half only (`src/rtwp_read.py agreement`, `outputs/rtwp_turn_2026/exploratory/topic_transform_agreement.json`), the way the market protocol measured it (section 5 there). Passing all 89,953 kept development-half reviews at once returns the fitted assignments exactly, because umap-learn recognises its training input, so that is a reproduction check only. Passing them in two halves (even and odd positions) forces the path new reviews take: 88.4% keep their fitted topic, against 93.6% for the market study's seed model, and the outlier share rises from 36.8% to 41.0%. Of the candidate topics' fitted reviews, transform keeps 93.9% for topic 1 (2,078 reviews), 70.3% for topic 3 (1,948), 81.7% for topic 11 (921), 88.0% for topic 15 (833), 92.5% for topic 19 (656) and 83.5% for topic 38 (430). Of the reviews transform assigns to each of these topics, 85.5% (topic 38) to 98.1% (topic 1) are fitted members. Phase B topics are therefore smaller and a little noisier than phase A's, most of all topic 3.
- **Sentiment:** VADER compound is computed inside the read for each kept review, as in phase A, and enters no criterion. Transformer sentiment is not computed on phase B data.
- **Statistics:** the `src/stats.py` functions phase A used, at the tagged commit. The only change since the candidate run is the precision merge (`stats.study_precision`), which decides which label file governs each aspect (section 13) and computes nothing on reviews.

## 6. Power

Computed by `src/rtwp_power.py --testable` from the development half only, on the 66 candidates and the frame of section 4 (`outputs/rtwp_turn_2026/exploratory/power_rtwp_turn_v3_cap0_testable.json`, seed 20261016, pinned by the read). The run reads the newest candidate run of the study's lexicon (run 4), takes each aspect's precision from its governing label file (`[aspect_labels]` in the study file) and drops the candidates of failing aspects. None was dropped. The v2 figures (`power_cap0.json` and `power_cap2000.json`, 70 candidates on a frame of 119,725) are superseded and kept as the planning record.

- **Method:** 500 synthetic samples per scenario. Each draws, for every game, a kept count from Binomial(frame size, development kept rate) and that many kept development-half reviews with replacement, applies the eligibility rule, and checks all 66 candidates with the primary criterion and a 1,000-resample cluster bootstrap. The simulation is `replication.simulate_power`, the code behind the market protocol's numbers.
- **Plug-in scenario:** each game's development data is the truth. Expected count 60.0, median 60, 5th to 95th percentile 56 to 63.
- **Half log lift scenario:** each game's theme flags are thinned on the dominant side so its log lift is halved, for the winner's curse on candidates selected by an interval rule. Expected count 46.7, median 47, 5th to 95th percentile 43 to 51.
- **By group:** the 14 combat-model candidates contribute 9.8 (plug-in) and 2.5 (half log lift) to those counts, the other 52 contribute 50.2 and 44.3.
- **Weakest candidates** (half log lift power at most 0.10): `realtime_chaos` in `crpg/turn_based` (0.01) and pooled `turn_based` (0.05), `waiting_turns` in pooled `turn_based` (0.03), and `pause_micro` in pooled `rtwp` (0.04), `strategy/rtwp` (0.09) and `dual_mode` (0.10). Each has a lower development bound between 1.03 and 1.78.
- **With a 2,000-per-game cap:** the frame would be 45,864 reviews and the expected count 55.2 (plug-in, 5th to 95th percentile 51 to 59) and 38.3 (half log lift, 33 to 43), with 6.2 and 0.9 from the combat-model candidates (`power_rtwp_turn_v3_cap2000_testable.json`). Not used (decision 1).
- **Not modelled:** labeler error in the audits, and the extra noise of topics assigned by `transform`, which keeps 88.4% of fitted assignments on new-data input and 70.3% of topic 3's members (section 5). The topic candidates' power is therefore optimistic, most for the four topic 3 candidates.

## 7. Decision rule

1. **Per candidate:** "replicated", "not replicated" or "not testable" by sections 2 and 3. An aspect candidate whose aspect fails the audit is "not testable" (decision 3). All nine aspects pass, so no candidate is lost that way. A candidate with fewer than 3 eligible phase B games is still "not testable".
2. **Per family:** "replicated" when more than half of its testable candidates replicate, "not replicated" otherwise, "not testable" when none is testable.
3. **Study level:** one count of the 66. A count of 43 or more, the 5th percentile of the half log lift scenario (section 6), is "consistent with power". Fewer is "below power".
4. **Combat-model count (reported, not deciding):** how many of the 14 combat-model candidates replicate, beside 9.8 expected (plug-in) and 2.5 (half log lift), with no verdict (decision 4).
5. **Ranking findings (reported, not deciding):** the 3 ranking findings of section 1, with no verdict.
6. **What changes in the write-up:** every candidate keeps its row and gains its phase B status, lift and interval. Replicated candidates may be called confirmatory. The rest stay exploratory and are marked as not replicated. None is removed or reworded.
7. **One read.** The read refuses to run unless the working tree is clean, HEAD carries the tag, no results file exists and every pinned file matches its SHA-256 (section 11). A fix after a failed read gets a new tag and is reported as a deviation. Criterion, frame, thresholds and instruments may not change in that commit.

## 8. Descriptive outputs

Phase B also reports, with no verdict and no count:

- every combat-model row that is not a candidate, for each aspect that passes its audit, in every cohort, `cohort_subgroup` and pooled subgroup scope with at least 3 eligible games: tagged reviews, macro complaint and praise shares, lift with its interval, and leave-one-game-out stability,
- the ranking rule (`stats._top_themes`) on the phase B sample: the top complaint aspect, its share and interval and the second aspect's, for the 3 scopes of the phase A ranking findings and any other scope where the rule holds,
- the dual-mode views of `src/dual_mode.py` (mode groups, and before and after each mode's arrival) with lexicon v3, computed inside the read on the frame reviews of the six dual-mode games. The phase A views in the design doc are the v3 rerun on the development half.

## 9. Reporting template

The read writes `outputs/rtwp_turn_2026/confirmatory/results.json` and `report.md` (ignored by Git).

- Header: commit, tag, frame SHA-256, reviews read and kept, games eligible in phase B and in phase A, audit precision per aspect with Wilson intervals, which audit attempt each aspect's precision comes from (section 13) and who labeled each audit (two separate fresh model labeler instances).
- Counts: candidates (66), testable, replicated, not replicated, not testable, full rule met, development lift inside the phase B interval, families replicated / not replicated / not testable, the expected counts (60.0 and 46.7), the pass line (43) and the study verdict, then the combat-model count against 9.8 and 2.5, with no verdict.
- Candidate table: all 66 rows with side, development lift and interval, games in scope, eligible in phase A and eligible in phase B, phase B lift and interval, status, full-rule result and stability.
- Descriptive tables from section 8.
- Deviations from this protocol, or "none".

## 10. Limits stated in advance

- Phase B uses the same games and the same pull windows. It tests whether the candidates hold on other reviews of these games, not on other games or later reviews.
- A candidate compares not-recommended with recommended reviews inside one scope. No candidate compares RTwP games with turn-based games directly. A difference between subgroups is read off two candidates side by side and is not itself tested.
- The combat-model candidates are the weakest part of the count. 2.5 of 14 are expected to replicate under halved effects, so a combat-model candidate that fails is weak evidence against it.
- The plug-in power uses selected development estimates, which are likely inflated. The half log lift scenario is the planning case.
- Topic membership in phase B comes from `transform`, which keeps 88.4% of development reviews' fitted topics on the new-data path and shrinks topic 3 the most (section 5).
- Precision for the six revised aspects comes from a revised lexicon and a different labeler instance at once, so the gain from the v2 to the v3 audit cannot be credited to the lexicon alone (section 13). `combat_pacing` passes at the floor exactly (32 of 40, Wilson interval 0.652 to 0.895).

## 11. Read code, pins and commands

`src/rtwp_read.py` runs phase B. Its tests (`tests/test_rtwp_read.py`) use synthetic data only.

- `frame` writes the frame file of section 4 from IDs and prints its size and SHA-256.
- `agreement` measures topic transform agreement on the development half (section 5).
- `gate` prints whether the read may run.
- `dry-run` runs the whole read path on up to 150 development-half reviews per game, with eligibility lowered to 40 kept and 8 not-recommended reviews and 200 bootstrap resamples, and writes to `outputs/rtwp_turn_2026/confirmatory_dry_run/`. It reads no test-half review and enforces no gate. It ran on October 15, 2026: 3,868 reviews read, 2,929 kept, 24 games eligible, all 66 candidates evaluated, and the report, ranking table and dual-mode views written. Its counts are development data on small samples and mean nothing for phase B.
- `read` is the confirmatory read. It refuses unless the working tree is clean (`git status --porcelain` is empty), HEAD carries the tag `rtwp-turn-protocol-v1`, this protocol is tracked, `outputs/rtwp_turn_2026/confirmatory/results.json` does not exist, and every pinned file below matches its SHA-256 at read time. Only then does it read the frame reviews' text, dates and recommend flags from the full database and run the instruments of section 5 on them. It applies the criterion of section 2 with `stats.macro_theme_metrics`, `stats.cluster_bootstrap_macro` (2,000 resamples, seed 42), `stats.leave_one_game_out` and `stats.classify_finding`, as phase A did, and `replication.meets_criterion`, the market replication's criterion.

| Pinned file | SHA-256 |
|---|---|
| `outputs/rtwp_turn_2026/exploratory/candidates_run_4.json` | `57c26c8245527dab90852f3cdc2b43590fdc927ff653e8869f7b9856db0295f2` |
| `studies/aspects_rtwp_turn_v3.toml` | `0b6907a312628913308c5a50dd05a9ca87ea1a92ef55e78c5622fc96c1037c8f` |
| `data/labels/aspect_audit_rtwp_turn_v2.csv` | `d5262bb4bffa3a9eb3fbc5219f9445760dbbdcc5644417167dda80588900101f` |
| `data/labels/aspect_audit_rtwp_turn_v3.csv` | `5cb56d300bfbd96f3587079a16df91b354d07f4460bcf7ca353926cc185b18e3` |
| `outputs/rtwp_turn_2026/bertopic_model_full_dev/model.pkl` | `dd037032288fa2df3c26df517c1fe0ea1517ffd5f173c6a2a11fd3ce9999e9d0` |
| `outputs/rtwp_turn_2026/exploratory/power_rtwp_turn_v3_cap0_testable.json` | `521cd42c18ffe1bbb23f6fcb8442e08858fb664ed21b9dc47da6a274ecd61274` |
| `data/frames/rtwp_turn_2026_phase_b_frame.tsv` | `4cf9fa3a44107836c96571096fd3ce9831dbc45071459b500e8456a42826965d` |

The lexicon, both label files and the frame file are committed with this protocol. The candidates file, the topic model and the power file live under `outputs/`, which Git ignores, so their hashes are recorded here for integrity. A reader of the repository can check the committed files byte for byte but must regenerate the three local ones, and a regenerated topic model need not match its hash.

Done before the commit, all on October 15, 2026: the v2 audit table (section 13), the v3 labels imported into `data/labels/aspect_audit_rtwp_turn_v3.csv`, `[aspect_labels]` in the study file naming each aspect's governing label file, the power rerun (section 6), the topic transform agreement (section 5), the frame file and its pin (section 4), the read code and its tests, and the decisions of section 12. Left: commit this protocol with the code, the study and lexicon files, both label files and guides and the frame file, tag the commit, and run the read once at a clean checkout of the tag.

```bash
# Done (development half and IDs only):
.venv/bin/python src/rtwp_power.py --cap 0 --testable
.venv/bin/python src/rtwp_power.py --cap 2000 --testable
.venv/bin/python src/rtwp_read.py frame
.venv/bin/python src/rtwp_read.py agreement
.venv/bin/python src/rtwp_read.py dry-run
# After the commit and tag, once:
.venv/bin/python src/rtwp_read.py gate
.venv/bin/python src/rtwp_read.py read
```

## 12. Decisions

All six are locked by the owner.

1. **Frame (locked).** The whole test half of the full database minus both audit sets, 119,485 reviews. A 2,000-per-game cap would read 45,864 and lower the expected count under halved effects from 46.7 to 38.3.
2. **Eligibility (locked).** Eligibility for the read is computed on the phase B sample itself (both halves share the full database). Development eligibility is reported beside it.
3. **Audits govern testability (locked).** Decision 3 stays as a rule: the model-labeler audits govern testability. Any later human relabel is reported as an agreement measure and a sensitivity only, never a change to which candidates were testable.
4. **Combat-model candidates (locked).** The 14 combat-model candidates are tested inside the 66 and their own count is reported beside it, against 9.8 expected (plug-in) and 2.5 (half), with no verdict.
5. **One count (locked).** 66 candidates against a pass line of 43 (5th percentile of the half log lift scenario). Expected 60.0 if the development effects are true, 46.7 at half log lift.
6. **Ranking findings (locked).** The 3 ranking findings are reported without a verdict.

## 13. Aspect audit: two attempts

The audit measures each aspect's precision on test-half sentences against the definitions of the labeling guide. An aspect passes at a precision of at least 0.80 on 30 or more sentences. Intervals are 95% Wilson intervals from `stats.load_audit_precision`. There were two attempts and there is no third. The result below is final.

**Attempt 1, lexicon v2.** 360 sentences, 40 for each of the 9 aspects, from 359 test-half reviews (`outputs/audit/rtwp_turn_2026_aspect_audit_rtwp_turn_v2_unlabeled.csv`, guide `data/labels/aspect_labeling_guide_rtwp_turn_v2.md`), labeled by a fresh model labeler instance into `data/labels/aspect_audit_rtwp_turn_v2.csv`. `mode_rtwp`, `realtime_chaos` and `bugs_performance` passed. `mode_turn_based`, `pause_micro`, `waiting_turns`, `combat_pacing`, `ui_clarity` and `price_dlc` failed.

| Aspect | Relevant of 40 | Precision (95% Wilson interval) | Result |
|---|---|---|---|
| `mode_turn_based` | 26 | 0.650 (0.495 to 0.779) | failed |
| `mode_rtwp` | 36 | 0.900 (0.769 to 0.960) | passed |
| `pause_micro` | 27 | 0.675 (0.520 to 0.799) | failed |
| `realtime_chaos` | 33 | 0.825 (0.681 to 0.913) | passed |
| `waiting_turns` | 30 | 0.750 (0.598 to 0.858) | failed |
| `combat_pacing` | 26 | 0.650 (0.495 to 0.779) | failed |
| `bugs_performance` | 37 | 0.925 (0.801 to 0.974) | passed |
| `ui_clarity` | 31 | 0.775 (0.625 to 0.877) | failed |
| `price_dlc` | 28 | 0.700 (0.546 to 0.819) | failed |

**Attempt 2, lexicon v3, the single retry.** `studies/aspects_rtwp_turn_v3.toml` revises only the six failed aspects. The three passed aspects are copied from v2 unchanged. The v3 builder knew which six aspects failed v2 but saw no v2 labels or test-half text, and developed v3 on development-half sentences only. The definitions in the guide did not change. The two audits were labeled by two separate fresh model labeler instances, each seeing only the guide and its own sentences, with ties going to `not_relevant`. This was the single retry and its result is final.

A lexicon change and a labeler change are confounded across the two audits. The six revised aspects were relabeled by a different labeler instance than the one that failed them, so the rise in their precision from attempt 1 to attempt 2 measures the narrower patterns and any difference between the two labelers together, and the two cannot be separated.

| Aspect | Relevant of 40 | Precision (95% Wilson interval) | Result |
|---|---|---|---|
| `mode_turn_based` | 40 | 1.000 (0.912 to 1.000) | passed |
| `pause_micro` | 38 | 0.950 (0.835 to 0.986) | passed |
| `waiting_turns` | 38 | 0.950 (0.835 to 0.986) | passed |
| `combat_pacing` | 32 | 0.800 (0.652 to 0.895) | passed |
| `ui_clarity` | 34 | 0.850 (0.709 to 0.929) | passed |
| `price_dlc` | 33 | 0.825 (0.681 to 0.913) | passed |

How v3 was built. The builder did not open the v2 label file, the v2 audit text or any other file under `outputs/audit/`, and read only development-half sentences from the analysis database: a seeded sample of 80 v2-tagged sentences per revised aspect (one per review), judged against the v2 guide's definitions, then a held-out sample of 40 more for five aspects. `waiting_turns` has only 80 v2-tagged development reviews, so all of them formed its sample and every v3-tagged development review was judged instead of a hold-out. After the hold-out was judged, four narrow fixes were made (strat* and JRPG as genre nouns, "with a pause" as a mode name, "drag ... down" and "so long as" no longer counted as length). The v3 export read only the `steam_review_id` column of the v2 export, to exclude those reviews.

| Aspect | v3 change |
|---|---|
| `mode_turn_based` | "turn-based" followed by a genre noun (strategy, RPG, 4X, games, tactical RPG and similar) no longer tags |
| `pause_micro` | "pause" inside the mode name (with pause, with-pause, /pause) no longer counts as pausing |
| `waiting_turns` | waiting must be for the AI, enemies or other players, or for one's own turn. Softlocks and hangs are excluded as bugs |
| `combat_pacing` | "tedious" alone no longer tags. "So many fights" and "takes a lot of time" are added |
| `ui_clarity` | "menu" no longer tags as menu music, menu screen or the main, help, difficulty, gameplay, options or settings menu |
| `price_dlc` | "DLC" needs a cost, value or content word, "on sale" a buying or waiting verb, "expensive" a degree word or game noun. Money amounts are added |

Development-sample precision, as judged by the builder. These are the builder's own judgments on development-half sentences, not an independent audit, and they do not decide anything. v3 keeps every relevant sentence of the 80-sentence samples except 2 each for `combat_pacing` and `price_dlc`. The v3 hold-out figures are those before the four fixes. After them the hold-out reads 0.89 for `mode_turn_based` (34 of 38), 0.97 for `pause_micro` (38 of 39) and 0.89 for `combat_pacing` (34 of 38), but those fixes were chosen on the hold-out, so the figures before them are the held-out estimate.

| Aspect | v2, sample of 80 | v3, same sample | v3, hold-out of 40 | Tagged development reviews, v2 to v3 |
|---|---|---|---|---|
| `mode_turn_based` | 0.68 (54 of 80) | 0.95 (54 of 57) | 0.85 (34 of 40) | 4,886 to 3,528 |
| `pause_micro` | 0.75 (60 of 80) | 0.91 (60 of 66) | 0.95 (38 of 40) | 1,495 to 1,274 |
| `waiting_turns` | 0.69 (55 of 80) | 0.90 (55 of 61) | no hold-out, 0.86 on all 65 v3-tagged (56) | 80 to 65 |
| `combat_pacing` | 0.75 (60 of 80) | 0.98 (58 of 59) | 0.85 (34 of 40) | 509 to 396 |
| `ui_clarity` | 0.83 (66 of 80) | 0.93 (66 of 71) | 0.93 (37 of 40) | 3,364 to 3,214 |
| `price_dlc` | 0.75 (60 of 80) | 0.91 (58 of 64) | 0.95 (38 of 40) | 10,636 to 8,308 |

The builder's v2 estimate for `ui_clarity` is above 0.80 although the aspect failed its audit, so the builder's standard was likely looser than the labeler's there, and every builder figure above may be optimistic.

**v3 audit set.** `src/aspects.py --export-audit` with lexicon v3 wrote 240 sentences, 40 for each revised aspect, to `outputs/audit/rtwp_turn_2026_aspect_audit_rtwp_turn_v3_unlabeled.csv` (ignored by Git). They come from 240 distinct test-half reviews, none of them among the 359 reviews of the v2 set. The eligible test-half pools after that exclusion were 3,369 reviews (`mode_turn_based`), 1,213 (`pause_micro`), 57 (`waiting_turns`), 369 (`combat_pacing`), 3,102 (`ui_clarity`) and 8,297 (`price_dlc`). The guide `data/labels/aspect_labeling_guide_rtwp_turn_v3.md` is the v2 guide with only file names, version and row counts changed, so both attempts are labeled against the same definitions.

**Precision rule.** Each aspect's precision comes from one attempt: the v3 labels (`data/labels/aspect_audit_rtwp_turn_v3.csv`) for the six revised aspects and the v2 labels (`data/labels/aspect_audit_rtwp_turn_v2.csv`) for `mode_rtwp`, `realtime_chaos` and `bugs_performance`, whose v3 patterns are identical to v2. All six passed the v3 audit, so every aspect enters phase B. No aspect is revised again.
