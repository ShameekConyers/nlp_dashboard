# What Are Customers Actually Saying? NLP-Powered Review Analysis

An NLP pipeline that turns Steam reviews into structured evidence: BERTopic topics crossed with sentiment, an audited phrase lexicon for named design questions, a transformer sentiment model adopted only after it beat VADER on held-out reviews, and every rate checked against Steam's own recommend flag. It ships two studies, each with a committed SQLite seed and a Streamlit dashboard:

- **Market analysis of historical and tactical strategy games** (September 2026, default): 29 games, 12,315 sampled reviews, 9,622 kept for analysis.
- **April 2026 cross-genre baseline**: 12 games across four genres, 2,382 reviews, kept reproducible.

---

## Market analysis of historical and tactical strategy games

### Business question

A design or marketing team planning a turn-based strategy game wants to know what players of comparable games praise and what drives them to leave a negative review, and how that differs between games that resolve turns in different ways (one side moves then the other, units alternate, or both sides plan and resolve at once). Reviews are the largest free source of that evidence. The question is which themes the review text supports with measured rates and intervals, and which ones it cannot settle.

### Dataset and sampling

- **Games:** 29 titles in three cohorts, set in `studies/strategy_market_2026.toml`. `turn_structure` holds 22 titles in three subgroups (IGO-UGO 7, alternating 7, WEGO 8, including two stand-ins). `historical_peer` holds 4 historical grand-strategy titles. `anchor` holds 3 large strategy titles (Civilization VI, Stellaris, Total War: WARHAMMER III). A `matched_basket` tag marks 9 titles with 2,000 to 14,000 reviews for a like-for-like check.
- **Pull (26 September 2026):** Steam appreviews, `filter=recent`, `language=english`, `review_type=all`, `purchase_type=all`, newest first, capped at 20,000 reviews per game. 22 games were pulled in full (every English review they have). Seven hit the cap and cover a recent window ending 26 September 2026: XCOM 2 from December 2021, BattleTech from October 2018, Divinity: Original Sin 2 from January 2023, Baldur's Gate 3 from February 2026, Civilization VI from August 2024, Stellaris from February 2024 and Total War: WARHAMMER III from March 2025. 209,223 reviews were pulled and 208,380 loaded (843 had empty text).
- **Seed (`data/market_seed.db`, 19.9 MiB with transformer scores):** a uniform random sample of up to 500 reviews per game (seed 42), with no stratification, so each game keeps its natural recommend rate. 12,315 reviews.
- **Kept for analysis:** 9,622. Dropped: 1,039 under 10 characters, 1,555 with fewer than 3 content tokens, and 99 detected as another language (lingua) although Steam listed them as English.
- **Eligibility:** a game counts toward cohort estimates if it has at least 100 kept reviews and at least 30 not-recommended kept reviews. 23 of 29 games qualify: 18 of 22 turn-structure titles, 2 of 4 historical peers and all 3 anchors. Into the Breach, War in the East 2, Baldur's Gate 3, Flashpoint Campaigns, Nobunaga's Ambition and Romance of the Three Kingdoms X are shown per game but greyed as low n.

### Method

1. **Preprocessing:** the April cleaning and sentinel rules, plus a language filter. A review is dropped as non-English when lingua's top language is not English at confidence 0.50 or higher.
2. **Sentiment:** two scorer families, compared against the recommend flag on a held-out test split (half the reviews, chosen by a hash of the review ID). VADER gives a compound score per review. Two pinned Hugging Face models give the probability that a review is positive: `siebert/sentiment-roberta-large-english` (revision `74cea614`) and `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (revision `714eb0fa`), a review counting as positive at probability 0.5 or higher. Reviews longer than 510 tokens keep their first 256 and last 254 tokens, because strategy reviews often give the verdict at the end. The rule was written before the test split was read. On the development half, VADER's threshold was tuned for balanced accuracy (0.40 won on a grid from -0.95 to 0.95) and the candidate with the higher balanced accuracy was selected. The test half was then read once. The selected model is adopted only if the paired-bootstrap 95% interval of its balanced-accuracy gain over VADER lies above 0 at both VADER thresholds (0.05 and 0.40). Balanced accuracy is the primary metric because about 81% of kept reviews recommend the game, so plain accuracy rewards calling everything positive. Scores are stored in `sentiment_scores`, where every row carries a model id that joins to the model name and pinned revision in `sentiment_models`: document scores from both candidates for every kept review and, for the selected model and the three-class sentence model described under the sentence audits, sentence scores for the 6,452 aspect-tagged sentences.
3. **Topics (primary lens):** BERTopic with `all-MiniLM-L6-v2` embeddings, BERTopic's default UMAP settings (seeded) and a minimum topic size of 40. Topic labels drop English and project stop words. The fit found 42 topics; 4,038 of 9,622 kept reviews (42.0%) are outliers with no topic.
4. **Design themes (supplementary lens):** a versioned phrase lexicon (`studies/aspects_strategy_v1.toml`, frozen after two passes on the development split) tags sentences with seven design aspects (waiting through enemy turns, first-strike advantage, initiative and turn order, combat preview and hit chance, enemy AI quality, maps, pacing) and four controls (bugs and performance, UI, translation, price and DLC). An aspect supports claims only if a precision audit of at least 30 test-split sentences finds 80% or more relevant. Audit sentences come from the seed's test split (up to 40 per aspect, one per review), except for three rare aspects (waiting through enemy turns, first-strike advantage, initiative and turn order): the seed's test split held only 7, 3 and 18 of them, so their 40 each come from the test split of the full local database, processed with the same cleaning, language and lexicon rules. The 433 audit labels (`data/labels/aspect_audit_strategy_v1.csv`) were made by an independent model labeler (Claude Opus) that saw only the labeling guide and the sentences, not the builder's work or the results. The guide offers no "unclear" value and tells the labeler to resolve ties as not relevant, so precision is conservative. Anyone can relabel the file and rerun `src/stats.py`, but reading the sentences for the three full-DB aspects needs the local full database: 111 of those 120 sentences come from reviews that are not in the committed seed.
5. **Metrics per theme:** complaint share (share of not-recommended reviews in the theme), praise share (same for recommended reviews), complaint lift (their ratio), the recommend share of the theme's reviews, the share that the adopted RoBERTa model scores positive (whole review), and their mean VADER score. Sentiment never enters the lift or the finding rule, so the findings do not depend on the scorer.
6. **Uncertainty:** Wilson 95% intervals for per-game rates, a review-level bootstrap (2,000 resamples) for per-game theme metrics, and a two-stage cluster bootstrap (games, then reviews) for cohort averages, which weight each eligible game equally.
7. **Finding rule (`FINDING_RULE` in `src/stats.py`):** a theme is a cohort finding only if the cohort has 3 or more eligible games, the lift interval lies entirely above 1 (complaint) or below 1 (praise), the result is stable when any one game is dropped, the theme has at least 5 mentions on each side, and, for aspects, the precision audit passed. Stable means the lift stays on the same side of 1 and the theme's complaint-share rank within its lens (aspects among passing aspects, topics among topics) moves by at most one place. A ranking finding ("most-cited complaint") also needs the top complaint theme's interval to clear the second theme's; praise rankings and the all-games scope are always descriptive. Everything else is descriptive.
8. **Rule changes made after results were seen.** Two, both disclosed here. The minimum-support condition was added after the first run, which read 13 collapsed [0, 0] intervals from topics with no complaint mentions as praise findings. Later, an independent review found the stability check ranked complaint shares across all themes at once (aspects and 42 topics mixed) instead of within each lens as the written plan specifies; correcting the code to the written rule moved the lift-finding count from 14 to 24. The same review found the ranking check had marked a praise-side ranking as a finding, which the plan does not allow; after the fix there are no ranking findings.

### Results

All numbers come from `data/market_seed.db` (`analysis_runs`, `game_stats`, `cohort_stats`, `theme_stats`) and can be re-derived with the commands in Quick Start.

**Natural recommend share.** Across the 22 fully pulled games, the seed share sits 0.3 points below Steam's English population share on average, so the sample reproduces the population. Among kept reviews the gap widens to 2.0 points, because short reviews that fail the text filter are mostly positive. Two capped anchors differ from their all-time population because their recent window is more critical: Stellaris 72.0% in the seed against 85.8% all-time, Total War: WARHAMMER III 63.6% against 71.3%.

| Scope | Games (eligible) | Kept reviews | Recommend share, macro (95% CI) | RoBERTa positive share, macro (95% CI) | Mean VADER, macro (95% CI) | Median hours at review, recommended / not |
|---|---|---|---|---|---|---|
| Turn structure | 22 (18) | 7,854 | 83.5% (80.5% to 86.5%) | 77.8% (75.1% to 80.7%) | 0.467 (0.421 to 0.513) | 32.6 / 7.2 |
| IGO-UGO | 7 (5) | 2,620 | 83.0% (78.6% to 87.9%) | 78.3% (74.3% to 82.6%) | 0.443 (0.381 to 0.503) | 32.5 / 7.9 |
| Alternating | 7 (6) | 2,685 | 88.4% (85.8% to 91.0%) | 82.0% (79.7% to 84.4%) | 0.532 (0.462 to 0.594) | 50.3 / 11.5 |
| WEGO | 8 (7) | 2,549 | 79.2% (72.9% to 85.3%) | 73.2% (67.9% to 78.8%) | 0.428 (0.348 to 0.511) | 20.2 / 4.3 |
| Historical peers | 4 (2) | 783 | 73.9% (66.3% to 84.3%) | 68.1% (61.0% to 78.0%) | 0.501 (0.438 to 0.573) | 25.2 / 10.7 |
| Anchors | 3 (3) | 985 | 67.7% (55.0% to 81.0%) | 57.9% (46.2% to 71.5%) | 0.273 (0.130 to 0.400) | 131.7 / 204.3 |

The recommend-share and sentiment macros use every game with at least 100 kept reviews, with two-stage cluster-bootstrap intervals; medians pool kept reviews in the scope. RoBERTa scores fewer reviews positive than recommend the game in every scope, by 4.8 to 9.9 points in the rows above and 3.0 to 6.6 in the matched baskets. Per-game RoBERTa and VADER sentiment is in the dashboard's Market Overview.

**Topics crossed with sentiment.** 378 topic lift comparisons were evaluated (42 topics across 9 cohort, subgroup and matched-basket scopes). Four are findings:

| Topic | Scope (eligible games) | Complaint share | Praise share | Lift (95% CI) | Recommended within topic | RoBERTa positive | Mean VADER |
|---|---|---|---|---|---|---|---|
| Crashes / Crash / Crashing / Screen | Turn-structure cohort (18) | 6.3% | 0.5% | 12.8 (7.2 to 29.3) | 21.7% | 9.7% | 0.048 |
| Crashes / Crash / Crashing / Screen | IGO-UGO (5) | 5.4% | 0.4% | 12.1 (4.9 to 52.1) | 25.5% | 18.5% | 0.206 |
| Crashes / Crash / Crashing / Screen | WEGO (7) | 7.7% | 0.7% | 11.1 (4.9 to 41.7) | 21.8% | 6.4% | -0.018 |
| Final / Fantasy / Tactics / Fft (praise) | Alternating, matched basket (3) | 16.5% | 35.1% | 0.47 (0.24 to 0.74) | 94.6% | 90.5% | 0.724 |

The crash topic also has lifts above 1 in the anchor cohort (18.2, 4.3 to 57.7), which fails the 5-mention floor on the praise side, and in historical peers (3.0) and alternating (18.6), which fail the stability check or have fewer than 3 eligible games. The praise row is a franchise-identity topic covering two of the three matched alternating games (Final Fantasy Tactics and Fell Seal, about 210 reviews each): reviews in that style lean positive, which says more about those games than about a theme.

Top three themes per scope by complaint share and by praise share. None is a ranking finding (18 ranking tests, 0 findings).

| Scope | Lens | Top complaint themes | Top praise themes |
|---|---|---|---|
| Turn structure | Aspects | Price and DLC 17.5%, Bugs and performance 16.1%, UI 12.4% | Price and DLC 12.3%, Bugs and performance 6.4%, UI 5.4% |
| Turn structure | Topics | Crashes 6.3%, Panzer / General 4.0%, Mission / Combat 3.8% | Final Fantasy Tactics 6.1%, Turn / Based / Strategy 4.9%, Mission / Combat 4.2% |
| IGO-UGO | Aspects | Price and DLC 18.5%, Bugs and performance 11.4%, Pacing 7.9% | Price and DLC 15.4%, Bugs and performance 4.3%, Enemy AI 3.8% |
| IGO-UGO | Topics | Panzer / General 12.7%, Free / Tutorial / Demo 5.7%, Crashes 5.4% | Panzer / General 14.3%, Advance Wars / Fire Emblem 6.1%, Turn / Based / Strategy 5.6% |
| Alternating | Aspects | Pacing 17.7%, Price and DLC 13.7%, Bugs and performance 12.0% | Price and DLC 10.5%, Pacing 5.4%, UI 4.3% |
| Alternating | Topics | Final Fantasy Tactics 8.8%, Crashes 5.4%, Mechs / Pilots 4.7% | Final Fantasy Tactics 17.8%, Heroes of Might and Magic 5.9%, Mechs / Pilots 5.6% |
| WEGO | Aspects | Bugs and performance 23.0%, Price and DLC 20.1%, UI 19.4% | Price and DLC 11.8%, Bugs and performance 10.0%, UI 7.5% |
| WEGO | Topics | Dominions 8.2%, Crashes 7.7%, Mission / Combat 6.9% | Mission / Combat 10.0%, Frozen Synapse 6.7%, Turn / Based / Strategy 5.3% |
| Historical peers | Aspects | UI 20.7%, Price and DLC 16.2%, Bugs and performance 13.8% | Price and DLC 12.2%, Bugs and performance 10.3%, Pacing 9.0% |
| Historical peers | Topics | Romance of the Three Kingdoms 52.1%, Crashes 6.9%, AI 3.1% | Romance of the Three Kingdoms 62.3%, Crashes 2.3%, Fun / Addictive 1.2% |
| Anchors | Aspects | Price and DLC 33.1%, Bugs and performance 23.7%, UI 5.0% | Price and DLC 17.6%, Bugs and performance 4.9%, Pacing 2.3% |
| Anchors | Topics | Total War: WARHAMMER 13.6%, Civilization 9.0%, DLC / Price 6.7% | Civilization 9.8%, Total War: WARHAMMER 5.1%, Space / Sci-Fi 5.1% |

Topic names above are shortened from their top-word labels. Game or franchise identity topics lead the topic lists in most scopes. The exceptions are the turn-structure cohort's complaint list, led by the crash topic, and WEGO's praise list, led by a general mission-and-combat topic. Cross-game topics include tutorials and the learning curve, DLC and price, hit chance and RNG, AI, soundtrack, mods and replayability; the dashboard's Topics x Sentiment view shows them per scope and per game.

**Sentiment scorers on held-out reviews.** On the development half (4,839 reviews) balanced accuracy was 69.8% for VADER at 0.05, 71.1% at the tuned 0.40, 92.4% for RoBERTa-large and 85.1% for DistilBERT, so RoBERTa-large was selected. Test half, 4,783 kept reviews, 913 not recommended, review-level bootstrap 95% intervals (Wilson for the last column):

| Scorer | Balanced accuracy | Accuracy | Macro-F1 | ROC AUC | Not-recommended scored positive |
|---|---|---|---|---|---|
| VADER, threshold 0.05 | 70.3% (68.7 to 72.0) | 78.2% (77.1 to 79.4) | 68.1% (66.6 to 69.7) | 0.750 (0.732 to 0.770) | 42.4% (39.2 to 45.6) |
| VADER, dev-tuned 0.40 | 71.3% (69.8 to 73.0) | 73.5% (72.3 to 74.7) | 65.8% (64.3 to 67.2) | 0.750 (0.732 to 0.770) | 32.2% (29.3 to 35.3) |
| RoBERTa-large (siebert) | 91.2% (90.2 to 92.2) | 90.8% (90.0 to 91.6) | 86.7% (85.5 to 87.8) | 0.972 (0.968 to 0.976) | 8.1% (6.5 to 10.1) |
| DistilBERT SST-2 | 84.8% (83.7 to 85.9) | 81.2% (80.2 to 82.3) | 76.0% (74.7 to 77.3) | 0.929 (0.921 to 0.936) | 9.5% (7.8 to 11.6) |

RoBERTa-large gains 19.9 points of balanced accuracy over tuned VADER (paired interval 18.1 to 21.6) and 20.9 over VADER at 0.05 (19.1 to 22.7), so it is adopted as the primary sentiment measure. Resampling games before reviews, a sensitivity check, gives 17.1 to 22.7 against tuned VADER. VADER stays in every table for continuity. Predicting "recommended" for every review would score 80.9% accuracy, which is why balanced accuracy is the metric and why the April 69.4% (on a 50/50 sample) is not comparable.

On the April balanced seed (1,559 kept reviews, nothing tuned on it), balanced accuracy is 70.7% for VADER at 0.05, 70.9% at 0.40, 87.8% (86.1 to 89.4) for RoBERTa-large and 83.4% for DistilBERT. On its three strategy titles (434 reviews, 261 not recommended), VADER scores 38.3% of not-recommended reviews positive at 0.05 and 27.2% at 0.40, against 6.9% for RoBERTa-large and 7.7% for DistilBERT.

Scored on the 113 praise or complaint comments hand-coded in the earlier desk study (59 praise, 54 complaint; mixed and neutral excluded), balanced accuracy is 86.7% (79.7 to 92.9) for RoBERTa-large, 76.6% (69.0 to 83.6) for DistilBERT and 67.6% (58.7 to 75.8) for VADER.

**Sentence sentiment.** Three pre-registered audits: two failed, the third passed, and the Design Themes view now shows sentence-level sentiment for aspect sentences. First audit (v1): an independent model labeler (Claude Opus) labelled 100 test-split aspect sentences, stratified across the seven passing aspects, as positive, negative or neither: 18, 61 and 21. It judged the writer's attitude in the whole sentence, not toward the aspect, and resolved genuine ties to neither. The gate, fixed before labelling, was a three-class macro-F1 of 0.70, with the model predicting neither when its positive probability falls between 0.2 and 0.8. RoBERTa-large scores 0.52. It gets 18 of 18 positive and 55 of 61 negative sentences right, but its probabilities never fall in the neither band, so all 21 neither sentences count as errors. On the 79 sentences labelled positive or negative its binary macro-F1 is 0.90.

Second audit (v2), designed and written down before any new score existed: the v1 labels became the development set, and two candidates were compared on them. The three-class `cardiffnlp/twitter-roberta-base-sentiment-latest` (revision `3216a57f`, argmax over negative, neutral and positive, no band) scored 0.72. RoBERTa-large with a symmetric neither band tuned on the same labels scored 0.55 at its best band (0.05 to 0.95), against 0.52 at the fixed band. The rule selected the higher development score and allowed one read of a fresh test set only if that score reached 0.70, so the three-class model went forward. The fresh set holds 200 test-split sentences that share none with v1, spread across the seven passing aspects as evenly as their unused sentences allowed (34 each for five aspects, 26 for combat preview, 4 for initiative). A separate fresh labeler with the same guide labelled them 54 positive, 101 negative and 45 neither. The three-class model scores 0.69 against the 0.70 gate, so it fails: per-class F1 0.75 positive, 0.83 negative and 0.50 neither (22 of 45 neither sentences right, 10 called positive and 13 negative). On the 155 sentences labelled positive or negative its binary macro-F1 is 0.89. A post-hoc bootstrap of the 200 sentences puts the macro-F1 between 0.62 and 0.76, which says the gate is close, not that it was met.

Third audit (v3), again written down before any new score existed, and after a written brief against a third attempt. The 300 v1 and v2 labels became the development set (72 positive, 162 negative, 66 neither). Four candidates: the same three-class model by argmax (0.699) or with a neutral-probability cut tuned on development, so a sentence is neither when P(neutral) is at least the cut and otherwise takes the larger of positive and negative (0.712 at 0.40, grid 0.10 to 0.70); and the larger `j-hartmann/sentiment-roberta-large-english-3-classes` (revision `81cdc0fe`) by argmax (0.668) or with the same tuned cut (0.677 at 0.70). The rule selected the highest development score, so the smaller model with the 0.40 cut went forward. The fresh set holds 200 test-split sentences that share none with v1 or v2. Initiative and combat preview had no unused test sentences left, so the set covers the other five passing aspects at 40 each. A separate fresh labeler with the same guide labelled them 44 positive, 111 negative and 45 neither. The selected rule scores 0.707 against the 0.70 gate and passes: per-class F1 0.80 positive, 0.81 negative and 0.51 neither (27 of 45 neither sentences right). Its main error is caution: 27 of 111 negative sentences and 6 of 44 positive ones are called neither, and only 3 sentences flip sign. On the 155 polar sentences its binary macro-F1 is 0.92, and per-aspect accuracy runs from 0.68 (bugs and performance) to 0.80 (UI). A post-hoc bootstrap puts the macro-F1 between 0.64 and 0.77, so the pass is as narrow as the v2 miss, and plain argmax would have scored 0.704 on the same set. The gate was fixed at 0.70 before the first label existed and never moved, and each attempt read its own fresh test set once; readers should still weigh three attempts against one gate when they judge the pass.

What the view shows: across all 6,452 aspect-tagged sentences the selected rule calls 38.4% negative, 34.1% neither and 27.5% positive, pooled over games. These shares are descriptive. The study's findings rest on the recommend flag and never use sentence scores. The RoBERTa-large sentence scores from v1 stay in the database, marked unvalidated; the larger three-class model's scores were dropped after the selection to keep the seed under its size cap (its scoring run stays in `analysis_runs`). Six audit files are committed: the unlabeled exports `data/labels/sentence_sentiment_audit_v{1,2,3}_unlabeled.csv`, which carry the sentence text copied from the seed, and the labels `sentence_sentiment_audit_v{1,2,3}.csv` (ids, aspect and label). Both development selections and all three test reads are stored as `analysis_runs` rows (`sentence_dev_selection`, `sentence_audit`) and shown in the Methods view.

**Design themes: precision audit.** Seven of eleven aspects pass the 80% floor. Four fail and support no claim: waiting through enemy turns, first-strike advantage, maps, and translation.

| Aspect | Relevant / labelled | Precision (Wilson 95% CI) | Passes |
|---|---|---|---|
| UI and interface | 37 / 40 | 92.5% (80.1% to 97.4%) | yes |
| Combat preview and hit chance | 31 / 33 | 93.9% (80.4% to 98.3%) | yes |
| Enemy AI quality | 36 / 40 | 90.0% (76.9% to 96.0%) | yes |
| Initiative and turn order | 35 / 40 | 87.5% (73.9% to 94.5%) | yes |
| Bugs and performance | 34 / 40 | 85.0% (70.9% to 92.9%) | yes |
| Price and DLC | 33 / 40 | 82.5% (68.1% to 91.3%) | yes |
| Pacing, grind and late game | 32 / 40 | 80.0% (65.2% to 89.5%) | yes, at the floor |
| Translation and text | 27 / 40 | 67.5% (52.0% to 79.9%) | no |
| Waiting through enemy turns | 25 / 40 | 62.5% (47.0% to 75.8%) | no |
| Maps, terrain and scenarios | 21 / 40 | 52.5% (37.5% to 67.1%) | no |
| First-strike advantage | 11 / 40 | 27.5% (16.1% to 42.8%) | no |

The first-strike result is a lexicon weakness. Most of its hits are "alpha strike" in the mech-combat sense (firing every weapon at once), not acting first. The lexicon stays frozen as `strategy_v1`, because changing it after the audit would tune on the reported numbers; failing aspects stay in the database, hidden in the dashboard by default. Four of the seven passing aspects are controls, and the passing design aspects are AI quality, initiative and turn order, combat previews, and pacing.

**Design themes: findings.** The seven passing aspects add 63 lift comparisons and 20 findings, all on the complaint side:

| Aspect | Scope (eligible games) | Complaint share | Praise share | Lift (95% CI) |
|---|---|---|---|---|
| Bugs and performance | Turn-structure cohort (18) | 16.1% | 6.4% | 2.53 (1.94 to 3.51) |
| Bugs and performance | IGO-UGO (5) | 11.4% | 4.3% | 2.66 (1.12 to 5.70) |
| Bugs and performance | Alternating (6) | 12.0% | 3.9% | 3.11 (1.90 to 4.66) |
| Bugs and performance | Alternating, matched basket (3) | 9.0% | 3.5% | 2.59 (1.18 to 4.64) |
| Bugs and performance | Anchors (3) | 23.7% | 4.9% | 4.86 (3.32 to 8.43) |
| Pacing | Turn-structure cohort (18) | 12.2% | 4.3% | 2.82 (1.90 to 3.96) |
| Pacing | IGO-UGO (5) | 7.9% | 3.0% | 2.61 (1.45 to 4.62) |
| Pacing | Alternating (6) | 17.7% | 5.4% | 3.27 (1.66 to 6.20) |
| Pacing | WEGO (7) | 10.5% | 4.3% | 2.44 (1.16 to 4.27) |
| UI and interface | Turn-structure cohort (18) | 12.4% | 5.4% | 2.29 (1.63 to 3.15) |
| UI and interface | IGO-UGO (5) | 7.3% | 3.8% | 1.91 (1.06 to 2.96) |
| UI and interface | IGO-UGO, matched basket (4) | 8.7% | 4.4% | 1.97 (1.14 to 3.16) |
| UI and interface | Alternating (6) | 8.6% | 4.3% | 2.01 (1.02 to 4.29) |
| UI and interface | WEGO (7) | 19.4% | 7.5% | 2.57 (1.51 to 4.13) |
| Price and DLC | Turn-structure cohort (18) | 17.5% | 12.3% | 1.42 (1.09 to 1.77) |
| Price and DLC | WEGO (7) | 20.1% | 11.8% | 1.71 (1.25 to 2.43) |
| Price and DLC | Anchors (3) | 33.1% | 17.6% | 1.88 (1.43 to 2.43) |
| Enemy AI quality | Turn-structure cohort (18) | 4.3% | 2.3% | 1.91 (1.05 to 3.37) |
| Enemy AI quality | Alternating (6) | 4.6% | 0.9% | 4.94 (1.79 to 19.30) |
| Combat preview and hit chance | Alternating (6) | 5.4% | 0.9% | 5.77 (1.16 to 13.24) |

No passing aspect is a praise finding. No subgroup finding, topic or aspect, reverses direction in the matched-basket rows, although many matched rows have intervals that include 1. The lexicon fires rarely on turn-structure aspects: across all 29 games, 14 reviews mention waiting through enemy turns, 8 first-strike advantage, 37 initiative and turn order and 65 combat previews or hit chance, against 1,316 for price and DLC.

**Check against an earlier desk study of the 22 turn-structure titles on Steam.**
- Steam's all-language positive share (`language=all`, `purchase_type=all`) matches the earlier figures within 1.0 point for all 22 titles. Recomputed subgroup medians: IGO-UGO 84.69% (earlier 84.69%), alternating 88.05% (88.06%), WEGO 81.98% (81.98%). Matched-basket medians: IGO-UGO 80.99% (4 titles), alternating 87.58% (3), WEGO 81.96% (2).
- Share of kept reviews mentioning any turn-structure aspect: alternating 2.7%, IGO-UGO 1.6%, WEGO 0.7%. Complaint lift: alternating 3.76 (1.41 to 7.06), IGO-UGO 2.18 (0.76 to 18.55), WEGO 2.23 (0.12 to 14.72). The earlier study's hand-coded comments leaned to complaint for IGO-UGO and alternating and to praise for WEGO; the measured direction agrees for the first two and not for WEGO, where the interval spans 1. These combined rows stay descriptive: the combined theme needs all four turn-structure aspects to pass the audit, and two failed. Of the two that passed, one subgroup lift is a finding: combat preview and hit chance in the alternating subgroup (5.77, 1.16 to 13.24).
- The frozen lexicon tags 27 of the 131 hand-coded turn-system comments (20.6%) with a turn-structure aspect or pacing, a rough recall floor for the lexicon. This number was seen once during a function check before the lexicon was frozen; no lexicon change was made in response. The check is not fully independent of the lexicon's development: 28 of the desk study's 185 quotes are public Steam reviews that also appear in the seed, 14 of them in the development split. The sentence-sentiment check on these comments below involved no tuning.

**Comparisons made:** 459: 441 lift comparisons (378 topic, 63 aspect) and 18 ranking tests. They produced 24 lift findings (4 topic, 20 aspect) and no ranking findings. Intervals are per-comparison 95% and uncorrected, so every finding is exploratory.

### Interpretation

Crash and stability complaints separate negative from positive reviews across the turn-structure titles by the widest, most stable margin: roughly one not-recommended review in sixteen sits in the crash topic, against one recommended review in two hundred. RoBERTa scores under one in ten of those reviews positive, while VADER rates them near neutral, which is the lexicon's known blind spot on critical strategy reviews.

Players write more critically than they vote. In every cohort a few points fewer reviews read as positive to RoBERTa than recommend the game, and the gap is widest for the anchors, whose recent windows are also the most critical.

Among design themes, pacing (grind, tedium, late-game drag, micromanagement) and the interface are mentioned roughly two to three times as often in negative reviews as in positive ones across the turn-structure titles, and the pattern holds in all three subgroups. Bugs and performance follow the same pattern in the cohort and in two of the three subgroups. The turn-system questions themselves (waiting, initiative, previews) come up in under 3% of kept reviews in each subgroup, too rarely in this sample to separate the subgroups.

The subgroup numbers describe these particular games. Genre, franchise and price travel with turn structure, and each subgroup holds 5 to 7 eligible titles, so no difference here is evidence about the turn rule itself.

### Limitations

- **Identity topics.** On a multi-game corpus BERTopic groups many reviews by game or franchise vocabulary. Those topics are real clusters but answer "which game" more than "which theme".
- **Outliers.** 42% of kept reviews have no topic, so topic shares describe the clustered 58%.
- **Lexicon recall is low.** Shares count mentions the lexicon detects, not true prevalence, and the turn-structure aspects fire on few reviews. A coverage check puts a number on the gap: 42.7% (40.4% to 44.9%) of not-recommended kept reviews match no aspect, and 49.2% match no aspect that passed the audit. Among topics, the tutorial and learning-curve topic is the clearest theme the lexicon misses, with 74% of its not-recommended reviews unmatched, and 35% of not-recommended reviews in the crash topic carry no bugs-and-performance match. The v1 lexicon stays frozen; these are recall gaps, not findings.
- **Transformer truncation and domain.** Reviews over 510 tokens lose their middle. The candidate models were trained on general review and social text, not on games, and the 0.5 cutoff was not tuned.
- **Sentence sentiment passed its gate on the third attempt, narrowly.** The selected rule scores 0.707 against a 0.70 gate, and the neither class is still its weak point (F1 0.51). The two failed attempts are reported above, each read its own fresh test set once, and the gate never moved. The sentence view is descriptive and no finding depends on it.
- **Windows differ.** Capped games cover a recent window and the rest their full history.
- **Confounding.** Cohorts differ in genre, age, price and franchise. Cohort averages weight games equally, and findings must survive dropping any one game, but nothing here is causal.
- **Audit labels are model-made.** Independent model labelers that saw only a guide and the sentences labelled the aspect audit (ties resolved as not relevant) and the sentence sentiment audit (ties resolved as neither). The labels live in `data/labels/` so they can be relabeled and rescored.
- **Four aspects failed the audit.** First-strike advantage (27.5%) mostly catches the mech-combat sense of "alpha strike", and maps (52.5%), waiting through enemy turns (62.5%) and translation (67.5%) also fall short. They make no claims in this version.
- **Infinite lifts in sparse rows.** When a theme has no recommended-review mentions in a bootstrap resample, its lift (complaint share over praise share) is infinite there. The percentile interval ignores those resamples, so some upper bounds sit lower than they would otherwise. The database stores an infinite point lift or bound as empty. None of the 958 affected rows is a finding.

---

## April 2026 cross-genre baseline

The original study, kept reproducible from `data/seed.db` and `studies/april_2026_baseline.toml` (the default for every command). Commit `6124c73` (tagged `april-2026-baseline`) is the last commit before the market study. `data/seed.db` changed once, when the topic labels were fixed; its hash has been fixed since then. The September transformer scores for the April reviews live in a separate committed file, `data/seed_transformer_scores.db`, so the seed itself never changes.

### Business question

Players leave thousands of free-text reviews on Steam, but the platform only surfaces a binary thumbs-up/down. Can an NLP pipeline extract more from that text (reliable sentiment scores, recurring themes, genre-level patterns) or does the noise in short, casual reviews make automated analysis unreliable? This project finds out using 2,382 reviews across 12 games and validates every result against Steam's own recommendation flag as ground truth.


### Dataset

All data comes from the [Steam Reviews API](https://store.steampowered.com/appreviews/) (public, zero authentication).

- **Volume:** 2,382 reviews across 12 games in 4 genres (FPS, Indie, RPG, Strategy)
- **Date range:** March 2023 to April 2026
- **Ground truth:** Steam recommendation flag (thumbs up/down) for sentiment validation
- **Seed database:** 2.3 MB, committed to Git, works out of the box with zero API calls

**Games included:** Baldur's Gate 3, Counter-Strike 2, Civilization VI, DOOM Eternal, Elden Ring, Hades, Hollow Knight, Stardew Valley, Stellaris, Team Fortress 2, The Witcher 3, Total War: WARHAMMER III (200 reviews each, 182 for Stardew Valley).


### Key findings

1. **A third of the reviews were unusable.** 823 of 2,382 reviews failed the sentinel filter (under 10 characters after cleaning, or fewer than 3 real tokens): emoji spam, single-word jokes, ASCII art. The filter caught all of these before they could pollute downstream analysis.
2. **VADER agrees with the Steam recommendation flag 69.4% of the time.** The remaining third is informative: players routinely recommend a game while writing paragraphs of complaints about specific features, or pan a game they clearly enjoy. The recommendation flag is a noisy ground truth.
3. **Sentiment skews positive overall** (average compound score of 0.131), consistent with Steam's self-selection bias. Engaged players who bother to write reviews tend to like what they're playing.
4. **Topic modeling found 3 clusters, two of them non-English.** After the September 2026 label fix (stop words removed from topic labels) and refit, BERTopic assigned all 1,559 kept reviews to one catch-all English cluster (1,513 reviews, labelled "Just / Like / Good / Time"), a Polish cluster (29) and a Russian cluster (17), with no outliers. The pipeline has no language filter for this study, so BERTopic grouped foreign-language reviews by shared vocabulary. At about 130 kept reviews per game, the English reviews do not separate into stable themes.


### Approach

#### Text Preprocessing

- HTML, BBCode, URL, and email removal
- Unicode encoding repair (ftfy)
- spaCy tokenization and lemmatization (`en_core_web_sm`)
- Domain-specific stopword filtering
- Sentinel exclusion: reviews under 10 characters after cleaning or under 3 tokens are flagged and excluded from NLP analysis (823 of 2,382 filtered out, leaving 1,559 for analysis)

#### Sentiment Analysis

- VADER (`nltk.sentiment`) compound scoring on preprocessed text
- Distribution across 1,559 kept reviews: 53.8% positive, 35.1% negative, 11.2% neutral
- Validated against Steam recommendation flag as ground truth (69.4% agreement rate)

September 2026 addition: on the same 1,559 kept reviews, the market study's transformer comparison gives balanced accuracy of 87.8% (86.1% to 89.4%) for RoBERTa-large and 83.4% for DistilBERT, against 70.7% for VADER (scores in `data/seed_transformer_scores.db`; see the market study's sentiment section).

The 69.4% tells you VADER captures the general direction of sentiment but breaks down where text tone and the binary recommendation diverge. A player who writes "love this game but the netcode is garbage and matchmaking takes forever" gets a mixed compound score, even though they hit thumbs-up. VADER is a blunt instrument on this kind of text.

#### Topic Modeling

- BERTopic with sentence-transformer embeddings and UMAP dimensionality reduction
- 3 topics extracted from the review corpus (no outlier documents); topic labels exclude English and project stop words
- Per-review topic assignments stored in SQLite for dashboard querying

Topic quality is weak on this corpus. One English cluster holds 97% of kept reviews, and the other two are Polish and Russian reviews that a language filter would remove. With 1,559 reviews across 12 games, BERTopic is working with a small corpus. The April 2026 fit reported 11 topics; the refit in the current environment clusters differently even with the original code, so the 11-topic result is not reproducible and is superseded.

#### Limitations

- **No language filter.** The pipeline processes all reviews regardless of language. BERTopic groups non-English reviews into clusters that look like topics but carry no analytical meaning. This inflates the topic count and dilutes English-language topic quality.
- **Small corpus per game.** With 200 reviews pulled per game (1,559 total after sentinel filtering), BERTopic is working near the lower bound for stable topic extraction. Topic boundaries are soft and some themes likely merge or split depending on the random seed.
- **VADER is lexicon-based.** It scores words in isolation without understanding context, sarcasm, or domain-specific slang. Gaming jargon ("this game slaps," "absolute banger") and mixed-sentiment paragraphs degrade accuracy. The 69.4% agreement rate reflects this ceiling.

---

## Dashboard

The Streamlit dashboard opens on the market study. The sidebar selector lists every `studies/*.toml` whose seed database exists, including studies you build yourself. Filters push down to SQLite queries.

- **Market Overview:** natural recommend share per game with Wilson intervals and the population share, cohort averages with cluster-bootstrap intervals and per-game dots, RoBERTa positive share and VADER per cohort and per game, median playtime by recommend flag.
- **Topics x Sentiment:** complaint lift per topic with intervals for each cohort, subgroup or matched basket, recommend share, RoBERTa positive share and VADER inside each topic, finding or descriptive status, and a topic by game heatmap.
- **Design Themes:** aspect complaint shares by cohort and per game, sentence-level sentiment of aspect sentences from the audited three-class model (negative, neither, positive shares per aspect, with a per-game table), example sentences, and a link into Review Explorer filtered by aspect. Aspects without a passing audit stay behind a toggle.
- **Cohort Comparison:** any two cohorts, subgroups or matched baskets side by side on any aspect or topic, from the precomputed statistics: theme, complaint and praise shares with intervals, lift, finding status, and each scope's recommend and sentiment shares. It makes no test of the difference. For the April baseline, whose files stay frozen, it compares genres live from the seed on recommend share, sentiment and topic shares with Wilson intervals.
- **Methods:** sampling parameters, pull window per game, drop counts, the finding rule, the comparison count, VADER validation, the adopted sentiment model, the sentence-level candidate selection and audit, the lexicon coverage check and the check against the earlier study.
- **Sentiment Analysis:** opens with the held-out model comparison for the selected study (market test split, or the April seed and its strategy titles) and the adoption verdict.
- **Overview, Sentiment Analysis, Topic Analysis, Review Explorer:** the original four views, available for both studies. Their charts and the sentiment slider use VADER.

Intervals and findings describe the unfiltered study. When sidebar filters are active, the market views show SQL point estimates and n with a notice.

---

## Tools

| Layer | Tool |
|-------|------|
| Runtime | Python 3.11+ |
| Database | SQLite 3 |
| Data ingestion | `requests` (Steam appreviews endpoint) |
| Text preprocessing | spaCy (`en_core_web_sm`), lingua language detection |
| Sentiment analysis | NLTK (VADER), Hugging Face Transformers and PyTorch (RoBERTa-large, DistilBERT, three-class RoBERTa-base for sentences) |
| Topic modeling | BERTopic, sentence-transformers, UMAP |
| Statistics | NumPy, SciPy (Wilson and bootstrap intervals, ROC AUC) |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |

---

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd nlp_dashboard
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm

# Run dashboard (uses the included seed databases, no API calls needed)
.venv/bin/streamlit run dashboard/app.py
```

Both shipped studies are frozen: their TOMLs set `frozen = true`, `pipeline.py` refuses them, and every command that writes (pull, load and seed, preprocessing, sentiment, topics, aspect tagging, statistics, transformer scoring) refuses their files. Passing `--unfreeze-i-know` is the only way to regenerate a shipped study, and doing so rewrites committed databases. A refused command prints `error: ...` and exits with status 2.

To rebuild the market study from the Steam API (about 1.5 hours of paced requests; cached pages are never pulled twice):

```bash
S=studies/strategy_market_2026.toml
U=--unfreeze-i-know
.venv/bin/python src/data_pull.py --study $S $U
.venv/bin/python src/db_setup.py --study $S --full $U
.venv/bin/python src/preprocessing.py --study $S $U
.venv/bin/python src/sentiment.py --study $S $U
.venv/bin/python src/topics.py --study $S $U
.venv/bin/python src/aspects.py --study $S $U
# Transformer scores (downloads the pinned model weights once, under 3 GB; about 15 minutes on Apple MPS)
.venv/bin/python src/sentiment_transformer.py score --study $S --units doc --models distilbert/distilbert-base-uncased-finetuned-sst-2-english $U
.venv/bin/python src/sentiment_transformer.py score --study $S --units doc,sentence --models siebert/sentiment-roberta-large-english $U
.venv/bin/python src/sentiment_transformer.py score --study studies/april_2026_baseline.toml --units doc $U
.venv/bin/python src/sentiment_transformer.py validate --market $S --april studies/april_2026_baseline.toml $U
.venv/bin/python src/sentiment_transformer.py audit-metrics --study $S --version v1 $U
# Second sentence audit: three-class candidate against a tuned band, selection on v1, one read of v2
.venv/bin/python src/sentiment_transformer.py score --study $S --units sentence --models cardiffnlp/twitter-roberta-base-sentiment-latest $U
.venv/bin/python src/sentiment_transformer.py dev-select --study $S --dev-versions v1 --models cardiffnlp/twitter-roberta-base-sentiment-latest --no-threshold --with-band $U
.venv/bin/python src/sentiment_transformer.py audit-metrics --study $S --version v2 $U
# Third sentence audit: two three-class models, argmax or a tuned neutral cut, selection on v1+v2, one read of v3
.venv/bin/python src/sentiment_transformer.py score --study $S --units sentence --models j-hartmann/sentiment-roberta-large-english-3-classes $U
.venv/bin/python src/sentiment_transformer.py dev-select --study $S --dev-versions v1,v2 $U
.venv/bin/python src/sentiment_transformer.py audit-metrics --study $S --version v3 $U
.venv/bin/python src/stats.py --study $S $U
.venv/bin/python src/stats.py --study $S --coverage $U
```

These flags reproduce the committed state, with one exception: after the v3 selection, the unselected `j-hartmann` sentence rows were deleted from the seed to keep it under its 20 MiB cap, so a full rerun ends about 0.25 MiB over the cap until the same rows are removed. Document scores come from both review-level candidates, sentence scores from the adopted RoBERTa-large and the selected three-class model. Scoring DistilBERT sentences as well would also push the seed past the cap. To draw a new audit set or import labels, use `export-audit --version <v>` and `import-labels <csv> --version <v>`. The commands above read the committed label files and never regenerate them.

Without `--study`, every command runs the April baseline as before (`data_pull.py`, `db_setup.py --full`, `preprocessing.py`, `sentiment.py`, `topics.py`); commands that write its files need `--unfreeze-i-know`.

---

## Run your own study

Any set of Steam app IDs, grouped into cohorts, runs through the same pipeline. Three commands build a study end to end:

```bash
# 1. Write studies/indie_2026.toml with one cohort, then add a second cohort.
#    --lookup fills names and genres from the public Steam store API (cached in data/raw/indie_2026/appdetails/).
.venv/bin/python src/study_cli.py new --name indie_2026 --title "Indie action and farming games" \
    --apps 1145360,367520 --cohort action --lookup
.venv/bin/python src/study_cli.py add --study studies/indie_2026.toml --apps 413150 --cohort farming --lookup

# 2. Pull, load, sample, preprocess, score VADER, fit topics, tag aspects and compute statistics.
.venv/bin/python src/pipeline.py --study studies/indie_2026.toml --full

# 3. Open the dashboard and pick the study in the sidebar.
.venv/bin/streamlit run dashboard/app.py
```

Without `--lookup`, pass `--names "A|B"` and `--genre G`. `study_cli.py new` writes every sampling parameter into the study file, and each has a flag:

| Flag | Default | Meaning |
|---|---|---|
| `--filter` | `recent` | Steam sort order; `recent` pulls newest first |
| `--language` | `english` | Steam review language |
| `--purchase-type` | `all` | `all`, `steam` or `non_steam_purchase` |
| `--raw-cap` | 20000 | Reviews pulled per game, newest first |
| `--seed-cap` | 500 | Reviews per game in the uniform seed sample |
| `--random-seed` | 42 | Seed for that sample |
| `--window-start`, `--window-end` | none | Keep only reviews created in this date range (UTC, applied at load) |
| `--review-type` | `all` | `positive` or `negative` only together with `--stratified` |
| `--lexicon` | `studies/aspects_generic_v1.toml` | Aspect lexicon, or `''` for none |
| `--min-topic-size`, `--umap-preset` | 40, `bertopic_default` | BERTopic settings |

Natural sampling is the default: every review type is pulled, so each game keeps its real recommend rate. A study that filters on the recommend flag must say so with `--stratified`; the flag is recorded in `pull_runs`, and the dashboard then labels every recommend rate "design-fixed". The study file is plain TOML and can be edited by hand; `study.load_study` rejects a file with an app missing its cohort, a duplicate app ID, or a filtered review type without `stratified = true`.

`pipeline.py` runs the stages in order and is idempotent: a second run with unchanged inputs changes no table. Pulled games are cached and never pulled twice, and rebuild stages (seed sample, aspect tagging, statistics) rerun only when their inputs change. Each stage reports `ran`, `no-op` (an idempotent stage that found nothing to do), `skipped` (a rebuild stage whose inputs are unchanged) or `not applicable`. Without `--full` it reruns the analysis on the existing seed, for example after editing the lexicon.

`study_cli.py new` refuses a name whose seed, full database or raw directory belongs to another study or already exists; `--force` overrides that for your own studies, but never overwrites a frozen study's file or reuses its paths. `add` refuses a frozen study. The topics stage saves the fitted BERTopic model (about 92 MB) under `outputs/<study>/bertopic_model/`, which Git ignores.

The generic lexicon (`bugs`, `performance`, `price`, `content`, `difficulty`, `ui`, `story`, `multiplayer`) is unaudited, and the dashboard says so: its aspect rows are descriptive until a study audits them with `src/aspects.py --study <file> --export-audit`, labels the exported sentences, imports them with `--import-labels`, and reruns the pipeline. Transformer sentiment scoring and its adoption test stay separate from the pipeline, so a new study reports VADER sentiment, and RoBERTa columns appear only where a validation run adopted the model (the market study). New full databases (`data/*_full.db`) and raw caches stay out of Git. A new study's seed (`data/<name>_seed.db`) is not ignored: if you commit it, it counts against the 25 MiB budget for committed databases, and the size test sums every `data/*_seed.db` with the April files.

---

## Project Structure

```
nlp_dashboard/
├── src/
│   ├── utils.py                # Shared constants, paths, logger
│   ├── study.py                # Study TOML loader and validation
│   ├── study_cli.py            # new/add commands that write study files
│   ├── pipeline.py             # Runs every stage in order, idempotently
│   ├── data_pull.py            # Steam API ingestion (April fixed pull, study census pull)
│   ├── db_setup.py             # Schema, loading, April and natural seed builders
│   ├── preprocessing.py        # Text cleaning, language filter (spaCy, lingua)
│   ├── sentiment.py            # VADER scoring + ground truth validation
│   ├── sentiment_transformer.py # Transformer scoring, Decision 11 validation, sentence audit
│   ├── validation.py           # Balanced accuracy, macro-F1, AUC, paired bootstrap
│   ├── topics.py               # BERTopic topic modeling
│   ├── aspects.py              # Phrase-lexicon aspect tagging and audit export
│   └── stats.py                # Intervals, cohort statistics, finding rule, coverage check
├── studies/
│   ├── april_2026_baseline.toml
│   ├── strategy_market_2026.toml
│   ├── aspects_strategy_v1.toml
│   └── aspects_generic_v1.toml # Unaudited lexicon for any genre
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   └── queries.py              # SQL query functions for dashboard views
├── data/
│   ├── seed.db                 # April baseline seed (2.3 MiB, hash fixed)
│   ├── seed_transformer_scores.db # April transformer scores (0.1 MiB)
│   ├── market_seed.db          # Market study seed with transformer scores (19.9 MiB)
│   └── labels/                 # Audit labeling guides and labels
├── tests/                      # 195 tests covering pull, schema, study CLI, pipeline, frozen studies, NLP, statistics, sentiment validation, dashboard
└── requirements.txt
```

The committed databases stay under 25 MB together: 23,330,816 bytes (22.2 MiB, 23.3 MB). The market seed is 20,856,832 bytes (19.9 MiB), under its 20 MiB cap. To stay within budget, only RoBERTa-large and the selected three-class model keep sentence scores (DistilBERT has document scores only, and the unselected three-class model's rows were dropped), and score rows store a small model id instead of repeating the model name and revision.

---

## Next Steps

- **Lexicon v2.** Rewrite the four failing aspects and audit them on fresh test-split sentences; `strategy_v1` stays the cited version.
- **Sentence sentiment, tighter labels.** The neither class holds the sentence gate back (F1 0.51 on v3), and it is also the softest label: the guide sends every tie to neither, so the class mixes neutral, balanced-mixed and undecidable sentences. A future version would split that class in the guide (neutral against mixed) and relabel, which changes the target and so counts as a new study design, not a fourth attempt at this gate.
- **Transformer validation per study.** A dev and test comparison inside any study, so a new study can adopt a transformer scorer on its own evidence.
