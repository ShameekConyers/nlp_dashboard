# What Are Customers Actually Saying? NLP-Powered Review Analysis

An NLP pipeline that turns Steam reviews into structured evidence: BERTopic topics crossed with sentiment, an audited phrase lexicon for named design questions, and every rate checked against Steam's own recommend flag. It ships two studies, each with a committed SQLite seed and a Streamlit dashboard:

- **Market analysis of historical and tactical strategy games** (September 2026, default): 29 games, 12,315 sampled reviews, 9,622 kept for analysis.
- **April 2026 cross-genre baseline**: 12 games across four genres, 2,382 reviews, kept reproducible.

---

## Market analysis of historical and tactical strategy games

### Business question

A design or marketing team planning a turn-based strategy game wants to know what players of comparable games praise and what drives them to leave a negative review, and how that differs between games that resolve turns in different ways (one side moves then the other, units alternate, or both sides plan and resolve at once). Reviews are the largest free source of that evidence. The question is which themes the review text supports with measured rates and intervals, and which ones it cannot settle.

### Dataset and sampling

- **Games:** 29 titles in three cohorts, set in `studies/strategy_market_2026.toml`. `turn_structure` holds 22 titles in three subgroups (IGO-UGO 7, alternating 7, WEGO 8, including two stand-ins). `historical_peer` holds 4 historical grand-strategy titles. `anchor` holds 3 large strategy titles (Civilization VI, Stellaris, Total War: WARHAMMER III). A `matched_basket` tag marks 9 titles with 2,000 to 14,000 reviews for a like-for-like check.
- **Pull (26 September 2026):** Steam appreviews, `filter=recent`, `language=english`, `review_type=all`, `purchase_type=all`, newest first, capped at 20,000 reviews per game. 22 games were pulled in full (every English review they have). Seven hit the cap and cover a recent window ending 26 September 2026: XCOM 2 from December 2021, BattleTech from October 2018, Divinity: Original Sin 2 from January 2023, Baldur's Gate 3 from February 2026, Civilization VI from August 2024, Stellaris from February 2024 and Total War: WARHAMMER III from March 2025. 209,223 reviews were pulled and 208,380 loaded (843 had empty text).
- **Seed (`data/market_seed.db`, 18.9 MB):** a uniform random sample of up to 500 reviews per game (seed 42), with no stratification, so each game keeps its natural recommend rate. 12,315 reviews.
- **Kept for analysis:** 9,622. Dropped: 1,039 under 10 characters, 1,555 with fewer than 3 content tokens, and 99 detected as another language (lingua) although Steam listed them as English.
- **Eligibility:** a game counts toward cohort estimates if it has at least 100 kept reviews and at least 30 not-recommended kept reviews. 23 of 29 games qualify: 18 of 22 turn-structure titles, 2 of 4 historical peers and all 3 anchors. Into the Breach, War in the East 2, Baldur's Gate 3, Flashpoint Campaigns, Nobunaga's Ambition and Romance of the Three Kingdoms X are shown per game but greyed as low n.

### Method

1. **Preprocessing:** the April cleaning and sentinel rules, plus a language filter. A review is dropped as non-English when lingua's top language is not English at confidence 0.50 or higher.
2. **Sentiment:** VADER compound score per review, validated against the recommend flag on a held-out test split (half the reviews, chosen by a hash of the review ID).
3. **Topics (primary lens):** BERTopic with `all-MiniLM-L6-v2` embeddings, BERTopic's default UMAP settings (seeded) and a minimum topic size of 40. Topic labels drop English and project stop words. The fit found 42 topics; 4,038 of 9,622 kept reviews (42.0%) are outliers with no topic.
4. **Design themes (supplementary lens):** a versioned phrase lexicon (`studies/aspects_strategy_v1.toml`, frozen after two passes on the development split) tags sentences with seven design aspects (waiting through enemy turns, first-strike advantage, initiative and turn order, combat preview and hit chance, enemy AI quality, maps, pacing) and four controls (bugs and performance, UI, translation, price and DLC). An aspect supports claims only if a precision audit of at least 30 test-split sentences finds 80% or more relevant. Audit sentences come from the seed's test split (up to 40 per aspect, one per review), except for three rare aspects (waiting through enemy turns, first-strike advantage, initiative and turn order): the seed's test split held only 7, 3 and 18 of them, so their 40 each come from the test split of the full local database, processed with the same cleaning, language and lexicon rules. The 433 audit labels (`data/labels/aspect_audit_strategy_v1.csv`) were made by an independent model labeler (Claude Opus) that saw only the labeling guide and the sentences, not the builder's work or the results. The guide offers no "unclear" value and tells the labeler to resolve ties as not relevant, so precision is conservative. Anyone can relabel the file and rerun `src/stats.py`, but reading the sentences for the three full-DB aspects needs the local full database: 111 of those 120 sentences come from reviews that are not in the committed seed.
5. **Metrics per theme:** complaint share (share of not-recommended reviews in the theme), praise share (same for recommended reviews), complaint lift (their ratio), and the mean VADER score and recommend share of the theme's reviews.
6. **Uncertainty:** Wilson 95% intervals for per-game rates, a review-level bootstrap (2,000 resamples) for per-game theme metrics, and a two-stage cluster bootstrap (games, then reviews) for cohort averages, which weight each eligible game equally.
7. **Finding rule (`FINDING_RULE` in `src/stats.py`):** a theme is a cohort finding only if the cohort has 3 or more eligible games, the lift interval lies entirely above 1 (complaint) or below 1 (praise), the result is stable when any one game is dropped, the theme has at least 5 mentions on each side, and, for aspects, the precision audit passed. Stable means the lift stays on the same side of 1 and the theme's complaint-share rank within its lens (aspects among passing aspects, topics among topics) moves by at most one place. A ranking finding ("most-cited complaint") also needs the top complaint theme's interval to clear the second theme's; praise rankings and the all-games scope are always descriptive. Everything else is descriptive.
8. **Rule changes made after results were seen.** Two, both disclosed here. The minimum-support condition was added after the first run, which read 13 collapsed [0, 0] intervals from topics with no complaint mentions as praise findings. Later, an independent review found the stability check ranked complaint shares across all themes at once (aspects and 42 topics mixed) instead of within each lens as the written plan specifies; correcting the code to the written rule moved the lift-finding count from 14 to 24. The same review found the ranking check had marked a praise-side ranking as a finding, which the plan does not allow; after the fix there are no ranking findings.

### Results

All numbers come from `data/market_seed.db` (`analysis_runs`, `game_stats`, `cohort_stats`, `theme_stats`) and can be re-derived with the commands in Quick Start.

**Natural recommend share.** Across the 22 fully pulled games, the seed share sits 0.3 points below Steam's English population share on average, so the sample reproduces the population. Among kept reviews the gap widens to 2.0 points, because short reviews that fail the text filter are mostly positive. Two capped anchors differ from their all-time population because their recent window is more critical: Stellaris 72.0% in the seed against 85.8% all-time, Total War: WARHAMMER III 63.6% against 71.3%.

| Scope | Games (eligible) | Kept reviews | Recommend share, macro (95% CI) | Mean VADER, macro (95% CI) | Median hours at review, recommended / not |
|---|---|---|---|---|---|
| Turn structure | 22 (18) | 7,854 | 83.5% (80.5% to 86.5%) | 0.467 (0.421 to 0.513) | 32.6 / 7.2 |
| IGO-UGO | 7 (5) | 2,620 | 83.0% (78.6% to 87.9%) | 0.443 (0.381 to 0.503) | 32.5 / 7.9 |
| Alternating | 7 (6) | 2,685 | 88.4% (85.8% to 91.0%) | 0.532 (0.462 to 0.594) | 50.3 / 11.5 |
| WEGO | 8 (7) | 2,549 | 79.2% (72.9% to 85.3%) | 0.428 (0.348 to 0.511) | 20.2 / 4.3 |
| Historical peers | 4 (2) | 783 | 73.9% (66.3% to 84.3%) | 0.501 (0.438 to 0.573) | 25.2 / 10.7 |
| Anchors | 3 (3) | 985 | 67.7% (55.0% to 81.0%) | 0.273 (0.130 to 0.400) | 131.7 / 204.3 |

The recommend-share and VADER macros use every game with at least 100 kept reviews, with two-stage cluster-bootstrap intervals; medians pool kept reviews in the scope. Per-game mean VADER is in the dashboard's Market Overview.

**Topics crossed with sentiment.** 378 topic lift comparisons were evaluated (42 topics across 9 cohort, subgroup and matched-basket scopes). Four are findings:

| Topic | Scope (eligible games) | Complaint share | Praise share | Lift (95% CI) | Recommended within topic | Mean VADER |
|---|---|---|---|---|---|---|
| Crashes / Crash / Crashing / Screen | Turn-structure cohort (18) | 6.3% | 0.5% | 12.8 (7.2 to 29.3) | 21.7% | 0.048 |
| Crashes / Crash / Crashing / Screen | IGO-UGO (5) | 5.4% | 0.4% | 12.1 (4.9 to 52.1) | 25.5% | 0.206 |
| Crashes / Crash / Crashing / Screen | WEGO (7) | 7.7% | 0.7% | 11.1 (4.9 to 41.7) | 21.8% | -0.018 |
| Final / Fantasy / Tactics / Fft (praise) | Alternating, matched basket (3) | 16.5% | 35.1% | 0.47 (0.24 to 0.74) | 94.6% | 0.724 |

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

**VADER on the natural sample** (test split, 4,783 kept reviews, 913 not recommended): accuracy 78.2%, balanced accuracy 70.3% (68.7% to 72.0%). It scores 42.4% of not-recommended reviews as positive. Predicting "recommended" for every review would score 80.9% accuracy, which is why balanced accuracy is the reported metric here and why the April 69.4% (on a 50/50 sample) is not comparable. Every VADER column in the dashboard carries this caveat.

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
- The frozen lexicon tags 27 of the 131 hand-coded turn-system comments (20.6%) with a turn-structure aspect or pacing, a rough recall floor for the lexicon. This number was seen once during a function check before the lexicon was frozen; no lexicon change was made in response.

**Comparisons made:** 459: 441 lift comparisons (378 topic, 63 aspect) and 18 ranking tests. They produced 24 lift findings (4 topic, 20 aspect) and no ranking findings. Intervals are per-comparison 95% and uncorrected, so every finding is exploratory.

### Interpretation

[USER REVIEWS] Crash and stability complaints separate negative from positive reviews across the turn-structure titles by the widest, most stable margin: roughly one not-recommended review in sixteen sits in the crash topic, against one recommended review in two hundred. VADER rates those reviews as near neutral, which is the lexicon's known blind spot on critical strategy reviews.

[USER REVIEWS] Among design themes, pacing (grind, tedium, late-game drag, micromanagement) and the interface are mentioned roughly two to three times as often in negative reviews as in positive ones across the turn-structure titles, and the pattern holds in all three subgroups. Bugs and performance follow the same pattern in the cohort and in two of the three subgroups. The turn-system questions themselves (waiting, initiative, previews) come up in under 3% of kept reviews in each subgroup, too rarely in this sample to separate the subgroups.

[USER REVIEWS] The subgroup numbers describe these particular games. Genre, franchise and price travel with turn structure, and each subgroup holds 5 to 7 eligible titles, so no difference here is evidence about the turn rule itself.

### Limitations

- **Identity topics.** On a multi-game corpus BERTopic groups many reviews by game or franchise vocabulary. Those topics are real clusters but answer "which game" more than "which theme".
- **Outliers.** 42% of kept reviews have no topic, so topic shares describe the clustered 58%.
- **Lexicon recall is low.** Shares count mentions the lexicon detects, not true prevalence, and the turn-structure aspects fire on few reviews.
- **Windows differ.** Capped games cover a recent window and the rest their full history.
- **Confounding.** Cohorts differ in genre, age, price and franchise. Cohort averages weight games equally, and findings must survive dropping any one game, but nothing here is causal.
- **Audit labels are model-made.** An independent model labeler that saw only the guide and the sentences labelled the aspect audit, resolving ties as not relevant. The labels live in `data/labels/` so they can be relabeled and rescored.
- **Four aspects failed the audit.** First-strike advantage (27.5%) mostly catches the mech-combat sense of "alpha strike", and maps (52.5%), waiting through enemy turns (62.5%) and translation (67.5%) also fall short. They make no claims in this version.
- **Infinite lifts in sparse rows.** When a theme has no recommended-review mentions in a bootstrap resample, its lift (complaint share over praise share) is infinite there. The percentile interval ignores those resamples, so some upper bounds sit lower than they would otherwise. The database stores an infinite point lift or bound as empty. None of the 958 affected rows is a finding.

---

## April 2026 cross-genre baseline

The original study, kept reproducible from `data/seed.db` and `studies/april_2026_baseline.toml` (the default for every command). Commit `6124c73` (tagged `april-2026-baseline`) is the last commit before the market study.

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
4. **Topic modeling found 3 clusters, two of them non-English.** After the September 2026 label fix (stop words removed from topic labels) and refit, BERTopic assigned all 1,559 kept reviews to one catch-all English cluster (1,513 reviews, labelled "Just / Like / Good / Time"), a Polish cluster (29) and a Russian cluster (17), with no outliers. The pipeline has no language filter for this study, so BERTopic grouped foreign-language reviews by shared vocabulary. [USER REVIEWS] At about 130 kept reviews per game, the English reviews do not separate into stable themes.


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

The Streamlit dashboard opens on the market study; a sidebar selector switches to the April baseline. Filters push down to SQLite queries.

- **Market Overview:** natural recommend share per game with Wilson intervals and the population share, cohort averages with cluster-bootstrap intervals and per-game dots, median playtime by recommend flag.
- **Topics x Sentiment:** complaint lift per topic with intervals for each cohort, subgroup or matched basket, VADER and recommend share inside each topic, finding or descriptive status, and a topic by game heatmap.
- **Design Themes:** aspect complaint shares by cohort and per game, example sentences, and a link into Review Explorer filtered by aspect. Aspects without a passing audit stay behind a toggle.
- **Methods:** sampling parameters, pull window per game, drop counts, the finding rule, the comparison count, VADER validation and the check against the earlier study.
- **Overview, Sentiment Analysis, Topic Analysis, Review Explorer:** the original four views, available for both studies.

Intervals and findings describe the unfiltered study. When sidebar filters are active, the market views show SQL point estimates and n with a notice.

---

## Tools

| Layer | Tool |
|-------|------|
| Runtime | Python 3.11+ |
| Database | SQLite 3 |
| Data ingestion | `requests` (Steam appreviews endpoint) |
| Text preprocessing | spaCy (`en_core_web_sm`), lingua language detection |
| Sentiment analysis | NLTK (VADER) |
| Topic modeling | BERTopic, sentence-transformers, UMAP |
| Statistics | NumPy (Wilson and bootstrap intervals) |
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

To rebuild the market study from the Steam API (about 1.5 hours of paced requests; cached pages are never pulled twice):

```bash
S=studies/strategy_market_2026.toml
.venv/bin/python src/data_pull.py --study $S
.venv/bin/python src/db_setup.py --study $S --full
.venv/bin/python src/preprocessing.py --study $S
.venv/bin/python src/sentiment.py --study $S
.venv/bin/python src/topics.py --study $S
.venv/bin/python src/aspects.py --study $S
.venv/bin/python src/stats.py --study $S
```

Without `--study`, every command runs the April baseline exactly as before (`data_pull.py`, `db_setup.py --full`, `preprocessing.py`, `sentiment.py`, `topics.py`).

---

## Project Structure

```
nlp_dashboard/
├── src/
│   ├── utils.py                # Shared constants, paths, logger
│   ├── study.py                # Study TOML loader
│   ├── data_pull.py            # Steam API ingestion (April fixed pull, study census pull)
│   ├── db_setup.py             # Schema, loading, April and natural seed builders
│   ├── preprocessing.py        # Text cleaning, language filter (spaCy, lingua)
│   ├── sentiment.py            # VADER scoring + ground truth validation
│   ├── topics.py               # BERTopic topic modeling
│   ├── aspects.py              # Phrase-lexicon aspect tagging and audit export
│   └── stats.py                # Intervals, cohort statistics, finding rule
├── studies/
│   ├── april_2026_baseline.toml
│   ├── strategy_market_2026.toml
│   └── aspects_strategy_v1.toml
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   └── queries.py              # SQL query functions for dashboard views
├── data/
│   ├── seed.db                 # April baseline seed (2.3 MB)
│   ├── market_seed.db          # Market study seed (18.9 MB)
│   └── labels/                 # Audit labeling guide and labels
├── tests/                      # 134 tests covering pull, schema, NLP, statistics, dashboard
└── requirements.txt
```

The committed databases stay under 25 MB together.

---

## Next Steps

- **Lexicon v2.** Rewrite the four failing aspects and audit them on fresh test-split sentences; `strategy_v1` stays the cited version.
- **Transformer sentiment.** Compare a transformer scorer with VADER on the held-out split and adopt it only if it wins.
- **Broader tool.** Any set of Steam app IDs and cohorts from the command line, with a cohort comparison view.
