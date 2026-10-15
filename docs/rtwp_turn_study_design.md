# Study design: real-time-with-pause versus turn-based combat

Phase A design and exploratory results, written October 15, 2026 and rerun the same day on the development half of the full database. Revised the same day for lexicon v3, the single retry after the first aspect audit failed six aspects (section 5), and again after the v3 audit passed all six, when the dual-mode views were rerun with v3.

**Result.** On the development half of the full database, 25 of 26 games are eligible and every CRPG and strategy scope keeps all of its games. With lexicon v3 and every aspect assumed to pass its precision audit, the finding rule yields 66 candidates (`stats.py --candidates`, run 4 of the analysis database). 14 of them are combat-model aspects. `pause_micro` (pausing and micromanagement) is a complaint candidate in 6 scopes and `combat_pacing` (slow or too many fights) in 3, in turn-based scopes as well as real-time-with-pause (RTwP) ones. `mode_turn_based` sits on the praise side among turn-based strategy games, lift 0.47 (95% interval 0.25 to 0.74). `mode_rtwp` is a candidate nowhere. The other 52 candidates are bugs and performance, UI clarity, price and DLC, and six topics led by a crash topic. The strongest seed-based combat signal, `pause_micro` among RTwP CRPGs (lift 4.13 on 2 games), shrinks to 1.51 (0.72 to 2.53) on all 5 games and is not a candidate. Under lexicon v2 the same rule gave 70 candidates (run 3), now superseded. The dual-mode views stay descriptive and use lexicon v3. Reviews that mention only the turn-based mode recommend the game more often than reviews that mention neither mode in Pathfinder: Kingmaker (80.0% against 73.2%) and Pillars of Eternity II: Deadfire (92.1% against 85.9%), with intervals that do not overlap. In Pillars of Eternity and Wrath of the Righteous the intervals overlap.

- 26 games, 240,711 English reviews pulled, 239,850 loaded, 119,766 in the development half, 89,953 kept development-half reviews, 17,430 of them not recommended.
- Lexicon `rtwp_turn_v3`, frozen. It revises the six aspects that failed the v2 audit and keeps `mode_rtwp`, `realtime_chaos` and `bugs_performance` from v2, which passed. v2 audit: 360 test-half sentences, three aspects pass. v3 audit: 240 test-half sentences, 40 for each revised aspect, from reviews outside the v2 set, all six pass. All nine aspects enter phase B (protocol, section 13).
- Phase B is specified in `docs/rtwp_turn_protocol.md`: the 66 candidates on 119,485 test-half reviews, 60.0 expected to replicate if the development effects are true and 46.7 under halved effects, with a pass line of 43.
- The seed-based phase A results of the first pass are superseded and kept under `outputs/rtwp_turn_2026/exploratory/superseded_seed_dev/`.

## 1. Question

What do players of comparable games praise and complain about under real-time-with-pause (RTwP) combat versus turn-based (TB) combat, and what do players of games that offer both modes say about the choice? The study is market analysis of public Steam reviews.

The study has two phases.

- **Phase A (this document, exploratory).** Every outcome statistic is computed on the development half of the hash split (`stats.dev_test_split`) of the full database. Lexicon development, the topic model, finding candidates and the dual-mode views all come from that half. No recommend flag of a test-half review is read.
- **Phase B (`docs/rtwp_turn_protocol.md`, confirmatory, not run).** A pre-registered read of the phase A candidates on the test half of the full database, after the aspect precision audit is labeled.

## 2. Cohorts and subgroups

| Cohort | Subgroup | Games |
|---|---|---|
| `crpg` | `rtwp` | Baldur's Gate: Enhanced Edition, Baldur's Gate II: Enhanced Edition, Icewind Dale: Enhanced Edition, Tyranny, Black Geyser: Couriers of Darkness |
| `crpg` | `turn_based` | Divinity: Original Sin 2, Baldur's Gate 3, Solasta: Crown of the Magister, Warhammer 40,000: Rogue Trader, Wasteland 3 |
| `strategy` | `rtwp` | Europa Universalis IV, Crusader Kings III, Hearts of Iron IV, Victoria 3, Stellaris |
| `strategy` | `turn_based` | Civilization VI, Old World, Humankind, Age of Wonders 4, Endless Legend |
| `dual_mode` | none | Pillars of Eternity, Pathfinder: Kingmaker, Pillars of Eternity II: Deadfire, Pathfinder: Wrath of the Righteous, Combat Mission: Shock Force 2, Combat Mission: Battle for Normandy |

The `dual_mode` cohort carries the `dual_mode` tag and is analysed on its own. Its games never enter the `crpg` or `strategy` scopes. Because the subgroup names `rtwp` and `turn_based` appear in both cohorts, the statistics add one `cohort_subgroup` scope per cohort and subgroup (for example `crpg/rtwp`). The plain `subgroup` scopes, which pool CRPG and strategy games, are reported but carry no question of their own.

In grand strategy and 4X games the combat model is the game clock: the RTwP titles run a pausable continuous clock and resolve battles in it, the TB titles advance in turns. In the CRPGs it is the party combat itself.

### Changes to the candidate list

- **Pillars of Eternity moved from `crpg/rtwp` to `dual_mode`.** Its store page now offers both modes, and Obsidian's announcement of April 6, 2026 says the turn-based mode is live. It is pulled in full like the other dual-mode titles and gets a before-and-after view.
- **Planescape: Torment: Enhanced Edition dropped.** No developer statement on its combat model was found on its store page or in its seven Steam announcements. The only first-party signal is a player-applied store tag. It is also a dialogue-led game with little combat, so it is a poor match for a combat-model question.
- **Black Geyser: Couriers of Darkness added to `crpg/rtwp`** in its place (tag `swap_in`). Its store page states real-time tactical combat with pause and names Baldur's Gate and Icewind Dale as its models. It also adds a 2022 release to a subgroup otherwise made of 2013 to 2016 releases.

### Classification evidence

Every app ID was resolved with `study_cli.py --lookup` (names and genres cached under `data/rtwp_turn_2026/raw/rtwp_turn_2026/appdetails/`). Store statements come from the Steam store API description of each game, fetched October 15, 2026. Developer statements come from the developer's Steam announcements (`ISteamNews/GetNewsForApp`, feed `steam_community_announcements`), and the URL given is the one that API returns for the post. English reviews is Steam's English total on the day of the pull (`query_summary.total_reviews` of the first page). Release date and list price come from the store API (US store, October 15, 2026).

| Game | App ID | Group | Mode | First-party statement | Source | Mode added | English reviews |
|---|---|---|---|---|---|---|---|
| Baldur's Gate: Enhanced Edition | 228280 | crpg/rtwp | RTwP | Store: "Real-Time-with-Pause Combat" | [store](https://store.steampowered.com/app/228280/) | none | 11,105 |
| Baldur's Gate II: Enhanced Edition | 257350 | crpg/rtwp | RTwP | Store: "Real-Time-with-Pause Combat" | [store](https://store.steampowered.com/app/257350/) | none | 6,605 |
| Icewind Dale: Enhanced Edition | 321800 | crpg/rtwp | RTwP | Store: "Real-Time-with-Pause Combat" | [store](https://store.steampowered.com/app/321800/) | none | 2,322 |
| Tyranny | 362960 | crpg/rtwp | RTwP | Store: "Tactical real-time-with-pause combat" | [store](https://store.steampowered.com/app/362960/) | none | 8,342 |
| Black Geyser: Couriers of Darkness | 1374930 | crpg/rtwp | RTwP | Store: "real time tactical combat with pause" | [store](https://store.steampowered.com/app/1374930/) | none | 688 |
| Divinity: Original Sin 2 - Definitive Edition | 435150 | crpg/turn_based | TB | Store: "deep, tactical, turn-based combat" | [store](https://store.steampowered.com/app/435150/) | none | 95,477 |
| Baldur's Gate 3 | 1086940 | crpg/turn_based | TB | Larian, Community Update #16 (2022-07-07): "Faster Turn-Based Combat" | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4474904295781453356) | none | 506,481 |
| Solasta: Crown of the Magister | 1096530 | crpg/turn_based | TB | Store: "squad-level, turn-based, tactical combat" | [store](https://store.steampowered.com/app/1096530/) | none | 15,580 |
| Warhammer 40,000: Rogue Trader | 2186680 | crpg/turn_based | TB | Store: "turn-based tactical combat" | [store](https://store.steampowered.com/app/2186680/) | none | 27,296 |
| Wasteland 3 | 719040 | crpg/turn_based | TB | Store: "tactical turn-based combat" | [store](https://store.steampowered.com/app/719040/) | none | 14,393 |
| Europa Universalis IV | 236850 | strategy/rtwp | RTwP, indirect | Hotfix 1.37.4 (2024-09-18) fixes a window "while the game is unpaused" that closed "when a day passed" | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/6341720536498264017) | none | 68,954 |
| Crusader Kings III | 1158310 | strategy/rtwp | RTwP, indirect | Dev Diary #147 (2024-04-30): an option to "automatically pause the game" | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/5759616966682601842) | none | 68,658 |
| Hearts of Iron IV | 394360 | strategy/rtwp | RTwP, indirect | Developer Diary, Tech (2022-09-14): playing "unpaused in Speed 4-5" | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4705721386244531546) | none | 163,878 |
| Victoria 3 | 529340 | strategy/rtwp | RTwP, indirect | Dev Diary #51 (2022-06-23): the tutorial shows "how to unpause the game" | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4474904295730225610) | none | 28,205 |
| Stellaris | 281990 | strategy/rtwp | RTwP, indirect | Dev Diary #270 (2022-10-06): choices made "while the game is paused" | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4613399405939390156) | none | 105,727 |
| Civilization VI | 289070 | strategy/turn_based | TB | Announcement (2016-05-11): "turn-based strategy franchise" | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/255955041673117655) | none | 150,876 |
| Old World | 597180 | strategy/turn_based | TB | Store: "Instead of moving every unit once per turn", units spend Orders | [store](https://store.steampowered.com/app/597180/) | none | 4,533 |
| Humankind | 1124300 | strategy/turn_based | TB | Store: each battle "plays out like a mini turn-based board game". Amplitude post (2021-11-18): "turn-based 4X strategy game" | [store](https://store.steampowered.com/app/1124300/), [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4227186657141223271) | none | 15,316 |
| Age of Wonders 4 | 1669000 | strategy/turn_based | TB | Store: "4X strategy and turn-based tactical combat" | [store](https://store.steampowered.com/app/1669000/) | none | 14,026 |
| Endless Legend | 289130 | strategy/turn_based | TB, flagged | Store: "4X turn-based fantasy strategy game" with a "simultaneous turn-based battle system" | [store](https://store.steampowered.com/app/289130/) | none | 11,832 |
| Pillars of Eternity | 291650 | dual_mode | both | Store: "either real-time with pause or turn based mode" | [store](https://store.steampowered.com/app/291650/) | 2026-04-06 | 16,185 |
| Pathfinder: Kingmaker | 640820 | dual_mode | both | Hotfix 1.0.6 (2018-10-05) names the "tactical pause". The turn-based mode came later (next table) | [post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/2447035533298507792) | 2020-08-18 | 22,341 |
| Pillars of Eternity II: Deadfire | 560130 | dual_mode | both | Store: "Patch 5.0 - Turn-based mode". Patch 4.1.0 added it (next table) | [store](https://store.steampowered.com/app/560130/) | 2019-01-24 | 11,362 |
| Pathfinder: Wrath of the Righteous | 1184370 | dual_mode | both | Store: "two combat modes", "real-time with pause or turn-based" | [store](https://store.steampowered.com/app/1184370/) | at launch | 24,279 |
| Combat Mission: Shock Force 2 | 1369370 | dual_mode | both | Store: "either Real Time or our WeGo hybrid turn based mode" | [store](https://store.steampowered.com/app/1369370/) | at launch | 667 |
| Combat Mission: Battle for Normandy | 2297010 | dual_mode | both | Store: "hybrid system for RealTime and WeGo (turn based) play" | [store](https://store.steampowered.com/app/2297010/) | at launch | 215 |

The five Paradox rows are marked indirect. Their store descriptions contain no "real-time" or "pause" wording. The cited posts show a continuous game clock that the player pauses and unpauses, which is what `rtwp` means for these titles. Store text was normalised to plain hyphens.

Endless Legend is flagged: its store page describes a "simultaneous turn-based battle system", so its battles resolve both sides' orders at once, closer to WEGO than to alternating turns. It stays in `turn_based` because it has no real-time clock. Old World is classified from the store's description of its turn and orders system. Combat Mission's real-time mode is described by the developer as real time, not as real time with pause. For those two games the RTwP column of the dual-mode views counts mentions of real-time play, which `mode_rtwp` also matches ("real-time mode", "real-time combat").

### Mode-addition dates

A mode event is the first availability of the turn-based mode on a game's default Steam branch. Opt-in beta branches do not count. Each event is stamped with the publication time (UTC) of the developer post that announced it, as `date` in the study's `[[mode_events]]`, so a review posted earlier on the same day counts as before.

The three dates were checked again on October 15, 2026 against the developer posts returned by `ISteamNews/GetNewsForApp`. Each kept date is the post that put the mode on the default branch for every player, and each opt-in beta post is dated separately below. Pillars of Eternity has no post between the beta of 2025-11-07 and the release post of 2026-04-06 17:32:35 UTC, whose patch notes say the mode "has been added". No date changed, so the before and after view was not recomputed for a date change.

| Game | Event date | What the developer post says | Source |
|---|---|---|---|
| Pillars of Eternity | 2026-04-06 | Turn-based mode live for everyone. An opt-in public beta branch opened 2025-11-07 | [release post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1828894815565733), [beta post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1815580768309076) |
| Pathfinder: Kingmaker | 2020-08-18 | Definitive Edition update, "Turn-based mode is now available". An opt-in beta opened 2020-07-20 | [release post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4828375451936483210), [beta post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4738301002615223217) |
| Pillars of Eternity II: Deadfire | 2019-01-24 | Patch 4.1.0 ships turn-based mode to all editions, labelled beta. Patch 5.0 took it out of beta on 2019-05-08 | [patch 4.1.0](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/2425653583383022807), [patch 5.0](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/2405396896205726743) |
| Pathfinder: Wrath of the Righteous | both at launch | Turn-based mode is in the beta patch notes of 2021-02-14, before the 2021-09-02 release | [beta notes](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/4703417046211672597) |
| Combat Mission: Shock Force 2 | both at launch | Launch post: orders "either in real time or with the innovative turn-based WeGo system" | [launch post](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/3788045105538455740) |
| Combat Mission: Battle for Normandy | both at launch | Store page: hybrid system for RealTime and WeGo play | [store page](https://store.steampowered.com/app/2297010/) |

## 3. Sampling

- Natural sampling as in the market study: `filter = "recent"`, English, every review type, every purchase type, off-topic review bombs excluded, newest first.
- Raw cap 10,000 per game. The six dual-mode titles carry `raw_cap = 20000` in the study file and are pulled in full when their English total is smaller, so reviews from before and after a mode event are both present.
- Analysis sample: the development half of the full database. The study file sets `stats_split = "dev"` and `stats_source = "full"`. The pipeline's split stage copies every development-half review of `rtwp_turn_2026_full.db` into `rtwp_turn_2026_full_dev.db`, choosing rows from review IDs alone, and every later stage (cleaning, VADER, topics, aspects, statistics) runs on that database. No test-half row, text or flag is copied or read. `tests/test_pipeline.py` poisons the test half's recommend flags and text and checks that none reaches the analysis database.
- The seed (a uniform 500 reviews per game, random seed 42) is still built. It was the analysis sample of the first pass, which left both CRPG subgroups with 2 eligible games, under the 3-game floor.
- The pull is fresh. The pipeline keeps one raw directory per study and refuses to write a frozen study's files, so reusing the market study's cached pulls would mean copying files outside the pipeline. The market pulls also used a 20,000 cap and are three weeks older.
- Local only: the databases, raw cache and pull log live under `data/rtwp_turn_2026/`, which Git ignores. They are outside the `data/*_seed.db` glob that the committed size budget sums.

**Counts.** The pull on October 15, 2026 collected 240,711 English reviews from 26 games with no rate-limit stop. 16 games stopped at their cap. Humankind stopped at 9,800 of its 15,316 English reviews because Steam's cursor ran out, so its oldest 5,516 are missing, including the launch weeks. Pathfinder: Kingmaker and Wrath of the Righteous hit the 20,000 dual-mode cap, so their oldest 2,341 and 4,279 English reviews are missing. The loader stored 239,850 reviews in the full database (the 861 others have empty text). 119,766 fall in the development half and 120,084 in the test half. 89,953 development-half reviews pass the cleaning, length, token and language rules.

| Game | Group | Pulled | Stop | Oldest review pulled | Dev reviews | Dev kept | Dev not recommended | Eligible on dev |
|---|---|---|---|---|---|---|---|---|
| Baldur's Gate: Enhanced Edition | crpg/rtwp | 10,000 | cap | 2015-04-09 | 4,956 | 3,787 | 340 | yes |
| Baldur's Gate II: Enhanced Edition | crpg/rtwp | 6,605 | end of list | 2013-11-16 | 3,185 | 2,422 | 236 | yes |
| Icewind Dale: Enhanced Edition | crpg/rtwp | 2,320 | end of list | 2014-10-30 | 1,158 | 944 | 114 | yes |
| Tyranny | crpg/rtwp | 8,342 | end of list | 2016-11-10 | 4,135 | 3,653 | 566 | yes |
| Black Geyser: Couriers of Darkness | crpg/rtwp | 688 | end of list | 2021-08-26 | 367 | 347 | 99 | yes |
| Divinity: Original Sin 2 - Definitive Edition | crpg/turn_based | 10,000 | cap | 2024-08-12 | 4,915 | 3,133 | 326 | yes |
| Baldur's Gate 3 | crpg/turn_based | 10,000 | cap | 2026-06-21 | 5,043 | 2,974 | 225 | yes |
| Solasta: Crown of the Magister | crpg/turn_based | 10,000 | cap | 2021-09-20 | 4,931 | 4,228 | 697 | yes |
| Warhammer 40,000: Rogue Trader | crpg/turn_based | 10,000 | cap | 2025-05-22 | 4,977 | 3,853 | 523 | yes |
| Wasteland 3 | crpg/turn_based | 10,000 | cap | 2020-11-25 | 5,011 | 4,024 | 734 | yes |
| Europa Universalis IV | strategy/rtwp | 10,000 | cap | 2023-08-29 | 5,071 | 2,928 | 473 | yes |
| Crusader Kings III | strategy/rtwp | 10,000 | cap | 2025-05-18 | 4,887 | 3,038 | 638 | yes |
| Hearts of Iron IV | strategy/rtwp | 10,000 | cap | 2026-02-03 | 4,929 | 2,787 | 454 | yes |
| Victoria 3 | strategy/rtwp | 10,000 | cap | 2023-12-27 | 5,015 | 3,642 | 1,328 | yes |
| Stellaris | strategy/rtwp | 10,000 | cap | 2025-05-08 | 4,972 | 3,563 | 1,253 | yes |
| Civilization VI | strategy/turn_based | 10,000 | cap | 2025-06-26 | 4,951 | 2,684 | 490 | yes |
| Old World | strategy/turn_based | 4,533 | end of list | 2022-05-19 | 2,291 | 1,964 | 336 | yes |
| Humankind | strategy/turn_based | 9,800 | end of list | 2021-09-05 | 4,915 | 3,959 | 1,426 | yes |
| Age of Wonders 4 | strategy/turn_based | 10,000 | cap | 2023-05-14 | 5,011 | 3,920 | 914 | yes |
| Endless Legend | strategy/turn_based | 10,000 | cap | 2015-01-25 | 4,921 | 3,991 | 700 | yes |
| Pillars of Eternity | dual_mode | 16,182 | end of list | 2015-03-26 | 8,125 | 6,770 | 941 | yes |
| Pathfinder: Kingmaker | dual_mode | 20,000 | cap | 2018-10-07 | 9,887 | 8,374 | 2,195 | yes |
| Pillars of Eternity II: Deadfire | dual_mode | 11,359 | end of list | 2018-05-08 | 5,675 | 4,578 | 608 | yes |
| Pathfinder: Wrath of the Righteous | dual_mode | 20,000 | cap | 2021-09-19 | 9,954 | 7,983 | 1,724 | yes |
| Combat Mission: Shock Force 2 | dual_mode | 667 | end of list | 2020-08-31 | 366 | 311 | 66 | yes |
| Combat Mission: Battle for Normandy | dual_mode | 215 | end of list | 2023-03-28 | 118 | 96 | 24 | no |

"Oldest review pulled" is the pull's `effective_window_start` (in each game's `.pull.json`). Dev reviews, kept and not-recommended counts come from `game_stats_run_2.csv` of the analysis database.

**Processing time** (this machine, October 15, 2026, development half): split 2 s, cleaning and language filter 12 min 55 s, VADER 51 s, BERTopic fit 6 min 19 s, aspects 1 min 18 s (v1), 1 min 30 s (v2) and 1 min 25 s (v3), stored statistics 8 min 58 s (v1), 9 min 49 s (v2) and 7 min 56 s (v3), candidate run 8 min 14 s (v2) and 8 min 33 s (v3), dual-mode views 6 min 13 s, audit export 10 min 31 s (v2) and 7 min 1 s (v3), power 3 min 17 s (whole frame) and 3 min 9 s (2,000 cap). Peak memory was 5.1 GB. Transformer sentiment was not attempted and not timed. It is descriptive only and outside the finding rule, and a run of up to 90 minutes on 89,953 reviews, on top of the stages above, would have reached the 3-hour compute limit set for this rerun.

## 4. Eligibility

The market study's rule is reused unchanged: a game is eligible for theme estimates when it has at least 100 kept English reviews and at least 30 not-recommended kept English reviews (`stats.MIN_KEPT_FOR_ELIGIBLE`, `stats.MIN_NOT_REC_FOR_ELIGIBLE`), and a scope needs 3 eligible games before any finding. In phase A the rule is applied to the development half, because that is the sample the statistics use. Phase B applies the same rule to its own sample, the test half of the same full database.

**Result on the development half: 25 of 26 games are eligible.**

| Scope | Eligible games | Not-recommended kept reviews in eligible games |
|---|---|---|
| crpg/rtwp | 5 of 5 | 1,355 (99 to 566 per game) |
| crpg/turn_based | 5 of 5 | 2,505 (225 to 734) |
| strategy/rtwp | 5 of 5 | 4,146 (454 to 1,328) |
| strategy/turn_based | 5 of 5 | 3,866 (336 to 1,426) |
| crpg | 10 of 10 | 3,860 |
| strategy | 10 of 10 | 8,012 |
| rtwp (pooled) | 10 of 10 | 5,501 |
| turn_based (pooled) | 10 of 10 | 6,371 |
| dual_mode | 5 of 6 | 5,534 (66 to 2,195) |

Combat Mission: Battle for Normandy is the only ineligible game: its whole development half is 118 reviews, 96 kept and 24 not recommended. Black Geyser (99 not recommended) and Shock Force 2 (66) are the smallest eligible games. Every scope except `dual_mode` has all its games, so every CRPG comparison can now become a finding.

## 5. Lexicon

`studies/aspects_rtwp_turn_v3.toml`, version `rtwp_turn_v3`. The three control aspects (`bugs_performance`, `ui_clarity`, `price_dlc`) are copied unchanged from `strategy_v1`. The six design aspects are new:

| Aspect | What it tags |
|---|---|
| `mode_turn_based` | Explicit talk about turn-based combat or a turn-based mode |
| `mode_rtwp` | Explicit talk about real-time-with-pause or real-time combat |
| `pause_micro` | Pausing to give orders, auto-pause, micromanagement, babysitting the party |
| `realtime_chaos` | Chaotic, messy or unreadable fights |
| `waiting_turns` | Waiting through AI or other turns, slow turn processing |
| `combat_pacing` | Slow or long encounters, too many fights, trash fights |

`waiting_turns` deliberately does not reuse the market name `turn_waiting`: `src/aspects.py` derives a turn-structure theme from that name, which does not apply here.

**Development of v1.** The lexicon was written and revised on development-half sentences only, in two passes on October 15, 2026, then frozen as `rtwp_turn_v1` (`studies/aspects_rtwp_turn_v1.toml`, unchanged). Pass 1 read matched sentences from the seed's development half. Pass 2 read development-half sentences from the full database for the three rare aspects. The revisions that came out of reading those sentences:

- `pause_micro` lost a bare `micro` pattern, which matched microtransactions and small amounts. It keeps `micro-manage` and its forms.
- `realtime_chaos` excludes Chaos as a faction or god (Chaos Space Marines, forces of Chaos), the chaotic alignments and "order versus chaos". "Mess" counts only in a phrase such as "a mess" or "messy", next to a combat word.
- `waiting_turns` skips "turn-based" through a lookahead and excludes "can't wait", waiting for a sale, patch or sequel, and "wait for my turn on the computer".

**v2.** The larger development half was checked for gaps by reading kept development-half reviews with a combat-model cue and no design tag (text only). 1,479 reviews say "real time" and 1,523 say "pause". 212 and 323 of them carry no design tag. Reading samples showed one real gap: v1's `mode_rtwp` missed "real time pause" without "with", "pause and play", "real-time CRPG" or "real-time RPG", and "played in real time". Four patterns cover them. They add 101 development-half reviews to `mode_rtwp` (1,106 under v1, 1,207 under v2), and a read of 20 of the 101 found 15 about real-time or real-time-with-pause combat. Candidate `pause_micro` additions ("pause button", "pause to think") added 40 reviews of mixed relevance and were left out. Every other aspect is unchanged. v2 was frozen on October 15, 2026, before its statistics were computed.

**v3, the single audit retry.** A fresh model labeler instance audited the v2 set. `mode_rtwp`, `realtime_chaos` and `bugs_performance` passed at 0.80. `mode_turn_based`, `pause_micro`, `waiting_turns`, `combat_pacing`, `ui_clarity` and `price_dlc` failed (exact figures in the protocol, section 13). v3 rewrites only those six and was built blind to the v2 labels and to all test-half text, from seeded samples of v2-tagged development-half sentences judged against the unchanged guide definitions. `mode_turn_based` no longer tags "turn-based" as a genre label (turn-based strategy, RPG, 4X, games). `pause_micro` no longer reads the mode name "with pause" as pausing. `waiting_turns` needs waiting for the AI, enemies, other players or one's own turn, and excludes softlocks and hangs. `combat_pacing` drops "tedious" alone. `ui_clarity` drops the main, settings, options and difficulty menus and menu music. `price_dlc` needs a cost, value or content word with "DLC" and a buying verb with "on sale", and adds money amounts. The builder's own development-sample precision for the six aspects rises from 0.68 to 0.83 under v2 to 0.85 to 0.95 under v3 on held-out development sentences (on every v3-tagged development review for `waiting_turns`, which has too few for a hold-out). These are builder judgments, not an audit (protocol, section 13). v3 was frozen on October 15, 2026, before its statistics were computed.

Kept development-half reviews with at least one match:

| Aspect | v2 | v3 |
|---|---|---|
| `mode_turn_based` | 4,886 | 3,528 |
| `mode_rtwp` | 1,207 | 1,207 |
| `pause_micro` | 1,495 | 1,274 |
| `realtime_chaos` | 146 | 146 |
| `waiting_turns` | 80 | 65 |
| `combat_pacing` | 509 | 396 |
| `bugs_performance` | 10,852 | 10,852 |
| `ui_clarity` | 3,364 | 3,214 |
| `price_dlc` | 10,636 | 8,308 |

**Audit sets.** Two attempts, both from the test half of the full database, the half phase B reads, with text only. The v2 set has 360 sentences, 40 for each of the 9 aspects, from 359 distinct reviews (`outputs/audit/rtwp_turn_2026_aspect_audit_rtwp_turn_v2_unlabeled.csv`, guide `data/labels/aspect_labeling_guide_rtwp_turn_v2.md`), and is labeled. The v3 set has 240 sentences, 40 for each of the six revised aspects, from 240 distinct reviews, none of them in the v2 set (`outputs/audit/rtwp_turn_2026_aspect_audit_rtwp_turn_v3_unlabeled.csv`, guide `data/labels/aspect_labeling_guide_rtwp_turn_v3.md`, which keeps the v2 definitions). A second fresh model labeler instance labeled it, and all six aspects pass. Phase B excludes the reviews of both sets. Precision comes from the v3 labels for the revised aspects and from the v2 labels for the three unchanged ones. Both CSVs are ignored by Git. The first pass's seed-based set is kept as `rtwp_turn_2026_aspect_audit_rtwp_turn_v1_unlabeled_superseded_seed.csv` and is not to be labeled. The stored statistics of phase A (run 3) were computed before the v3 labels existed and read no precision, so every aspect row in phase A is descriptive.

## 6. Analyses

**Cohort analyses.** The pipeline's statistics with the market study's finding rule (`stats.FINDING_RULE`), its intervals (Wilson for game rates, a 2,000-resample two-stage cluster bootstrap for macro theme metrics, seed 42) and its leave-one-game-out check, on the development half of the full database. `stats_split = "dev"` restricts every statistic of the stats stage to that half and `stats_source = "full"` points it at the development database. Aspect findings need an audited precision of at least 0.80 on 30 or more test-half sentences, so until the audit is labeled every aspect row is descriptive and only topics can meet the rule. `stats.py --candidates` repeats the statistics on a copy with every aspect assumed to pass, which gives the candidates phase B tests.

**Dual-mode, within game.** For each dual-mode game, reviews are grouped by the mode aspects they carry (`mode_turn_based` only, `mode_rtwp` only, both, neither), and each group's recommend share is reported with a Wilson interval (`src/dual_mode.py`). This asks whether players who talk about the turn-based mode recommend the game more or less often than players who talk about the real-time mode.

**Dual-mode, before and after.** For the three games that added a turn-based mode after launch, reviews created before the mode event are compared with reviews created on or after it: recommend share, the share that mentions each mode, and each aspect's share among recommended and not-recommended reviews. This view is labelled descriptive because time is confounded with every other patch, sale, expansion and change of audience between the two periods.

Both dual-mode views run on the development half of the full database (`--source full`) with the seed's cleaning, length, token and language rules, sentencizer and lexicon `rtwp_turn_v3`. They were rerun with v3 because its `mode_turn_based` is narrower. The v2 views are kept as `dual_mode_full_dev_v2.json` and `.md`.

**Topics.** BERTopic is fitted on all 89,953 kept development-half reviews with the study's settings (`umap_preset = "bertopic_default"`, `min_topic_size = 40`), with no sampling. It found 196 topics and left 33,146 reviews as outliers. The fit reads review text only. The model is saved at `outputs/rtwp_turn_2026/bertopic_model_full_dev/model.pkl` (ignored by Git), SHA-256 `dd037032288fa2df3c26df517c1fe0ea1517ffd5f173c6a2a11fd3ce9999e9d0`. The first pass's seed model stays at `outputs/rtwp_turn_2026/bertopic_model/model.pkl` and is not used. Its topic numbers are a different model's. Phase B assigns every test-half review a topic with `BERTopic.transform` of the full-development model.

## 7. Exploratory results (development half)

All numbers in this section come from the development half of the full database, lexicon `rtwp_turn_v3`. The v2 statistics (stats run 2, candidates run 3) stay in the folder and are superseded for the six revised aspects (`SUPERSEDED_BY_V3.txt` there). Files are in `outputs/rtwp_turn_2026/exploratory/` (ignored by Git). Run numbers are those of the analysis database `rtwp_turn_2026_full_dev.db`.

**Stored statistics (stats run 3).** 26 games, 25 eligible, 1,764 lift comparisons. With no v3 labels yet, stats reads no precision, so only topics can meet the rule, and 27 topic rows do (the topic rows of the candidate table below). VADER on the development half: balanced accuracy 0.699 (0.695 to 0.702) against the recommend flag, 44.9% of not-recommended reviews scored positive.

**Candidates (run 4, `--candidates`).** The same statistics on a copy of the analysis database, with every aspect assumed to pass its audit: 1,845 lift comparisons, 66 candidates, 64 on the complaint side and 2 on the praise side. These are what phase B tests. Against v2's run 3, v3 drops `combat_pacing` in `dual_mode`, `strategy`, `crpg/rtwp` and `strategy/turn_based`, `mode_turn_based` in `strategy` and `pause_micro` in `crpg`, and adds price and DLC in `crpg/turn_based` and `strategy/turn_based`. Rows of unchanged themes are identical in both runs. Lifts are complaint lifts: the share of not-recommended reviews that cite the theme over the share of recommended reviews that do, macro-averaged over eligible games.

| Theme | Scope | Lift (95% interval) |
|---|---|---|
| `combat_pacing` | `crpg` | 4.61 (2.00 to 9.71) |
| `combat_pacing` | `turn_based` (pooled) | 4.43 (2.07 to 10.13) |
| `combat_pacing` | `crpg/turn_based` | 5.89 (2.79 to 14.75) |
| `pause_micro` | `dual_mode` | 1.90 (1.23 to 2.63) |
| `pause_micro` | `strategy` | 1.90 (1.16 to 2.97) |
| `pause_micro` | `rtwp` (pooled) | 1.60 (1.03 to 2.30) |
| `pause_micro` | `turn_based` (pooled) | 2.33 (1.21 to 4.18) |
| `pause_micro` | `crpg/turn_based` | 5.21 (2.08 to 15.90) |
| `pause_micro` | `strategy/rtwp` | 1.78 (1.29 to 2.81) |
| `mode_turn_based` (praise) | `turn_based` (pooled) | 0.63 (0.39 to 0.99) |
| `mode_turn_based` (praise) | `strategy/turn_based` | 0.47 (0.25 to 0.74) |
| `realtime_chaos` | `turn_based` (pooled) | 4.76 (1.78 to 13.67) |
| `realtime_chaos` | `crpg/turn_based` | 4.69 (1.23 to 17.92) |
| `waiting_turns` | `turn_based` (pooled) | 2.96 (1.26 to 6.42) |
| Bugs and performance | `crpg` | 2.87 (2.36 to 3.59) |
| Bugs and performance | `dual_mode` | 2.21 (1.68 to 3.14) |
| Bugs and performance | `strategy` | 3.89 (3.13 to 4.91) |
| Bugs and performance | `rtwp` (pooled) | 3.37 (2.56 to 4.56) |
| Bugs and performance | `turn_based` (pooled) | 3.17 (2.71 to 3.86) |
| Bugs and performance | `crpg/rtwp` | 2.90 (2.10 to 4.24) |
| Bugs and performance | `crpg/turn_based` | 2.84 (2.47 to 3.37) |
| Bugs and performance | `strategy/rtwp` | 4.24 (3.40 to 5.90) |
| Bugs and performance | `strategy/turn_based` | 3.62 (2.69 to 5.05) |
| UI clarity | `crpg` | 2.48 (1.69 to 3.83) |
| UI clarity | `dual_mode` | 2.37 (1.72 to 3.28) |
| UI clarity | `strategy` | 2.65 (2.02 to 3.67) |
| UI clarity | `rtwp` (pooled) | 2.53 (1.74 to 3.68) |
| UI clarity | `turn_based` (pooled) | 2.59 (1.93 to 3.83) |
| UI clarity | `crpg/rtwp` | 2.07 (1.28 to 3.24) |
| UI clarity | `crpg/turn_based` | 3.45 (2.10 to 7.11) |
| UI clarity | `strategy/rtwp` | 3.86 (2.88 to 5.40) |
| UI clarity | `strategy/turn_based` | 2.11 (1.48 to 2.95) |
| Price and DLC | `dual_mode` | 1.64 (1.23 to 2.11) |
| Price and DLC | `strategy` | 1.90 (1.52 to 2.28) |
| Price and DLC | `rtwp` (pooled) | 1.94 (1.33 to 2.47) |
| Price and DLC | `turn_based` (pooled) | 1.41 (1.13 to 1.65) |
| Price and DLC | `crpg/turn_based` | 1.49 (1.06 to 1.82) |
| Price and DLC | `strategy/rtwp` | 2.33 (2.00 to 2.72) |
| Price and DLC | `strategy/turn_based` | 1.35 (1.02 to 1.65) |
| Topic 1, DLC | `turn_based` (pooled) | 2.02 (1.27 to 3.27) |
| Topic 1, DLC | `strategy/rtwp` | 1.54 (1.20 to 1.92) |
| Topic 3, party and enemies | `crpg` | 3.05 (1.88 to 4.64) |
| Topic 3, party and enemies | `dual_mode` | 3.55 (2.62 to 4.44) |
| Topic 3, party and enemies | `crpg/rtwp` | 2.25 (1.09 to 4.44) |
| Topic 3, party and enemies | `crpg/turn_based` | 4.29 (3.24 to 5.55) |
| Topic 11, crashes | `crpg` | 24.41 (14.00 to 45.01) |
| Topic 11, crashes | `dual_mode` | 17.27 (7.26 to 52.95) |
| Topic 11, crashes | `strategy` | 10.33 (7.28 to 15.69) |
| Topic 11, crashes | `rtwp` (pooled) | 11.82 (7.28 to 21.11) |
| Topic 11, crashes | `turn_based` (pooled) | 14.90 (9.56 to 31.04) |
| Topic 11, crashes | `crpg/rtwp` | 23.46 (9.89 to 66.31) |
| Topic 11, crashes | `strategy/rtwp` | 9.03 (5.30 to 14.18) |
| Topic 11, crashes | `strategy/turn_based` | 12.23 (7.94 to 29.12) |
| Topic 15, AI and armies | `strategy` | 6.23 (4.63 to 8.99) |
| Topic 15, AI and armies | `rtwp` (pooled) | 8.58 (4.31 to 14.72) |
| Topic 15, AI and armies | `strategy/rtwp` | 9.50 (4.65 to 17.38) |
| Topic 15, AI and armies | `strategy/turn_based` | 5.30 (3.90 to 7.33) |
| Topic 19, multiplayer and desync | `crpg` | 11.31 (6.46 to 18.00) |
| Topic 19, multiplayer and desync | `strategy` | 9.08 (5.97 to 12.37) |
| Topic 19, multiplayer and desync | `rtwp` (pooled) | 12.10 (7.26 to 18.38) |
| Topic 19, multiplayer and desync | `turn_based` (pooled) | 8.42 (5.66 to 11.48) |
| Topic 19, multiplayer and desync | `crpg/rtwp` | 15.01 (7.77 to 34.55) |
| Topic 19, multiplayer and desync | `strategy/turn_based` | 9.02 (5.85 to 14.04) |
| Topic 38, Paradox DLC | `strategy` | 5.22 (3.49 to 7.83) |
| Topic 38, Paradox DLC | `rtwp` (pooled) | 5.15 (3.29 to 8.27) |
| Topic 38, Paradox DLC | `strategy/rtwp` | 5.18 (3.39 to 7.89) |

BERTopic labels the topics "Dlc / Dlcs / Base / Buy" (1), "Party / Just / Enemies / Level" (3), "Crashes / Crash / Crashing / Crashed" (11), "Ai / Units / War / Army" (15), "Multiplayer / Desync / Desyncs / Unplayable" (19) and "Paradox / Dlc / Dlcs / Base" (38). Several candidates overlap: topic 11 with bugs and performance, topics 1 and 38 with price and DLC, and every pooled subgroup scope with its cohort scopes. The run also has 3 ranking findings: bugs and performance is the most-cited complaint aspect in `crpg` (24.8% of not-recommended reviews, 18.0% to 32.8%), in `crpg/rtwp` (25.8%, 19.0% to 34.8%) and in pooled `turn_based` (23.3%, 17.2% to 31.1%). The third is new under v3. Rankings compare aspects with each other, so narrowing the revised aspects can change one although bugs and performance itself is unchanged.

**Combat-model signals.** The combat-model rows of the candidate run in the four `cohort_subgroup` scopes and `dual_mode` with at least 10 tagged reviews, with the macro complaint share (among not-recommended reviews), the macro praise share (among recommended reviews) and the complaint lift. "Tagged" counts reviews in the scope's eligible games. Leave-one-game-out stability is assessed with every aspect assumed to pass. The cohort and pooled rows are in `candidates_theme_stats_run_4.csv`, and those that are candidates are in the table above.

| Aspect | Scope | Tagged | Complaint share | Praise share | Lift (95% interval) | Stable | Candidate |
|---|---|---|---|---|---|---|---|
| `mode_turn_based` | crpg/rtwp | 139 | 1.8% | 1.6% | 1.14 (0.53 to 2.16) | yes | no |
| `mode_turn_based` | crpg/turn_based | 834 | 3.3% | 4.8% | 0.69 (0.39 to 1.23) | yes | no |
| `mode_turn_based` | strategy/rtwp | 11 | 0.2% | 0.0% | 4.43 (0.00 to 39.11) | yes | no |
| `mode_turn_based` | strategy/turn_based | 303 | 0.9% | 1.9% | 0.47 (0.25 to 0.74) | yes | praise |
| `mode_turn_based` | dual_mode | 2,230 | 5.7% | 9.2% | 0.62 (0.27 to 1.02) | yes | no |
| `mode_rtwp` | crpg/rtwp | 224 | 2.8% | 3.1% | 0.90 (0.42 to 2.09) | no | no |
| `mode_rtwp` | crpg/turn_based | 25 | 0.2% | 0.1% | 1.13 (0.00 to 3.66) | no | no |
| `mode_rtwp` | dual_mode | 947 | 2.9% | 3.4% | 0.84 (0.59 to 1.14) | yes | no |
| `pause_micro` | crpg/rtwp | 218 | 3.0% | 2.0% | 1.51 (0.72 to 2.53) | yes | no |
| `pause_micro` | crpg/turn_based | 38 | 0.7% | 0.1% | 5.21 (2.08 to 15.90) | yes | complaint |
| `pause_micro` | strategy/rtwp | 184 | 1.6% | 0.9% | 1.78 (1.29 to 2.81) | yes | complaint |
| `pause_micro` | strategy/turn_based | 199 | 2.2% | 1.1% | 2.00 (0.77 to 3.68) | yes | no |
| `pause_micro` | dual_mode | 631 | 4.1% | 2.2% | 1.90 (1.23 to 2.63) | yes | complaint |
| `realtime_chaos` | crpg/rtwp | 26 | 0.5% | 0.3% | 1.61 (0.00 to 5.65) | yes | no |
| `realtime_chaos` | crpg/turn_based | 18 | 0.3% | 0.1% | 4.69 (1.23 to 17.92) | yes | complaint |
| `realtime_chaos` | dual_mode | 91 | 0.6% | 0.5% | 1.27 (0.14 to 5.07) | no | no |
| `waiting_turns` | crpg/turn_based | 20 | 0.3% | 0.1% | 5.74 (0.59 to 23.91) | yes | no |
| `waiting_turns` | strategy/turn_based | 40 | 0.5% | 0.2% | 2.21 (0.51 to 4.30) | yes | no |
| `combat_pacing` | crpg/rtwp | 35 | 0.5% | 0.2% | 2.39 (0.00 to 6.29) | no | no |
| `combat_pacing` | crpg/turn_based | 102 | 2.1% | 0.4% | 5.89 (2.79 to 14.75) | yes | complaint |
| `combat_pacing` | strategy/turn_based | 50 | 0.4% | 0.2% | 2.01 (0.89 to 10.49) | yes | no |
| `combat_pacing` | dual_mode | 204 | 1.4% | 0.5% | 2.62 (0.83 to 4.84) | yes | no |

Left out for fewer than 10 tagged reviews: `waiting_turns` in both RTwP subgroups (one tagged in all), `mode_rtwp` and `realtime_chaos` in both strategy subgroups, and `combat_pacing` in `strategy/rtwp`. What the development half suggests, before any test:

- Complaints about slow or too many fights (`combat_pacing`) go with not recommending in turn-based CRPGs (5.89, a candidate). In RTwP CRPGs the lift is 2.39 on 35 tagged reviews with an interval reaching 0 and no candidate.
- Pausing and micromanagement (`pause_micro`) is a complaint marker in most scopes, not a mark of RTwP games. Among RTwP CRPGs it is 1.51 and not a candidate. In turn-based CRPGs it is 5.21 on 38 tagged reviews.
- Turn-based talk leans to the praise side in turn-based strategy games (0.47) and, without a candidate, in turn-based CRPGs (0.69) and the dual-mode games (0.62).
- RTwP talk is neutral wherever it is common (lifts 0.84 to 0.91).
- Messy fights (`realtime_chaos`) and waiting through turns (`waiting_turns`) are complaint candidates only in turn-based scopes and rest on 18 to 60 tagged reviews.

**Dual-mode views (full database, development half).** `src/dual_mode.py --source full` read 34,125 development-half reviews of the six games and kept 28,112. Recommend share by the mode a review mentions, with Wilson intervals:

| Game | Turn-based only | RTwP only | Both | Neither |
|---|---|---|---|---|
| Pillars of Eternity | 80.1% (73.6 to 85.3), n=176 | 85.5% (79.5 to 90.0), n=173 | 90.0% (80.8 to 95.1), n=70 | 86.2% (85.4 to 87.1), n=6,351 |
| Pathfinder: Kingmaker | 80.0% (76.5 to 83.1), n=560 | 84.1% (73.2 to 91.1), n=63 | 76.2% (69.2 to 82.1), n=164 | 73.2% (72.2 to 74.2), n=7,587 |
| Pillars of Eternity II: Deadfire | 92.1% (89.2 to 94.3), n=444 | 89.6% (80.0 to 94.8), n=67 | 89.1% (83.9 to 92.8), n=193 | 85.9% (84.8 to 87.0), n=3,874 |
| Pathfinder: Wrath of the Righteous | 76.8% (72.6 to 80.6), n=423 | 76.3% (60.8 to 87.0), n=38 | 79.2% (72.5 to 84.6), n=173 | 78.5% (77.5 to 79.4), n=7,349 |
| Combat Mission: Shock Force 2 | 100.0% (86.2 to 100.0), n=24 | 2 of 3 | 3 of 3 | 76.9% (71.6 to 81.4), n=281 |
| Combat Mission: Battle for Normandy | 100.0% (74.1 to 100.0), n=11 | none | none | 71.8% (61.4 to 80.2), n=85 |

Before and after the turn-based mode reached the default branch (descriptive, section 8):

| Game | Period | Reviews | Recommend | Mention turn-based | Mention RTwP |
|---|---|---|---|---|---|
| Pillars of Eternity | before 2026-04-06 | 6,588 | 86.2% (85.4 to 87.0) | 2.8% | 3.4% |
| Pillars of Eternity | after | 182 | 81.9% (75.6 to 86.8) | 33.5% | 11.5% |
| Pathfinder: Kingmaker | before 2020-08-18 | 3,827 | 78.3% (77.0 to 79.6) | 7.2% | 2.6% |
| Pathfinder: Kingmaker | after | 4,547 | 70.0% (68.7 to 71.3) | 9.9% | 2.8% |
| Pillars of Eternity II: Deadfire | before 2019-01-24 | 1,010 | 83.3% (80.8 to 85.4) | 2.8% | 1.9% |
| Pillars of Eternity II: Deadfire | after | 3,568 | 87.7% (86.6 to 88.7) | 17.1% | 6.8% |

Turn-based talk rises after each event, most in Pillars of Eternity, where a third of the 182 later reviews mention the mode. Kingmaker's recommend share fell after its event and Deadfire's rose, with intervals that do not overlap in either game. Time confounds both, and 7.2% of Kingmaker's reviews mentioned turn-based play before the mode existed. The narrower v3 `mode_turn_based` moves reviews out of the turn-based-only and both groups. In the four CRPGs the turn-based-only groups shrink by 6% to 12% against v2 (Pillars of Eternity 199 to 176, Kingmaker 606 to 560, Deadfire 471 to 444, Wrath of the Righteous 466 to 423) and their recommend shares move by at most 1.8 points. The v2 views are kept as `dual_mode_full_dev_v2.json` and `.md`, and the v1 versions in `superseded_seed_dev/`.

## 8. Confounds stated in advance

- **Release era.** Four of the five RTwP CRPGs are 2013 to 2016 Steam releases and three of them are enhanced editions of older games. Every TB CRPG is a 2017 to 2023 release. Any RTwP versus TB difference among CRPGs is also an old versus new difference.
- **Franchise and studio.** Three RTwP CRPGs are Beamdog enhanced editions of Infinity Engine games. Every RTwP strategy game is a Paradox Development Studio title. Humankind and Endless Legend are both Amplitude games. Divinity: Original Sin 2 and Baldur's Gate 3 are both Larian games. A subgroup effect can be a studio effect.
- **Review volume.** The 10,000 most recent reviews of Baldur's Gate 3 go back only to June 21, 2026 and those of Hearts of Iron IV to February 3, 2026, while games pulled to the end of their list cover their whole Steam life. A capped game's reviews describe its current state, an uncapped game's its whole history. The window is recorded per game (`effective_window_start` in the pull records, and the counts table above).
- **Price.** List prices run from $19.99 (the enhanced editions, Deadfire, Kingmaker, Wrath of the Righteous) to $59.99 (Baldur's Gate 3, Civilization VI, Combat Mission). The store API lists 17 to 33 DLC entries for each RTwP strategy title against 7 to 19 for each TB strategy title, which can raise `price_dlc` talk for reasons unrelated to combat.
- **Genre inside a cohort.** The strategy cohort mixes grand strategy (RTwP) with 4X (TB), so the strategy comparison is also a grand strategy versus 4X comparison.
- **Turn structure inside a subgroup.** Endless Legend's simultaneous battles resolve both sides' orders at once and sit close to WEGO, so the turn-based strategy subgroup holds one game whose battles do not alternate turns (section 2).
- **Dual-mode before and after.** Time confounds everything that changed between the periods, including the patches and editions that shipped with the turn-based mode (Kingmaker's Definitive Edition, Deadfire's patch 4.1.0). Kingmaker's oldest pulled review is from October 7, 2018, twelve days after release, because the 20,000 cap cut its oldest 2,341 English reviews. Those missing reviews fall in its launch weeks, inside its before period, so that period misses the launch weeks. Pillars of Eternity's opt-in beta of the turn-based mode ran from 2025-11-07, inside its before period, and the mode was announced in March 2025, so some before-period reviews were written by players who knew of it or had tried it. Pillars of Eternity has only six months of reviews after its event.
- **Self-selection.** Players choose games partly by combat model, so reviews describe what the people who bought each game think, not what one player would think of both.

## 9. Plan for phase B

Phase B is pre-registered in `docs/rtwp_turn_protocol.md`, committed and tagged `rtwp-turn-protocol-v1` before the read. In short: test the 66 phase A candidates, with the same instruments and rule, on every test-half review in the full database except the 359 reviews of the v2 audit set and the 240 of the v3 set, 119,485 in all, pinned as a list of review IDs. Precision comes from the v3 labels for the six revised aspects and the v2 labels for the three unchanged ones, and all nine pass. Development-half reviews, including every review read in phase A, never enter phase B. Power on the development half: 60.0 of 66 expected to replicate if the development effects are the truth, 46.7 if each game's log lift is halved, pass line 43, and 9.8 and 2.5 of the 14 combat-model candidates, whose own count is reported with no verdict. `src/rtwp_read.py` runs the read once, behind a gate on the tag, a clean tree, an absent results file and the pinned files' hashes.

## 10. Commands

```bash
.venv/bin/python src/data_pull.py --study studies/rtwp_turn_2026.toml
.venv/bin/python src/pipeline.py --study studies/rtwp_turn_2026.toml --full
O=outputs/rtwp_turn_2026/exploratory
.venv/bin/python src/stats.py --study studies/rtwp_turn_2026.toml --export $O
.venv/bin/python src/stats.py --study studies/rtwp_turn_2026.toml --candidates $O
.venv/bin/python src/dual_mode.py --study studies/rtwp_turn_2026.toml --source full --split dev --out $O
.venv/bin/python src/aspects.py --study studies/rtwp_turn_2026.toml --export-audit
.venv/bin/python src/rtwp_power.py --cap 0 --testable
.venv/bin/python src/rtwp_power.py --cap 2000 --testable
.venv/bin/python src/rtwp_read.py frame
.venv/bin/python src/rtwp_read.py agreement
# Phase B, once, at a clean checkout of the tag rtwp-turn-protocol-v1:
.venv/bin/python src/rtwp_read.py read
```

With `stats_source = "full"` the pipeline runs a split stage after the seed and every later stage on `rtwp_turn_2026_full_dev.db`. The dashboard reads the seed's two halves, so it is not used in phase A.
