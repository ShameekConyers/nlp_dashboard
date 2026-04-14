# What Are Customers Actually Saying? NLP-Powered Review Analysis

An NLP pipeline that transforms unstructured Steam game reviews into structured insights using sentiment analysis and topic modeling, visualized through an interactive Streamlit dashboard.

---

## Business Question

Players leave thousands of free-text reviews on Steam, but the platform only surfaces a binary thumbs-up/down. Can an NLP pipeline extract more from that text — reliable sentiment scores, recurring themes, genre-level patterns — or does the noise in short, casual reviews make automated analysis unreliable? This project finds out using 2,382 reviews across 12 games and validates every result against Steam's own recommendation flag as ground truth.

---

## Dataset

All data comes from the [Steam Reviews API](https://store.steampowered.com/appreviews/) (public, zero authentication).

- **Volume:** 2,382 reviews across 12 games in 4 genres (FPS, Indie, RPG, Strategy)
- **Date range:** March 2023 to April 2026
- **Ground truth:** Steam recommendation flag (thumbs up/down) for sentiment validation
- **Seed database:** 2.3 MB, committed to Git, works out of the box with zero API calls

**Games included:** Baldur's Gate 3, Counter-Strike 2, Civilization VI, DOOM Eternal, Elden Ring, Hades, Hollow Knight, Stardew Valley, Stellaris, Team Fortress 2, The Witcher 3, Total War: WARHAMMER III (200 reviews each, 182 for Stardew Valley).

---

## Key Findings

1. **Nearly half the reviews were unusable.** 1,181 of 2,382 reviews had fewer than 5 tokens after preprocessing — emoji spam, single-word jokes, ASCII art. The sentinel filter caught all of these before they could pollute downstream analysis.
2. **VADER agrees with the Steam recommendation flag 62.8% of the time.** That sounds low, but the mismatch is informative: players routinely recommend a game while writing paragraphs of complaints about specific features, or pan a game they clearly enjoy. The recommendation flag is a noisy ground truth.
3. **Sentiment skews positive overall** (average compound score of 0.135), consistent with Steam's self-selection bias — engaged players who bother to write reviews tend to like what they're playing.
4. **Topic modeling surfaced 11 themes, but several are non-English clusters** (Polish, Russian, German, Turkish). The pipeline has no language filter, so BERTopic grouped foreign-language reviews by shared vocabulary rather than shared meaning. Language detection is the most obvious preprocessing improvement.
5. **The high-signal topics that do emerge are genre-flavored** — gameplay mechanics dominate FPS reviews, narrative and world-building surface in RPG clusters, and "cozy" vocabulary anchors the Indie topics.

---

## Approach

### Text Preprocessing

- HTML, URL, and email removal; Unicode encoding normalization
- spaCy tokenization and lemmatization (`en_core_web_sm`)
- Domain-specific stopword filtering
- Sentinel exclusion: reviews under 5 tokens flagged and excluded from NLP analysis (1,181 of 2,382 filtered out, leaving 1,201 for analysis)

### Sentiment Analysis

- VADER (`nltk.sentiment`) compound scoring on preprocessed text
- Distribution across 1,201 kept reviews: 55.3% positive, 36.4% negative, 8.3% neutral
- Validated against Steam recommendation flag as ground truth (62.8% agreement rate)

The 62.8% tells you VADER captures the general direction of sentiment but breaks down where text tone and the binary recommendation diverge. A player who writes "love this game but the netcode is garbage and matchmaking takes forever" gets a mixed compound score, even though they hit thumbs-up. VADER is a blunt instrument on this kind of text.

### Topic Modeling

- BERTopic with sentence-transformer embeddings and UMAP dimensionality reduction
- 11 topics extracted from the review corpus (excluding the outlier topic)
- Per-review topic assignments stored in SQLite for dashboard querying

Topic quality is mixed. The English-language topics are interpretable and track real discussion themes (gameplay mechanics, story/narrative, value/pricing). The non-English clusters are noise that a language filter would clean up. With 1,201 reviews across 12 games, BERTopic is working with a small corpus — more data per game would sharpen the topic boundaries.

---

## Dashboard

The Streamlit dashboard provides four interactive views with SQL-backed filtering. All filters push down to SQLite queries rather than in-memory DataFrame filtering.

- **Overview** — KPI cards (total reviews, average sentiment, top genre) and high-level distribution charts
- **Sentiment Analysis** — Sentiment score distributions, ground truth validation against the recommendation flag, per-game sentiment comparison
- **Topic Analysis** — Topic distribution, word clouds, topic-sentiment cross-tabulation
- **Review Explorer** — Filterable, paginated table of individual reviews with sentiment scores and topic assignments

Filters include genre, game, sentiment range, and topic.

---

## Tools

| Layer | Tool |
|-------|------|
| Runtime | Python 3.11+ |
| Database | SQLite 3 |
| Data ingestion | `steamreviews` |
| Text preprocessing | spaCy (`en_core_web_sm`) |
| Sentiment analysis | NLTK (VADER) |
| Topic modeling | BERTopic, sentence-transformers, UMAP |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |

---

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd nlp_dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run dashboard (uses included seed database — no API calls needed)
streamlit run dashboard/app.py
```

To run the full pipeline from the Steam API:

```bash
python src/data_pull.py
python src/db_setup.py --full
python src/preprocessing.py
python src/sentiment.py
python src/topics.py
streamlit run dashboard/app.py
```

---

## Project Structure

```
nlp_dashboard/
├── src/
│   ├── utils.py                # Shared constants, paths, logger
│   ├── data_pull.py            # Steam API ingestion
│   ├── db_setup.py             # Schema creation + seed/full loading
│   ├── preprocessing.py        # Text cleaning pipeline (spaCy)
│   ├── sentiment.py            # VADER scoring + ground truth validation
│   └── topics.py               # BERTopic topic modeling
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   └── queries.py              # SQL query functions for dashboard views
├── data/
│   └── seed.db                 # Included — works on clone (<25 MB)
├── tests/                      # 90 tests covering schema, dedup, NLP, dashboard
└── requirements.txt
```

---

## Limitations and Next Steps

### Limitations

- **No language filter.** The pipeline processes all reviews regardless of language. BERTopic groups non-English reviews into clusters that look like topics but carry no analytical meaning. This inflates the topic count and dilutes English-language topic quality.
- **Small corpus per game.** With ~200 reviews per game (1,201 total after sentinel filtering), BERTopic is working near the lower bound for stable topic extraction. Topic boundaries are soft and some themes likely merge or split depending on the random seed.
- **VADER is lexicon-based.** It scores words in isolation without understanding context, sarcasm, or domain-specific slang. Gaming jargon ("this game slaps," "absolute banger") and mixed-sentiment paragraphs degrade accuracy. The 62.8% agreement rate reflects this ceiling.

### Next Steps

- **Language detection** — filter non-English reviews in preprocessing so BERTopic only clusters reviews it can meaningfully compare
- **Manual topic relabeling** — replace auto-generated top-word labels with human-readable names that a stakeholder can scan
- **Transformer-based sentiment** — swap VADER for a fine-tuned DistilBERT to handle sarcasm, mixed opinions, and gaming jargon that a lexicon misses
- **More data per game** — pulling 1,000+ reviews per game would sharpen topic boundaries and reduce outlier noise
- **Temporal analysis** — track sentiment trends over time to catch the impact of patches, DLC releases, or controversies
