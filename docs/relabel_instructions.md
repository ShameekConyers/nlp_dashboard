# Blind relabel: instructions for the owner

Two files to label, about 2.5 hours in total. Your labels are compared with the model labeler's labels and used to recompute the aspect precision audit and the sentence sentiment gate (protocol section 9, `docs/replication_protocol.md`).

## Before you start

Stay blind to the earlier labels until both files are finished. Do not open:

- `outputs/relabel_2026_key/` (the key),
- `data/labels/aspect_audit_strategy_v1.csv` and `data/labels/sentence_sentiment_audit_v3.csv` (the committed model labels),
- `outputs/audit/` (the labeled audit copies),
- the dashboard's Design Themes view, which shows sentence sentiment and example sentences.

## Files

1. `outputs/relabel_2026/aspect_relevance_blind.csv`: 160 rows. Columns `row_id`, `aspect`, `aspect_label`, `sentence`, `label`.
2. `outputs/relabel_2026/sentence_sentiment_blind.csv`: 200 rows. Columns `row_id`, `sentence`, `label`.

Fill only the `label` column of every row. Leave the other columns and the row order as they are, and save in place as CSV (UTF-8). Rows are shuffled. Label each row on its own.

## File 1: aspect relevance

Write `relevant` or `not_relevant`. These rules are copied from `data/labels/aspect_labeling_guide.md`.

Judge only the `sentence` against the `aspect` on that row. Praise and complaint both count as relevant. Do not look up the game. Label `relevant` when the sentence talks about the aspect as defined below, in the game being reviewed or in a direct comparison with another game. Label `not_relevant` when the matched words mean something else, when the aspect is only a passing word with no content (for example a list of tags), or when the sentence is about something outside the game (a sale, a studio, real-world events). When a reasonable reader could go either way, choose `not_relevant`.

| Aspect | Relevant when the sentence is about | Not relevant examples |
|---|---|---|
| `pacing` | Game pacing: slow or tedious play, grind, repetition, late-game drag, snowballing, micromanagement load, turn or mission timers | "Slow" loading screens or performance (that is bugs_performance) |
| `bugs_performance` | Bugs, crashes, glitches, frame rate, stutter, loading times, optimization | "Bugs" as in-game insect enemies |
| `ui_clarity` | User interface, menus, tooltips, HUD, and how clear the interface makes information | "Interface" of a real-world device unrelated to the game |
| `price_dlc` | Price, value for money, sales and discounts as they bear on value, DLC and its cost or content, microtransactions | A sale mentioned only as the reason for buying, with no view on price or value |

## File 2: sentence sentiment

Write `positive`, `negative` or `neither`. These rules are copied from `data/labels/sentence_sentiment_labeling_guide.md`.

Read only the `sentence`. Decide what attitude the writer expresses in that sentence toward the game or the part of the game it discusses. Do not look up the game or the rest of the review.

- `positive`: the sentence praises, recommends, or expresses enjoyment or approval. "The AI is actually smart and punishes mistakes" is positive.
- `negative`: the sentence complains, warns, or expresses frustration or disapproval. "Crashes every hour and the UI hides basic information" is negative.
- `neither`: the sentence is neutral, factual, a question, a plain description, clearly mixed with no side stronger, or impossible to read as either side. "The game has a hit chance display on every shot" is neither. "Great maps but the AI is brain dead" is mixed. Label it by the side that dominates, and use `neither` only when neither side dominates.

Sarcasm counts as what the writer means: "Oh great, another crash" is negative. Judge the sentence alone, even when you suspect the rest of the review says something else. When a reasonable reader could go either way between `positive` or `negative` and `neither`, choose `neither`.

## When both files are done

```bash
.venv/bin/python src/relabel.py score
```

It refuses to score while any row is blank or holds another value. Results go to `outputs/relabel_2026/relabel_scores.json`: kappa and AC1 with intervals, each aspect's precision on your labels against the 0.80 floor, and the sentence rule's macro-F1 on your labels against the 0.70 gate. After scoring you may open the key.
