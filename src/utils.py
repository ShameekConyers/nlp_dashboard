"""Shared constants, path resolution, and logging setup for the NLP dashboard project."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.study import Study

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
SEED_DB_PATH: Path = DATA_DIR / "seed.db"
FULL_DB_PATH: Path = DATA_DIR / "full.db"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Game catalog — {display_name: (app_id, genre)}
# 12 games across RPG, FPS, strategy, and indie with high review volume.
# ---------------------------------------------------------------------------

GAME_CATALOG: dict[str, tuple[int, str]] = {
    # RPG
    "The Witcher 3": (292030, "RPG"),
    "Elden Ring": (1245620, "RPG"),
    "Baldurs Gate 3": (1086940, "RPG"),
    # FPS
    "Counter-Strike 2": (730, "FPS"),
    "Team Fortress 2": (440, "FPS"),
    "DOOM Eternal": (782330, "FPS"),
    # Strategy
    "Civilization VI": (289070, "Strategy"),
    "Stellaris": (281990, "Strategy"),
    "Total War WARHAMMER III": (1142710, "Strategy"),
    # Indie
    "Stardew Valley": (413150, "Indie"),
    "Hades": (1145360, "Indie"),
    "Hollow Knight": (367520, "Indie"),
}

GAME_IDS: dict[str, int] = {name: info[0] for name, info in GAME_CATALOG.items()}
"""Human name -> Steam app ID for every target game."""

GAME_GENRES: dict[str, str] = {name: info[1] for name, info in GAME_CATALOG.items()}
"""Human name -> genre for every target game."""


# ---------------------------------------------------------------------------
# Preprocessing thresholds (Phase 2)
# ---------------------------------------------------------------------------

MIN_CLEAN_TEXT_CHARS: int = 10
"""Minimum cleaned_text length (in characters) for a review to be kept."""

MIN_TOKEN_COUNT: int = 3
"""Minimum number of tokens (after filtering) for a review to be kept."""

SPACY_MODEL_NAME: str = "en_core_web_sm"
"""spaCy model used by the preprocessing pipeline."""

PROJECT_STOPWORDS: frozenset[str] = frozenset({"game", "play", "player", "steam"})
"""Project-specific stopwords layered on top of spaCy's English defaults.

These four tokens dominate Steam reviews (>80% frequency) and bury real topic
signal. Filtered in `preprocessing.Preprocessor.tokenize_batch`.
"""

LANGUAGE_MIN_CONFIDENCE: float = 0.50
"""A review is dropped as non-English when another language is detected at this confidence."""

DETECTOR_LANGUAGES: tuple[str, ...] = (
    "BOKMAL", "BULGARIAN", "CHINESE", "CZECH", "DANISH", "DUTCH", "ENGLISH",
    "FINNISH", "FRENCH", "GERMAN", "GREEK", "HUNGARIAN", "INDONESIAN", "ITALIAN",
    "JAPANESE", "KOREAN", "POLISH", "PORTUGUESE", "ROMANIAN", "RUSSIAN",
    "SPANISH", "SWEDISH", "THAI", "TURKISH", "UKRAINIAN", "VIETNAMESE",
)
"""Lingua languages matching Steam's review languages (Norwegian as Bokmal)."""

# ---------------------------------------------------------------------------
# Sentiment thresholds (Phase 3)
# ---------------------------------------------------------------------------

VADER_POSITIVE_THRESHOLD: float = 0.05
"""Compound score >= this is classified as positive sentiment."""

VADER_NEGATIVE_THRESHOLD: float = -0.05
"""Compound score <= this is classified as negative sentiment."""

TRANSFORMER_CANDIDATES: dict[str, str] = {
    "siebert/sentiment-roberta-large-english": "74cea614e245b0832c770ec9aa51bd58df965b9c",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english": (
        "714eb0fa89d2f80546fda750413ed43d93601a13"
    ),
}
"""Transformer sentiment candidates (Decision 11): Hugging Face name -> pinned revision."""

TRANSFORMER_POSITIVE_CUTOFF: float = 0.5
"""A transformer scores a text positive when its positive probability is at least this."""

HEAD_TOKENS: int = 256
"""Tokens kept from the start of a long input (head+tail truncation)."""

TAIL_TOKENS: int = 254
"""Tokens kept from the end of a long input; head, tail and two special tokens make 512."""

# ---------------------------------------------------------------------------
# Topic modeling (Phase 4)
# ---------------------------------------------------------------------------

BERTOPIC_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
"""Sentence-transformer model used by BERTopic for document embeddings."""

BERTOPIC_MIN_TOPIC_SIZE: int = 10
"""Minimum number of documents required to form a topic."""

BERTOPIC_RANDOM_STATE: int = 42
"""Random seed for UMAP dimensionality reduction (reproducibility)."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to stdout at INFO level.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_db_path(mode: str = "seed", study: "Study | None" = None) -> Path:
    """Return the correct database path for the given mode.

    Args:
        mode: Either 'seed' or 'full'.
        study: Optional study. When given, its ``seed_db`` or ``full_db``
            is returned. When None, the April baseline paths are used.

    Returns:
        Path to the appropriate SQLite database file.

    Raises:
        ValueError: If mode is not 'seed' or 'full'.
    """
    if mode not in ("seed", "full"):
        raise ValueError(f"Unknown mode '{mode}'. Expected 'seed' or 'full'.")
    if study is not None:
        return study.seed_db if mode == "seed" else study.full_db
    return SEED_DB_PATH if mode == "seed" else FULL_DB_PATH
