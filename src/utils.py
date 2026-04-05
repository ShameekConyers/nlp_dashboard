"""Shared constants, path resolution, and logging setup for the NLP dashboard project."""

import logging
from pathlib import Path

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

def get_db_path(mode: str = "seed") -> Path:
    """Return the correct database path for the given mode.

    Args:
        mode: Either 'seed' or 'full'.

    Returns:
        Path to the appropriate SQLite database file.

    Raises:
        ValueError: If mode is not 'seed' or 'full'.
    """
    if mode == "seed":
        return SEED_DB_PATH
    if mode == "full":
        return FULL_DB_PATH
    raise ValueError(f"Unknown mode '{mode}'. Expected 'seed' or 'full'.")
