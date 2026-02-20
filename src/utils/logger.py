import logging
import logging.config
from pathlib import Path

def setup_logger(name: str = "deep_research_agent") -> logging.Logger:
    """
    Configure and return a logger instance.
    Uses basic config first, then upgrades with YAML config if available.
    Avoids importing settings at module-level to prevent circular imports.
    """
    # Ensure log directory exists
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        from config.settings import settings
        if settings.logging_config:
            logging.config.dictConfig(settings.logging_config)
        else:
            _basic_config()
    except Exception:
        # Fallback: if settings fails to load, use basic config
        _basic_config()

    return logging.getLogger(name)

def _basic_config():
    """Fallback basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

logger = setup_logger()
