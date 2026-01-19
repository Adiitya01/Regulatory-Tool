import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys


def configure_logging(log_level: int = logging.INFO, log_dir: str = None):
    """Configure root logger with console and rotating file handlers.

    Call this once at program startup (modules can then call `logging.getLogger(__name__)`).
    """
    log_dir_path = Path(log_dir) if log_dir else Path.cwd()
    try:
        log_dir_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_dir_path = Path.cwd()

    log_file = log_dir_path / "dhf_processor.log"

    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Allow overriding via environment variable LOG_LEVEL (e.g., DEBUG, INFO)
    try:
        import os
        env_level = os.environ.get('LOG_LEVEL')
        if env_level:
            env_level = env_level.upper()
            if env_level in logging._nameToLevel:
                log_level = logging._nameToLevel[env_level]
    except Exception:
        pass

    root = logging.getLogger()
    root.setLevel(log_level)

    # Ensure a RotatingFileHandler for our log file exists (avoid duplicates)
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '') == str(log_file)]
    if not file_handlers:
        fh = RotatingFileHandler(str(log_file), maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8')
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Add a dedicated stdout StreamHandler so logs are visible in the terminal
    # Name it so we don't add duplicates.
    stdout_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and getattr(h, 'name', None) == 'dhf_stdout']
    if not stdout_handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(fmt)
        ch.name = 'dhf_stdout'
        root.addHandler(ch)


def get_logger(name: str):
    return logging.getLogger(name)


# Auto-configure when imported by the main application
configure_logging()
