import logging
import sys
from pathlib import Path
from typing import Optional
from src.config import get_settings


NOISY_HTTP_LOGGERS = (
    "tornado.access",
    "tornado.application",
    "tornado.general",
    "urllib3.connectionpool",
    "urllib3",
    "httpx",
    "httpcore",
)


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    suppress_http_logs: bool = True,
) -> None:
    settings = get_settings()
    log_file = log_file or settings.log_file
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(message)s"
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)

    if suppress_http_logs:
        for logger_name in NOISY_HTTP_LOGGERS:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(logging.WARNING)
            noisy_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

