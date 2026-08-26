"""Structured logging setup for Finance RAG.

Configures rich console logging with structured fields for debugging
pipeline execution, retrieval quality, and cost tracking.
"""

import logging
import sys
from typing import Any


def _make_handler() -> logging.Handler:
    """Create a log handler that is safe on all platforms.

    On Windows the Rich LegacyWindowsTerm renderer injects Unicode arrows
    (U+2192) into every log line regardless of our encoding settings, causing
    UnicodeEncodeError on cp1252 terminals. We avoid this by using a plain
    UTF-8 StreamHandler with a simple coloured formatter instead.
    """
    handler = logging.StreamHandler(
        stream=open(  # noqa: WPS515  — intentional unbuffered UTF-8 wrapper
            sys.stdout.fileno(),
            mode="w",
            encoding="utf-8",
            buffering=1,
            closefd=False,
        )
    )
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    return handler


# Module-level logger cache
_configured = False


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure UTF-8-safe logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path to also write logs to.
    """
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler] = [_make_handler()]

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,   # override any handler already installed by third-party libs
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(name)


def log_pipeline_step(
    logger: logging.Logger,
    step: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Log a pipeline execution step with structured details.

    Args:
        logger: The logger instance.
        step: Name of the pipeline step (e.g., "retrieval", "grading").
        details: Optional dictionary of step-specific details.
    """
    detail_str = ""
    if details:
        detail_str = " | " + " | ".join(f"{k}={v}" for k, v in details.items())
    logger.info(f">> {step}{detail_str}")
