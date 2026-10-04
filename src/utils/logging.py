"""
Logging Configuration

Structured logging setup for the application.
"""

import structlog
import logging
import sys


def configure_logging(level: str = "INFO", format_type: str = "json"):
    """
    Configure structured logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_type: Output format (json or console)
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )

    # Configure structlog
    if format_type == "json":
        processors = [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    else:
        processors = [
            structlog.dev.ConsoleRenderer()
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
