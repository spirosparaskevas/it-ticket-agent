"""Structured logs: one JSON object per line on stdout. API keys are never passed to the logger."""

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    numeric = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            # request_id and session_id, bound once per request in main.py, go on every line.
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # tests reconfigure and capture logs
    )
