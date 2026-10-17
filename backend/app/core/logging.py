"""Logging setup shared by the API and background tasks."""

import logging


def configure_logging(level: str) -> None:
    """Configure root logging with a consistent, grep-friendly format."""
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
