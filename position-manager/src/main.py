"""Uvicorn entrypoint for Position Manager service."""

from __future__ import annotations

import asyncio

import uvicorn

from .api.main import app  # noqa: F401
from .config.logging import configure_logging, get_logger
from .config.settings import settings


configure_logging()
logger = get_logger(__name__)


def main() -> None:
    """Start the uvicorn server."""
    logger.info(
        "position_manager_starting",
        port=settings.position_manager_port,
        service=settings.position_manager_service_name,
    )

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.position_manager_port,
        reload=False,
        loop="asyncio",
        factory=False,
    )


if __name__ == "__main__":
    main()



