"""FastAPI application setup for Position Manager."""

from __future__ import annotations

from fastapi import FastAPI

from ..config.database import DatabaseConnection
from ..config.logging import configure_logging, get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..config.settings import settings
from ..exceptions import QueueError
from .middleware.logging import logging_middleware
from .routes import health as health_routes
from .routes import portfolio as portfolio_routes
from .routes import positions as positions_routes


configure_logging()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Position Manager",
        version="1.0.0",
    )

    # Middleware
    app.middleware("http")(logging_middleware)

    # Routes
    app.include_router(health_routes.router)
    app.include_router(positions_routes.router)
    app.include_router(portfolio_routes.router)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("app_startup_begin", service=settings.position_manager_service_name)
        # Initialize database connection (hard requirement)
        await DatabaseConnection.create_pool()

        # Initialize RabbitMQ connection, but do not fail startup if it's temporarily unavailable.
        # Health and consumers will reflect queue connectivity separately.
        try:
            await RabbitMQConnection.create_connection()
        except QueueError as e:
            logger.error(
                "rabbitmq_startup_connection_failed_non_fatal",
                error=str(e),
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
            )
        logger.info("app_startup_completed", service=settings.position_manager_service_name)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("app_shutdown_begin", service=settings.position_manager_service_name)
        await DatabaseConnection.close_pool()
        await RabbitMQConnection.close_connection()
        logger.info("app_shutdown_completed", service=settings.position_manager_service_name)

    return app


app = create_app()



