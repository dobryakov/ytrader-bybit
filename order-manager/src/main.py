"""Service entry point for Order Manager microservice.

Handles service startup, dependency initialization, and graceful shutdown.
"""

import signal
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from .api.main import create_app
from .config.database import DatabaseConnection
from .config.rabbitmq import RabbitMQConnection
from .config.logging import get_logger, configure_logging
from .utils.bybit_client import get_bybit_client, close_bybit_client
from .utils.tracing import generate_trace_id, set_trace_id
from .config.settings import settings

# Configure logging first
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    trace_id = generate_trace_id()
    set_trace_id(trace_id)
    logger.info(
        "application_starting",
        service=settings.order_manager_service_name,
        trace_id=trace_id,
    )

    try:
        # Initialize database connection pool
        await DatabaseConnection.create_pool()
        logger.info("database_pool_initialized", trace_id=trace_id)

        # Initialize RabbitMQ connection
        await RabbitMQConnection.create_connection()
        logger.info("rabbitmq_connection_initialized", trace_id=trace_id)

        # Initialize Bybit client
        get_bybit_client()
        logger.info("bybit_client_initialized", trace_id=trace_id)

        # TODO: Initialize signal consumer
        # TODO: Initialize event subscriber
        # TODO: Start background tasks (position snapshots, validation)

        logger.info("application_started", port=settings.order_manager_port, trace_id=trace_id)
    except Exception as e:
        logger.error(
            "application_startup_failed",
            error=str(e),
            error_type=type(e).__name__,
            trace_id=trace_id,
            exc_info=True,
        )
        raise

    yield

    # Shutdown
    logger.info("application_shutting_down", service=settings.order_manager_service_name)
    
    try:
        # Close Bybit client
        await close_bybit_client()
        logger.info("bybit_client_closed")

        # Close RabbitMQ connection
        await RabbitMQConnection.close_connection()
        logger.info("rabbitmq_connection_closed")

        # Close database pool
        await DatabaseConnection.close_pool()
        logger.info("database_pool_closed")

        logger.info("application_shutdown_complete")
    except Exception as e:
        logger.error(
            "application_shutdown_error",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )


# Create app with lifespan
app = create_app(lifespan_context=lifespan)


def main():
    """Main entry point for the service."""
    # Register signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("shutdown_signal_received", signal=signum)
        # Uvicorn will handle graceful shutdown

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run uvicorn server
    uvicorn.run(
        "order_manager.src.main:app",
        host="0.0.0.0",
        port=settings.order_manager_port,
        log_config=None,  # Use our structured logging
        access_log=False,  # We handle logging via middleware
    )


if __name__ == "__main__":
    main()

