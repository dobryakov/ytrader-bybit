"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from .config.settings import settings
from .config.logging import setup_logging, get_logger
from .api.health import router as health_router
from .services.database.connection import DatabaseConnection
from .services.queue.connection import QueueConnection

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("application_starting", service=settings.ws_gateway_service_name)

    try:
        # Initialize database connection pool
        await DatabaseConnection.create_pool()
        logger.info("database_pool_initialized")

        # Initialize RabbitMQ connection
        await QueueConnection.create_connection()
        logger.info("rabbitmq_connection_initialized")

        logger.info("application_started", port=settings.ws_gateway_port)
    except Exception as e:
        logger.error("application_startup_failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("application_shutting_down")

    try:
        # Close RabbitMQ connection
        await QueueConnection.close_connection()
        logger.info("rabbitmq_connection_closed")

        # Close database connection pool
        await DatabaseConnection.close_pool()
        logger.info("database_pool_closed")

        logger.info("application_shutdown_complete")
    except Exception as e:
        logger.error("application_shutdown_error", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="WebSocket Gateway Service",
    description="WebSocket Gateway for Bybit Data Aggregation and Routing",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(health_router)

