"""
Main application entry point.

Initializes FastAPI application with routing, middleware, and startup/shutdown handlers.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config.settings import settings
from .config.logging import configure_logging, get_logger
from .config.exceptions import ModelServiceError
from .database.connection import db_pool
from .config.rabbitmq import rabbitmq_manager
from .api.router import api_router, APIKeyMiddleware, TraceIDMiddleware
from .api.health import router as health_router

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown operations:
    - Database connection pool creation
    - RabbitMQ connection establishment
    - Configuration validation
    """
    # Startup
    logger.info("Starting model service", service_name=settings.model_service_service_name)

    try:
        # Validate configuration
        settings.validate_on_startup()
        logger.info("Configuration validated successfully")

        # Create database connection pool
        await db_pool.create_pool()
        logger.info("Database connection pool created")

        # Connect to RabbitMQ
        await rabbitmq_manager.connect()
        logger.info("RabbitMQ connection established")

        logger.info("Model service started successfully")
    except Exception as e:
        logger.error("Failed to start model service", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down model service")

    try:
        # Close RabbitMQ connection
        await rabbitmq_manager.disconnect()
        logger.info("RabbitMQ connection closed")

        # Close database connection pool
        await db_pool.close_pool()
        logger.info("Database connection pool closed")

        logger.info("Model service shut down successfully")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e), exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="Model Service",
    description="Trading Decision and ML Training Microservice",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(TraceIDMiddleware)
app.add_middleware(APIKeyMiddleware)

# Register routers
app.include_router(health_router)
app.include_router(api_router)


@app.exception_handler(ModelServiceError)
async def model_service_error_handler(request, exc: ModelServiceError):
    """
    Global exception handler for ModelServiceError.

    Args:
        request: FastAPI request
        exc: ModelServiceError exception

    Returns:
        JSON error response
    """
    logger.error("Model service error", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        JSON error response
    """
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"},
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "model-service",
        "version": "1.0.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "model_service.src.main:app",
        host="0.0.0.0",
        port=settings.model_service_port,
        log_level=settings.model_service_log_level.lower(),
        reload=False,
    )

