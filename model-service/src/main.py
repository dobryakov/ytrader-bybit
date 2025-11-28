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
from .services.market_data_subscriber import MarketDataSubscriber
from .consumers.market_data_consumer import market_data_consumer
from .consumers.execution_event_consumer import ExecutionEventConsumer
from .publishers.signal_publisher import signal_publisher
from .services.warmup_orchestrator import warmup_orchestrator
from .services.training_orchestrator import training_orchestrator
from .database.repositories.model_version_repo import ModelVersionRepository

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

        # Initialize signal publisher
        await signal_publisher.initialize()
        logger.info("Signal publisher initialized")

        # Check if trained model exists
        model_version_repo = ModelVersionRepository()
        active_model = await model_version_repo.get_active_by_strategy(None)  # Check for default strategy
        has_trained_model = active_model is not None

        if not has_trained_model and settings.warmup_mode_enabled:
            logger.info("No trained model found, starting warm-up mode")

            # Subscribe to market data channels
            subscriber = MarketDataSubscriber(
                ws_gateway_url=settings.ws_gateway_url,
                api_key=settings.ws_gateway_api_key,
            )

            # Get trading strategies and default assets
            strategies = settings.trading_strategy_list
            assets = ["BTCUSDT", "ETHUSDT"]  # TODO: Make configurable

            if strategies:
                try:
                    # Subscribe to all required channels for all assets
                    subscriptions = await subscriber.subscribe_all_channels(assets, "model-service")
                    logger.info(
                        "Subscribed to market data channels",
                        subscriptions=subscriptions,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to subscribe to market data channels",
                        error=str(e),
                        exc_info=True,
                    )
                    # Continue anyway - warm-up mode can work with fallback values

            # Start market data consumer
            try:
                await market_data_consumer.start()
                logger.info("Market data consumer started")
            except Exception as e:
                logger.error(
                    "Failed to start market data consumer",
                    error=str(e),
                    exc_info=True,
                )
                # Continue anyway - warm-up mode can work with fallback values

            # Start warm-up orchestrator
            try:
                await warmup_orchestrator.start()
                logger.info("Warm-up orchestrator started")
            except Exception as e:
                logger.error(
                    "Failed to start warm-up orchestrator",
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Start execution event consumer for training pipeline
        async def handle_execution_event(event):
            """Handle execution event for training."""
            try:
                # Add event to training orchestrator buffer
                await training_orchestrator.add_execution_event(event)
                # Check if training should be triggered
                await training_orchestrator.check_and_trigger_training(event.strategy_id)
            except Exception as e:
                logger.error("Error handling execution event", event_id=event.event_id, error=str(e), exc_info=True)

        execution_event_consumer = ExecutionEventConsumer(event_callback=handle_execution_event)
        app.state.execution_event_consumer = execution_event_consumer  # Store for shutdown
        try:
            await execution_event_consumer.start()
            logger.info("Execution event consumer started")
        except Exception as e:
            logger.error("Failed to start execution event consumer", error=str(e), exc_info=True)
            # Continue anyway - training can be triggered manually

        logger.info("Model service started successfully")
    except Exception as e:
        logger.error("Failed to start model service", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down model service")

    try:
        # Cancel any ongoing training
        await training_orchestrator._cancel_current_training()
        logger.info("Training cancelled")

        # Stop execution event consumer
        if hasattr(app.state, "execution_event_consumer"):
            await app.state.execution_event_consumer.stop()
            logger.info("Execution event consumer stopped")

        # Stop warm-up orchestrator
        await warmup_orchestrator.stop()
        logger.info("Warm-up orchestrator stopped")

        # Stop market data consumer
        await market_data_consumer.stop()
        logger.info("Market data consumer stopped")

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

