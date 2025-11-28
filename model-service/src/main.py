"""
Main application entry point.

Initializes FastAPI application with routing, middleware, and startup/shutdown handlers.
"""

import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .config.settings import settings
from .config.logging import configure_logging, get_logger
from .config.exceptions import ModelServiceError
from .database.connection import db_pool
from .config.rabbitmq import rabbitmq_manager
from .api.router import api_router, APIKeyMiddleware, TraceIDMiddleware
from .api.middleware import RequestResponseLoggingMiddleware
from .api.health import router as health_router
from .services.market_data_subscriber import MarketDataSubscriber
from .consumers.market_data_consumer import market_data_consumer
from .consumers.execution_event_consumer import ExecutionEventConsumer
from .publishers.signal_publisher import signal_publisher
from .services.warmup_orchestrator import warmup_orchestrator
from .services.intelligent_orchestrator import intelligent_orchestrator
from .services.training_orchestrator import training_orchestrator
from .services.mode_transition import mode_transition
from .services.quality_monitor import quality_monitor
from .database.repositories.model_version_repo import ModelVersionRepository

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Global shutdown event
_shutdown_event = asyncio.Event()


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

        # Subscribe to market data channels (needed for both warm-up and intelligent modes)
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
                # Continue anyway - can work with fallback values

        # Start market data consumer (needed for both modes)
        try:
            await market_data_consumer.start()
            logger.info("Market data consumer started")
        except Exception as e:
            logger.error(
                "Failed to start market data consumer",
                error=str(e),
                exc_info=True,
            )
            # Continue anyway - can work with fallback values

        # Check if trained model exists and start appropriate orchestrator
        model_version_repo = ModelVersionRepository()
        active_model = await model_version_repo.get_active_by_strategy(None)  # Check for default strategy
        has_trained_model = active_model is not None

        if has_trained_model:
            logger.info("Active model found, starting intelligent signal generation", model_version=active_model["version"])
            try:
                # Start intelligent orchestrator
                await intelligent_orchestrator.start()
                logger.info("Intelligent orchestrator started")
            except Exception as e:
                logger.error(
                    "Failed to start intelligent orchestrator",
                    error=str(e),
                    exc_info=True,
                )
                raise
        elif settings.warmup_mode_enabled:
            logger.info("No trained model found, starting warm-up mode")
            try:
                # Start warm-up orchestrator
                await warmup_orchestrator.start()
                logger.info("Warm-up orchestrator started")
            except Exception as e:
                logger.error(
                    "Failed to start warm-up orchestrator",
                    error=str(e),
                    exc_info=True,
                )
                raise
        else:
            logger.warning("No trained model and warm-up mode disabled, signal generation will not start")

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

        # Start quality monitor for periodic quality evaluation
        try:
            await quality_monitor.start()
            logger.info("Quality monitor started")
        except Exception as e:
            logger.error("Failed to start quality monitor", error=str(e), exc_info=True)
            # Continue anyway - quality evaluation can be triggered manually

        logger.info("Model service started successfully")
    except Exception as e:
        logger.error("Failed to start model service", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down model service")

    # Set shutdown timeout (30 seconds)
    shutdown_timeout = 30.0

    try:
        # Create shutdown tasks with timeout
        shutdown_tasks = []

        # Cancel any ongoing training
        async def cancel_training():
            try:
                await asyncio.wait_for(training_orchestrator._cancel_current_training(), timeout=5.0)
                logger.info("Training cancelled")
            except asyncio.TimeoutError:
                logger.warning("Training cancellation timed out")
            except Exception as e:
                logger.error("Error cancelling training", error=str(e), exc_info=True)

        shutdown_tasks.append(cancel_training())

        # Stop quality monitor
        async def stop_quality_monitor():
            try:
                await asyncio.wait_for(quality_monitor.stop(), timeout=5.0)
                logger.info("Quality monitor stopped")
            except asyncio.TimeoutError:
                logger.warning("Quality monitor stop timed out")
            except Exception as e:
                logger.error("Error stopping quality monitor", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_quality_monitor())

        # Stop execution event consumer
        async def stop_execution_consumer():
            try:
                if hasattr(app.state, "execution_event_consumer"):
                    await asyncio.wait_for(app.state.execution_event_consumer.stop(), timeout=5.0)
                    logger.info("Execution event consumer stopped")
            except asyncio.TimeoutError:
                logger.warning("Execution event consumer stop timed out")
            except Exception as e:
                logger.error("Error stopping execution event consumer", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_execution_consumer())

        # Stop intelligent orchestrator
        async def stop_intelligent_orchestrator():
            try:
                await asyncio.wait_for(intelligent_orchestrator.stop(), timeout=5.0)
                logger.info("Intelligent orchestrator stopped")
            except asyncio.TimeoutError:
                logger.warning("Intelligent orchestrator stop timed out")
            except Exception as e:
                logger.error("Error stopping intelligent orchestrator", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_intelligent_orchestrator())

        # Stop warm-up orchestrator
        async def stop_warmup_orchestrator():
            try:
                await asyncio.wait_for(warmup_orchestrator.stop(), timeout=5.0)
                logger.info("Warm-up orchestrator stopped")
            except asyncio.TimeoutError:
                logger.warning("Warm-up orchestrator stop timed out")
            except Exception as e:
                logger.error("Error stopping warm-up orchestrator", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_warmup_orchestrator())

        # Stop market data consumer
        async def stop_market_data_consumer():
            try:
                await asyncio.wait_for(market_data_consumer.stop(), timeout=5.0)
                logger.info("Market data consumer stopped")
            except asyncio.TimeoutError:
                logger.warning("Market data consumer stop timed out")
            except Exception as e:
                logger.error("Error stopping market data consumer", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_market_data_consumer())

        # Execute shutdown tasks in parallel with overall timeout
        try:
            await asyncio.wait_for(asyncio.gather(*shutdown_tasks, return_exceptions=True), timeout=shutdown_timeout)
        except asyncio.TimeoutError:
            logger.warning("Shutdown tasks timed out, forcing shutdown")

        # Close RabbitMQ connection
        try:
            await asyncio.wait_for(rabbitmq_manager.disconnect(), timeout=5.0)
            logger.info("RabbitMQ connection closed")
        except asyncio.TimeoutError:
            logger.warning("RabbitMQ disconnect timed out")
        except Exception as e:
            logger.error("Error closing RabbitMQ connection", error=str(e), exc_info=True)

        # Close database connection pool
        try:
            await asyncio.wait_for(db_pool.close_pool(), timeout=5.0)
            logger.info("Database connection pool closed")
        except asyncio.TimeoutError:
            logger.warning("Database pool close timed out")
        except Exception as e:
            logger.error("Error closing database connection pool", error=str(e), exc_info=True)

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

# Add middleware (order matters - TraceID first, then logging, then auth)
app.add_middleware(TraceIDMiddleware)
app.add_middleware(RequestResponseLoggingMiddleware)
app.add_middleware(APIKeyMiddleware)

# Register routers
app.include_router(health_router)
app.include_router(api_router)


@app.exception_handler(ModelServiceError)
async def model_service_error_handler(request, exc: ModelServiceError):
    """
    Global exception handler for ModelServiceError with proper status code mapping.

    Args:
        request: FastAPI request
        exc: ModelServiceError exception

    Returns:
        JSON error response with appropriate status code
    """
    from fastapi import status
    from ..config.exceptions import (
        ConfigurationError,
        DatabaseConnectionError,
        DatabaseQueryError,
        MessageQueueConnectionError,
        MessageQueuePublishError,
        MessageQueueConsumeError,
        ModelNotFoundError,
        ModelLoadError,
        ModelSaveError,
        ModelTrainingError,
        DatasetInsufficientError,
        SignalValidationError,
        RateLimitExceededError,
    )

    # Map exception types to HTTP status codes
    status_code_map = {
        ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,  # Configuration issues are server errors
        DatabaseConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,  # Database unavailable
        DatabaseQueryError: status.HTTP_500_INTERNAL_SERVER_ERROR,  # Query errors are server errors
        MessageQueueConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,  # Message queue unavailable
        MessageQueuePublishError: status.HTTP_503_SERVICE_UNAVAILABLE,  # Publishing failed
        MessageQueueConsumeError: status.HTTP_503_SERVICE_UNAVAILABLE,  # Consumption failed
        ModelNotFoundError: status.HTTP_404_NOT_FOUND,  # Model not found
        ModelLoadError: status.HTTP_500_INTERNAL_SERVER_ERROR,  # Model loading failed
        ModelSaveError: status.HTTP_500_INTERNAL_SERVER_ERROR,  # Model saving failed
        ModelTrainingError: status.HTTP_500_INTERNAL_SERVER_ERROR,  # Training failed
        DatasetInsufficientError: status.HTTP_400_BAD_REQUEST,  # Insufficient data
        SignalValidationError: status.HTTP_400_BAD_REQUEST,  # Invalid signal
        RateLimitExceededError: status.HTTP_429_TOO_MANY_REQUESTS,  # Rate limit exceeded
    }

    # Get status code for exception type
    status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    logger.error(
        "Model service error",
        error=str(exc),
        error_type=type(exc).__name__,
        status_code=status_code,
        exc_info=True,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "type": "ModelServiceError",
        },
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
    from fastapi import status

    # Map common exception types to status codes
    if isinstance(exc, ValueError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_detail = "Invalid request: " + str(exc)
    elif isinstance(exc, KeyError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_detail = "Missing required field: " + str(exc)
    elif isinstance(exc, PermissionError):
        status_code = status.HTTP_403_FORBIDDEN
        error_detail = "Permission denied: " + str(exc)
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        error_detail = "An unexpected error occurred"

    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        status_code=status_code,
        exc_info=True,
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": "Internal server error",
            "detail": error_detail,
            "type": type(exc).__name__,
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "model-service",
        "version": "1.0.0",
        "status": "running",
    }


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal", signal=signum)
        _shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    import uvicorn

    # Setup signal handlers
    setup_signal_handlers()

    uvicorn.run(
        "model_service.src.main:app",
        host="0.0.0.0",
        port=settings.model_service_port,
        log_level=settings.model_service_log_level.lower(),
        reload=False,
    )

