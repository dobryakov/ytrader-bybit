"""
Main application entry point.

Initializes FastAPI application with routing, middleware, and startup/shutdown handlers.
"""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID
from datetime import timezone
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
# Market data subscription and consumer removed - now using Feature Service
# Execution event consumer removed - training pipeline now uses Feature Service datasets
from .consumers.position_update_consumer import position_update_consumer
from .consumers.feature_consumer import feature_consumer
from .consumers.dataset_ready_consumer import DatasetReadyConsumer
from .consumers.execution_event_consumer import ExecutionEventConsumer
from .publishers.signal_publisher import signal_publisher
from .services.warmup_orchestrator import warmup_orchestrator
from .services.intelligent_orchestrator import intelligent_orchestrator
from .services.training_orchestrator import training_orchestrator
from .tasks.target_evaluation_task import target_evaluation_task
from .tasks.retraining_task import retraining_task
from .services.mode_transition import mode_transition
from .services.quality_monitor import quality_monitor
from .database.repositories.model_version_repo import ModelVersionRepository
from .services.prediction_trading_linker import prediction_trading_linker
from .services.position_based_signal_generator import position_based_signal_generator
from .models.execution_event import OrderExecutionEvent
from decimal import Decimal

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

        # Market data subscription removed - now using Feature Service for market data
        logger.info("Market data now provided by Feature Service (via features queue or REST API)")

        # Start feature consumer if queue is enabled (for Feature Service integration)
        if settings.feature_service_use_queue:
            try:
                await feature_consumer.start()
                logger.info("Feature consumer started")
            except Exception as e:
                logger.error(
                    "Failed to start feature consumer",
                    error=str(e),
                    exc_info=True,
                )
                # Continue anyway - can fallback to REST API

        # Market data consumer removed - now using Feature Service for market data
        logger.info("Market data consumer removed - using Feature Service instead")

        # Check if trained model exists and start appropriate orchestrator
        model_version_repo = ModelVersionRepository()
        
        # Check for active models across all configured strategies
        has_trained_model = False
        active_model = None
        strategies = settings.trading_strategy_list
        
        if strategies:
            # Check each strategy for active model
            for strategy_id in strategies:
                model = await model_version_repo.get_active_by_strategy(strategy_id)
                if model:
                    has_trained_model = True
                    active_model = model
                    logger.info(
                        "Active model found for strategy",
                        strategy_id=strategy_id,
                        model_version=model["version"],
                    )
                    break
        else:
            # If no strategies configured, check for default strategy (None)
            active_model = await model_version_repo.get_active_by_strategy(None)
            has_trained_model = active_model is not None
            if has_trained_model:
                logger.info(
                    "Active model found for default strategy",
                    model_version=active_model["version"],
                )

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

        # Note: Training pipeline no longer uses execution_events
        # Training is triggered by time-based retraining or scheduled triggers
        # Dataset building is handled by Feature Service
        logger.info("Training pipeline uses Feature Service datasets (market-data-only training)")

        # Start position update consumer for cache invalidation and exit strategy evaluation
        app.state.position_update_consumer = position_update_consumer  # Store for shutdown
        try:
            await position_update_consumer.start()
            logger.info("Position update consumer started")
        except Exception as e:
            logger.error("Failed to start position update consumer", error=str(e), exc_info=True)
            # Continue anyway - cache will work but won't auto-invalidate on updates
            # If exit strategy is enabled, start fallback mode
            if settings.exit_strategy_enabled:
                try:
                    await position_based_signal_generator.start_fallback_mode()
                    logger.warning("Started fallback mode for exit strategy evaluation")
                except Exception as fallback_error:
                    logger.error("Failed to start fallback mode", error=str(fallback_error), exc_info=True)

        # Start quality monitor for periodic quality evaluation
        try:
            await quality_monitor.start()
            logger.info("Quality monitor started")
        except Exception as e:
            logger.error("Failed to start quality monitor", error=str(e), exc_info=True)
            # Continue anyway - quality evaluation can be triggered manually

        # Start dataset ready consumer for Feature Service dataset notifications
        async def handle_dataset_ready(dataset_id: UUID, symbol: Optional[str], trace_id: Optional[str]):
            """Handle dataset ready notification from Feature Service."""
            try:
                # Trigger training with ready dataset via TrainingOrchestrator
                await training_orchestrator.handle_dataset_ready(dataset_id, symbol, trace_id)
            except Exception as e:
                logger.error(
                    "Error handling dataset ready notification",
                    dataset_id=str(dataset_id),
                    symbol=symbol,
                    error=str(e),
                    trace_id=trace_id,
                    exc_info=True,
                )

        dataset_ready_consumer = DatasetReadyConsumer(dataset_ready_callback=handle_dataset_ready)
        app.state.dataset_ready_consumer = dataset_ready_consumer  # Store for shutdown
        try:
            await dataset_ready_consumer.start()
            logger.info("Dataset ready consumer started")
        except Exception as e:
            logger.error("Failed to start dataset ready consumer", error=str(e), exc_info=True)
            # Continue anyway - training can be triggered manually or via polling

        # Start target evaluation background task
        try:
            await target_evaluation_task.start()
            app.state.target_evaluation_task = target_evaluation_task
            logger.info("Target evaluation task started")
        except Exception as e:
            logger.error("Failed to start target evaluation task", error=str(e), exc_info=True)

        # Start retraining background task
        try:
            await retraining_task.start()
            app.state.retraining_task = retraining_task
            logger.info("Retraining task started")
        except Exception as e:
            logger.error("Failed to start retraining task", error=str(e), exc_info=True)

        # Start execution event consumer (for execution_events persistence & prediction trading linkage)
        async def handle_execution_event(event: OrderExecutionEvent) -> None:
            """Handle execution event for prediction trading linker."""
            try:
                pool = await db_pool.get_pool()
                
                # Get order UUID from orders table by bybit_order_id
                # event.order_id is bybit_order_id (string), we need orders.id (UUID)
                order_query = """
                    SELECT id
                    FROM orders
                    WHERE order_id = $1
                    LIMIT 1
                """
                order_row = await pool.fetchrow(order_query, event.order_id)
                
                if not order_row:
                    logger.debug(
                        "No order found in database",
                        bybit_order_id=event.order_id,
                        signal_id=event.signal_id,
                    )
                    return
                
                order_uuid = order_row["id"]
                
                # Query position_orders to get relationship_type and position_id
                position_order_query = """
                    SELECT por.relationship_type, por.size_delta, por.execution_price, por.position_id
                    FROM position_orders por
                    WHERE por.order_id = $1
                    LIMIT 1
                """
                position_order_row = await pool.fetchrow(position_order_query, order_uuid)
                
                if not position_order_row:
                    logger.debug(
                        "No position_order found for order",
                        order_id=str(order_uuid),
                        bybit_order_id=event.order_id,
                        signal_id=event.signal_id,
                    )
                    return
                
                relationship_type = position_order_row["relationship_type"]
                position_id = position_order_row["position_id"]
                
                # Calculate realized_pnl_delta for this specific order execution
                # For closed or decreased positions, calculate PnL based on entry and exit prices
                realized_pnl_delta = Decimal("0")
                
                # Get existing prediction trading results (used for both realized_pnl calculation and creation check)
                existing_results = await prediction_trading_linker.prediction_trading_results_repo.get_by_signal_id(event.signal_id)
                
                if relationship_type in ("closed", "decreased") and existing_results and len(existing_results) > 0:
                        # Use the first (most recent) open result that's not yet closed
                        result = None
                        for res in existing_results:
                            if not res.get("is_closed", False):
                                result = res
                                break
                        
                        if result:
                            entry_price = Decimal(str(result.get("entry_price", 0)))
                            position_size_at_entry = Decimal(str(result.get("position_size_at_entry", 0)))
                            
                            if entry_price > 0 and position_size_at_entry != 0:
                                # Determine position direction from size_at_entry
                                # Positive = long, Negative = short
                                is_long = position_size_at_entry > 0
                                
                                # Calculate closed quantity
                                closed_quantity = Decimal(str(event.execution_quantity))
                                
                                exit_price = Decimal(str(event.execution_price))
                                execution_fees = Decimal(str(event.execution_fees))
                                
                                # Calculate realized PnL
                                if is_long:
                                    # Long position: profit = (exit_price - entry_price) * quantity - fees
                                    # For SELL order closing long position
                                    realized_pnl_delta = (exit_price - entry_price) * closed_quantity - execution_fees
                                else:
                                    # Short position: profit = (entry_price - exit_price) * quantity - fees
                                    # For BUY order closing short position
                                    realized_pnl_delta = (entry_price - exit_price) * closed_quantity - execution_fees
                                
                                logger.info(
                                    "Calculated realized PnL delta",
                                    signal_id=event.signal_id,
                                    relationship_type=relationship_type,
                                    entry_price=str(entry_price),
                                    exit_price=str(exit_price),
                                    closed_quantity=str(closed_quantity),
                                    execution_fees=str(execution_fees),
                                    realized_pnl_delta=str(realized_pnl_delta),
                                    is_long=is_long,
                                )
                
                # Check if prediction trading result exists, if not create it
                if not existing_results:
                    # Normalize entry_timestamp to timezone-aware UTC, then to naive for PostgreSQL
                    # This prevents asyncpg errors when comparing datetime objects
                    entry_timestamp = event.executed_at
                    if entry_timestamp.tzinfo is None:
                        entry_timestamp = entry_timestamp.replace(tzinfo=timezone.utc)
                    else:
                        entry_timestamp = entry_timestamp.astimezone(timezone.utc)
                    entry_timestamp = entry_timestamp.replace(tzinfo=None)  # Convert to naive for PostgreSQL
                    
                    # Create prediction trading result for any relationship_type if prediction_target exists
                    # This ensures we track all predictions that result in trading activity
                    result = await prediction_trading_linker.link_prediction_to_trading(
                        signal_id=event.signal_id,
                        entry_signal_id=event.signal_id,
                        entry_price=Decimal(str(event.execution_price)),
                        entry_timestamp=entry_timestamp,
                        position_size_at_entry=Decimal(str(event.execution_quantity)),
                    )
                    if result:
                        logger.info(
                            "Prediction trading result created from execution event",
                            signal_id=event.signal_id,
                            order_id=str(order_uuid),
                            relationship_type=relationship_type,
                            result_id=result.get("id"),
                        )
                    else:
                        logger.debug(
                            "No prediction target found for signal, skipping prediction trading result creation",
                            signal_id=event.signal_id,
                            order_id=str(order_uuid),
                        )
                
                # Update prediction trading result (will do nothing if it doesn't exist)
                await prediction_trading_linker.update_trading_result_on_order_fill(
                    signal_id=event.signal_id,
                    order_id=order_uuid,
                    execution_price=Decimal(str(event.execution_price)),
                    execution_quantity=Decimal(str(event.execution_quantity)),
                    realized_pnl_delta=realized_pnl_delta,
                    relationship_type=relationship_type,
                )
                
                logger.debug(
                    "Execution event processed for prediction trading linker",
                    signal_id=event.signal_id,
                    order_id=str(order_uuid),
                    bybit_order_id=event.order_id,
                    relationship_type=relationship_type,
                )
            except Exception as e:
                logger.error(
                    "Error handling execution event for prediction trading linker",
                    signal_id=event.signal_id,
                    order_id=event.order_id,
                    error=str(e),
                    exc_info=True,
                )
        
        try:
            execution_event_consumer = ExecutionEventConsumer(event_callback=handle_execution_event)
            app.state.execution_event_consumer = execution_event_consumer
            await execution_event_consumer.start()
            logger.info("Execution event consumer started")
        except Exception as e:
            logger.error("Failed to start execution event consumer", error=str(e), exc_info=True)

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

        # Gracefully shut down training orchestrator
        async def shutdown_training_orchestrator():
            try:
                await asyncio.wait_for(training_orchestrator.shutdown(timeout=10.0), timeout=15.0)
                logger.info("Training orchestrator shut down gracefully")
            except asyncio.TimeoutError:
                logger.warning("Training orchestrator shutdown timed out")
            except Exception as e:
                logger.error("Error shutting down training orchestrator", error=str(e), exc_info=True)

        shutdown_tasks.append(shutdown_training_orchestrator())

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

        # Stop target evaluation task
        async def stop_target_evaluation_task():
            try:
                if hasattr(app.state, "target_evaluation_task"):
                    await asyncio.wait_for(app.state.target_evaluation_task.stop(), timeout=5.0)
                    logger.info("Target evaluation task stopped")
            except asyncio.TimeoutError:
                logger.warning("Target evaluation task stop timed out")
            except Exception as e:
                logger.error("Error stopping target evaluation task", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_target_evaluation_task())

        # Stop retraining task
        async def stop_retraining_task():
            try:
                if hasattr(app.state, "retraining_task"):
                    await asyncio.wait_for(app.state.retraining_task.stop(), timeout=10.0)
                    logger.info("Retraining task stopped")
            except asyncio.TimeoutError:
                logger.warning("Retraining task stop timed out")
            except Exception as e:
                logger.error("Error stopping retraining task", error=str(e), exc_info=True)
        shutdown_tasks.append(stop_retraining_task())

        # Stop execution event consumer
        async def stop_execution_event_consumer():
            try:
                if hasattr(app.state, "execution_event_consumer"):
                    await asyncio.wait_for(app.state.execution_event_consumer.stop(), timeout=5.0)
                    logger.info("Execution event consumer stopped")
            except asyncio.TimeoutError:
                logger.warning("Execution event consumer stop timed out")
            except Exception as e:
                logger.error("Error stopping execution event consumer", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_execution_event_consumer())

        # Stop position update consumer
        async def stop_position_update_consumer():
            try:
                if hasattr(app.state, "position_update_consumer"):
                    await asyncio.wait_for(app.state.position_update_consumer.stop(), timeout=5.0)
                    logger.info("Position update consumer stopped")
            except asyncio.TimeoutError:
                logger.warning("Position update consumer stop timed out")
            except Exception as e:
                logger.error("Error stopping position update consumer", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_position_update_consumer())

        # Stop position-based signal generator fallback mode if active
        async def stop_position_based_signal_generator():
            try:
                await asyncio.wait_for(position_based_signal_generator.stop_fallback_mode(), timeout=5.0)
                logger.info("Position-based signal generator stopped")
            except asyncio.TimeoutError:
                logger.warning("Position-based signal generator stop timed out")
            except Exception as e:
                logger.error("Error stopping position-based signal generator", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_position_based_signal_generator())

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

        # Stop feature consumer
        async def stop_feature_consumer():
            try:
                if settings.feature_service_use_queue:
                    await asyncio.wait_for(feature_consumer.stop(), timeout=5.0)
                    logger.info("Feature consumer stopped")
            except asyncio.TimeoutError:
                logger.warning("Feature consumer stop timed out")
            except Exception as e:
                logger.error("Error stopping feature consumer", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_feature_consumer())

        # Market data consumer removed - no longer needed

        # Stop dataset ready consumer
        async def stop_dataset_ready_consumer():
            try:
                if hasattr(app.state, "dataset_ready_consumer"):
                    await asyncio.wait_for(app.state.dataset_ready_consumer.stop(), timeout=5.0)
                    logger.info("Dataset ready consumer stopped")
            except asyncio.TimeoutError:
                logger.warning("Dataset ready consumer stop timed out")
            except Exception as e:
                logger.error("Error stopping dataset ready consumer", error=str(e), exc_info=True)

        shutdown_tasks.append(stop_dataset_ready_consumer())

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

