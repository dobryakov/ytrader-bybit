"""Service entry point for Order Manager microservice.

Handles service startup, dependency initialization, and graceful shutdown.
"""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI

from .api.main import create_app
from .config.database import DatabaseConnection
from .config.rabbitmq import RabbitMQConnection
from .config.logging import get_logger, configure_logging
from .utils.bybit_client import get_bybit_client, close_bybit_client
from .utils.tracing import generate_trace_id, set_trace_id
from .config.settings import settings
from .consumers.signal_consumer import SignalConsumer
from .services.event_subscriber import EventSubscriber
from .services.order_state_sync import OrderStateSync
from .services.position_manager import PositionManager

# Configure logging first
configure_logging()
logger = get_logger(__name__)


class PositionSnapshotTask:
    """Background task for periodic position snapshots."""

    def __init__(self):
        """Initialize snapshot task."""
        self._task: Optional[asyncio.Task] = None
        self._should_run = False
        self._position_manager = PositionManager()

    async def start(self) -> None:
        """Start the snapshot task."""
        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._snapshot_loop())
        logger.info(
            "position_snapshot_task_started",
            interval=settings.order_manager_position_snapshot_interval,
        )

    async def stop(self) -> None:
        """Stop the snapshot task."""
        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("position_snapshot_task_stopped")

    async def _snapshot_loop(self) -> None:
        """Continuously create position snapshots at regular intervals."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                await asyncio.sleep(settings.order_manager_position_snapshot_interval)

                if not self._should_run:
                    break

                # Get all positions
                positions = await self._position_manager.get_all_positions()

                # Create snapshots for all positions
                snapshot_count = 0
                for position in positions:
                    try:
                        await self._position_manager.create_position_snapshot(
                            position, trace_id=trace_id
                        )
                        snapshot_count += 1
                    except Exception as e:
                        logger.error(
                            "position_snapshot_failed",
                            position_id=str(position.id),
                            asset=position.asset,
                            error=str(e),
                            trace_id=trace_id,
                        )

                if snapshot_count > 0:
                    logger.info(
                        "position_snapshots_created",
                        count=snapshot_count,
                        trace_id=trace_id,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "position_snapshot_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Continue loop even if error occurs
                await asyncio.sleep(60)  # Wait before retrying


class PositionValidationTask:
    """Background task for periodic position validation."""

    def __init__(self):
        """Initialize validation task."""
        self._task: Optional[asyncio.Task] = None
        self._should_run = False
        self._position_manager = PositionManager()

    async def start(self) -> None:
        """Start the validation task."""
        self._should_run = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._validation_loop())
        logger.info(
            "position_validation_task_started",
            interval=settings.order_manager_position_validation_interval,
        )

    async def stop(self) -> None:
        """Stop the validation task."""
        self._should_run = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("position_validation_task_stopped")

    async def _validation_loop(self) -> None:
        """Continuously validate positions at regular intervals."""
        trace_id = generate_trace_id()
        set_trace_id(trace_id)

        while self._should_run:
            try:
                await asyncio.sleep(settings.order_manager_position_validation_interval)

                if not self._should_run:
                    break

                # Get all positions
                positions = await self._position_manager.get_all_positions()

                # Validate all positions
                validated_count = 0
                fixed_count = 0
                error_count = 0

                for position in positions:
                    try:
                        is_valid, error_msg, updated_position = (
                            await self._position_manager.validate_position(
                                position.asset, position.mode, fix_discrepancies=True
                            )
                        )

                        if is_valid:
                            validated_count += 1
                            if updated_position is not None:
                                fixed_count += 1
                                logger.info(
                                    "position_discrepancy_fixed",
                                    asset=position.asset,
                                    mode=position.mode,
                                    trace_id=trace_id,
                                )
                        else:
                            error_count += 1
                            logger.warning(
                                "position_validation_failed",
                                asset=position.asset,
                                mode=position.mode,
                                error=error_msg,
                                trace_id=trace_id,
                            )

                    except Exception as e:
                        error_count += 1
                        logger.error(
                            "position_validation_error",
                            asset=position.asset,
                            mode=position.mode,
                            error=str(e),
                            trace_id=trace_id,
                        )

                logger.info(
                    "position_validation_completed",
                    validated_count=validated_count,
                    fixed_count=fixed_count,
                    error_count=error_count,
                    total_positions=len(positions),
                    trace_id=trace_id,
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "position_validation_loop_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    trace_id=trace_id,
                    exc_info=True,
                )
                # Continue loop even if error occurs
                await asyncio.sleep(60)  # Wait before retrying


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

        # Initialize and start signal consumer
        signal_consumer = SignalConsumer()
        await signal_consumer.start()
        logger.info("signal_consumer_started", trace_id=trace_id)

        # Store consumer for shutdown
        app.state.signal_consumer = signal_consumer

        # Perform startup reconciliation - sync active orders with Bybit
        order_state_sync = OrderStateSync()
        try:
            sync_result = await order_state_sync.sync_active_orders(trace_id=trace_id)
            logger.info(
                "startup_reconciliation_completed",
                synced_count=sync_result["synced_count"],
                discrepancies_count=len(sync_result["discrepancies"]),
                trace_id=trace_id,
            )
        except Exception as e:
            logger.warning(
                "startup_reconciliation_failed",
                error=str(e),
                trace_id=trace_id,
            )
            # Don't fail startup if reconciliation fails - service can still operate

        # Initialize and start event subscriber for order execution events
        event_subscriber = EventSubscriber()
        await event_subscriber.start()
        logger.info("event_subscriber_started", trace_id=trace_id)

        # Store event subscriber for shutdown
        app.state.event_subscriber = event_subscriber

        # Start background tasks (position snapshots, validation)
        snapshot_task = PositionSnapshotTask()
        await snapshot_task.start()
        app.state.snapshot_task = snapshot_task
        logger.info("position_snapshot_task_started", trace_id=trace_id)

        validation_task = PositionValidationTask()
        await validation_task.start()
        app.state.validation_task = validation_task
        logger.info("position_validation_task_started", trace_id=trace_id)

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
        # Stop position validation task
        if hasattr(app.state, "validation_task"):
            await app.state.validation_task.stop()
            logger.info("position_validation_task_stopped")

        # Stop position snapshot task
        if hasattr(app.state, "snapshot_task"):
            await app.state.snapshot_task.stop()
            logger.info("position_snapshot_task_stopped")

        # Stop event subscriber
        if hasattr(app.state, "event_subscriber"):
            await app.state.event_subscriber.stop()
            logger.info("event_subscriber_stopped")

        # Stop signal consumer
        if hasattr(app.state, "signal_consumer"):
            await app.state.signal_consumer.stop()
            logger.info("signal_consumer_stopped")

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

