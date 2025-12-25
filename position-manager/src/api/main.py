"""FastAPI application setup for Position Manager."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI

from ..config.database import DatabaseConnection
from ..config.logging import configure_logging, get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..config.settings import settings
from ..consumers import PositionOrderLinkerConsumer, WebSocketPositionConsumer
from ..exceptions import QueueError
from ..services.ws_gateway_client import WSGatewayClient
from ..tasks import (
    PositionBybitSyncTask,
    PositionSnapshotCleanupTask,
    PositionSnapshotTask,
    PositionValidationTask,
)
from .error_handlers import register_error_handlers
from .middleware.logging import logging_middleware
from .middleware.security import security_middleware
from .routes.portfolio import get_portfolio_manager
from .routes import health as health_routes
from .routes import portfolio as portfolio_routes
from .routes import positions as positions_routes
from .routes import metrics as metrics_routes
from .routes import stats as stats_routes


configure_logging()
logger = get_logger(__name__)

# Global instances so we can start/stop background components with the app lifecycle.
_ws_consumer: Optional[WebSocketPositionConsumer] = None
_order_consumer: Optional[PositionOrderLinkerConsumer] = None
_snapshot_task: Optional[PositionSnapshotTask] = None
_validation_task: Optional[PositionValidationTask] = None
_bybit_sync_task: Optional[PositionBybitSyncTask] = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Position Manager",
        version="1.0.0",
    )

    # Global error handlers
    register_error_handlers(app)

    # Middleware (security first, then logging)
    app.middleware("http")(security_middleware)
    app.middleware("http")(logging_middleware)

    # Routes
    app.include_router(health_routes.router)
    app.include_router(metrics_routes.router)
    app.include_router(positions_routes.router)
    app.include_router(portfolio_routes.router)
    app.include_router(stats_routes.router)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("app_startup_begin", service=settings.position_manager_service_name)
        # Initialize database connection (hard requirement)
        await DatabaseConnection.create_pool()

        # Warm portfolio metrics cache for default view to reduce latency
        # on first risk-management queries (T068).
        try:
            pm = get_portfolio_manager()
            await pm.get_portfolio_metrics(include_positions=False)
            logger.info("portfolio_metrics_cache_warm_completed")
        except Exception:  # pragma: no cover - best-effort
            logger.warning("portfolio_metrics_cache_warm_failed")

        # Initialize RabbitMQ connection, but do not fail startup if it's temporarily unavailable.
        # Health and consumers will reflect queue connectivity separately.
        global _ws_consumer, _order_consumer, _snapshot_task, _validation_task, _bybit_sync_task
        
        # Try to connect to RabbitMQ
        try:
            await RabbitMQConnection.create_connection()
        except QueueError as e:
            logger.error(
                "rabbitmq_startup_connection_failed_non_fatal",
                error=str(e),
                host=settings.rabbitmq_host,
                port=settings.rabbitmq_port,
            )

        # Subscribe to position updates from ws-gateway (non-fatal if fails)
        try:
            ws_gateway_client = WSGatewayClient()
            await ws_gateway_client.subscribe_to_position()
            logger.info("ws_gateway_position_subscription_completed")
        except QueueError as e:
            logger.warning(
                "ws_gateway_subscription_failed_non_fatal",
                error=str(e),
                url=settings.ws_gateway_url,
            )

        # Start background consumers for WS and order events (if RabbitMQ is available)
        try:
            _ws_consumer = WebSocketPositionConsumer()
            _order_consumer = PositionOrderLinkerConsumer()
            _ws_consumer.spawn()
            _order_consumer.spawn()
            logger.info("position_manager_consumers_started")
        except Exception as e:
            logger.warning(
                "position_manager_consumers_start_failed_non_fatal",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Run one-off snapshot cleanup on startup (US4 T081).
        try:
            cleanup_task = PositionSnapshotCleanupTask()
            await cleanup_task.run_once()
        except Exception as e:
            logger.warning(
                "position_snapshot_cleanup_failed_non_fatal",
                error=str(e),
                error_type=type(e).__name__,
            )

        # Start periodic snapshot task (US4 T077/T077a/T083).
        try:
            _snapshot_task = PositionSnapshotTask()
            await _snapshot_task.start()
        except Exception as e:
            logger.error(
                "position_snapshot_task_start_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

        # Start periodic validation task (US5 T087/T087a/T091).
        try:
            _validation_task = PositionValidationTask()
            await _validation_task.start()
        except Exception as e:
            logger.error(
                "position_validation_task_start_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

        # Start periodic Bybit sync task (critical - must start even if other components fail)
        try:
            _bybit_sync_task = PositionBybitSyncTask()
            await _bybit_sync_task.start()
        except Exception as e:
            logger.error(
                "position_bybit_sync_task_start_failed",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
        logger.info("app_startup_completed", service=settings.position_manager_service_name)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("app_shutdown_begin", service=settings.position_manager_service_name)

        # Stop consumers and background tasks first so they no longer use connections.
        global _ws_consumer, _order_consumer, _snapshot_task, _validation_task, _bybit_sync_task
        if _bybit_sync_task is not None:
            await _bybit_sync_task.stop()
            _bybit_sync_task = None
        if _validation_task is not None:
            await _validation_task.stop()
            _validation_task = None
        if _snapshot_task is not None:
            await _snapshot_task.stop()
            _snapshot_task = None
        if _ws_consumer is not None:
            await _ws_consumer.stop()
            _ws_consumer = None
        if _order_consumer is not None:
            await _order_consumer.stop()
            _order_consumer = None

        await DatabaseConnection.close_pool()
        await RabbitMQConnection.close_connection()
        logger.info("app_shutdown_completed", service=settings.position_manager_service_name)

    return app


app = create_app()



