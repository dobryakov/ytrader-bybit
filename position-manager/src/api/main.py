"""FastAPI application setup for Position Manager."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI

from ..config.database import DatabaseConnection
from ..config.logging import configure_logging, get_logger
from ..config.rabbitmq import RabbitMQConnection
from ..config.settings import settings
from ..consumers import OrderPositionConsumer, WebSocketPositionConsumer
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
_order_consumer: Optional[OrderPositionConsumer] = None
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

    # Middleware
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
        global _ws_consumer, _order_consumer, _snapshot_task, _validation_task
        try:
            await RabbitMQConnection.create_connection()

            # Subscribe to position updates from ws-gateway
            try:
                ws_gateway_client = WSGatewayClient()
                subscription_success = await ws_gateway_client.subscribe_to_position()
                if subscription_success:
                    logger.info("ws_gateway_position_subscription_completed")
                else:
                    logger.warning(
                        "ws_gateway_position_subscription_failed_non_fatal",
                        message="Position updates may not be received until subscription is created manually",
                    )
            except Exception as e:
                # Non-fatal: subscription can be created manually or retried later
                logger.warning(
                    "ws_gateway_position_subscription_error_non_fatal",
                    error=str(e),
                    error_type=type(e).__name__,
                )

            # Start background consumers for WS and order events.
            _ws_consumer = WebSocketPositionConsumer()
            _order_consumer = OrderPositionConsumer()
            _ws_consumer.spawn()
            _order_consumer.spawn()
            logger.info("position_manager_consumers_started")

            # Run one-off snapshot cleanup on startup (US4 T081).
            cleanup_task = PositionSnapshotCleanupTask()
            await cleanup_task.run_once()

            # Start periodic snapshot task (US4 T077/T077a/T083).
            _snapshot_task = PositionSnapshotTask()
            await _snapshot_task.start()

            # Start periodic validation task (US5 T087/T087a/T091).
            _validation_task = PositionValidationTask()
            await _validation_task.start()

            # Start periodic Bybit sync task
            _bybit_sync_task = PositionBybitSyncTask()
            await _bybit_sync_task.start()
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



