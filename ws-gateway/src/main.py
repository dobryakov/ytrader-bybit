"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .api.health import router as health_router
from .api.middleware import APIKeyAuthMiddleware, RequestLoggingMiddleware
from .api.middleware.security import SecurityMiddleware
from .api.v1 import balances_router, subscriptions_router
from .config.logging import get_logger, setup_logging
from .config.settings import settings
from .utils.tracing import generate_trace_id, set_trace_id
from .services.database.connection import DatabaseConnection
from .services.queue.connection import QueueConnection
from .services.queue.publisher import get_publisher, close_publisher
from .services.queue.setup import setup_queues
from .services.queue.retention import start_monitoring, stop_monitoring
from .services.queue.monitoring import start_backlog_monitoring, stop_backlog_monitoring
from .services.websocket.connection import get_connection
from .services.websocket.connection_manager import get_connection_manager
from .services.websocket.heartbeat import HeartbeatManager
from .services.websocket.reconnection import ReconnectionManager
from .services.websocket.stream_monitor import StreamActivityMonitor
from .services.subscription.subscription_monitor import SubscriptionMonitor

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("application_starting", service=settings.ws_gateway_service_name)

    # Initialize WebSocket connection components
    websocket_connection = None
    reconnection_manager = None
    heartbeat_manager = None
    subscription_monitor = None
    stream_activity_monitor = None

    try:
        # Initialize database connection pool
        await DatabaseConnection.create_pool()
        logger.info("database_pool_initialized")

        # Initialize RabbitMQ connection
        await QueueConnection.create_connection()
        logger.info("rabbitmq_connection_initialized")

        # Setup queues (durability, retention)
        await setup_queues()
        logger.info("queues_initialized")

        # Initialize queue publisher
        await get_publisher()
        logger.info("queue_publisher_initialized")

        # Start queue retention monitoring
        await start_monitoring()
        logger.info("queue_retention_monitoring_started")

        # Start queue backlog monitoring (EC7: Monitor slow subscriber consumption)
        await start_backlog_monitoring()
        logger.info("queue_backlog_monitoring_started")

        # Start subscription monitoring (monitor stale subscriptions)
        subscription_monitor = SubscriptionMonitor()
        await subscription_monitor.start()
        logger.info("subscription_monitoring_started")

        # Start stream activity monitoring (monitor last_message_at and trigger reconnects)
        stream_activity_monitor = StreamActivityMonitor()
        await stream_activity_monitor.start()
        logger.info("stream_activity_monitoring_started")

        # Initialize WebSocket connection
        websocket_connection = get_connection()
        reconnection_manager = ReconnectionManager(websocket_connection)
        heartbeat_manager = HeartbeatManager(websocket_connection)

        # Register disconnection callback with connection
        websocket_connection.set_disconnection_callback(
            reconnection_manager.handle_disconnection
        )

        # Start reconnection manager
        await reconnection_manager.start()

        # Attempt initial connection
        try:
            await websocket_connection.connect()
            # Start heartbeat after successful connection
            await heartbeat_manager.start()
            # Resubscribe to active subscriptions after initial connection (private)
            await reconnection_manager._resubscribe_for_connection(websocket_connection)
            
            # Also resubscribe to public subscriptions if any exist
            # Public connection will be created lazily when needed, and reconnection
            # manager will be set up automatically by ConnectionManager
            connection_manager = get_connection_manager()
            try:
                public_connection = await connection_manager.get_public_connection()
                if public_connection and public_connection.is_connected:
                    # Get the reconnection manager that was automatically set up
                    public_reconnect_manager = connection_manager.get_public_reconnection_manager()
                    if public_reconnect_manager:
                        await public_reconnect_manager._resubscribe_for_connection(public_connection)
                        logger.info("public_connection_resubscribed_on_startup")
            except Exception as e:
                logger.warning(
                    "public_connection_resubscribe_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Non-critical, public connection will be created when subscription is requested
            
            logger.info("websocket_connection_initialized")
        except Exception as e:
            trace_id = generate_trace_id()
            set_trace_id(trace_id)
            logger.warning(
                "websocket_initial_connection_failed",
                error=str(e),
                error_type=type(e).__name__,
                environment=settings.bybit_environment,
                trace_id=trace_id,
                exc_info=True,
            )
            # Reconnection manager will handle retries
            await reconnection_manager.handle_disconnection()

        logger.info("application_started", port=settings.ws_gateway_port)
    except Exception as e:
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
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
    logger.info("application_shutting_down")

    try:
        # Stop heartbeat manager
        if heartbeat_manager:
            await heartbeat_manager.stop()
            logger.info("heartbeat_manager_stopped")

        # Stop reconnection manager
        if reconnection_manager:
            await reconnection_manager.stop()
            logger.info("reconnection_manager_stopped")

        # Close WebSocket connection
        if websocket_connection:
            await websocket_connection.disconnect()
            logger.info("websocket_connection_closed")

        # Stop subscription monitoring
        if subscription_monitor:
            await subscription_monitor.stop()
            logger.info("subscription_monitoring_stopped")

        # Stop stream activity monitoring
        if stream_activity_monitor:
            await stream_activity_monitor.stop()
            logger.info("stream_activity_monitoring_stopped")

        # Stop queue backlog monitoring
        await stop_backlog_monitoring()
        logger.info("queue_backlog_monitoring_stopped")

        # Stop queue retention monitoring
        await stop_monitoring()
        logger.info("queue_retention_monitoring_stopped")

        # Close queue publisher
        await close_publisher()
        logger.info("queue_publisher_closed")

        # Close RabbitMQ connection
        await QueueConnection.close_connection()
        logger.info("rabbitmq_connection_closed")

        # Close database connection pool
        await DatabaseConnection.close_pool()
        logger.info("database_pool_closed")

        logger.info("application_shutdown_complete")
    except Exception as e:
        trace_id = generate_trace_id()
        set_trace_id(trace_id)
        logger.error(
            "application_shutdown_error",
            error=str(e),
            error_type=type(e).__name__,
            trace_id=trace_id,
            exc_info=True,
        )


# Create FastAPI application
app = FastAPI(
    title="WebSocket Gateway Service",
    description="WebSocket Gateway for Bybit Data Aggregation and Routing",
    version="1.0.0",
    lifespan=lifespan,
)

#
# Middleware (security first, then logging, then auth)
#
app.add_middleware(SecurityMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyAuthMiddleware)

#
# Routers
#
app.include_router(health_router)
app.include_router(subscriptions_router)
app.include_router(balances_router)

# Temporary test routers for subscriptions and data viewing
try:
    from .api.test_subscribe import router as test_subscribe_router

    app.include_router(test_subscribe_router)
except ImportError:
    pass

try:
    from .api.view_data import router as view_data_router

    app.include_router(view_data_router)
except ImportError:
    pass


#
# Exception handlers
#
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors, returning 404 for suspicious paths."""
    path = request.url.path
    
    # Check if path looks suspicious (even if normalized by uvicorn)
    suspicious_patterns = ["/etc/passwd", "/etc/shadow", "/proc/", "/sys/", "/dev/", "passwd", "shadow"]
    if any(suspicious in path.lower() for suspicious in suspicious_patterns):
        logger.warning(
            "Suspicious path in validation error - returning 404",
            path=path,
            client_ip=request.client.host if request.client else None,
        )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Not found"}
        )
    
    # For other validation errors, return standard 422
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions, returning 404 for suspicious paths."""
    path = request.url.path
    
    # Check if path looks suspicious (even if normalized by uvicorn)
    suspicious_patterns = ["/etc/passwd", "/etc/shadow", "/proc/", "/sys/", "/dev/", "passwd", "shadow"]
    if any(suspicious in path.lower() for suspicious in suspicious_patterns):
        logger.warning(
            "Suspicious path in exception - returning 404",
            path=path,
            client_ip=request.client.host if request.client else None,
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": "Not found"}
        )
    
    # For other exceptions, log and return 500
    logger.error(
        "Unhandled exception",
        path=path,
        client_ip=request.client.host if request.client else None,
        error_type=type(exc).__name__,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

