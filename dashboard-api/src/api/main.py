"""FastAPI application setup and configuration."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .middleware.auth import APIKeyAuthMiddleware
from ..config.settings import settings
from ..config.logging import configure_logging, get_logger
from ..config.database import DatabaseConnection
from ..exceptions import DashboardAPIError, DatabaseError
from .routes import positions, orders, signals, models, metrics, charts, datasets

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("dashboard_api_starting")
    try:
        await DatabaseConnection.create_pool()
        logger.info("dashboard_api_started")
    except Exception as e:
        logger.error("dashboard_api_startup_failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("dashboard_api_shutting_down")
    await DatabaseConnection.close_pool()
    logger.info("dashboard_api_shutdown_complete")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Dashboard API",
        description="API for trading system dashboard",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    # Note: allow_origins=["*"] and allow_credentials=True cannot be used together
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=False,  # Must be False when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Add API key authentication middleware
    app.add_middleware(APIKeyAuthMiddleware, api_prefix="/api")

    # Register global exception handlers
    @app.exception_handler(DashboardAPIError)
    async def dashboard_error_handler(request: Request, exc: DashboardAPIError):
        """Handle DashboardAPIError exceptions with consistent error response."""
        logger.error(
            "dashboard_api_error",
            error_type=type(exc).__name__,
            message=exc.message,
            trace_id=exc.trace_id,
        )

        status_code_map = {
            DatabaseError: 503,
        }
        status_code = status_code_map.get(type(exc), 500)

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": exc.message,
                }
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        logger.error("unexpected_error", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred",
                }
            },
        )

    # Register routers
    app.include_router(positions.router, prefix="/api/v1", tags=["positions"])
    app.include_router(orders.router, prefix="/api/v1", tags=["orders"])
    app.include_router(signals.router, prefix="/api/v1", tags=["signals"])
    app.include_router(models.router, prefix="/api/v1", tags=["models"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
    app.include_router(charts.router, prefix="/api/v1", tags=["charts"])
    app.include_router(datasets.router, prefix="/api/v1", tags=["datasets"])

    # Health check endpoints
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/live")
    async def live():
        """Liveness probe."""
        return {"status": "alive"}

    @app.get("/ready")
    async def ready():
        """Readiness probe."""
        if DatabaseConnection.is_connected():
            return {"status": "ready"}
        return JSONResponse(status_code=503, content={"status": "not ready"})

    return app


# Create app instance
app = create_app()

