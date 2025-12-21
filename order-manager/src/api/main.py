"""FastAPI application setup and configuration."""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .middleware.auth import APIKeyAuthMiddleware
from .middleware.logging import LoggingMiddleware
from .middleware.security import SecurityMiddleware
from ..config.settings import settings
from ..config.logging import get_logger, configure_logging
from ..exceptions import (
    OrderManagerError,
    ConfigurationError,
    DatabaseError,
    QueueError,
    BybitAPIError,
    OrderExecutionError,
    RiskLimitError,
)
from ..utils.tracing import get_or_create_trace_id

# Configure logging
configure_logging()
logger = get_logger(__name__)


def create_app(lifespan_context=None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        lifespan_context: Optional lifespan context manager for startup/shutdown

    Returns:
        Configured FastAPI application instance
    """
    app_kwargs = {
        "title": "Order Manager API",
        "description": "Order management and execution microservice for trading signals",
        "version": "1.0.0",
    }
    if lifespan_context:
        app_kwargs["lifespan"] = lifespan_context
    
    app = FastAPI(**app_kwargs)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add security middleware first (before other middleware)
    app.add_middleware(SecurityMiddleware)

    # Add request/response logging middleware (before auth middleware to log all requests)
    app.add_middleware(LoggingMiddleware)

    # Add API key authentication middleware
    app.add_middleware(APIKeyAuthMiddleware, api_prefix="/api")

    # Register global exception handlers
    @app.exception_handler(OrderManagerError)
    async def order_manager_error_handler(request: Request, exc: OrderManagerError):
        """Handle OrderManagerError exceptions with consistent error response."""
        trace_id = get_or_create_trace_id()
        logger.error(
            "order_manager_error",
            error_type=type(exc).__name__,
            message=exc.message,
            trace_id=trace_id or exc.trace_id,
            exc_info=True,
        )

        # Map exception types to HTTP status codes
        status_code_map = {
            ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
            DatabaseError: status.HTTP_503_SERVICE_UNAVAILABLE,
            QueueError: status.HTTP_503_SERVICE_UNAVAILABLE,
            BybitAPIError: status.HTTP_502_BAD_GATEWAY,
            OrderExecutionError: status.HTTP_400_BAD_REQUEST,
            RiskLimitError: status.HTTP_400_BAD_REQUEST,
        }

        status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": exc.message,
                    "trace_id": trace_id or exc.trace_id,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTPException with consistent error response."""
        trace_id = get_or_create_trace_id()
        logger.warning(
            "http_exception",
            status_code=exc.status_code,
            detail=exc.detail,
            trace_id=trace_id,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "HTTPException",
                    "message": exc.detail,
                    "trace_id": trace_id,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors with consistent error response."""
        trace_id = get_or_create_trace_id()
        path = request.url.path
        
        # Check if path looks suspicious (even if normalized by uvicorn)
        suspicious_patterns = ["/etc/passwd", "/etc/shadow", "/proc/", "/sys/", "/dev/", "passwd", "shadow"]
        if any(suspicious in path.lower() for suspicious in suspicious_patterns):
            logger.warning(
                "Suspicious path in validation error - returning 404",
                path=path,
                client_ip=request.client.host if request.client else None,
                trace_id=trace_id,
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": {
                        "type": "NotFound",
                        "message": "Not found",
                        "trace_id": trace_id,
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
            )
        
        logger.warning(
            "validation_error",
            errors=exc.errors(),
            trace_id=trace_id,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "type": "ValidationError",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "trace_id": trace_id,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions with consistent error response."""
        trace_id = get_or_create_trace_id()
        path = request.url.path
        
        # Check if path looks suspicious (even if normalized by uvicorn)
        suspicious_patterns = ["/etc/passwd", "/etc/shadow", "/proc/", "/sys/", "/dev/", "passwd", "shadow"]
        if any(suspicious in path.lower() for suspicious in suspicious_patterns):
            logger.warning(
                "Suspicious path in exception - returning 404",
                path=path,
                client_ip=request.client.host if request.client else None,
                error_type=type(exc).__name__,
                trace_id=trace_id,
            )
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": {
                        "type": "NotFound",
                        "message": "Not found",
                        "trace_id": trace_id,
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
            )
        
        logger.error(
            "unexpected_error",
            error_type=type(exc).__name__,
            error=str(exc),
            trace_id=trace_id,
            exc_info=True,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "type": "InternalServerError",
                    "message": "An unexpected error occurred",
                    "trace_id": trace_id,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )

    # Register routes
    from .routes import health, orders, positions, sync

    app.include_router(health.router, tags=["health"])
    app.include_router(orders.router, prefix="/api/v1", tags=["orders"])
    app.include_router(positions.router, prefix="/api/v1", tags=["positions"])
    app.include_router(sync.router, prefix="/api/v1", tags=["sync"])

    return app


# Create app instance
app = create_app()

