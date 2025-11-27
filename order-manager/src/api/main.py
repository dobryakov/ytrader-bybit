"""FastAPI application setup and configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware.auth import APIKeyAuthMiddleware
from ..config.settings import settings
from ..config.logging import get_logger, configure_logging

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

    # Add API key authentication middleware
    app.add_middleware(APIKeyAuthMiddleware, api_prefix="/api")

    # Register routes will be added here
    # from .routes import health, orders, positions, sync
    # app.include_router(health.router, tags=["health"])
    # app.include_router(orders.router, prefix="/api/v1", tags=["orders"])
    # app.include_router(positions.router, prefix="/api/v1", tags=["positions"])
    # app.include_router(sync.router, prefix="/api/v1", tags=["sync"])

    return app


# Create app instance
app = create_app()

