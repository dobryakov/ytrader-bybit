"""
Main application entry point for Feature Service.

Initializes FastAPI application with basic routing and health check.
"""

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from src.api.health import router as health_router
from src.api.middleware.auth import verify_api_key
from src.logging import setup_logging, get_logger
from src.config import config

# Setup logging
setup_logging(level=config.feature_service_log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Feature Service",
    description="Service for real-time feature computation and dataset building",
    version="0.1.0",
)

# Include routers
app.include_router(health_router)

# Add authentication middleware to all routes except health
@app.middleware("http")
async def auth_middleware(request, call_next):
    """Authentication middleware."""
    try:
        await verify_api_key(request)
    except Exception:
        # Health endpoint is allowed without auth
        if request.url.path not in ["/health", "/", "/docs", "/openapi.json"]:
            raise
    response = await call_next(request)
    return response


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        content={
            "service": "feature-service",
            "version": "0.1.0",
            "status": "running",
        }
    )


@app.on_event("startup")
async def startup():
    """Application startup event."""
    logger.info("Feature Service starting up")


@app.on_event("shutdown")
async def shutdown():
    """Application shutdown event."""
    logger.info("Feature Service shutting down")

