"""
Main application entry point for Feature Service.

Initializes FastAPI application with basic routing and health check.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Feature Service",
    description="Service for real-time feature computation and dataset building",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "feature-service",
        }
    )


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

