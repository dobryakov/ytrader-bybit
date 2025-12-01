"""Integration tests for REST API endpoints."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from decimal import Decimal

from fastapi.testclient import TestClient
from src.api.main import create_app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint returns correct format."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data
    assert "database_connected" in data
    assert "queue_connected" in data
    assert "timestamp" in data
    assert data["service"] == "order-manager"


def test_live_endpoint(client):
    """Test liveness probe endpoint."""
    response = client.get("/live")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_ready_endpoint(client):
    """Test readiness probe endpoint."""
    response = client.get("/ready")
    
    # Should return 200 or 503 depending on dependencies
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "dependencies" in data
    assert "timestamp" in data


def test_orders_endpoint_unauthorized(client):
    """Test orders endpoint requires authentication."""
    response = client.get("/api/v1/orders")
    
    # Should require API key
    assert response.status_code == 401


def test_positions_endpoint_unauthorized(client):
    """Test positions endpoint requires authentication."""
    response = client.get("/api/v1/positions")
    
    # Should require API key
    assert response.status_code == 401


def test_positions_endpoint_exists(client):
    """Test positions endpoint exists (requires authentication)."""
    # Without API key, should return 401
    response = client.get("/api/v1/positions")
    
    # Should require API key authentication
    assert response.status_code == 401


def test_sync_endpoint_unauthorized(client):
    """Test sync endpoint requires authentication."""
    response = client.post("/api/v1/sync", json={"scope": "active"})
    
    # Should require API key
    assert response.status_code == 401

