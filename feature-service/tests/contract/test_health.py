"""
Contract tests for health check endpoint.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app


class TestHealthEndpoint:
    """Contract tests for health check endpoint."""
    
    def test_health_endpoint_returns_200(self):
        """Test that health endpoint returns 200 status."""
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.status_code == 200
    
    def test_health_endpoint_returns_json(self):
        """Test that health endpoint returns JSON response."""
        client = TestClient(app)
        
        response = client.get("/health")
        
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert isinstance(data, dict)
    
    def test_health_endpoint_includes_status(self):
        """Test that health endpoint includes status field."""
        client = TestClient(app)
        
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy", "degraded"]
    
    def test_health_endpoint_includes_service_name(self):
        """Test that health endpoint includes service name."""
        client = TestClient(app)
        
        response = client.get("/health")
        data = response.json()
        
        assert "service" in data
        assert data["service"] == "feature-service"
    
    def test_health_endpoint_includes_timestamp(self):
        """Test that health endpoint includes timestamp."""
        client = TestClient(app)
        
        response = client.get("/health")
        data = response.json()
        
        # Timestamp may be in different formats, just check it exists
        # (adjust based on actual implementation)
        assert "timestamp" in data or "created_at" in data
    
    def test_health_endpoint_does_not_require_auth(self):
        """Test that health endpoint does not require authentication."""
        client = TestClient(app)
        
        # Should not require API key
        response = client.get("/health")
        
        assert response.status_code == 200

