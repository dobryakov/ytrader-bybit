"""Integration tests for Grafana container and health endpoint."""

import pytest
import httpx
import os
from typing import Optional


def get_grafana_url() -> str:
    """Get Grafana URL based on environment (Docker or local)."""
    # In Docker, use service name; locally, use localhost
    grafana_host = os.getenv("GRAFANA_HOST", "localhost")
    grafana_port = os.getenv("GRAFANA_PORT", "4700")
    
    # If running in Docker container, use service name
    if os.getenv("DOCKER_ENV") == "true" or grafana_host != "localhost":
        # In Docker, Grafana runs on port 3000 internally
        return f"http://grafana:3000"
    
    return f"http://{grafana_host}:{grafana_port}"


@pytest.mark.integration
def test_grafana_container_starts():
    """
    Test that Grafana container starts successfully.
    
    This test verifies:
    - Grafana container is running
    - Health endpoint is accessible
    """
    grafana_url = get_grafana_url()
    
    # Test health endpoint
    response = httpx.get(f"{grafana_url}/api/health", timeout=10.0)
    
    assert response.status_code == 200, f"Grafana health endpoint returned {response.status_code}"
    
    health_data = response.json()
    assert "database" in health_data or "status" in health_data, "Health response missing expected fields"


@pytest.mark.integration
def test_grafana_health_endpoint_responds():
    """
    Test that Grafana /api/health endpoint responds correctly.
    
    This test verifies:
    - Health endpoint returns 200 OK
    - Response contains expected health status information
    """
    grafana_url = get_grafana_url()
    
    try:
        response = httpx.get(f"{grafana_url}/api/health", timeout=10.0)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        # Grafana health endpoint returns JSON with status information
        health_data = response.json()
        assert isinstance(health_data, dict), "Health response should be a JSON object"
        
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")
    except httpx.TimeoutException as e:
        pytest.fail(f"Grafana health endpoint timed out: {e}")


@pytest.mark.integration
def test_grafana_api_accessible():
    """
    Test that Grafana API is accessible (requires authentication for most endpoints).
    
    This test verifies:
    - Grafana API base endpoint is reachable
    - API responds (may require authentication for full access)
    """
    grafana_url = get_grafana_url()
    
    # Test API root endpoint (may require auth, but should not return connection error)
    try:
        response = httpx.get(f"{grafana_url}/api/health", timeout=10.0)
        # Health endpoint should be accessible without auth
        assert response.status_code in [200, 401], f"Unexpected status code: {response.status_code}"
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana API at {grafana_url}: {e}")

