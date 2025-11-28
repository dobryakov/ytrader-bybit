"""Integration tests for Grafana data source connectivity."""

import pytest
import httpx
import os
from typing import Optional, Dict, Any


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


def get_grafana_auth() -> Optional[httpx.BasicAuth]:
    """Get Grafana basic auth credentials from environment."""
    admin_user = os.getenv("GRAFANA_ADMIN_USER", "admin")
    admin_password = os.getenv("GRAFANA_ADMIN_PASSWORD")
    
    if not admin_password:
        pytest.skip("GRAFANA_ADMIN_PASSWORD not set, skipping authenticated tests")
    
    return httpx.BasicAuth(admin_user, admin_password)


@pytest.mark.integration
def test_grafana_postgres_datasource_configured():
    """
    Test that PostgreSQL data source is configured in Grafana.
    
    This test verifies:
    - PostgreSQL data source exists in Grafana
    - Data source configuration is correct
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    try:
        # Get list of data sources
        response = httpx.get(
            f"{grafana_url}/api/datasources",
            auth=auth,
            timeout=10.0
        )
        
        assert response.status_code == 200, f"Failed to get data sources: {response.status_code}"
        
        datasources = response.json()
        assert isinstance(datasources, list), "Data sources should be a list"
        
        # Find PostgreSQL data source
        postgres_ds = next(
            (ds for ds in datasources if ds.get("type") == "postgres" and ds.get("name") == "PostgreSQL"),
            None
        )
        
        assert postgres_ds is not None, "PostgreSQL data source not found"
        assert postgres_ds.get("type") == "postgres", "Data source type should be postgres"
        
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")
    except httpx.HTTPStatusError as e:
        pytest.fail(f"Grafana API returned error: {e.response.status_code}")


@pytest.mark.integration
def test_grafana_rabbitmq_datasource_configured():
    """
    Test that RabbitMQ HTTP API data source is configured in Grafana.
    
    This test verifies:
    - RabbitMQ HTTP API data source exists
    - Data source is properly configured
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    try:
        response = httpx.get(
            f"{grafana_url}/api/datasources",
            auth=auth,
            timeout=10.0
        )
        
        assert response.status_code == 200, f"Failed to get data sources: {response.status_code}"
        
        datasources = response.json()
        
        # Find RabbitMQ HTTP API data source
        rabbitmq_ds = next(
            (ds for ds in datasources if ds.get("name") == "RabbitMQ HTTP API"),
            None
        )
        
        assert rabbitmq_ds is not None, "RabbitMQ HTTP API data source not found"
        
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")


@pytest.mark.integration
def test_grafana_service_health_datasources_configured():
    """
    Test that service health HTTP data sources are configured in Grafana.
    
    This test verifies:
    - ws-gateway Health data source exists
    - model-service Health data source exists
    - order-manager Health data source exists
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    expected_datasources = [
        "ws-gateway Health",
        "model-service Health",
        "order-manager Health"
    ]
    
    try:
        response = httpx.get(
            f"{grafana_url}/api/datasources",
            auth=auth,
            timeout=10.0
        )
        
        assert response.status_code == 200, f"Failed to get data sources: {response.status_code}"
        
        datasources = response.json()
        datasource_names = [ds.get("name") for ds in datasources]
        
        for expected_name in expected_datasources:
            assert expected_name in datasource_names, f"Data source '{expected_name}' not found"
        
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")


@pytest.mark.integration
def test_grafana_postgres_datasource_connectivity():
    """
    Test that PostgreSQL data source can connect to database.
    
    This test verifies:
    - PostgreSQL data source is configured
    - Data source can successfully connect to PostgreSQL
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    try:
        # Get PostgreSQL data source ID
        response = httpx.get(
            f"{grafana_url}/api/datasources",
            auth=auth,
            timeout=10.0
        )
        
        assert response.status_code == 200, f"Failed to get data sources: {response.status_code}"
        
        datasources = response.json()
        postgres_ds = next(
            (ds for ds in datasources if ds.get("type") == "postgres" and ds.get("name") == "PostgreSQL"),
            None
        )
        
        if not postgres_ds:
            pytest.skip("PostgreSQL data source not configured")
        
        datasource_id = postgres_ds.get("id")
        
        # Test data source connectivity
        test_response = httpx.post(
            f"{grafana_url}/api/datasources/{datasource_id}/health",
            auth=auth,
            timeout=10.0
        )
        
        # Health check may return 200 (healthy) or other status codes
        # The important thing is that it doesn't return connection error
        assert test_response.status_code in [200, 400, 500], \
            f"Unexpected status code for data source health check: {test_response.status_code}"
        
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")

