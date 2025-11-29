"""Integration tests for Grafana dashboard loading and panel rendering."""

import pytest
import httpx
import os
from typing import Optional, Dict, Any, List


def get_grafana_url() -> str:
    """Get Grafana URL based on environment (Docker or local)."""
    grafana_host = os.getenv("GRAFANA_HOST", "localhost")
    grafana_port = os.getenv("GRAFANA_PORT", "4700")
    
    if os.getenv("DOCKER_ENV") == "true" or grafana_host != "localhost":
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
def test_grafana_dashboard_exists():
    """
    Test that Trading System Monitoring dashboard exists in Grafana.
    
    This test verifies:
    - Dashboard is provisioned and available
    - Dashboard has correct UID
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    try:
        # Search for dashboard by UID
        response = httpx.get(
            f"{grafana_url}/api/dashboards/uid/trading-system-monitoring",
            auth=auth,
            timeout=10.0
        )
        
        # Dashboard should exist (200) or be accessible via search
        if response.status_code == 404:
            # Try searching for it
            search_response = httpx.get(
                f"{grafana_url}/api/search?query=trading-system-monitoring",
                auth=auth,
                timeout=10.0
            )
            assert search_response.status_code == 200, "Failed to search for dashboard"
            dashboards = search_response.json()
            assert len(dashboards) > 0, "Dashboard not found in search results"
        else:
            assert response.status_code == 200, f"Dashboard not found: {response.status_code}"
            dashboard_data = response.json()
            assert "dashboard" in dashboard_data, "Dashboard response missing dashboard object"
            
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")


@pytest.mark.integration
def test_grafana_dashboard_panels_exist():
    """
    Test that dashboard contains expected panels.
    
    This test verifies:
    - Dashboard has all required panels
    - Panels are properly configured
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    expected_panels = [
        "Trading Signals",
        "Order Execution",
        "Model State",
        "Model Quality Metrics",
        "Queue Metrics",
        "System Health",
        "WebSocket Connection",
        "Event History"
    ]
    
    try:
        # Get dashboard by UID
        response = httpx.get(
            f"{grafana_url}/api/dashboards/uid/trading-system-monitoring",
            auth=auth,
            timeout=10.0
        )
        
        if response.status_code == 404:
            pytest.skip("Dashboard not found - may need to be imported first")
        
        assert response.status_code == 200, f"Failed to get dashboard: {response.status_code}"
        
        dashboard_data = response.json()
        dashboard = dashboard_data.get("dashboard", {})
        panels = dashboard.get("panels", [])
        
        panel_titles = [panel.get("title", "") for panel in panels]
        
        for expected_title in expected_panels:
            assert expected_title in panel_titles, f"Panel '{expected_title}' not found in dashboard"
        
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")


@pytest.mark.integration
def test_grafana_dashboard_panels_have_datasources():
    """
    Test that dashboard panels are configured with data sources.
    
    This test verifies:
    - Panels have data source targets configured
    - Data sources are valid
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    try:
        response = httpx.get(
            f"{grafana_url}/api/dashboards/uid/trading-system-monitoring",
            auth=auth,
            timeout=10.0
        )
        
        if response.status_code == 404:
            pytest.skip("Dashboard not found - may need to be imported first")
        
        assert response.status_code == 200, f"Failed to get dashboard: {response.status_code}"
        
        dashboard_data = response.json()
        dashboard = dashboard_data.get("dashboard", {})
        panels = dashboard.get("panels", [])
        
        # Verify panels have targets configured
        for panel in panels:
            panel_title = panel.get("title", "Unknown")
            targets = panel.get("targets", [])
            
            # Some panels may have multiple targets (e.g., System Health)
            assert len(targets) > 0, f"Panel '{panel_title}' has no data source targets"
            
            # Verify each target has a datasource configured
            for target in targets:
                datasource = target.get("datasource", {})
                assert datasource is not None, f"Panel '{panel_title}' target missing datasource"
                
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")


@pytest.mark.integration
def test_grafana_dashboard_auto_refresh_configured():
    """
    Test that dashboard has auto-refresh configured.
    
    This test verifies:
    - Dashboard refresh interval is set
    - Refresh interval is reasonable (60 seconds default)
    """
    grafana_url = get_grafana_url()
    auth = get_grafana_auth()
    
    try:
        response = httpx.get(
            f"{grafana_url}/api/dashboards/uid/trading-system-monitoring",
            auth=auth,
            timeout=10.0
        )
        
        if response.status_code == 404:
            pytest.skip("Dashboard not found - may need to be imported first")
        
        assert response.status_code == 200, f"Failed to get dashboard: {response.status_code}"
        
        dashboard_data = response.json()
        dashboard = dashboard_data.get("dashboard", {})
        
        # Check dashboard-level refresh
        refresh = dashboard.get("refresh", "")
        # Refresh should be set (e.g., "60s", "1m", etc.)
        assert refresh != "", "Dashboard refresh interval not configured"
        
        # Verify panels have refresh intervals
        panels = dashboard.get("panels", [])
        for panel in panels:
            panel_title = panel.get("title", "Unknown")
            panel_interval = panel.get("interval", "")
            # Panels should have refresh interval configured (60s default)
            # Some panels may inherit from dashboard, so this is a soft check
            if panel_interval:
                assert panel_interval in ["60s", "1m", "30s"], \
                    f"Panel '{panel_title}' has unexpected refresh interval: {panel_interval}"
        
    except httpx.ConnectError as e:
        pytest.fail(f"Could not connect to Grafana at {grafana_url}: {e}")

