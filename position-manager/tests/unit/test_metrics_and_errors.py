from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_metrics_endpoint_ok() -> None:
    """Basic smoke test for /metrics endpoint."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "timestamp" in data
    assert "database_connected" in data
    assert "queue_connected" in data
    assert "portfolio_cache_hit_rate" in data
    assert "validation_statistics" in data


def test_error_schema_for_unauthorized() -> None:
    """Unauthorized request should return structured error envelope."""
    # Missing API key for a protected route
    resp = client.get("/api/v1/positions")
    assert resp.status_code == 401
    data = resp.json()
    # Either native HTTPException body or our global handler should expose a message
    assert "detail" in data or "message" in data


